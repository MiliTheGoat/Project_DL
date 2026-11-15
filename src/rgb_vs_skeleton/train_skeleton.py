"""Train CTR-GCN on the NTU skeletons (16 classes) from scratch, with live monitoring in wandb.

Run from the Project_DL folder:
    uv run src/rgb_vs_skeleton/train_skeleton.py               # uses configs/experiments/skeleton_experiments/exp1.yaml
    uv run src/rgb_vs_skeleton/train_skeleton.py +exp=exp2     # uses .../exp2.yaml

All hyper-parameters are in the experiment yaml. Paths, the class list, the split, the number of frames per clip and
the skeleton format (skeleton_format: "3d25" or "2d13") come from configs/main_config.yaml. The epoch loop is the same function as for the RGB model (train_RGB.run_one_epoch).

Saved in <output_dir>/<name>/ (same files and same summary keys as the RGB run, so the two can be compared):
    last.pt, best.pt, history.csv, summary.json
The test set is NOT used here. It is evaluated once, later, by the benchmark script.
"""
import json
import math
import random
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from loaders import skeleton_dataloader
from model_CTR_GCN import build_ctr_gcn, count_parameters, flops
from train_RGB import run_one_epoch, save_history, set_seed

EXPERIMENT_DIR = "configs/experiments/skeleton_experiments"


def learning_rate(epoch, exp):
    """Learning rate of one epoch (epoch counts from 0): linear warm-up, then step or cosine decay."""
    if epoch < exp.warmup_epochs:
        return exp.lr * (epoch + 1) / exp.warmup_epochs
    if exp.schedule == "step":
        steps_passed = sum(1 for step in exp.lr_steps if epoch >= step)
        return exp.lr * exp.lr_decay ** steps_passed
    if exp.schedule == "cosine":
        progress = (epoch - exp.warmup_epochs) / max(1, exp.epochs - exp.warmup_epochs)
        return exp.lr * 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError("schedule must be 'step' or 'cosine'")


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    exp_name = cfg.get("exp", "exp1")
    exp = OmegaConf.load(to_absolute_path(f"{EXPERIMENT_DIR}/{exp_name}.yaml"))
    print(f"[config] experiment file: {EXPERIMENT_DIR}/{exp_name}.yaml  (run name: {exp.name})")
    set_seed(exp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(exp.mixed_precision) and device.type == "cuda"
    actions = [int(c.ntu60) for c in cfg.classes]
    num_classes = len(actions)
    num_frames = cfg.skeleton_num_frames
    run_dir = Path(to_absolute_path(exp.output_dir)) / exp.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    loaders, datasets = skeleton_dataloader(
        skeleton_dir=to_absolute_path(cfg.processed_skeleton_FOLDER),
        cache_file=to_absolute_path(cfg.skeleton_cache_file), actions=actions,
        split_file=to_absolute_path(cfg.split_file), num_frames=num_frames,
        batch_size=exp.batch_size, num_workers=exp.num_workers, skeleton_format=cfg.skeleton_format)
    datasets["train"].train = bool(exp.augment)                       # switch the random rotation / scaling on or off
    train_loader, val_loader = loaders["train"], loaders["val"]
    if exp.train_fraction < 1.0:                                      # train on a random part of the training clips
        n_keep = int(len(datasets["train"]) * exp.train_fraction)
        keep = random.Random(exp.seed).sample(range(len(datasets["train"])), n_keep)
        train_loader = DataLoader(Subset(datasets["train"], keep), batch_size=exp.batch_size, shuffle=True,
                                  num_workers=exp.num_workers, pin_memory=True, drop_last=True)
    clip_shape = tuple(datasets["train"][0][0].shape)                 # (3, T, 25, 2) or (2, T, 13, 2)
    print(f"[data] train {len(train_loader.dataset)} clips, val {len(val_loader.dataset)} clips, "
          f"skeleton format {cfg.skeleton_format}, clip shape {clip_shape}, device {device}")

    # ---- model, loss, optimizer
    model = build_ctr_gcn(num_classes, dropout=exp.dropout, skeleton_format=cfg.skeleton_format)
    total_params = count_parameters(model)
    gflops = flops(model.eval(), torch.zeros(1, *clip_shape))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=exp.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=exp.lr, momentum=exp.momentum, nesterov=exp.nesterov,
                                weight_decay=exp.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- how many epochs, and where to continue from
    last_epoch = exp.epochs if exp.test_n_epochs == 0 else min(exp.test_n_epochs, exp.epochs)
    checkpoint_path = run_dir / "last.pt"
    history, best_acc, wandb_id = [], 0.0, None
    if checkpoint_path.exists() and not exp.restart:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_acc, wandb_id = checkpoint["history"], checkpoint["best_acc"], checkpoint["wandb_id"]
        print(f"[resume] found {checkpoint_path}: {len(history)} epochs already done, best val acc {best_acc:.3f}")
    start_epoch = len(history)
    if start_epoch >= last_epoch:
        print(f"[done] {start_epoch} epochs are already trained and this run stops at epoch {last_epoch}. "
              f"Raise test_n_epochs (or set it to 0 for the full {exp.epochs} epochs) to train more.")
        return
    print(f"[plan] training epochs {start_epoch + 1}..{last_epoch} of a {exp.epochs}-epoch schedule | "
          f"{total_params:,} parameters, all trained from scratch")

    # ---- wandb (the same run is continued after a restart of the script)
    run = wandb.init(project=exp.wandb_project, name=exp.name, id=wandb_id, resume="allow", mode=exp.wandb_mode,
                     config=OmegaConf.to_container(exp))
    wandb.config.update({"params": total_params, "gflops": gflops, "train_clips": len(train_loader.dataset)},
                        allow_val_change=True)

    for epoch in range(start_epoch, last_epoch):
        lr = learning_rate(epoch, exp)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train = run_one_epoch(model, train_loader, criterion, device, num_classes, use_amp, optimizer, scaler,
                              exp.log_every_steps)
        val = run_one_epoch(model, val_loader, criterion, device, num_classes, use_amp)
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0

        row = {"epoch": epoch + 1, "stage": "all", "lr": lr,
               "train_loss": train["loss"], "train_acc": train["acc"],
               "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1"],
               "train_time_s": train["time"], "val_time_s": val["time"], "data_time_s": train["data_time"],
               "train_clips_per_s": train["clips"] / train["time"], "gpu_memory_mb": gpu_memory}
        history.append(row)
        wandb.log({"epoch": row["epoch"], "lr": lr, "train/loss": row["train_loss"], "train/acc": row["train_acc"],
                   "val/loss": row["val_loss"], "val/acc": row["val_acc"], "val/f1": row["val_f1"],
                   "time/train_epoch_s": row["train_time_s"], "time/data_wait_s": row["data_time_s"],
                   "time/train_clips_per_s": row["train_clips_per_s"], "gpu_memory_mb": gpu_memory})
        print(f"epoch {epoch + 1:>2}/{last_epoch} lr {lr:.4f} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
              f" | val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} f1 {row['val_f1']:.3f} | {row['train_time_s']:.0f} s")

        if val["acc"] > best_acc:                                     # keep the best model
            best_acc = val["acc"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "history": history, "best_acc": best_acc, "wandb_id": run.id}, checkpoint_path)
        save_history(history, run_dir / "history.csv")

    # ---- numbers for the comparison with the RGB model (same keys as in train_RGB.py)
    best_row = max(history, key=lambda r: r["val_acc"])
    train_time = sum(r["train_time_s"] for r in history)
    summary = {
        "name": exp.name, "model": "CTR-GCN", "input": "x".join(map(str, clip_shape)),
        "skeleton_format": cfg.skeleton_format,
        "parameters": total_params, "gflops_per_clip": gflops,
        "epochs_trained": len(history), "schedule_epochs": exp.epochs, "train_clips": len(train_loader.dataset),
        "total_train_time_s": train_time, "mean_epoch_time_s": train_time / len(history),
        "time_to_best_epoch_s": sum(r["train_time_s"] for r in history[:best_row["epoch"]]),
        "mean_train_clips_per_s": sum(r["train_clips_per_s"] for r in history) / len(history),
        "data_wait_share": sum(r["data_time_s"] for r in history) / train_time,
        "peak_gpu_memory_mb": max(r["gpu_memory_mb"] for r in history),
        "best_epoch": best_row["epoch"], "best_val_acc": best_row["val_acc"], "best_val_f1": best_row["val_f1"],
        "final_train_acc": history[-1]["train_acc"], "final_val_acc": history[-1]["val_acc"],
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for key, value in summary.items():
        wandb.summary[key] = value
    wandb.finish()
    print(f"\n[saved] {run_dir}: last.pt, best.pt, history.csv, summary.json")
    print(f"[result] best val acc {summary['best_val_acc']:.3f} at epoch {summary['best_epoch']} | "
          f"training time {train_time / 60:.1f} min | {summary['mean_train_clips_per_s']:.1f} clips/s")


if __name__ == "__main__":
    main()