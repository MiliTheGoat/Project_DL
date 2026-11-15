"""Train X3D-M on the NTU RGB clips (16 classes), with live monitoring in wandb.

Run from the Project_DL folder:
    uv run src/rgb_vs_skeleton/train_RGB.py                 # uses configs/experiments/RGB_/experiment/exp1.yaml
    uv run src/rgb_vs_skeleton/train_RGB.py +exp=exp2       # uses .../exp2.yaml

All hyper-parameters are in the experiment yaml. Paths, the class list and the split come from configs/main_config.yaml.

What is saved in <output_dir>/<name>/ :
    last.pt        everything needed to continue (model, optimizer, epoch, history) - written after every epoch
    best.pt        the model weights of the epoch with the best validation accuracy
    history.csv    one row per epoch: losses, accuracies, learning rate, times, clips per second, GPU memory
    summary.json   the numbers for the RGB-vs-skeleton comparison (parameters, GFLOPs, training time, best accuracy, ...)
The test set is NOT used here. It is evaluated once, later, by the benchmark script.
"""
import csv
import json
import math
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from loaders import rgb_dataloader
from model_X3D_M import build_x3d_m, count_parameters, flops, set_trainable

EXPERIMENT_DIR = "configs/experiments/RGB_experiments"


# ------------------------------------------------------------------------------------------------ small helpers
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_plan(stages):
    """Turn the stages of the yaml into one entry per epoch: which part is trained and with which learning rate.
    Inside a stage the learning rate follows a cosine curve from `lr` down towards 0."""
    plan = []
    for stage in stages:
        for k in range(stage.epochs):
            lr = stage.lr * 0.5 * (1 + math.cos(math.pi * k / stage.epochs))
            plan.append({"stage": stage.train, "lr": lr})
    return plan


def freeze_batchnorm_of_frozen_layers(model):
    """Frozen layers must keep their Kinetics batch-norm statistics, so their batch-norm layers stay in eval mode."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d) and not module.weight.requires_grad:
            module.eval()


def accuracy_and_f1(labels, predictions, num_classes):
    """labels, predictions: lists of class numbers -> (accuracy, macro F1)."""
    confusion = torch.zeros(num_classes, num_classes)
    for label, prediction in zip(labels, predictions):
        confusion[label, prediction] += 1
    correct = confusion.diag()
    precision = correct / confusion.sum(dim=0).clamp(min=1)
    recall = correct / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    return (correct.sum() / confusion.sum()).item(), f1.mean().item()


def run_one_epoch(model, loader, criterion, device, num_classes, use_amp, optimizer=None, scaler=None, log_every=0):
    """One pass over `loader`. With an optimizer the model is trained, without it the model is only evaluated.
    Returns a dict with loss, accuracy, macro F1, total time, time spent waiting for data, and the number of clips."""
    training = optimizer is not None
    if training:
        model.train()
        freeze_batchnorm_of_frozen_layers(model)
    else:
        model.eval()

    total_loss, all_labels, all_predictions = 0.0, [], []
    data_time, start = 0.0, time.perf_counter()
    wait_start = time.perf_counter()
    for step, (clips, labels) in enumerate(loader, 1):
        data_time += time.perf_counter() - wait_start                 # time spent decoding videos
        clips, labels = clips.to(device), labels.to(device)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(clips)
                loss = criterion(outputs.float(), labels)
        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if log_every and step % log_every == 0:
                wandb.log({"train/step_loss": loss.item()})
                print(f"    step {step}/{len(loader)}  loss {loss.item():.3f}")

        total_loss += loss.item() * len(labels)
        all_labels += labels.tolist()
        all_predictions += outputs.argmax(dim=1).tolist()
        wait_start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    accuracy, f1 = accuracy_and_f1(all_labels, all_predictions, num_classes)
    return {"loss": total_loss / len(all_labels), "acc": accuracy, "f1": f1,
            "time": time.perf_counter() - start, "data_time": data_time, "clips": len(all_labels)}


def save_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# ------------------------------------------------------------------------------------------------ training
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
    run_dir = Path(to_absolute_path(exp.output_dir)) / exp.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    loaders, datasets = rgb_dataloader(
        video_dir=to_absolute_path(cfg.processed_RGB_FOLDER), actions=actions,
        split_file=to_absolute_path(cfg.split_file), num_frames=exp.num_frames, side=exp.side, size=exp.size,
        batch_size=exp.batch_size, num_workers=exp.num_workers)
    train_loader, val_loader = loaders["train"], loaders["val"]
    if exp.train_fraction < 1.0:                                      # train on a random part of the training clips
        n_keep = int(len(datasets["train"]) * exp.train_fraction)
        keep = random.Random(exp.seed).sample(range(len(datasets["train"])), n_keep)
        train_loader = DataLoader(Subset(datasets["train"], keep), batch_size=exp.batch_size, shuffle=True,
                                  num_workers=exp.num_workers, pin_memory=True, drop_last=True)
    print(f"[data] train {len(train_loader.dataset)} clips, val {len(val_loader.dataset)} clips, "
          f"clip shape (3, {exp.num_frames}, {exp.size}, {exp.size}), device {device}")

    # ---- model
    model = build_x3d_m(num_classes, pretrained=exp.pretrained, dropout=exp.dropout)
    total_params = count_parameters(model)
    gflops = flops(model.eval(), torch.zeros(1, 3, exp.num_frames, exp.size, exp.size))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=exp.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ---- how many epochs, and where to continue from
    plan = make_plan(exp.stages)
    last_epoch = len(plan) if exp.test_n_epochs == 0 else min(exp.test_n_epochs, len(plan))
    checkpoint_path = run_dir / "last.pt"
    checkpoint, history, best_acc, wandb_id, start_epoch = None, [], 0.0, None, 0
    if checkpoint_path.exists() and not exp.restart:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        history, best_acc, wandb_id = checkpoint["history"], checkpoint["best_acc"], checkpoint["wandb_id"]
        start_epoch = len(history)
        print(f"[resume] found {checkpoint_path}: {start_epoch} epochs already done, best val acc {best_acc:.3f}")
    if start_epoch >= last_epoch:
        print(f"[done] {start_epoch} epochs are already trained and this run stops at epoch {last_epoch}. "
              f"Raise test_n_epochs (or set it to 0 for the full {len(plan)} epochs) to train more.")
        return
    print(f"[plan] training epochs {start_epoch + 1}..{last_epoch} of a {len(plan)}-epoch schedule")

    # ---- wandb (the same run is continued after a restart of the script)
    run = wandb.init(project=exp.wandb_project, name=exp.name, id=wandb_id, resume="allow", mode=exp.wandb_mode,
                     config=OmegaConf.to_container(exp))
    wandb.config.update({"params": total_params, "gflops": gflops, "train_clips": len(train_loader.dataset)},
                        allow_val_change=True)

    optimizer, current_stage = None, None
    for epoch in range(start_epoch, last_epoch):
        stage, lr = plan[epoch]["stage"], plan[epoch]["lr"]
        if stage != current_stage:                                    # a new stage: unfreeze layers, new optimizer
            set_trainable(model, stage)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=exp.weight_decay)
            if checkpoint is not None and epoch == start_epoch and checkpoint["stage"] == stage:
                optimizer.load_state_dict(checkpoint["optimizer"])    # continuing in the middle of a stage
            current_stage = stage
            print(f"[stage] '{stage}': {count_parameters(model, trainable_only=True):,} of {total_params:,} parameters are trained")
        for group in optimizer.param_groups:
            group["lr"] = lr

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train = run_one_epoch(model, train_loader, criterion, device, num_classes, use_amp, optimizer, scaler,
                              exp.log_every_steps)
        val = run_one_epoch(model, val_loader, criterion, device, num_classes, use_amp)
        gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0

        row = {"epoch": epoch + 1, "stage": stage, "lr": lr,
               "train_loss": train["loss"], "train_acc": train["acc"],
               "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1"],
               "train_time_s": train["time"], "val_time_s": val["time"], "data_time_s": train["data_time"],
               "train_clips_per_s": train["clips"] / train["time"], "gpu_memory_mb": gpu_memory}
        history.append(row)
        wandb.log({"epoch": row["epoch"], "lr": lr, "train/loss": row["train_loss"], "train/acc": row["train_acc"],
                   "val/loss": row["val_loss"], "val/acc": row["val_acc"], "val/f1": row["val_f1"],
                   "time/train_epoch_s": row["train_time_s"], "time/data_wait_s": row["data_time_s"],
                   "time/train_clips_per_s": row["train_clips_per_s"], "gpu_memory_mb": gpu_memory})
        print(f"epoch {epoch + 1:>2}/{last_epoch} [{stage}] lr {lr:.5f} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f}"
              f" | val loss {row['val_loss']:.3f} acc {row['val_acc']:.3f} f1 {row['val_f1']:.3f}"
              f" | {row['train_time_s']:.0f} s ({row['data_time_s'] / row['train_time_s']:.0%} waiting for data)")

        if val["acc"] > best_acc:                                     # keep the best model
            best_acc = val["acc"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "stage": stage,
                    "history": history, "best_acc": best_acc, "wandb_id": run.id}, checkpoint_path)
        save_history(history, run_dir / "history.csv")

    # ---- numbers for the comparison with the skeleton model
    best_row = max(history, key=lambda r: r["val_acc"])
    train_time = sum(r["train_time_s"] for r in history)
    summary = {
        "name": exp.name, "model": "X3D-M", "input": f"3x{exp.num_frames}x{exp.size}x{exp.size}",
        "parameters": total_params, "gflops_per_clip": gflops,
        "epochs_trained": len(history), "schedule_epochs": len(plan), "train_clips": len(train_loader.dataset),
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