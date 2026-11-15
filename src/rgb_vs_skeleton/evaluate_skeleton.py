"""Evaluate a trained CTR-GCN on a TEST set: accuracy, per-class accuracy, and time per video sample.

Run from the Project_DL folder (in a notebook: put a ! in front and use python instead of uv run):
    uv run src/rgb_vs_skeleton/evaluate_skeleton.py +exp=exp1                    # NTU test set (the 20 test performers)
    uv run src/rgb_vs_skeleton/evaluate_skeleton.py +exp=exp1 +test_on=hmdb      # HMDB-51 skeletons (cross-dataset test)

The experiment yaml (configs/experiments/skeleton_experiments/<exp>.yaml) tells which model folder to use:
<output_dir>/<name>/best.pt. Results are printed and saved next to the model: test_results.json (NTU) or
test_results_hmdb.json (HMDB). Use a test set ONCE per final model; choose models with the validation set only.

+test_on=hmdb needs the skeletons made by pose_estimation.py (<skeleton_estimates_HMDB1_FOLDER>/hmdb_skeleton_2d13.npz)
and a model trained with skeleton_format: 2d13. The cost of pose estimation (pose_summary.json) is added to the timing,
because for a video the skeleton pipeline is pose estimation + CTR-GCN.
"""
import json
import time
from collections import Counter
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from loaders import hmdb_skeleton_dataloader, skeleton_dataloader
from model_CTR_GCN import build_ctr_gcn, count_parameters
from train_RGB import accuracy_and_f1

EXPERIMENT_DIR = "configs/experiments/skeleton_experiments"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict_test_set(model, loader, device, use_amp):
    """Run the whole test set once. Returns labels, predictions and the time spent in the model."""
    labels, predictions, model_time = [], [], 0.0
    for step, (clips, batch_labels) in enumerate(loader, 1):
        clips = clips.to(device)
        sync(device)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(clips)
        sync(device)
        model_time += time.perf_counter() - start
        labels += batch_labels.tolist()
        predictions += outputs.argmax(dim=1).tolist()
        if step % 10 == 0 or step == len(loader):
            running_acc = sum(p == y for p, y in zip(predictions, labels)) / len(labels)
            print(f"  [{len(labels):>5}/{len(loader.dataset)} clips]  running accuracy {running_acc:.3f}")
    return labels, predictions, model_time


@torch.no_grad()
def time_one_sample(model, dataset, device, n=100):
    """Latency of ONE video sample (batch size 1), as a deployed model would see it. -> milliseconds per clip."""
    clips = [dataset[i][0].unsqueeze(0).to(device) for i in range(min(n, len(dataset)))]
    for clip in clips[:10]:                                   # warm-up: the first calls are always slower
        model(clip)
    sync(device)
    times = []
    for clip in clips:
        start = time.perf_counter()
        model(clip)
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.95) - 1]       # median and 95th percentile


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    exp_name = cfg.get("exp", "exp1")
    test_on = cfg.get("test_on", "ntu")
    exp = OmegaConf.load(to_absolute_path(f"{EXPERIMENT_DIR}/{exp_name}.yaml"))
    run_dir = Path(to_absolute_path(exp.output_dir)) / exp.name
    weights = run_dir / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found - train experiment '{exp_name}' first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(exp.mixed_precision) and device.type == "cuda"
    actions = [int(c.ntu60) for c in cfg.classes]
    class_names = [f"A{c.ntu60:03d} {c.hmdb}" for c in cfg.classes]
    print(f"[evaluate] experiment {exp_name} -> {weights} | device {device} | test on {test_on}")

    # ---- data: only a test set is used
    pose = None                                              # cost of pose estimation, only known for HMDB
    if test_on == "ntu":
        loaders, datasets = skeleton_dataloader(
            skeleton_dir=to_absolute_path(cfg.processed_skeleton_FOLDER),
            cache_file=to_absolute_path(cfg.skeleton_cache_file), actions=actions,
            split_file=to_absolute_path(cfg.split_file), num_frames=cfg.skeleton_num_frames,
            batch_size=exp.batch_size, num_workers=0,        # skeletons are already in RAM: extra worker processes
            skeleton_format=cfg.skeleton_format)             # only copy them (and run out of memory on Windows)
        test_loader, test_set = loaders["test"], datasets["test"]
        performers = sorted({name[8:12] for name in test_set.names})
        print(f"[data] NTU test set: {len(test_set)} clips, {len(performers)} performers, {len(actions)} classes, "
              f"skeleton format {cfg.skeleton_format}")
        results_file = run_dir / "test_results.json"
    elif test_on == "hmdb":
        estimates_dir = Path(to_absolute_path(cfg.skeleton_estimates_HMDB1_FOLDER))
        skeleton_file = estimates_dir / "hmdb_skeleton_2d13.npz"
        if cfg.skeleton_format != "2d13":
            raise ValueError("HMDB skeletons exist only in the 2d13 format: set skeleton_format: 2d13 and use a model trained with it")
        if not skeleton_file.exists():
            raise FileNotFoundError(f"{skeleton_file} not found - run pose_estimation.py first")
        test_loader, test_set = hmdb_skeleton_dataloader(skeleton_file, [str(c.hmdb) for c in cfg.classes], exp.batch_size)
        if test_set.x.shape[2] != cfg.skeleton_num_frames:
            raise ValueError(f"the HMDB skeletons have {test_set.x.shape[2]} frames per clip, the yaml says {cfg.skeleton_num_frames}")
        print(f"[data] HMDB-51 skeletons: {len(test_set)} clips with a good skeleton, {len(actions)} classes, from {skeleton_file}")
        if (estimates_dir / "pose_summary.json").exists():
            pose = json.loads((estimates_dir / "pose_summary.json").read_text())
            print(f"[data] they are {pose['good_clips']} of {pose['clips']} HMDB clips ({pose['good_share']:.0%}); "
                  f"the others had no good skeleton and are NOT counted in the accuracy below")
        results_file = run_dir / "test_results_hmdb.json"
    else:
        raise ValueError("test_on must be 'ntu' or 'hmdb'")

    # ---- model: the weights of the best validation epoch
    model = build_ctr_gcn(len(actions), dropout=exp.dropout, skeleton_format=cfg.skeleton_format)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.to(device).eval()
    history_file = run_dir / "history.csv"
    if history_file.exists():
        rows = history_file.read_text().strip().split("\n")[1:]
        print(f"[model] {count_parameters(model):,} parameters | trained {len(rows)} epochs")

    # ---- 1. the whole test set
    print("\n[1/2] predicting the test set ...")
    total_start = time.perf_counter()
    labels, predictions, model_time = predict_test_set(model, test_loader, device, use_amp)
    total_time = time.perf_counter() - total_start
    accuracy, macro_f1 = accuracy_and_f1(labels, predictions, len(actions))

    # ---- 2. one video sample at a time
    print("\n[2/2] timing one video sample at a time (batch size 1) ...")
    median_ms, p95_ms = time_one_sample(model, test_set, device)

    # ---- results
    print("\n" + "=" * 70)
    print(f"TEST RESULTS  {exp.name}  (CTR-GCN on {test_on.upper()}, {len(labels)} clips)")
    print("=" * 70)
    print(f"  accuracy        {accuracy:.4f}   ({sum(p == y for p, y in zip(predictions, labels))} of {len(labels)} correct)")
    print(f"  macro F1        {macro_f1:.4f}")
    print("\n  per class        correct / clips   accuracy")
    per_class = {}
    for k, name in enumerate(class_names):
        clips_k = [p for p, y in zip(predictions, labels) if y == k]
        correct = sum(p == k for p in clips_k)
        per_class[name] = correct / max(1, len(clips_k))
        print(f"  {name:<18}{correct:>5} / {len(clips_k):<5}    {per_class[name]:.3f}")
    mistakes = Counter((y, p) for p, y in zip(predictions, labels) if p != y)
    print("\n  most common mistakes (true class -> predicted)")
    for (y, p), n in mistakes.most_common(5):
        print(f"  {class_names[y]:<18} -> {class_names[p]:<18} {n:>4} clips")
    print("\n  timing")
    print(f"  one video sample (batch 1)  {median_ms:.2f} ms median, {p95_ms:.2f} ms for 95% of clips")
    print(f"  whole test set              {total_time:.1f} s total, {model_time:.1f} s of it in the model")
    print(f"  throughput                  {len(labels) / model_time:.0f} clips/s in the model (batch {exp.batch_size})")
    if pose:
        print(f"  pose estimation             {pose['pose_ms_per_clip']:.0f} ms per clip ({pose['pose_ms_per_frame']:.1f} ms x "
              f"{pose['pose_frames_per_clip']:.0f} frames, {pose['pose_model']} on {pose['device']})")
        print(f"  whole skeleton pipeline     about {pose['pose_ms_per_clip'] + median_ms:.0f} ms per video = pose estimation + CTR-GCN")
    print("=" * 70)

    results = {"name": exp.name, "model": "CTR-GCN", "experiment": exp_name, "test_on": test_on, "test_clips": len(labels),
               "skeleton_format": cfg.skeleton_format,
               "test_accuracy": accuracy, "test_macro_f1": macro_f1, "per_class_accuracy": per_class,
               "latency_ms_batch1_median": median_ms, "latency_ms_batch1_p95": p95_ms,
               "test_total_time_s": total_time, "test_model_time_s": model_time,
               "throughput_clips_per_s": len(labels) / model_time, "batch_size": exp.batch_size,
               "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    if pose:
        results.update({"hmdb_clips_total": pose["clips"], "hmdb_good_share": pose["good_share"],
                        "pose_model": pose["pose_model"], "pose_device": pose["device"],
                        "pose_ms_per_clip": pose["pose_ms_per_clip"],
                        "pipeline_ms_per_video": pose["pose_ms_per_clip"] + median_ms})
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {results_file}")


if __name__ == "__main__":
    main()