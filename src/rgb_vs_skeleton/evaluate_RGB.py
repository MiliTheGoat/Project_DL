"""Evaluate a trained X3D-M on a TEST set: accuracy, per-class accuracy, and time per video sample.

Run from the Project_DL folder (in a notebook: put a ! in front):
    python src/rgb_vs_skeleton/evaluate_RGB.py +exp=exp1                    # NTU test set (the 20 test performers)
    python src/rgb_vs_skeleton/evaluate_RGB.py +exp=exp1 +test_on=hmdb      # HMDB-51 clips (cross-dataset test)

The experiment yaml tells which model folder to use: <output_dir>/<name>/best.pt (the best validation epoch).
Results are printed and saved next to the model: test_results.json (NTU) or test_results_hmdb.json (HMDB), with the
same keys as the skeleton results, so the two models can be compared.
Use a test set ONCE per final model; choose models and settings with the validation set only.

Where the videos come from: the first of these folders that holds every clip of the split
    shrunk_RGB_FOLDER (small .mp4 on your computer) -> /content/data/rgb (unpacked by the Colab notebook) -> processed_RGB_FOLDER
The small videos are what the model was trained on in Colab, so they are also the right ones for testing.
"""
import csv
import json
import time
from collections import Counter
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from evaluate_skeleton import predict_test_set, time_one_sample
from loaders import hmdb_dataloader, rgb_dataloader
from model_X3D_M import build_x3d_m, count_parameters
from train_RGB import EXPERIMENT_DIR, accuracy_and_f1


def find_video_folder(cfg):
    """The first folder that holds a video for EVERY clip of the split file (small versions first)."""
    with open(to_absolute_path(cfg.split_file), newline="") as f:
        needed = {row["name"] for row in csv.DictReader(f)}
    candidates = [to_absolute_path(cfg.shrunk_RGB_FOLDER), "/content/data/rgb", to_absolute_path(cfg.processed_RGB_FOLDER)]
    for folder in candidates:
        folder = Path(folder)
        if not folder.exists():
            continue
        found = {p.stem for p in folder.glob("*.mp4")} | {p.stem for p in folder.glob("*.avi")}
        if needed <= found:
            return folder
        if found:
            print(f"[data] skipped {folder}: it has only {len(needed & found)} of the {len(needed)} clips of the split")
    raise FileNotFoundError(f"no folder with all {len(needed)} clips of the split; looked in {candidates}")


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

    # ---- data: only a test set is used, in exactly the clip format the model was trained with
    if test_on == "ntu":
        video_dir = find_video_folder(cfg)
        loaders, datasets = rgb_dataloader(
            video_dir=video_dir, actions=actions, split_file=to_absolute_path(cfg.split_file),
            num_frames=exp.num_frames, side=exp.side, size=exp.size,
            batch_size=exp.batch_size, num_workers=exp.num_workers)
        test_loader, test_set = loaders["test"], datasets["test"]
        performers = sorted({name[8:12] for name in test_set.names})
        print(f"[data] NTU test set: {len(test_set)} clips, {len(performers)} performers, videos from {video_dir}")
        results_file = run_dir / "test_results.json"
    elif test_on == "hmdb":
        video_dir = Path(to_absolute_path(cfg.processed_HMDB1_FOLDER))
        test_loader, test_set = hmdb_dataloader(
            hmdb_dir=video_dir, hmdb_classes=[str(c.hmdb) for c in cfg.classes],
            num_frames=exp.num_frames, side=exp.side, size=exp.size,
            batch_size=exp.batch_size, num_workers=exp.num_workers)
        print(f"[data] HMDB-51: {len(test_set)} clips of our {len(actions)} classes, videos from {video_dir}")
        results_file = run_dir / "test_results_hmdb.json"
    else:
        raise ValueError("test_on must be 'ntu' or 'hmdb'")

    # ---- model: the weights of the best validation epoch (no Kinetics download needed, they are replaced anyway)
    model = build_x3d_m(len(actions), pretrained=False, dropout=exp.dropout)
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
    print(f"TEST RESULTS  {exp.name}  (X3D-M on {test_on.upper()}, {len(labels)} clips)")
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
    video_ms = 1000 * (total_time - model_time) / len(labels)
    print("\n  timing")
    print(f"  one video sample (batch 1)  {median_ms:.2f} ms median, {p95_ms:.2f} ms for 95% of clips   (model only)")
    print(f"  whole test set              {total_time:.1f} s total, {model_time:.1f} s of it in the model")
    print(f"  throughput                  {len(labels) / model_time:.0f} clips/s in the model (batch {exp.batch_size})")
    print(f"  reading the videos          about {video_ms:.0f} ms per clip (decoding and resizing, {exp.num_workers} workers)")
    print("=" * 70)

    results = {"name": exp.name, "model": "X3D-M", "experiment": exp_name, "test_on": test_on, "test_clips": len(labels),
               "input": f"3x{exp.num_frames}x{exp.size}x{exp.size}",
               "test_accuracy": accuracy, "test_macro_f1": macro_f1, "per_class_accuracy": per_class,
               "latency_ms_batch1_median": median_ms, "latency_ms_batch1_p95": p95_ms,
               "test_total_time_s": total_time, "test_model_time_s": model_time,
               "throughput_clips_per_s": len(labels) / model_time, "batch_size": exp.batch_size,
               "video_reading_ms_per_clip": video_ms,
               "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {results_file}")


if __name__ == "__main__":
    main()