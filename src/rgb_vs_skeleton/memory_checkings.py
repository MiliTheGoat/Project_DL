"""Disk-size report of all project data: what is in every folder / file, how much shrinking saved, and the totals.

Run from the Project_DL folder (all paths are read from configs/main_config.yaml):
    uv run src/rgb_vs_skeleton/memory_checkings.py
    uv run src/rgb_vs_skeleton/memory_checkings.py +mem_per_class=true      # also a table per class for every folder
"""
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

NTU_ACTION_RE = re.compile(r"A(\d{3})(?:\.|_)")          # ...R001A008.avi -> A008
GB = 1024 ** 3
MB = 1024 ** 2


def scan(path):
    """Folder or single file -> dict(files, bytes, types, per_class), or None if it does not exist.
    types: e.g. {'.avi': 15050}.  per_class: NTU action id (A008) or HMDB sub-folder -> [files, bytes]."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return {"files": 1, "bytes": path.stat().st_size, "types": Counter([path.suffix]), "per_class": {}}
    n_files, total, types = 0, 0, Counter()
    per_class = defaultdict(lambda: [0, 0])
    for root, _, files in os.walk(path):
        for name in files:
            if name.endswith((".part", ".tmp")) or "_part." in name:      # unfinished files are not data
                continue
            size = os.path.getsize(os.path.join(root, name))
            match = NTU_ACTION_RE.search(name)
            if match:
                key = f"A{match.group(1)}"
            elif Path(root) != path:
                key = Path(root).relative_to(path).parts[0]
            else:
                key = "(other)"
            per_class[key][0] += 1
            per_class[key][1] += size
            types[Path(name).suffix] += 1
            n_files += 1
            total += size
    return {"files": n_files, "bytes": total, "types": types, "per_class": dict(per_class)}


def describe_types(types):
    """{'.avi': 15050} -> '15,050 .avi'"""
    return ", ".join(f"{n:,} {suffix or '(no ending)'}" for suffix, n in types.most_common(3))


def print_per_class(label, info):
    print(f"\n    [{label}] per class")
    print(f"    {'class':<14}{'files':>8}{'size':>14}")
    for key in sorted(info["per_class"]):
        n, size = info["per_class"][key]
        print(f"    {key:<14}{n:>8,}{size / MB:>11,.1f} MB")


def print_saving(name, before, after):
    if before and after and after["bytes"] > 0:
        print(f"  {name:<10}{before['bytes'] / GB:>8.2f} GB -> {after['bytes'] / GB:>7.3f} GB   "
              f"{before['bytes'] / after['bytes']:>6.1f}x smaller   (saves {(before['bytes'] - after['bytes']) / GB:.2f} GB)")
    else:
        print(f"  {name:<10}not shrunk yet (run shrink_data.py)")


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    # (label, path from the yaml, goes to Google Drive?)
    items = [
        ("NTU RGB original",        cfg.processed_RGB_FOLDER,      False),
        ("NTU RGB small",           cfg.shrunk_RGB_FOLDER,         False),
        ("NTU RGB zip",             cfg.shrunk_RGB_ZIP,            True),
        ("NTU skeleton text",       cfg.processed_skeleton_FOLDER, False),
        ("NTU skeleton cache",      cfg.skeleton_cache_file,       True),
        ("HMDB-51",                 cfg.processed_HMDB1_FOLDER,    True),
        ("split file",              cfg.split_file,                True),
    ]
    results = {}
    print(f"\n{'data':<22}{'what is there':<34}{'size':>12}   {'for Colab':<10}path")
    print("-" * 110)
    for label, relative_path, for_colab in items:
        info = scan(to_absolute_path(relative_path))
        results[label] = info
        if info is None:
            print(f"{label:<22}{'(not there yet)':<34}{'-':>12}   {'':<10}{relative_path}")
        else:
            size = f"{info['bytes'] / GB:.2f} GB" if info["bytes"] >= 0.01 * GB else f"{info['bytes'] / MB:.2f} MB"
            print(f"{label:<22}{describe_types(info['types']):<34}{size:>12}   {'yes' if for_colab else '':<10}{relative_path}")

    if cfg.get("mem_per_class", False):
        for label, info in results.items():
            if info and len(info["per_class"]) > 1:
                print_per_class(label, info)

    print("\nHOW MUCH SHRINKING HELPED")
    print_saving("RGB", results["NTU RGB original"], results["NTU RGB small"])
    print_saving("skeleton", results["NTU skeleton text"], results["NTU skeleton cache"])

    on_disk = sum(info["bytes"] for info in results.values() if info)
    for_colab = sum(results[label]["bytes"] for label, _, colab in items if colab and results[label])
    originals = sum(results[label]["bytes"] for label in ("NTU RGB original", "NTU skeleton text") if results[label])
    print("\nTOTALS")
    print(f"  everything listed above, on this computer : {on_disk / GB:>8.2f} GB")
    print(f"  original NTU data (RGB + skeleton text)   : {originals / GB:>8.2f} GB")
    print(f"  to upload to Google Drive (marked 'yes')  : {for_colab / GB:>8.2f} GB")
    print(f"  free space on this drive                  : {shutil.disk_usage(to_absolute_path('.')).free / GB:>8.1f} GB")


if __name__ == "__main__":
    main()