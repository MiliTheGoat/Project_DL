"""Skeletons for HMDB-51 from a pose estimator, in the shared "2d13" format, keeping only the clips with a good skeleton.

Run from the Project_DL folder (all settings are in configs/main_config.yaml):
    uv run src/rgb_vs_skeleton/pose_estimation.py                      # estimate, select the good clips, print a report
    uv run src/rgb_vs_skeleton/pose_estimation.py hmdb_prune=true      # ... and move the other RGB clips out of the HMDB folder
Needs:  uv add ultralytics      (the pose model is downloaded on first use; a GPU is much faster than a CPU)

The three steps
  1. ESTIMATE (slow, done once)  The pose model looks at skeleton_num_frames frames of every HMDB clip and gives, for every
     person, 17 COCO keypoints with a confidence. We follow the people over the frames, keep the one or two that are
     biggest and clearest, and store their 13 shared joints. Saved per class in <output>/raw/<class>.npz, so a class
     that is finished is not estimated again.
  2. SELECT (fast, repeat as often as you like)  A clip is GOOD when a person is found in enough frames, enough of the
     13 joints are found, and the torso (both shoulders and both hips) is visible. Clips that show only a head or half a
     body fail. Good clips are normalised with the same function as the NTU 2D skeletons and saved together:
         <output>/hmdb_skeleton_2d13.npz   x (clips, 2, T, 13, 2), names, classes     <- the cross-dataset test set
         <output>/pose_quality.csv         every clip: its numbers, good or not, and why not
         <output>/pose_summary.json        good clips per class, and the cost of pose estimation (for the report)
     Changing the thresholds in the yaml only repeats this step.
  3. PRUNE (only with hmdb_prune=true)  The RGB clips WITHOUT a good skeleton are MOVED from the HMDB folder to
     excluded_HMDB1_FOLDER (same class sub-folders), so the RGB model is tested on exactly the same clips. Nothing is
     deleted: every run first moves all excluded clips back, so new thresholds give a new, correct selection.
"""
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from skeleton_format import COCO17_TO_13, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, normalize_2d

MIN_GOOD_SHARE_TO_PRUNE = 0.6        # pruning is refused if fewer than half of all clips have a good skeleton


# ------------------------------------------------------------------------------------------------ step 1: estimate
def read_frames(path, num_frames):
    """num_frames frames spread evenly over the video -> (list of different frames, position of each wanted frame in it).
    A video with fewer frames than num_frames gives every frame once; `positions` then repeats some of them."""
    video = cv2.VideoCapture(str(path))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        video.release()
        raise ValueError("cannot read the video")
    wanted = np.linspace(0, n_frames - 1, num_frames).round().astype(int)
    frames, index_of = [], {}
    for t in range(wanted.max() + 1):
        ok, frame = video.read()
        if not ok:
            break
        if t in wanted:
            index_of[t] = len(frames)
            frames.append(frame)
    video.release()
    if not frames:
        raise ValueError("no frame could be read")
    last = len(frames) - 1
    positions = [index_of.get(t, last) for t in wanted]
    return frames, positions


def detect_people(model, frames, device):
    """Run the pose model. -> for every frame a list of people, each (box x1 y1 x2 y2, keypoints (17, 3) = x, y, confidence)."""
    people = []
    for start in range(0, len(frames), 16):                               # 16 frames at a time
        for result in model.predict(frames[start:start + 16], verbose=False, device=device):
            if result.keypoints is None or len(result.boxes) == 0:
                people.append([])
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            keypoints = result.keypoints.data.cpu().numpy()
            people.append([(boxes[i], keypoints[i]) for i in range(len(boxes))])
    return people


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap(a, b):
    """How much two boxes cover each other: 0 = not at all, 1 = identical."""
    shared = box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    total = box_area(a) + box_area(b) - shared
    return shared / total if total > 0 else 0.0


def follow_people(people):
    """Give every person a number that stays the same from frame to frame: a detection continues the earlier person
    whose last box it overlaps most; if it overlaps nobody, it is a new person."""
    last_box = {}                                                         # person number -> box in its latest frame
    followed = []
    for detections in people:
        frame, taken = [], set()
        for box, keypoints in sorted(detections, key=lambda d: box_area(d[0]), reverse=True):
            best, best_overlap = None, 0.2
            for number, old_box in last_box.items():
                if number not in taken and overlap(box, old_box) > best_overlap:
                    best, best_overlap = number, overlap(box, old_box)
            if best is None:
                best = len(last_box)
            last_box[best] = box
            taken.add(best)
            frame.append((best, box, keypoints))
        followed.append(frame)
    return followed


def keep_main_people(followed, n_people=2):
    """The people that matter are the ones that are biggest and clearest over the whole clip.
    -> (frames, n_people, 13 joints, 3 = x, y, confidence); a person missing in a frame is all zeros."""
    score = defaultdict(float)
    for frame in followed:
        for number, box, keypoints in frame:
            score[number] += box_area(box) * keypoints[:, 2].mean()
    main = sorted(score, key=score.get, reverse=True)[:n_people]
    result = np.zeros((len(followed), n_people, 13, 3), dtype=np.float32)
    for t, frame in enumerate(followed):
        for number, box, keypoints in frame:
            if number in main:
                result[t, main.index(number)] = keypoints[COCO17_TO_13]
    return result


def estimate_class(model, clip_paths, num_frames, device):
    """All clips of one class -> keypoints (clips, T, 2, 13, 3), the clip names, and the time the pose model needed."""
    keypoints, names, pose_seconds, pose_frames = [], [], 0.0, 0
    for path in clip_paths:
        try:
            frames, positions = read_frames(path, num_frames)
        except ValueError as error:
            print(f"    skipped {path.name}: {error}")
            continue
        start = time.perf_counter()
        people = detect_people(model, frames, device)
        if device != "cpu":
            torch.cuda.synchronize()
        pose_seconds += time.perf_counter() - start
        pose_frames += len(frames)
        main_people = keep_main_people(follow_people(people))
        keypoints.append(main_people[positions])                          # back to exactly num_frames frames
        names.append(path.stem)
    return np.stack(keypoints), names, pose_seconds, pose_frames


def estimate_all(hmdb_dir, raw_dir, hmdb_classes, num_frames, model_name):
    """Step 1 for every class. A class whose raw file fits the clips on disk is not estimated again."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    model, device = None, (0 if torch.cuda.is_available() else "cpu")
    for hmdb_class in hmdb_classes:
        clip_paths = sorted((hmdb_dir / hmdb_class).glob("*.avi"))
        if not clip_paths:
            raise FileNotFoundError(f"no .avi clips in {hmdb_dir / hmdb_class}")
        raw_file = raw_dir / f"{hmdb_class}.npz"
        if raw_file.exists():
            raw = np.load(raw_file)
            same_clips = set(raw["attempted"].tolist()) == {p.stem for p in clip_paths}
            if same_clips and raw["keypoints"].shape[1] == num_frames and str(raw["model"]) == model_name:
                print(f"[estimate] {hmdb_class:<12} already done ({len(raw['names'])} clips)")
                continue
        if model is None:                                                 # load the pose model only if there is work
            from ultralytics import YOLO
            print(f"[estimate] loading {model_name} on {'the GPU' if device == 0 else 'the CPU (slow)'}")
            model = YOLO(model_name)
        start = time.perf_counter()
        keypoints, names, pose_seconds, pose_frames = estimate_class(model, clip_paths, num_frames, device)
        np.savez_compressed(raw_file, keypoints=keypoints, names=np.array(names), model=np.array(model_name),
                            attempted=np.array([p.stem for p in clip_paths]),
                            pose_seconds=pose_seconds, pose_frames=pose_frames,
                            device=np.array(torch.cuda.get_device_name(0) if device == 0 else "cpu"))
        print(f"[estimate] {hmdb_class:<12} {len(names)} clips in {time.perf_counter() - start:.0f} s "
              f"({1000 * pose_seconds / max(1, pose_frames):.1f} ms per frame in the pose model)")


# ------------------------------------------------------------------------------------------------ step 2: select
def clip_quality(keypoints, joint_confidence):
    """keypoints: (T, 2, 13, 3). Looks at the main person only.
    -> share of frames with a person, average number of joints found in those frames, share of frames with a full torso."""
    found = keypoints[:, 0, :, 2] >= joint_confidence                      # (T, 13)
    person_there = found.any(axis=1)
    detect_rate = float(person_there.mean())
    joints = float(found[person_there].sum(axis=1).mean()) if person_there.any() else 0.0
    torso_rate = float(found[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].all(axis=1).mean())
    return detect_rate, joints, torso_rate


def fill_gaps(data):
    """data: (T, 13, 2, M). A joint that is lost for some frames keeps its last known position (the NTU skeletons do
    the same). A joint that is never found stays (0, 0)."""
    for m in range(data.shape[3]):
        for j in range(data.shape[1]):
            seen = np.abs(data[:, j, :, m]).sum(axis=1) > 0
            if not seen.any():
                continue
            last = data[np.argmax(seen), j, :, m].copy()                   # before its first appearance: the first position
            for t in range(len(data)):
                if seen[t]:
                    last = data[t, j, :, m].copy()
                else:
                    data[t, j, :, m] = last
    return data


def to_model_input(keypoints, joint_confidence, two_people):
    """keypoints (T, 2, 13, 3) in pixels -> CTR-GCN input (2, T, 13, 2) in the shared "2d13" format."""
    xy = keypoints[..., :2] * (keypoints[..., 2:] >= joint_confidence)      # joints that are not sure enough -> (0, 0)
    if not two_people:
        xy[:, 1] = 0                                                       # single-person action: ignore everybody else
    data = xy.transpose(0, 2, 3, 1).copy()                                 # (T, 2, 13, 2) -> (T, 13, 2, M)
    data = normalize_2d(fill_gaps(data))
    return data.transpose(2, 0, 1, 3)                                      # -> (2, T, 13, M)


def select_good_clips(raw_dir, output_dir, classes, cfg):
    """Step 2. classes: list of (hmdb class, is it a two-person action). -> {hmdb class: [names of the good clips]}"""
    rows, x, names, labels, good_names = [], [], [], [], {}
    pose_seconds, pose_frames, device, model_name = 0.0, 0, "", ""
    for hmdb_class, two_people in classes:
        raw = np.load(raw_dir / f"{hmdb_class}.npz")
        pose_seconds += float(raw["pose_seconds"])
        pose_frames += int(raw["pose_frames"])
        device, model_name = str(raw["device"]), str(raw["model"])
        good_names[hmdb_class] = []
        for keypoints, name in zip(raw["keypoints"], raw["names"].tolist()):
            detect_rate, joints, torso_rate = clip_quality(keypoints, cfg.pose_joint_confidence)
            if detect_rate < cfg.pose_min_detect_rate:
                problem = "no person found in most frames"
            elif joints < cfg.pose_min_joints:
                problem = "too few joints (only a head or part of the body is visible)"
            elif torso_rate < cfg.pose_min_torso_rate:
                problem = "shoulders and hips are not visible together"
            else:
                problem = ""
            if not problem:
                try:
                    clip = to_model_input(keypoints, cfg.pose_joint_confidence, two_people)
                except ValueError:                                         # the torso is needed to normalise the skeleton
                    problem = "shoulders and hips are not visible together"
            if not problem:
                x.append(clip)
                names.append(name)
                labels.append(hmdb_class)
                good_names[hmdb_class].append(name)
            rows.append({"class": hmdb_class, "clip": name, "detect_rate": round(detect_rate, 3), "joints_found": round(joints, 2),
                         "torso_rate": round(torso_rate, 3), "good": int(not problem), "problem": problem})

    if not x:
        raise ValueError("not a single clip has a good skeleton - lower the pose_min_... thresholds in the yaml")
    np.savez_compressed(output_dir / "hmdb_skeleton_2d13.npz", x=np.stack(x).astype(np.float32), names=np.array(names),
                        classes=np.array(labels), skeleton_format=np.array("2d13"))
    with open(output_dir / "pose_quality.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- report
    print(f"\n[select] good clip = person in >= {cfg.pose_min_detect_rate:.0%} of the frames, >= {cfg.pose_min_joints} of 13 joints, "
          f"torso in >= {cfg.pose_min_torso_rate:.0%} of the frames (joint confidence >= {cfg.pose_joint_confidence})")
    print(f"{'class':<14}{'clips':>7}{'good':>7}{'share':>8}   most common problem of the excluded clips")
    per_class = {}
    for hmdb_class, _ in classes:
        class_rows = [r for r in rows if r["class"] == hmdb_class]
        n_good = sum(r["good"] for r in class_rows)
        problems = [r["problem"] for r in class_rows if not r["good"]]
        top_problem = max(set(problems), key=problems.count) if problems else "-"
        per_class[hmdb_class] = {"clips": len(class_rows), "good": n_good}
        warning = "   <-- FEW CLIPS" if n_good < cfg.pose_min_clips_per_class else ""
        print(f"{hmdb_class:<14}{len(class_rows):>7}{n_good:>7}{n_good / len(class_rows):>8.0%}   {top_problem}{warning}")
    good_share = len(x) / len(rows)
    print(f"{'ALL':<14}{len(rows):>7}{len(x):>7}{good_share:>8.0%}")
    if good_share < MIN_GOOD_SHARE_TO_PRUNE:
        print(f"\n[WARNING] only {good_share:.0%} of the clips have a good skeleton. Before testing on so little data: lower "
              f"pose_min_joints / pose_min_torso_rate / pose_joint_confidence, or try a bigger pose_model. "
              f"Look at pose_quality.csv to see what fails.")

    ms_per_frame = 1000 * pose_seconds / max(1, pose_frames)
    frames_per_clip = pose_frames / len(rows)
    summary = {"pose_model": model_name, "device": device, "clips": len(rows), "good_clips": len(x), "good_share": good_share,
               "per_class": per_class, "pose_ms_per_frame": ms_per_frame, "pose_frames_per_clip": frames_per_clip,
               "pose_ms_per_clip": ms_per_frame * frames_per_clip,
               "thresholds": {"joint_confidence": cfg.pose_joint_confidence, "min_detect_rate": cfg.pose_min_detect_rate,
                              "min_joints": cfg.pose_min_joints, "min_torso_rate": cfg.pose_min_torso_rate}}
    with open(output_dir / "pose_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[cost] pose estimation with {model_name} on {device}: {ms_per_frame:.1f} ms per frame x {frames_per_clip:.0f} frames "
          f"= {ms_per_frame * frames_per_clip:.0f} ms per clip   (add this to the cost of the skeleton model)")
    print(f"[saved] {output_dir}: hmdb_skeleton_2d13.npz {np.stack(x).shape}, pose_quality.csv, pose_summary.json")
    return good_names, good_share


# ------------------------------------------------------------------------------------------------ step 3: prune
def restore_excluded(hmdb_dir, excluded_dir):
    """Move every clip that an earlier run excluded back into the HMDB folder."""
    moved = 0
    for clip in excluded_dir.glob("*/*.avi"):
        shutil.move(str(clip), str(hmdb_dir / clip.parent.name / clip.name))
        moved += 1
    if moved:
        print(f"[prune] moved {moved} earlier excluded clips back to {hmdb_dir}")


def prune_hmdb_rgb(hmdb_dir, excluded_dir, good_names):
    """Move the RGB clips without a good skeleton to excluded_dir/<class>/, so RGB and skeleton are tested on the same clips."""
    moved = 0
    for hmdb_class, names in good_names.items():
        (excluded_dir / hmdb_class).mkdir(parents=True, exist_ok=True)
        for clip in sorted((hmdb_dir / hmdb_class).glob("*.avi")):
            if clip.stem not in names:
                shutil.move(str(clip), str(excluded_dir / hmdb_class / clip.name))
                moved += 1
    left = sum(len(list((hmdb_dir / c).glob("*.avi"))) for c in good_names)
    print(f"[prune] moved {moved} RGB clips without a good skeleton to {excluded_dir}; {left} clips stay in {hmdb_dir}")


# ------------------------------------------------------------------------------------------------ main
@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    hmdb_dir = Path(to_absolute_path(cfg.processed_HMDB1_FOLDER))
    output_dir = Path(to_absolute_path(cfg.skeleton_estimates_HMDB1_FOLDER))
    excluded_dir = Path(to_absolute_path(cfg.excluded_HMDB1_FOLDER))
    classes = [(str(c.hmdb), 50 <= int(c.ntu60) <= 60) for c in cfg.classes]          # (hmdb class, two-person action?)
    output_dir.mkdir(parents=True, exist_ok=True)

    restore_excluded(hmdb_dir, excluded_dir)                  # always start from the complete HMDB folder
    estimate_all(hmdb_dir, output_dir / "raw", [c for c, _ in classes], cfg.skeleton_num_frames, cfg.pose_model)
    good_names, good_share = select_good_clips(output_dir / "raw", output_dir, classes, cfg)

    if not cfg.hmdb_prune:
        print("\n[prune] not done (hmdb_prune: false). The HMDB RGB folder still holds all clips.")
    elif good_share < MIN_GOOD_SHARE_TO_PRUNE:
        print(f"\n[prune] REFUSED: only {good_share:.0%} of the clips are good. Fix the selection first (see the warning above).")
    else:
        print()
        prune_hmdb_rgb(hmdb_dir, excluded_dir, good_names)


if __name__ == "__main__":
    main()