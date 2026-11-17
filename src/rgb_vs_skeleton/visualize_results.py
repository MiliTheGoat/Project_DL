"""Videos and pictures for the report: the HMDB skeletons, and correct / wrong predictions of both models.

Run from the Project_DL folder (on a machine that has the videos, the skeletons and the trained models):
    uv run src/rgb_vs_skeleton/visualize_results.py                       # seed from the yaml
    uv run src/rgb_vs_skeleton/visualize_results.py results_seed=1        # other clips; the files of seed 0 stay
On the HPC use .venv/bin/python instead of uv run.

What it makes (every file name ends with the seed, so a new seed ADDS files and never replaces old ones):

  reports/HMDB1/                         the pose estimates, drawn over the HMDB video (one good clip per class)
      videos/<class>_seed0.mp4
      photos/<class>_seed0.png           the middle frame of the same clip

  reports/results/                       per class one clip the model got RIGHT and one it got WRONG
      rgb/              X3D-M   on the NTU test set         each of these four folders has
      skeleton/         CTR-GCN on the NTU test set             videos/<class>_correct_seed0.mp4, <class>_wrong_seed0.mp4
      hmdb1/rgb/        X3D-M   on HMDB                         photos/<class>_correct_seed0.png, <class>_wrong_seed0.png
      hmdb1/skeleton/   CTR-GCN on HMDB
      overview_correct_seed0.png         all classes x the four cases in one picture (8 columns x 8 rows):
      overview_wrong_seed0.png           classes 1-8 on the left, 9-16 on the right, four cases next to each other

Green bar = predicted right, red bar = predicted wrong (it says which class the model chose instead).
A class without a wrong clip (100% correct) gets no wrong video / photo and a grey square in the overview.
RGB cases show the video. Skeleton cases show exactly what CTR-GCN gets: the 13 normalised joints, nothing else.

The predictions of every clip are computed once with best.pt and saved next to the model (predictions.csv or
predictions_hmdb.csv); later runs only read that file. X3D-M on the NTU test set takes minutes on a GPU, long on a CPU.
A case whose model or data is not on this machine is skipped with a message.
"""
import csv
import random
from pathlib import Path

import cv2
import hydra
import imageio_ffmpeg
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from evaluate_RGB import find_video_folder
from evaluate_skeleton import predict_test_set
from loaders import hmdb_dataloader, hmdb_skeleton_dataloader, rgb_dataloader, skeleton_dataloader
from model_CTR_GCN import build_ctr_gcn
from model_X3D_M import build_x3d_m
from pose_estimation import read_frames
from skeleton_format import BONES_13

CELL = 384                       # every video and picture is a square of this many pixels (a multiple of 16 for mp4)
BAR = 26                         # height of the text bars
VIDEO_FRAMES = 48                # frames taken from an RGB video ...
FPS = 16                         # ... and how fast they are played
GAP = 6                          # space between the squares of the overview pictures

GREEN, RED, GREY = (70, 150, 60), (50, 50, 200), (60, 60, 60)        # colours are blue-green-red, as OpenCV wants
PERSON_COLORS = [(90, 220, 90), (60, 160, 255)]                      # main person green, second person orange

# (model, test set, title, folder inside reports/results)
CASES = [("rgb", "ntu", "X3D-M on NTU", "rgb"),
         ("skeleton", "ntu", "CTR-GCN on NTU", "skeleton"),
         ("rgb", "hmdb", "X3D-M on HMDB", "hmdb1/rgb"),
         ("skeleton", "hmdb", "CTR-GCN on HMDB", "hmdb1/skeleton")]


# ------------------------------------------------------------------------------------------------ drawing
def to_square(frame):
    """The whole frame inside a CELL x CELL square (black bars fill the rest).
    -> (square, scale, x offset, y offset): a point p of the frame lands at p * scale + offset."""
    h, w = frame.shape[:2]
    scale = CELL / max(h, w)
    small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    y, x = (CELL - small.shape[0]) // 2, (CELL - small.shape[1]) // 2
    square = np.zeros((CELL, CELL, 3), np.uint8)
    square[y:y + small.shape[0], x:x + small.shape[1]] = small
    return square, scale, x, y


def draw_skeleton(image, points, found, color):
    """points: (13, 2) in pixels of the image. found: (13,) which joints exist. Draws bones, then white joints."""
    points = points.round().astype(int)
    for a, b in BONES_13:                                              # bones are numbered from 1
        if found[a - 1] and found[b - 1]:
            cv2.line(image, tuple(points[a - 1]), tuple(points[b - 1]), color, 2, cv2.LINE_AA)
    for j in range(len(points)):
        if found[j]:
            cv2.circle(image, tuple(points[j]), 3, (255, 255, 255), -1, cv2.LINE_AA)


def add_bars(image, top_text, top_color, bottom_text):
    """A coloured bar with text at the top and a dark bar with text at the bottom. Returns a new image."""
    image = image.copy()
    cv2.rectangle(image, (0, 0), (CELL, BAR), top_color, -1)
    cv2.rectangle(image, (0, CELL - BAR), (CELL, CELL), (0, 0, 0), -1)
    cv2.putText(image, top_text, (6, BAR - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, bottom_text, (6, CELL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return image


def empty_square(text):
    image = np.full((CELL, CELL, 3), 45, np.uint8)
    cv2.putText(image, text, (12, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    return image


# ------------------------------------------------------------------------------------------------ clip -> list of squares
def rgb_frames(path):
    """VIDEO_FRAMES frames spread evenly over a video, each as a square."""
    frames, positions = read_frames(path, VIDEO_FRAMES)
    return [to_square(frames[p])[0] for p in positions]


def skeleton_frames(clip):
    """clip: CTR-GCN input (2 = x y, T frames, 13 joints, M persons) -> the skeleton drawn on a dark square, per frame.
    The drawing is zoomed so that the whole clip fits in the square. A joint at exactly (0, 0) was not found."""
    points = clip.transpose(1, 3, 2, 0)                                # (T, M, 13, 2)
    found = np.abs(points).sum(axis=3) > 0                             # (T, M, 13)
    scale, offset = 1.0, np.zeros(2)
    if found.any():
        low, high = points[found].min(axis=0), points[found].max(axis=0)
        scale = 0.75 * CELL / max((high - low).max(), 1e-6)
        offset = CELL / 2 - scale * (low + high) / 2
    frames = []
    for t in range(len(points)):
        image = np.full((CELL, CELL, 3), 30, np.uint8)
        for m in range(points.shape[1]):
            draw_skeleton(image, points[t, m] * scale + offset, found[t, m], PERSON_COLORS[m])
        frames.append(image)
    return frames


def pose_over_video_frames(path, keypoints, confidence, n_people):
    """keypoints: the raw pose estimates of one HMDB clip (T, 2 persons, 13, 3 = x y confidence), in video pixels.
    -> the T frames the pose model looked at, as squares, with the skeletons drawn over them."""
    frames, positions = read_frames(path, len(keypoints))
    result = []
    for t, p in enumerate(positions):
        image, scale, x, y = to_square(frames[p])
        for m in range(n_people):
            points = keypoints[t, m, :, :2] * scale + (x, y)
            draw_skeleton(image, points, keypoints[t, m, :, 2] >= confidence, PERSON_COLORS[m])
        result.append(image)
    return result


# ------------------------------------------------------------------------------------------------ saving
def save_video(path, frames):
    """Squares -> an .mp4 that every player and browser can show (H.264)."""
    writer = imageio_ffmpeg.write_frames(str(path), (CELL, CELL), fps=FPS, pix_fmt_in="bgr24")
    writer.send(None)                                                  # starts ffmpeg
    for frame in frames:
        writer.send(np.ascontiguousarray(frame))
    writer.close()


def save_example(folder, file_name, frames, top_text, top_color, bottom_text):
    """One clip -> folder/videos/<file_name>.mp4 and folder/photos/<file_name>.png. Returns the photo (the middle frame)."""
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    (folder / "photos").mkdir(parents=True, exist_ok=True)
    frames = [add_bars(frame, top_text, top_color, bottom_text) for frame in frames]
    photo = frames[len(frames) // 2]
    save_video(folder / "videos" / f"{file_name}.mp4", frames)
    cv2.imwrite(str(folder / "photos" / f"{file_name}.png"), photo)
    return photo


def save_overview(path, squares):
    """squares[class][case] -> one picture: per class the cases next to each other, half of the classes left, half right."""
    n_cases = len(CASES)
    n_rows = (len(squares) + 1) // 2
    step = CELL + GAP
    picture = np.full((GAP + n_rows * step, GAP + 2 * n_cases * step, 3), 24, np.uint8)
    for k, row in enumerate(squares):
        block, r = divmod(k, n_rows)                                   # block 0 = left half, block 1 = right half
        for c, square in enumerate(row):
            y, x = GAP + r * step, GAP + (block * n_cases + c) * step
            picture[y:y + CELL, x:x + CELL] = square
    cv2.imwrite(str(path), picture)
    print(f"[saved] {path}  ({picture.shape[1]} x {picture.shape[0]} pixels)")


# ------------------------------------------------------------------------------------------------ part 1: HMDB skeletons
def visualize_hmdb(cfg, seed):
    """One good HMDB clip per class with its estimated skeleton drawn over the video."""
    out_dir = Path(to_absolute_path(cfg.reports_HMDB1_FOLDER))
    estimates_dir = Path(to_absolute_path(cfg.skeleton_estimates_HMDB1_FOLDER))
    video_dirs = [Path(to_absolute_path(cfg.processed_HMDB1_FOLDER)), Path(to_absolute_path(cfg.excluded_HMDB1_FOLDER))]
    good = np.load(estimates_dir / "hmdb_skeleton_2d13.npz")           # the clips that passed the quality check
    good_names = set(zip(good["classes"].tolist(), good["names"].tolist()))
    print(f"\n[1/2] HMDB skeletons over the videos -> {out_dir}")
    for c in cfg.classes:
        hmdb_class, two_people = str(c.hmdb), 50 <= int(c.ntu60) <= 60
        raw = np.load(estimates_dir / "raw" / f"{hmdb_class}.npz")
        names = raw["names"].tolist()
        choices = [i for i, name in enumerate(names) if (hmdb_class, name) in good_names]
        if not choices:
            print(f"  {hmdb_class:<12} no clip with a good skeleton")
            continue
        i = random.Random(f"{seed}-{hmdb_class}").choice(choices)      # same seed = same clip
        paths = [d / hmdb_class / f"{names[i]}.avi" for d in video_dirs]
        paths = [p for p in paths if p.exists()]
        if not paths:
            print(f"  {hmdb_class:<12} video {names[i]}.avi not found")
            continue
        frames = pose_over_video_frames(paths[0], raw["keypoints"][i], cfg.pose_joint_confidence, 2 if two_people else 1)
        save_example(out_dir, f"{hmdb_class}_seed{seed}", frames, hmdb_class, GREY, f"HMDB + {cfg.pose_model}")
        print(f"  {hmdb_class:<12} {names[i]}")


# ------------------------------------------------------------------------------------------------ part 2: right and wrong
def load_case(cfg, kind, test_on):
    """-> (test dataset, the class the model predicts for each of its clips).
    The predictions are read from the model folder, or computed with best.pt and saved there if they are missing."""
    actions = [int(c.ntu60) for c in cfg.classes]
    hmdb_classes = [str(c.hmdb) for c in cfg.classes]
    split_file = to_absolute_path(cfg.split_file)

    # ---- the test set and an empty model, built exactly as in evaluate_RGB.py / evaluate_skeleton.py
    if kind == "rgb":
        exp = OmegaConf.load(to_absolute_path(f"configs/experiments/RGB_experiments/{cfg.results_rgb_exp}.yaml"))
        if test_on == "ntu":
            loaders, datasets = rgb_dataloader(
                video_dir=find_video_folder(cfg), actions=actions, split_file=split_file, num_frames=exp.num_frames,
                side=exp.side, size=exp.size, batch_size=exp.batch_size, num_workers=exp.num_workers)
            loader, dataset = loaders["test"], datasets["test"]
        else:
            loader, dataset = hmdb_dataloader(
                hmdb_dir=to_absolute_path(cfg.processed_HMDB1_FOLDER), hmdb_classes=hmdb_classes, num_frames=exp.num_frames,
                side=exp.side, size=exp.size, batch_size=exp.batch_size, num_workers=exp.num_workers)
        model = build_x3d_m(len(actions), pretrained=False, dropout=exp.dropout)
    else:
        exp = OmegaConf.load(to_absolute_path(f"configs/experiments/skeleton_experiments/{cfg.results_skeleton_exp}.yaml"))
        if test_on == "ntu":
            loaders, datasets = skeleton_dataloader(
                skeleton_dir=to_absolute_path(cfg.processed_skeleton_FOLDER),
                cache_file=to_absolute_path(cfg.skeleton_cache_file), actions=actions, split_file=split_file,
                num_frames=cfg.skeleton_num_frames, batch_size=exp.batch_size, num_workers=0,
                skeleton_format=cfg.skeleton_format)
            loader, dataset = loaders["test"], datasets["test"]
        else:
            skeleton_file = Path(to_absolute_path(cfg.skeleton_estimates_HMDB1_FOLDER)) / "hmdb_skeleton_2d13.npz"
            loader, dataset = hmdb_skeleton_dataloader(skeleton_file, hmdb_classes, exp.batch_size)
        model = build_ctr_gcn(len(actions), dropout=exp.dropout, skeleton_format=cfg.skeleton_format)

    # ---- predictions saved by an earlier run (only used if they belong to exactly these clips)
    run_dir = Path(to_absolute_path(exp.output_dir)) / exp.name
    predictions_file = run_dir / ("predictions.csv" if test_on == "ntu" else "predictions_hmdb.csv")
    if predictions_file.exists():
        with open(predictions_file, newline="") as f:
            rows = list(csv.DictReader(f))
        if [row["name"] for row in rows] == dataset.names:
            print(f"  predictions read from {predictions_file}")
            return dataset, [int(row["predicted"]) for row in rows]

    # ---- otherwise: run the model once over the test set
    weights = run_dir / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.to(device).eval()
    print(f"  predicting {len(dataset)} clips with {weights} on {device} ...")
    use_amp = bool(exp.mixed_precision) and device.type == "cuda"
    labels, predictions, model_time = predict_test_set(model, loader, device, use_amp)
    with open(predictions_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "true", "predicted"])
        writer.writerows(zip(dataset.names, labels, predictions))
    print(f"  [saved] {predictions_file}")
    return dataset, predictions


def clip_frames(dataset, i, kind):
    """Clip i of a test set as a list of squares: the video for the RGB model, the drawn joints for the skeleton model."""
    if kind == "skeleton":
        return skeleton_frames(dataset.x[i])
    if hasattr(dataset, "paths"):                                      # HMDB: the videos are in class folders
        return rgb_frames(dataset.paths[i])
    return rgb_frames(dataset.video_path(dataset.names[i]))            # NTU


def visualize_results(cfg, seed):
    """Per case and class: one clip predicted right and one predicted wrong (videos, photos, two overview pictures)."""
    out_dir = Path(to_absolute_path(cfg.reports_results_FOLDER))
    class_names = [str(c.hmdb) for c in cfg.classes]
    correct_squares = [[] for _ in class_names]                        # [class][case] -> photo, for the overviews
    wrong_squares = [[] for _ in class_names]
    print(f"\n[2/2] right and wrong predictions -> {out_dir}")
    for kind, test_on, title, folder in CASES:
        print(f"\n{title}")
        try:
            dataset, predictions = load_case(cfg, kind, test_on)
        except FileNotFoundError as error:                             # model or data is not on this machine
            print(f"  SKIPPED: {error}")
            for k in range(len(class_names)):
                correct_squares[k].append(empty_square(f"{title}: not available"))
                wrong_squares[k].append(empty_square(f"{title}: not available"))
            continue
        labels = dataset.labels
        accuracy = sum(p == y for p, y in zip(predictions, labels)) / len(labels)
        print(f"  {len(labels)} clips, accuracy {accuracy:.3f}")
        for k, class_name in enumerate(class_names):
            rng = random.Random(f"{seed}-{title}-{class_name}")        # same seed = same clips
            right = [i for i in range(len(labels)) if labels[i] == k and predictions[i] == k]
            wrong = [i for i in range(len(labels)) if labels[i] == k and predictions[i] != k]
            if right:
                i = rng.choice(right)
                photo = save_example(out_dir / folder, f"{class_name}_correct_seed{seed}", clip_frames(dataset, i, kind),
                                     f"{class_name}: correct", GREEN, title)
            else:
                photo = empty_square(f"{class_name}: never correct" if right + wrong else f"{class_name}: no clips")
            correct_squares[k].append(photo)
            if wrong:
                i = rng.choice(wrong)
                photo = save_example(out_dir / folder, f"{class_name}_wrong_seed{seed}", clip_frames(dataset, i, kind),
                                     f"{class_name}: WRONG -> {class_names[predictions[i]]}", RED, title)
            else:
                photo = empty_square(f"{class_name}: no wrong clip" if right else f"{class_name}: no clips")
            wrong_squares[k].append(photo)
            print(f"  {class_name:<12} right {len(right):>4}   wrong {len(wrong):>4}")
    print()
    save_overview(out_dir / f"overview_correct_seed{seed}.png", correct_squares)
    save_overview(out_dir / f"overview_wrong_seed{seed}.png", wrong_squares)


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    seed = cfg.results_seed
    print(f"[visualize results] seed {seed} | RGB experiment {cfg.results_rgb_exp} | skeleton experiment {cfg.results_skeleton_exp}")
    visualize_hmdb(cfg, seed)
    visualize_results(cfg, seed)


if __name__ == "__main__":
    main()