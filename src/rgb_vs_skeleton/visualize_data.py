"""One big overview image: for every class a square frame from NTU RGB | NTU skeleton | HMDB (n_classes x 3 squares).
Layout (default 16 classes): 6 columns x 8 rows = [RGB | skeleton | HMDB] for classes 1-8 on the left, the same three
columns again for classes 9-16 on the right.

Run from the Project_DL folder:
    uv run src/rgb_vs_skeleton/visualize_data.py dry_run=true      # only print which clips / frames would be used
    uv run src/rgb_vs_skeleton/visualize_data.py                   # middle frame of one clip per class
    uv run src/rgb_vs_skeleton/visualize_data.py +viz_frame=random +viz_seed=3 +viz_cell_px=640 +viz_fit=pad +viz_rows=4
Needs:  uv add opencv-python

Output: <reports_class_images_datasets_FOLDER>/class_overview.png   (default folder: reports/figures)
Classes follow the order of the yaml: block 1 (left) = classes 1..viz_rows, block 2 = the next viz_rows classes, and so on.

Optional config keys (use `+key=value` on the command line if they are not in the yaml; defaults in brackets):
    viz_frame [middle]    middle = frame in the middle of each clip | random = random frame (20-80% of the clip)
    viz_cell_px [512]     side of every square in pixels (picture = 3 columns x n_classes rows of these)
    viz_fit [crop]        crop = centre square crop (fills the square) | pad = whole frame with black bars
    viz_seed [0]          which clips / frames are drawn (same seed = same picture)
    viz_rows [8]          classes per block = number of rows; 16 classes -> 2 blocks side by side = 6 columns x 8 rows
                          (viz_rows=16 gives the old layout: one block of 3 columns x 16 rows)
    viz_gap_px [8]        space between squares
The skeleton square shows the frame of the SAME clip at the same relative time, drawn from the 2D colour-space joints in the
.skeleton file, with the same crop as the RGB square. Body selection follows the dataset authors' FAQ (largest summed variance =
main person; two persons only for the mutual actions A50-A60).
"""
import random
from pathlib import Path

import cv2
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

NTU_COLOR_W, NTU_COLOR_H = 1920, 1080
EDGES_1B = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
BODY_COLORS = [(90, 220, 90), (60, 160, 255)]          # BGR: main person green, second person orange
NTU_CAMERA = 2                                         # preferred camera id (my belief: frontal view); None = any
BG = (24, 24, 24)
COLS = ["NTU RGB", "NTU skeleton", "HMDB"]


# ------------------------------------------------------------------ geometry: frame -> square
def square_transform(w, h, cell, fit):
    """Returns (scale, offset_x, offset_y) so that point p in a w x h frame lands at p*scale + offset in the cell x cell square."""
    if fit == "crop":
        side = min(w, h)
        s = cell / side
        return s, -(w - side) / 2 * s, -(h - side) / 2 * s
    s = cell / max(w, h)
    return s, (cell - w * s) / 2, (cell - h * s) / 2


def to_square(img, cell, fit):
    h, w = img.shape[:2]
    s, ox, oy = square_transform(w, h, cell, fit)
    M = np.array([[s, 0, ox], [0, s, oy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (cell, cell), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ------------------------------------------------------------------ video frame
def video_frame(path, frac):
    """Frame at relative position frac (0..1). -> (BGR image, n_frames). Frames are grabbed sequentially (reliable for avi/mp4)."""
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:                                           # unreliable header -> count by decoding
            n = 0
            while cap.grab():
                n += 1
            cap.release()
            cap = cv2.VideoCapture(str(path))
        if n <= 0:
            raise ValueError(f"OpenCV could not decode {path}")
        idx = int(round(frac * (n - 1)))
        for _ in range(idx):
            if not cap.grab():
                break
        ok, img = cap.read()
        if not ok:                                           # frame count lied: fall back to the first frame
            cap.release()
            cap = cv2.VideoCapture(str(path))
            ok, img = cap.read()
            if not ok:
                raise ValueError(f"OpenCV could not decode {path}")
        return img, n
    finally:
        cap.release()


# ------------------------------------------------------------------ skeleton frame
def read_skeleton_tracks(path):
    lines = Path(path).read_text().split("\n")
    i = 0
    n = int(lines[i]); i += 1
    tracks = {}
    for t in range(n):
        n_bodies = int(lines[i]); i += 1
        for _ in range(n_bodies):
            body_id = lines[i].split()[0]; i += 1
            n_j = int(lines[i]); i += 1
            tracks.setdefault(body_id, {})[t] = np.array([lines[i + j].split()[:7] for j in range(n_j)], dtype=np.float32)
            i += n_j
    return n, tracks


def select_bodies(tracks, n_people):
    """Authors' heuristic: main subject = largest summed variance of the X, Y, Z joint values; ghost bodies (tables, seats) lose."""
    def score(frames):
        a = np.stack(list(frames.values()))[..., :3]
        return float(sum(np.var(a[..., c]) for c in range(3)))
    return sorted(tracks.values(), key=score, reverse=True)[:n_people]


def skeleton_square(path, frac, n_people, cell, fit):
    n, tracks = read_skeleton_tracks(path)
    if not tracks:
        raise ValueError(f"no bodies in {path}")
    t = int(round(frac * (n - 1)))
    s, ox, oy = square_transform(NTU_COLOR_W, NTU_COLOR_H, cell, fit)
    canvas = np.zeros((cell, cell, 3), np.uint8)
    lw, r = max(2, cell // 90), max(3, cell // 70)
    for slot, body in enumerate(select_bodies(tracks, n_people)):
        frames = sorted(body)                                 # nearest frame where this body exists
        tt = min(frames, key=lambda f: abs(f - t))
        pts = body[tt][:, 5:7] * s + np.array([ox, oy])
        ok = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 1e5).all(axis=1)
        for a, b in EDGES_1B:
            if ok[a - 1] and ok[b - 1]:
                cv2.line(canvas, tuple(int(v) for v in pts[a - 1]), tuple(int(v) for v in pts[b - 1]), BODY_COLORS[slot], lw, cv2.LINE_AA)
        for j in range(len(pts)):
            if ok[j]:
                cv2.circle(canvas, tuple(int(v) for v in pts[j]), r, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


# ------------------------------------------------------------------ cells
def caption(img, text):
    h, w = img.shape[:2]
    scale = max(0.5, w / 900)
    th = int(34 * scale) + 6
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, th), (0, 0, 0), -1)
    img = cv2.addWeighted(bar, 0.6, img, 0.4, 0)
    cv2.putText(img, text, (8, int(th * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(scale * 1.5)), cv2.LINE_AA)
    return img


def placeholder(cell, text):
    img = np.full((cell, cell, 3), 40, np.uint8)
    cv2.putText(img, text, (12, cell // 2), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, cell / 900), (170, 170, 170), 1, cv2.LINE_AA)
    return img


def pick_ntu_clip(rgb_dir, skel_dir, ntu_id, rng):
    """A clip that exists as BOTH rgb and skeleton (the skeleton square must show the same clip); preferred camera first."""
    rgb = {p.stem: p for p in Path(rgb_dir).glob(f"*A{ntu_id:03d}.avi")}
    skel = {p.stem: p for p in Path(skel_dir).glob(f"*A{ntu_id:03d}.skeleton")}
    names = sorted(set(rgb) & set(skel))
    if not names:
        return None
    pref = [n for n in names if NTU_CAMERA and f"C{NTU_CAMERA:03d}" in n] or names
    name = rng.choice(pref)
    return name, rgb[name], skel[name]


def pick_hmdb_clip(hmdb_dir, hmdb_class, rng):
    clips = sorted((Path(hmdb_dir) / hmdb_class).glob("*.avi"))
    return rng.choice(clips) if clips else None


def build_row(ntu_id, hmdb_class, rgb_dir, skel_dir, hmdb_dir, cell, fit, frame_mode, rng, dry_run=False):
    """-> list of 3 squares (BGR) [rgb, skeleton, hmdb] + info string"""
    frac = 0.5 if frame_mode == "middle" else rng.uniform(0.2, 0.8)
    frac_h = 0.5 if frame_mode == "middle" else rng.uniform(0.2, 0.8)
    tag = f"A{ntu_id:03d} {hmdb_class}"
    ntu = pick_ntu_clip(rgb_dir, skel_dir, ntu_id, rng)
    hm = pick_hmdb_clip(hmdb_dir, hmdb_class, rng) if Path(hmdb_dir, hmdb_class).exists() else None
    info = f"{tag}: NTU {ntu[0] if ntu else 'NONE (needs rgb AND skeleton)'} @{frac:.0%} | HMDB {hm.name[-34:] if hm else 'NONE'} @{frac_h:.0%}"
    if dry_run:
        return None, info
    cells = []
    if ntu:
        name, rgb_path, skel_path = ntu
        try:
            img, _ = video_frame(rgb_path, frac)
            cells.append(caption(to_square(img, cell, fit), f"{tag} | NTU RGB | {name[8:]}"))
        except Exception as e:
            print(f"[viz] {tag}: RGB failed - {e}")
            cells.append(caption(placeholder(cell, "RGB frame failed"), f"{tag} | NTU RGB"))
        try:
            cells.append(caption(skeleton_square(skel_path, frac, 2 if 50 <= ntu_id <= 60 else 1, cell, fit), f"{tag} | NTU skeleton"))
        except Exception as e:
            print(f"[viz] {tag}: skeleton failed - {e}")
            cells.append(caption(placeholder(cell, "skeleton failed"), f"{tag} | NTU skeleton"))
    else:
        cells += [caption(placeholder(cell, "no NTU clip with rgb + skeleton"), f"{tag} | NTU RGB"),
                  caption(placeholder(cell, "no NTU clip with rgb + skeleton"), f"{tag} | NTU skeleton")]
    if hm:
        try:
            img, _ = video_frame(hm, frac_h)
            cells.append(caption(to_square(img, cell, fit), f"{tag} | HMDB"))
        except Exception as e:
            print(f"[viz] {tag}: HMDB failed - {e}")
            cells.append(caption(placeholder(cell, "HMDB frame failed"), f"{tag} | HMDB"))
    else:
        cells.append(caption(placeholder(cell, f"no HMDB clip for '{hmdb_class}'"), f"{tag} | HMDB"))
    return cells, info


def visualize_data(rgb_dir, skeleton_dir, hmdb_dir, out_dir, classes, frame_mode="middle", cell=512, fit="crop",
                   seed=0, rows_per_block=8, gap=8, dry_run=False):
    """classes: list of (ntu60_id, hmdb_class). Writes <out_dir>/class_overview.png."""
    rgb_dir, skeleton_dir, hmdb_dir, out_dir = map(Path, (rgb_dir, skeleton_dir, hmdb_dir, out_dir))
    if frame_mode not in ("middle", "random"):
        raise ValueError("viz_frame must be 'middle' or 'random'")
    if fit not in ("crop", "pad"):
        raise ValueError("viz_fit must be 'crop' or 'pad'")
    cell, gap = int(cell), int(gap)
    print(f"[viz start] classes={len(classes)}  frame={frame_mode}  cell={cell}px  fit={fit}  seed={seed}  rows_per_block={rows_per_block}  dry_run={dry_run}\n"
          f"            rgb={rgb_dir}\n            skeleton={skeleton_dir}\n            hmdb={hmdb_dir}\n            out={out_dir}")
    for d in (rgb_dir, skeleton_dir):
        if not d.exists():
            raise FileNotFoundError(f"folder not found: {d}")
    if not hmdb_dir.exists():
        print(f"[viz] HMDB folder not found ({hmdb_dir}) - HMDB squares will be placeholders")
    rows, missing = [], []
    for ntu_id, hmdb_class in classes:
        rng = random.Random(f"{seed}-{ntu_id}-{hmdb_class}")           # same picture for the same seed
        cells, info = build_row(ntu_id, hmdb_class, rgb_dir, skeleton_dir, hmdb_dir, cell, fit, frame_mode, rng, dry_run)
        print(f"[viz] {info}")
        if "NONE" in info:
            missing.append(f"A{ntu_id:03d}/{hmdb_class}")
        if cells:
            rows.append(cells)
    if dry_run:
        print(f"[viz] DRY RUN: would draw {len(classes)} x 3 = {len(classes) * 3} squares"
              + (f" | classes with a missing source: {missing}" if missing else ""))
        return None
    per_block = max(1, int(rows_per_block))
    n_blocks = -(-len(rows) // per_block)                       # ceil
    nr, nc = per_block, 3 * n_blocks
    grid = [[None] * nc for _ in range(nr)]
    for k, three in enumerate(rows):                            # class k -> block k // per_block, row k % per_block
        b, r = divmod(k, per_block)
        for j in range(3):
            grid[r][3 * b + j] = three[j]
    print(f"[viz] layout: {nc} columns x {nr} rows  ({' | '.join(COLS * n_blocks)})")
    H, W = nr * cell + (nr + 1) * gap, nc * cell + (nc + 1) * gap
    canvas = np.full((H, W, 3), BG, np.uint8)
    for i, row in enumerate(grid):
        for j, img in enumerate(row):
            y, x = gap + i * (cell + gap), gap + j * (cell + gap)
            if img is not None:                                  # empty cells (last block not full) stay background
                canvas[y:y + cell, x:x + cell] = img
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "class_overview_2.png"
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write {path} (image {W}x{H}px)")
    print(f"[viz done] {len(rows)} classes x 3 = {len(rows) * 3} squares -> {path}  ({W}x{H}px, {path.stat().st_size / 1e6:.1f} MB)"
          + (f" | classes with a missing source: {missing}" if missing else ""))
    return path


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    visualize_data(
        rgb_dir=to_absolute_path(cfg.processed_RGB_FOLDER),
        skeleton_dir=to_absolute_path(cfg.processed_skeleton_FOLDER),
        hmdb_dir=to_absolute_path(cfg.processed_HMDB1_FOLDER),
        out_dir=to_absolute_path(cfg.reports_class_images_datasets_FOLDER),
        classes=[(int(c.ntu60), str(c.hmdb)) for c in cfg.classes],
        frame_mode=cfg.get("viz_frame", "middle"),
        cell=cfg.get("viz_cell_px", 512),
        fit=cfg.get("viz_fit", "crop"),
        seed=cfg.get("viz_seed", 2),
        rows_per_block=cfg.get("viz_rows", 8),
        gap=cfg.get("viz_gap_px", 8),
        dry_run=cfg.dry_run,
    )


if __name__ == "__main__":
    main()