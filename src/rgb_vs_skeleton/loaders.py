"""Data loaders: NTU RGB+D videos for X3D-S and NTU skeletons for CTR-GCN, with the official cross-subject split.

Run from the Project_DL folder (settings come from configs/main_config.yaml):
    uv run src/rgb_vs_skeleton/loaders.py                       # builds the split + skeleton cache, prints all shapes
    uv run src/rgb_vs_skeleton/loaders.py +rgb_size=182         # change the RGB crop size

THE SPLIT (official NTU RGB+D cross-subject protocol)
    train = the 20 official training performers, minus 4 performers that we hold out as validation
    val   = performers 4, 13, 28, 35 (our choice; the official protocol has no validation set)
    test  = the other 20 performers (the official test set)
    The split is decided from the clip name alone (performer id P###), so every camera angle and repetition of one
    performer lands in the same split; there is no leakage between train and test. Every class gets the same number
    of clips inside each split (the smallest class decides). The result is saved ONCE to data/processed/splits.csv and
    both loaders read that file, so RGB and skeleton models see exactly the same clips.

RGB  -> X3D-S input  (C=3, T=13, H=160, W=160), float32, normalised with mean 0.45 / std 0.225
    13 frames are spread evenly over the clip, each frame is resized so its short side is 182 px and centre-cropped to
    160 x 160 (the actor is centred in NTU). Decoding happens when a clip is requested; nothing is kept in RAM.
    Decoding 1080p .avi is slow (seconds per clip). Pre-shrink the videos once (ffmpeg, short side 182) to .mp4 and the
    loader picks them up automatically (it looks for <name>.mp4 first, then <name>.avi).

HMDB -> same format as the NTU RGB clips, for cross-dataset TESTING of the RGB model (never for training)
    data/external/processed/<class>/<clip>.avi; the label is the index of that class in the yaml list, so label 5 means
    "sit" for HMDB exactly as it means A8 "sit down" for NTU. HMDB clips are small (320x240), so decoding is fast.

SKELETON -> CTR-GCN input, float32, in one of two formats (yaml key skeleton_format, details in skeleton_format.py):
    "3d25"  (C=3, T=64, V=25, M=2)  the Kinect skeleton in 3D, only NTU has it
    "2d13"  (C=2, T=64, V=13, M=2)  13 joints in the image; HMDB gets the same format from pose_estimation.py
    The .skeleton text files are parsed ONCE into data/processed/skeleton_cache.npz (parsing text is slow). For each clip:
    the main person is the body with the largest motion (dataset authors' rule: ghost "bodies" such as tables are static),
    two persons are kept only for the mutual actions A50-A60, frames where a body is missing repeat the last seen pose,
    3D: the skeleton is centred on the spine and rotated so the body faces the camera. 2D: the origin is the middle of
    the hips and the unit of length is the torso length. Then the clip is resampled to 64 frames.
"""
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from skeleton_format import NTU25_TO_13, normalize_2d

# ------------------------------------------------------------------------------------------------ names and split
NAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")
XSUB_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}   # official
VAL_SUBJECTS = {4, 13, 28, 35}                                                                  # our hold-out
SPLITS = ("train", "val", "test")


def parse_name(name):
    """'S001C002P003R001A008' -> dict(setup=1, camera=2, subject=3, rep=1, action=8)"""
    m = NAME_RE.search(name)
    if m is None:
        raise ValueError(f"not an NTU clip name: {name}")
    s, c, p, r, a = map(int, m.groups())
    return {"setup": s, "camera": c, "subject": p, "rep": r, "action": a}


def split_of(name):
    """Official cross-subject split, with 4 training performers moved to validation."""
    subject = parse_name(name)["subject"]
    if subject in VAL_SUBJECTS:
        return "val"
    if subject in XSUB_TRAIN_SUBJECTS:
        return "train"
    return "test"


def make_split(names, balance=True, seed=0):
    """names -> {'train': [...], 'val': [...], 'test': [...]} with the same number of clips per class in each split."""
    rng = random.Random(seed)
    splits = {key: defaultdict(list) for key in SPLITS}            # split -> action -> [names]
    for name in sorted(names):
        splits[split_of(name)][parse_name(name)["action"]].append(name)

    result = {}
    for key in SPLITS:
        per_class = splits[key]
        if balance:
            smallest = min(len(clips) for clips in per_class.values())
            result[key] = sorted(n for clips in per_class.values() for n in rng.sample(clips, smallest))
        else:
            result[key] = sorted(n for clips in per_class.values() for n in clips)
    return result


def load_or_make_split(names, split_file, balance=True, seed=0):
    """First call: make the split from `names` and save it. Later calls: read the saved split (same clips for everyone)."""
    split_file = Path(split_file)
    if split_file.exists():
        result = {key: [] for key in SPLITS}
        with open(split_file, newline="") as f:
            for row in csv.DictReader(f):
                result[row["split"]].append(row["name"])
        saved = set(n for clips in result.values() for n in clips)
        missing = saved - set(names)
        if missing:
            print(f"[split] WARNING: {len(missing)} clips of {split_file.name} are not in this data source, "
                  f"e.g. {sorted(missing)[:3]} - they are skipped here")
            result = {key: [n for n in clips if n in set(names)] for key, clips in result.items()}
        print(f"[split] loaded {split_file}")
        return result

    result = make_split(names, balance, seed)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "split"])
        for key in SPLITS:
            writer.writerows([name, key] for name in result[key])
    print(f"[split] made the official cross-subject split (val = performers {sorted(VAL_SUBJECTS)}) -> {split_file}")
    return result


def print_split_summary(datasets, kind):
    total = sum(len(d) for d in datasets.values())
    for key, ds in datasets.items():
        x, _ = ds[0]
        per_class = Counter(ds.labels)
        subjects = sorted({parse_name(n)["subject"] for n in ds.names})
        print(f"[{kind} {key:<5}] {len(ds):>6} clips ({len(ds) / total:4.0%})  shape ({len(ds)}, {', '.join(map(str, x.shape))})"
              f"  | {min(per_class.values())} clips per class | {len(subjects)} performers")


# ------------------------------------------------------------------------------------------------ RGB (X3D-S)
X3D_MEAN = (0.45, 0.45, 0.45)
X3D_STD = (0.225, 0.225, 0.225)


class RGBDataset(Dataset):
    """One item = (clip, label). clip: float32 (3, num_frames, size, size). label: 0..n_classes-1."""

    def __init__(self, video_dir, names, actions, num_frames=13, side=182, size=160, train=False):
        self.video_dir = Path(video_dir)
        self.names = list(names)
        self.labels = [actions.index(parse_name(n)["action"]) for n in self.names]
        self.num_frames, self.side, self.size, self.train = num_frames, side, size, train
        self.mean = torch.tensor(X3D_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(X3D_STD).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.names)

    def video_path(self, name):
        for ext in (".mp4", ".avi"):                       # a pre-shrunk .mp4 wins over the original .avi
            path = self.video_dir / f"{name}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"no video for {name} in {self.video_dir}")

    def frame_indices(self, n_frames):
        """Cut the clip into num_frames equal segments, take one frame per segment (random in training, middle otherwise)."""
        edges = np.linspace(0, n_frames, self.num_frames + 1)
        if self.train:
            picks = [np.random.uniform(a, b) for a, b in zip(edges[:-1], edges[1:])]
        else:
            picks = (edges[:-1] + edges[1:]) / 2
        return np.clip(np.floor(picks).astype(int), 0, n_frames - 1)

    def shrink(self, frame):
        """BGR frame of any size -> RGB (size, size): short side to `side` pixels, then a centre crop."""
        h, w = frame.shape[:2]
        scale = self.side / min(h, w)
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        top, left = (h - self.size) // 2, (w - self.size) // 2
        return cv2.cvtColor(frame[top:top + self.size, left:left + self.size], cv2.COLOR_BGR2RGB)

    def read_clip(self, path):
        """-> uint8 array (num_frames, size, size, 3)"""
        cap = cv2.VideoCapture(str(path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            cap.release()
            raise ValueError(f"cannot read {path}")
        wanted = self.frame_indices(n_frames)
        frames = {}
        t = 0
        while t <= wanted.max() and cap.grab():            # walk once through the video, decode only wanted frames
            if t in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    frames[t] = self.shrink(frame)
            t += 1
        cap.release()
        if not frames:
            raise ValueError(f"no frame decoded from {path}")
        clip, last = [], frames[min(frames)]
        for t in wanted:                                   # a video shorter than its header says: repeat the last frame
            last = frames.get(t, last)
            clip.append(last)
        return np.stack(clip)

    def __getitem__(self, i):
        clip = self.read_clip(self.video_path(self.names[i]))
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0      # (T,H,W,C) -> (C,T,H,W)
        return (clip - self.mean) / self.std, self.labels[i]


def rgb_dataloader(video_dir, actions, split_file, num_frames=13, side=182, size=160, batch_size=8, num_workers=2,
                   balance=True, seed=0):
    """-> ({'train': DataLoader, 'val': ..., 'test': ...}, {'train': RGBDataset, ...})"""
    video_dir = Path(video_dir)
    names = sorted({p.stem for p in list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4"))
                    if parse_name(p.stem)["action"] in actions})
    if not names:
        raise FileNotFoundError(f"no videos of the chosen classes in {video_dir}")
    split = load_or_make_split(names, split_file, balance, seed)
    datasets, loaders = {}, {}
    for key in SPLITS:
        datasets[key] = RGBDataset(video_dir, split[key], actions, num_frames, side, size, train=(key == "train"))
        loaders[key] = DataLoader(datasets[key], batch_size=batch_size, shuffle=(key == "train"),
                                  num_workers=num_workers, pin_memory=True, drop_last=(key == "train"))
    return loaders, datasets


# ------------------------------------------------------------------------------------------------ HMDB (cross-dataset test)
class HMDBDataset(RGBDataset):
    """Same clip format as RGBDataset; the clip list comes from the class sub-folders, labels from the yaml order."""

    def __init__(self, hmdb_dir, hmdb_classes, num_frames=13, side=182, size=160):
        self.video_dir = Path(hmdb_dir)
        self.paths, self.labels = [], []
        for label, cls in enumerate(hmdb_classes):
            clips = sorted((self.video_dir / cls).glob("*.avi"))
            if not clips:
                print(f"[hmdb] WARNING: no clips for class '{cls}' in {self.video_dir}")
            self.paths += clips
            self.labels += [label] * len(clips)
        self.names = [p.stem for p in self.paths]
        self.num_frames, self.side, self.size, self.train = num_frames, side, size, False
        self.mean = torch.tensor(X3D_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(X3D_STD).view(3, 1, 1, 1)

    def __getitem__(self, i):
        clip = self.read_clip(self.paths[i])
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        return (clip - self.mean) / self.std, self.labels[i]


def hmdb_dataloader(hmdb_dir, hmdb_classes, num_frames=13, side=182, size=160, batch_size=8, num_workers=2):
    """-> (DataLoader, HMDBDataset) holding ALL HMDB clips of our classes: a test set only."""
    dataset = HMDBDataset(hmdb_dir, hmdb_classes, num_frames, side, size)
    if len(dataset) == 0:
        raise FileNotFoundError(f"no HMDB clips of our classes in {hmdb_dir}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader, dataset


# ------------------------------------------------------------------------------------------------ skeleton (CTR-GCN)
NUM_JOINTS = 25
MAX_PERSONS = 2


def read_skeleton_file(path):
    """.skeleton text file -> {body_id: {frame: (25, 7) array}} and the number of frames.
    The 7 numbers of a joint: x y z (metres, Kinect space), depthX depthY, colorX colorY (pixels in the colour video)."""
    lines = Path(path).read_text().split("\n")
    i = 0
    n_frames = int(lines[i]); i += 1
    bodies = {}
    for t in range(n_frames):
        n_bodies = int(lines[i]); i += 1
        for _ in range(n_bodies):
            body_id = lines[i].split()[0]; i += 1
            n_joints = int(lines[i]); i += 1
            joints = np.array([lines[i + j].split()[:7] for j in range(n_joints)], dtype=np.float32)
            i += n_joints
            bodies.setdefault(body_id, {})[t] = joints
    return bodies, n_frames


def motion(frames):
    """How much a body moves over its frames (sum of the variance of x, y, z). Ghost bodies hardly move."""
    xyz = np.stack(list(frames.values()))
    return float(xyz[..., 0].var() + xyz[..., 1].var() + xyz[..., 2].var())


def pick_persons(bodies, n_frames, n_persons):
    """Keep the n_persons bodies with the most motion -> array (T, 25, 7, 2); a missing person stays all zeros."""
    ranked = sorted(bodies.values(), key=motion, reverse=True)[:n_persons]
    data = np.zeros((n_frames, NUM_JOINTS, 7, MAX_PERSONS), dtype=np.float32)
    for m, frames in enumerate(ranked):
        last = frames[min(frames)]
        for t in range(n_frames):                          # frames where this body is missing repeat the last pose
            last = frames.get(t, last)
            data[t, :, :, m] = last
    return data


def center_and_rotate(data):
    """3D: put the spine base of the main person (first frame) at the origin and make the body face the camera."""
    first = data[0, :, :, 0]
    origin = first[1]                                      # joint 2 (index 1) = middle of the spine
    up = first[20] - first[0]                              # spine base -> spine shoulder
    up = up / (np.linalg.norm(up) + 1e-6)
    right = first[8] - first[4]                            # left shoulder -> right shoulder
    right = right - np.dot(right, up) * up
    right = right / (np.linalg.norm(right) + 1e-6)
    forward = np.cross(right, up)
    rotation = np.stack([right, up, forward])              # rows = new x, y, z axes
    present = np.abs(data).sum(axis=2, keepdims=True) > 0  # keep missing persons at exactly zero
    data = (data - origin[None, None, :, None]) * present
    return np.einsum("ij,tvjm->tvim", rotation, data).astype(np.float32)


def kinect_to_2d13(data):
    """(T, 25, 7, M) -> (T, 13, 2, M): where the 13 shared joints are in the colour video, in pixels.
    A joint whose position is not a normal number (the Kinect lost it) becomes (0, 0) = not found."""
    xy = data[:, NTU25_TO_13, 5:7, :].copy()
    bad = ~np.isfinite(xy).all(axis=2) | (np.abs(xy) > 10000).any(axis=2)        # (T, 13, M)
    xy[np.repeat(bad[:, :, None, :], 2, axis=2)] = 0
    return xy


def resample(data, num_frames):
    """Pick num_frames frames spread evenly over the clip."""
    idx = np.linspace(0, len(data) - 1, num_frames).round().astype(int)
    return data[idx]


def skeleton_to_model_input(path, num_frames=64, skeleton_format="3d25"):
    """One .skeleton file -> CTR-GCN input: (3, T, 25, 2) for "3d25" or (2, T, 13, 2) for "2d13"."""
    name = Path(path).stem
    n_persons = 2 if 50 <= parse_name(name)["action"] <= 60 else 1
    bodies, n_frames = read_skeleton_file(path)
    if not bodies:
        raise ValueError(f"no body in {name}")
    data = pick_persons(bodies, n_frames, n_persons)       # (T, 25, 7, M)
    if skeleton_format == "3d25":
        data = center_and_rotate(data[:, :, :3, :])
    elif skeleton_format == "2d13":
        data = normalize_2d(kinect_to_2d13(data))          # the same normalisation as for the HMDB pose estimates
    else:
        raise ValueError("skeleton_format must be '3d25' or '2d13'")
    data = resample(data, num_frames)
    return data.transpose(2, 0, 1, 3)                      # (T, V, C, M) -> (C, T, V, M)


def format_of_cache(cached):
    """Caches made before the 2D format existed have no label: they are 3D."""
    return str(cached["skeleton_format"]) if "skeleton_format" in cached.files else "3d25"


def prepare_skeleton_cache(skeleton_dir, cache_file, actions, num_frames=64, skeleton_format="3d25"):
    """Parse every .skeleton file of the chosen classes once and store them together in one .npz."""
    skeleton_dir, cache_file = Path(skeleton_dir), Path(cache_file)
    files = sorted(p for p in skeleton_dir.glob("*.skeleton") if parse_name(p.stem)["action"] in actions)
    cached = np.load(cache_file) if cache_file.exists() else None
    cache_fits = cached is not None and cached["x"].shape[2] == num_frames and format_of_cache(cached) == skeleton_format
    if not files:
        if cached is None:
            raise FileNotFoundError(f"no .skeleton files of the chosen classes in {skeleton_dir}")
        if not cache_fits:                                 # e.g. on Colab: only the cache was uploaded, not the text files
            raise ValueError(f"{cache_file} holds {format_of_cache(cached)} skeletons with {cached['x'].shape[2]} frames, "
                             f"but the yaml asks for {skeleton_format} with {num_frames} frames")
        print(f"[skeleton] no .skeleton text files here - using the existing cache {cache_file}")
        return cache_file
    if cache_fits and set(cached["names"].tolist()) == {p.stem for p in files}:
        print(f"[skeleton] cache is up to date: {cache_file}")
        return cache_file
    if cached is not None:
        print("[skeleton] cache does not match the yaml or the files on disk - rebuilding")
    print(f"[skeleton] parsing {len(files)} .skeleton files into the {skeleton_format} format (a few minutes) ...")
    x, names = [], []
    for k, path in enumerate(files, 1):
        try:
            x.append(skeleton_to_model_input(path, num_frames, skeleton_format))
            names.append(path.stem)
        except ValueError as e:
            print(f"[skeleton] skipped {path.name}: {e}")
        if k % 2000 == 0:
            print(f"[skeleton] {k}/{len(files)}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, x=np.stack(x), names=np.array(names), skeleton_format=np.array(skeleton_format))
    print(f"[skeleton] saved {len(names)} clips as array {np.stack(x).shape} -> {cache_file}")
    return cache_file


def augment_skeleton(clip):
    """Training only: turn the skeleton a little and make it slightly bigger or smaller."""
    if clip.shape[0] == 3:                                 # 3D: turn around the vertical axis (x and z change)
        angle, a, b = np.deg2rad(np.random.uniform(-15, 15)), 0, 2
    else:                                                  # 2D: turn inside the image (x and y change)
        angle, a, b = np.deg2rad(np.random.uniform(-10, 10)), 0, 1
    cos, sin = np.cos(angle), np.sin(angle)
    first, second = clip[a].clone(), clip[b].clone()
    clip[a], clip[b] = cos * first + sin * second, -sin * first + cos * second
    return clip * np.random.uniform(0.9, 1.1)


class SkeletonDataset(Dataset):
    """One item = (clip, label). clip: float32 (3, T, 25, 2) or (2, T, 13, 2), depending on the skeleton format."""

    def __init__(self, cache_file, names, actions, train=False):
        cached = np.load(cache_file)
        index = {name: i for i, name in enumerate(cached["names"].tolist())}
        self.names = [n for n in names if n in index]
        self.x = cached["x"][[index[n] for n in self.names]]
        self.labels = [actions.index(parse_name(n)["action"]) for n in self.names]
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        clip = torch.from_numpy(self.x[i].copy())
        if self.train:
            clip = augment_skeleton(clip)
        return clip, self.labels[i]


def skeleton_dataloader(skeleton_dir, cache_file, actions, split_file, num_frames=64, batch_size=32, num_workers=2,
                        balance=True, seed=0, skeleton_format="3d25"):
    """-> ({'train': DataLoader, 'val': ..., 'test': ...}, {'train': SkeletonDataset, ...})"""
    cache_file = prepare_skeleton_cache(skeleton_dir, cache_file, actions, num_frames, skeleton_format)
    names = np.load(cache_file)["names"].tolist()
    split = load_or_make_split(names, split_file, balance, seed)
    datasets, loaders = {}, {}
    for key in SPLITS:
        datasets[key] = SkeletonDataset(cache_file, split[key], actions, train=(key == "train"))
        loaders[key] = DataLoader(datasets[key], batch_size=batch_size, shuffle=(key == "train"),
                                  num_workers=num_workers, pin_memory=True, drop_last=(key == "train"))
    return loaders, datasets


# ------------------------------------------------------------------------------------------------ HMDB skeletons (cross-dataset test)
class HMDBSkeletonDataset(Dataset):
    """The good HMDB skeletons made by pose_estimation.py: (2, T, 13, 2) clips in the same format as NTU "2d13"."""

    def __init__(self, skeleton_file, hmdb_classes):
        saved = np.load(skeleton_file)
        self.x = saved["x"]
        self.names = saved["names"].tolist()
        self.labels = [hmdb_classes.index(c) for c in saved["classes"].tolist()]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i].copy()), self.labels[i]


def hmdb_skeleton_dataloader(skeleton_file, hmdb_classes, batch_size=32):
    """-> (DataLoader, HMDBSkeletonDataset): a test set only, never used for training."""
    dataset = HMDBSkeletonDataset(skeleton_file, hmdb_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), dataset


# ------------------------------------------------------------------------------------------------ quick check
@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    actions = [int(c.ntu60) for c in cfg.classes]
    split_file = to_absolute_path(cfg.split_file)
    rgb_dir, skel_dir = Path(to_absolute_path(cfg.processed_RGB_FOLDER)), Path(to_absolute_path(cfg.processed_skeleton_FOLDER))

    # the split is made from the clips that exist in BOTH folders, so the two models always see the same clips
    rgb_names = {p.stem for p in list(rgb_dir.glob("*.avi")) + list(rgb_dir.glob("*.mp4"))}
    skel_names = {p.stem for p in skel_dir.glob("*.skeleton")}
    both = sorted(n for n in rgb_names & skel_names if parse_name(n)["action"] in actions)
    print(f"[data] {len(rgb_names)} rgb clips, {len(skel_names)} skeleton clips, {len(both)} clips of our classes in both")
    load_or_make_split(both, split_file, cfg.split_balance, cfg.split_seed)

    rgb_loaders, rgb_datasets = rgb_dataloader(
        video_dir=to_absolute_path(cfg.processed_RGB_FOLDER), actions=actions, split_file=split_file,
        num_frames=cfg.rgb_num_frames, side=cfg.rgb_side, size=cfg.rgb_size,
        batch_size=cfg.rgb_batch_size, num_workers=cfg.num_workers)
    print_split_summary(rgb_datasets, "rgb")
    x, y = next(iter(rgb_loaders["train"]))
    print(f"[rgb batch] clips {tuple(x.shape)} {x.dtype}  labels {y.tolist()}   = X3D-S input (N, C, T, H, W)\n")

    skel_loaders, skel_datasets = skeleton_dataloader(
        skeleton_dir=to_absolute_path(cfg.processed_skeleton_FOLDER),
        cache_file=to_absolute_path(cfg.skeleton_cache_file),
        actions=actions, split_file=split_file, num_frames=cfg.skeleton_num_frames,
        batch_size=cfg.skeleton_batch_size, num_workers=cfg.num_workers, skeleton_format=cfg.skeleton_format)
    print_split_summary(skel_datasets, "skel")
    x, y = next(iter(skel_loaders["train"]))
    print(f"[skel batch] clips {tuple(x.shape)} {x.dtype}  labels {y.tolist()}   = CTR-GCN input (N, C, T, V, M)")

    same = all(rgb_datasets[k].names == skel_datasets[k].names for k in SPLITS)
    print(f"\n[check] RGB and skeleton use exactly the same clips in every split: {same}\n")

    hmdb_loader, hmdb_dataset = hmdb_dataloader(
        hmdb_dir=to_absolute_path(cfg.processed_HMDB1_FOLDER), hmdb_classes=[str(c.hmdb) for c in cfg.classes],
        num_frames=cfg.rgb_num_frames, side=cfg.rgb_side, size=cfg.rgb_size,
        batch_size=cfg.rgb_batch_size, num_workers=cfg.num_workers)
    x, y = next(iter(hmdb_loader))
    per_class = Counter(hmdb_dataset.labels)
    print(f"[hmdb test ] {len(hmdb_dataset):>6} clips  shape ({len(hmdb_dataset)}, {', '.join(map(str, x.shape[1:]))})"
          f"  | {min(per_class.values())}-{max(per_class.values())} clips per class (not balanced: test only)")
    print(f"[hmdb batch] clips {tuple(x.shape)} {x.dtype}  labels {y.tolist()}   = same format as the NTU RGB clips")

    hmdb_skeletons = Path(to_absolute_path(cfg.skeleton_estimates_HMDB1_FOLDER)) / "hmdb_skeleton_2d13.npz"
    if cfg.skeleton_format == "2d13" and hmdb_skeletons.exists():          # made by pose_estimation.py
        loader, dataset = hmdb_skeleton_dataloader(hmdb_skeletons, [str(c.hmdb) for c in cfg.classes], cfg.skeleton_batch_size)
        x, y = next(iter(loader))
        per_class = Counter(dataset.labels)
        print(f"\n[hmdb skel ] {len(dataset):>6} clips  shape ({len(dataset)}, {', '.join(map(str, x.shape[1:]))})"
              f"  | {min(per_class.values())}-{max(per_class.values())} clips per class (only clips with a good skeleton)")


if __name__ == "__main__":
    main()