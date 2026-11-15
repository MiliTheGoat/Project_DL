"""Shrink the data once on your own computer, so that Google Colab only needs small files.

Run from the Project_DL folder:
    uv run src/rgb_vs_skeleton/shrink_data.py
It is safe to stop and run again: finished clips are skipped. The original files are never changed or deleted.

RGB       data/processed/rgb/*.avi (1920x1080)  ->  data/processed/rgb_small/*.mp4 (short side 256 px, same frames)
          -> packed into ONE file data/processed/rgb.zip, which the Colab notebook unpacks to its fast local disk
SKELETON  data/processed/skeleton/*.skeleton (text)  ->  data/processed/skeleton_cache.npz (made by loaders.py)
          Only built if it is missing. It already is the small version of the skeleton data.

Upload to Google Drive: src/, configs/, notebooks/, data/processed/rgb.zip, data/processed/splits.csv,
data/processed/skeleton_cache.npz. NOT needed there: data/raw, data/processed/rgb, data/processed/rgb_small.

Video conversion uses ffmpeg when it is available (fast, small files):  uv add imageio-ffmpeg
Without ffmpeg it falls back to OpenCV (works everywhere, but slower and the files are bigger).
"""
import random
import shutil
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from loaders import prepare_skeleton_cache


def find_ffmpeg():
    """Path of an ffmpeg program, or None."""
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def shrink_with_ffmpeg(ffmpeg, source, target, short_side):
    command = [ffmpeg, "-loglevel", "error", "-y", "-i", str(source),
               "-vf", f"scale=-2:{short_side}",            # height = short_side, width keeps the shape of the picture
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", str(target)]
    subprocess.run(command, check=True)


def shrink_with_opencv(source, target, short_side):
    reader = cv2.VideoCapture(str(source))
    fps = reader.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = short_side / min(h, w)
        new_w, new_h = round(w * scale / 2) * 2, round(h * scale / 2) * 2      # even numbers, as video codecs want
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))
        writer.write(frame)
    reader.release()
    if writer is None:
        raise ValueError("no frame could be read")
    writer.release()


def shrink_one(source, target, short_side, ffmpeg):
    """Returns None when the clip is fine, otherwise the error text."""
    if target.exists() and target.stat().st_size > 0:
        return None                                        # done in an earlier run
    temporary = target.with_name(target.stem + "_part.mp4")
    try:
        if ffmpeg:
            shrink_with_ffmpeg(ffmpeg, source, temporary, short_side)
        else:
            shrink_with_opencv(source, temporary, short_side)
        temporary.replace(target)                          # a clip only gets its final name when it is complete
        return None
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return f"{source.name}: {error}"


def frame_count(path):
    video = cv2.VideoCapture(str(path))
    n = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    return n, height


def folder_size_gb(folder, pattern):
    return sum(p.stat().st_size for p in Path(folder).glob(pattern)) / 1024 ** 3


def shrink_rgb(source_dir, target_dir, zip_path, short_side, workers):
    source_dir, target_dir, zip_path = Path(source_dir), Path(target_dir), Path(zip_path)
    sources = sorted(source_dir.glob("*.avi"))
    if not sources:
        raise FileNotFoundError(f"no .avi videos in {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    todo = [s for s in sources if not (target_dir / f"{s.stem}.mp4").exists()]
    print(f"[rgb] {len(sources)} videos, {len(sources) - len(todo)} already shrunk, {len(todo)} to do | "
          f"short side {short_side} px | {workers} at a time | using {'ffmpeg' if ffmpeg else 'OpenCV (slow; uv add imageio-ffmpeg)'}")

    errors, start = [], time.perf_counter()

    def job(source):
        return shrink_one(source, target_dir / f"{source.stem}.mp4", short_side, ffmpeg)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, error in enumerate(pool.map(job, todo), 1):
            if error:
                errors.append(error)
            if done % 100 == 0 or done == len(todo):
                seconds = time.perf_counter() - start
                left = (len(todo) - done) * seconds / done
                print(f"[rgb] {done}/{len(todo)}  {seconds / done:.2f} s per clip  about {left / 60:.0f} min left")

    # check a few clips: same number of frames as the original, and the new height
    shrunk = sorted(target_dir.glob("*.mp4"))
    for target in random.Random(0).sample(shrunk, min(20, len(shrunk))):
        frames_new, height = frame_count(target)
        frames_old, _ = frame_count(source_dir / f"{target.stem}.avi")
        if abs(frames_new - frames_old) > 1 or height != short_side:
            errors.append(f"{target.name}: {frames_new} frames (original {frames_old}), height {height}")

    before, after = folder_size_gb(source_dir, "*.avi"), folder_size_gb(target_dir, "*.mp4")
    print(f"[rgb] {len(shrunk)} of {len(sources)} clips shrunk: {before:.1f} GB -> {after:.2f} GB")
    if errors:
        print(f"[rgb] {len(errors)} PROBLEMS (run the script again to retry; the zip is not made until all clips are fine):")
        for error in errors[:10]:
            print("     ", error)
        return

    # one zip for Google Drive (no compression: mp4 is already compressed)
    names_in_zip = set()
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            names_in_zip = set(z.namelist())
    if names_in_zip == {f"rgb/{p.name}" for p in shrunk}:
        print(f"[rgb] zip is up to date: {zip_path}")
        return
    print(f"[rgb] packing {len(shrunk)} clips into {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for p in shrunk:
            z.write(p, f"rgb/{p.name}")
    print(f"[rgb] zip ready: {zip_path} ({zip_path.stat().st_size / 1024 ** 3:.2f} GB) - upload this file to Google Drive")


def shrink_skeleton(skeleton_dir, cache_file, actions, num_frames, skeleton_format):
    cache_file = Path(cache_file)
    text_gb = folder_size_gb(skeleton_dir, "*.skeleton")
    prepare_skeleton_cache(skeleton_dir, cache_file, actions, num_frames, skeleton_format)   # only if missing / outdated
    print(f"[skeleton] {text_gb:.2f} GB of text files -> {cache_file.stat().st_size / 1024 ** 3:.3f} GB cache "
          f"({cache_file}) - upload this file to Google Drive")


@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    shrink_rgb(source_dir=to_absolute_path(cfg.processed_RGB_FOLDER),
               target_dir=to_absolute_path(cfg.shrunk_RGB_FOLDER),
               zip_path=to_absolute_path(cfg.shrunk_RGB_ZIP),
               short_side=cfg.shrink_short_side, workers=cfg.shrink_workers)
    print()
    shrink_skeleton(skeleton_dir=to_absolute_path(cfg.processed_skeleton_FOLDER),
                    cache_file=to_absolute_path(cfg.skeleton_cache_file),
                    actions=[int(c.ntu60) for c in cfg.classes], num_frames=cfg.skeleton_num_frames,
                    skeleton_format=cfg.skeleton_format)


if __name__ == "__main__":
    main()