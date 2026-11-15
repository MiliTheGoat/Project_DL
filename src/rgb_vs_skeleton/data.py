"""NTU RGB+D 60: from the big RGB zips to ONLY the clips of our classes.
Run from the Project_DL folder (settings come from configs/main_config.yaml):
    python src/preprocess_rgb.py                      # real run
    python src/preprocess_rgb.py dry_run=true         # just print what would happen
    python src/preprocess_rgb.py delete_zip_after_extract=false

For every RGB zip found (setup 001..017; it is fine if only some exist) the function
  1. lists the videos inside the zip (nothing is extracted yet),
  2. keeps only clips of the chosen classes that are NOT on the missing-skeleton list,
  3. extracts just those clips into the output folder (flat, named like S001C001P001R001A008.avi),
  4. checks that every kept clip is on disk with the right size,
  5. only then deletes the zip.
Extracting selectively gives the same end state as "unzip everything, then delete the rest", but never
writes the ~90% of the archive you would throw away (matters on Colab / HPC disk quotas).
"""
import re
import shutil
import zipfile
import difflib
import subprocess
import tempfile
from pathlib import Path
from collections import Counter

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

NAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")
ZIP_RE = re.compile(r"_rgb_s(\d{3})\.zip$", re.IGNORECASE)   # nturgbd_rgb_s001.zip (NOT nturgbd_skeletons_...: "nturgbd" contains "rgb")
NUM_SETUPS = 17


def load_missing(missing_skeleton_file) -> set:
    """Clip names from the authors' file (its header sentences are ignored)."""
    p = Path(missing_skeleton_file)
    if not p.exists():
        raise FileNotFoundError(f"missing-skeleton file not found: {p}")
    names = set(re.findall(r"S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}", p.read_text()))
    print(f"[missing list] {len(names)} clips will be excluded ({p})")
    return names


def find_rgb_zips(input_dir: Path):
    """-> [(setup_number, path)] sorted; accepts any subset of setups 1..17."""
    found = []
    for f in sorted(input_dir.glob("*.zip")):
        m = ZIP_RE.search(f.name)
        if m and 1 <= int(m.group(1)) <= NUM_SETUPS:
            found.append((int(m.group(1)), f))
    return sorted(found)


def preprocess_rgb(input_dir, output_dir, ntu_ids, missing_skeleton_file, delete_zip=True, dry_run=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    wanted = {int(a) for a in ntu_ids}
    bad = sorted(a for a in wanted if not 1 <= a <= 60)
    if bad:
        raise ValueError(f"classes {bad} are not in NTU RGB+D 60 (A61-A120 are NTU-120 only)")
    print(f"[start] input={input_dir}  output={output_dir}  classes={sorted(wanted)}  "
          f"delete_zip={delete_zip}  dry_run={dry_run}")
    missing = load_missing(missing_skeleton_file)

    zips = find_rgb_zips(input_dir)
    if not zips:
        raise FileNotFoundError(f"no RGB zip (e.g. nturgbd_rgb_s001.zip) found in {input_dir}")
    print(f"[found] {len(zips)} RGB zip(s): setups {[s for s, _ in zips]}  "
          f"(not found: {[s for s in range(1, NUM_SETUPS + 1) if s not in dict(zips)]})")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    totals = {"extracted": 0, "kept_already": 0, "other_class": 0, "missing_skel": 0, "zips_deleted": 0}
    for setup, zpath in zips:
        print(f"\n[setup {setup:03d}] opening {zpath.name} ({zpath.stat().st_size / 1e9:.1f} GB)")
        with zipfile.ZipFile(zpath) as zf:
            keep, other_class, missing_skel, not_ntu = [], 0, 0, 0
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".avi"):
                    continue
                m = NAME_RE.search(Path(info.filename).name)
                if m is None:
                    not_ntu += 1
                elif int(m.group(5)) not in wanted:
                    other_class += 1
                elif m.group(0) in missing:
                    missing_skel += 1
                else:
                    keep.append((info, m.group(0)))
            print(f"[setup {setup:03d}] videos in zip: {len(keep) + other_class + missing_skel + not_ntu} | "
                  f"to keep: {len(keep)} | other classes (skipped): {other_class} | "
                  f"on missing list (skipped): {missing_skel} | unrecognised names: {not_ntu}")
            totals["other_class"] += other_class
            totals["missing_skel"] += missing_skel
            if dry_run:
                print(f"[setup {setup:03d}] DRY RUN: would extract {len(keep)} clips"
                      f"{' and delete the zip' if delete_zip else ''}")
                continue

            for i, (info, name) in enumerate(keep, 1):
                target = output_dir / f"{name}.avi"
                if target.exists() and target.stat().st_size == info.file_size:
                    totals["kept_already"] += 1            # resume support after an interrupted run
                    continue
                tmp = target.with_name(target.name + ".part")
                with zf.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(target)
                totals["extracted"] += 1
                if i % 200 == 0 or i == len(keep):
                    print(f"[setup {setup:03d}] extracted {i}/{len(keep)}")

            # SAFETY: a zip is only ever deleted if clips were really extracted from it and all are verified on disk
            ok = bool(keep) and all((output_dir / f"{n}.avi").exists() and (output_dir / f"{n}.avi").stat().st_size == i.file_size
                                    for i, n in keep)
            print(f"[setup {setup:03d}] verification: "
                  f"{'all kept clips are on disk' if ok else 'PROBLEM - no clips of interest found or some missing'}")
        # the zip is closed here (required on Windows before it can be deleted)
        if ok and delete_zip:
            zpath.unlink()
            totals["zips_deleted"] += 1
            print(f"[setup {setup:03d}] deleted {zpath.name}")
        elif not ok:
            print(f"[setup {setup:03d}] zip KEPT because verification failed - rerun the script")
        else:
            print(f"[setup {setup:03d}] zip kept (delete_zip=False)")

    print(f"\n[done] {totals}")
    if not dry_run:
        counts = {a: len(list(output_dir.glob(f"*A{a:03d}.avi"))) for a in sorted(wanted)}
        print(f"[clips per class in {output_dir}] {counts}")
    return totals

def find_skeleton_zips(input_dir: Path):
    """Skeleton archive(s), e.g. nturgbd_skeletons_s001_to_s017.zip (name just has to contain 'skeleton')."""
    return sorted(f for f in Path(input_dir).glob("*.zip") if "skeleton" in f.name.lower())

def preprocess_skeleton(input_dir, output_dir, ntu_ids, missing_skeleton_file, prune_unwanted=True, dry_run=False):
    """Extract ONLY the .skeleton files of the chosen classes (minus the missing-skeleton list). The zip is NEVER deleted,
    so you can re-run this whenever the class list changes: only the difference is extracted (existing files are
    skipped) and, if prune_unwanted=True, files of classes you removed are deleted from the output folder."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    wanted = {int(a) for a in ntu_ids}
    bad = sorted(a for a in wanted if not 1 <= a <= 60)
    if bad:
        raise ValueError(f"classes {bad} are not in NTU RGB+D 60 (A61-A120 are NTU-120 only)")
    print(f"[skeleton start] input={input_dir}  output={output_dir}  classes={sorted(wanted)}  "
          f"prune_unwanted={prune_unwanted}  dry_run={dry_run}")
    missing = load_missing(missing_skeleton_file)

    zips = find_skeleton_zips(input_dir)
    if not zips:
        raise FileNotFoundError(f"no skeleton zip (e.g. nturgbd_skeletons_s001_to_s017.zip) found in {input_dir}")
    print(f"[skeleton found] {[z.name for z in zips]}")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    totals = {"extracted": 0, "already_there": 0, "other_class": 0, "missing_skel": 0, "pruned": 0}
    expected = {}                                            # clip name -> size it must have on disk
    for zpath in zips:
        print(f"\n[skeleton] opening {zpath.name} ({zpath.stat().st_size / 1e9:.1f} GB)")
        with zipfile.ZipFile(zpath) as zf:
            todo, other_class, missing_skel, not_ntu = [], 0, 0, 0
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".skeleton"):
                    continue
                m = NAME_RE.search(Path(info.filename).name)
                if m is None:
                    not_ntu += 1
                elif int(m.group(5)) not in wanted:
                    other_class += 1
                elif m.group(0) in missing:
                    missing_skel += 1
                else:
                    expected[m.group(0)] = info.file_size
                    target = output_dir / f"{m.group(0)}.skeleton"
                    if target.exists() and target.stat().st_size == info.file_size:
                        totals["already_there"] += 1         # earlier run already extracted it
                    else:
                        todo.append((info, m.group(0)))
            totals["other_class"] += other_class
            totals["missing_skel"] += missing_skel
            print(f"[skeleton] to keep: {len(todo) + totals['already_there']} | already extracted: {totals['already_there']} | "
                  f"still to extract: {len(todo)} | other classes (skipped): {other_class} | "
                  f"on missing list (skipped): {missing_skel} | unrecognised names: {not_ntu}")
            if dry_run:
                continue
            for i, (info, name) in enumerate(todo, 1):
                target = output_dir / f"{name}.skeleton"
                tmp = target.with_name(target.name + ".part")
                with zf.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(target)
                totals["extracted"] += 1
                if i % 500 == 0 or i == len(todo):
                    print(f"[skeleton] extracted {i}/{len(todo)}")

    # sync: remove .skeleton files of classes that are no longer wanted (safe: the zip is kept)
    stale = [f for f in output_dir.glob("*.skeleton")
             if NAME_RE.search(f.name) and NAME_RE.search(f.name).group(0) not in expected] if output_dir.exists() else []
    if prune_unwanted and expected:
        print(f"[skeleton] {'would prune' if dry_run else 'pruning'} {len(stale)} file(s) of classes/clips no longer wanted")
        if not dry_run:
            for f in stale:
                f.unlink()
            totals["pruned"] = len(stale)
    elif stale:
        print(f"[skeleton] {len(stale)} extra file(s) left in place (prune_unwanted=False)")

    if not dry_run:
        ok = all((output_dir / f"{n}.skeleton").exists() and (output_dir / f"{n}.skeleton").stat().st_size == s
                 for n, s in expected.items())
        print(f"[skeleton] verification: {'all wanted clips are on disk' if ok else 'PROBLEM - some clips missing, rerun'}")
        counts = {a: sum(1 for n in expected if int(n[-3:]) == a) for a in sorted(wanted)}
        print(f"[skeleton clips per class] {counts}")
    print(f"[skeleton done] {totals} | zip(s) kept (skeleton zips are never deleted)")
    return totals



HMDB51_CLASSES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble",
    "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit", "hug", "jump", "kick",
    "kick_ball", "kiss", "laugh", "pick", "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse",
    "run", "shake_hands", "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
    "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]

def check_hmdb_labels(hmdb_classes):
    """Every label must be an exact HMDB-51 folder name; otherwise stop with a hint (e.g. 'sit_down' -> 'sit')."""
    bad = [c for c in hmdb_classes if c not in HMDB51_CLASSES]
    if bad:
        hints = {c: difflib.get_close_matches(c, HMDB51_CLASSES, n=2, cutoff=0.4) for c in bad}
        raise ValueError(f"not HMDB-51 folder names: {hints}. Use the short folder names, e.g. 'sit' not 'sit_down'.")
    dup = sorted({c for c in hmdb_classes if list(hmdb_classes).count(c) > 1})
    if dup:
        raise ValueError(f"HMDB classes used twice in the config: {dup}")
 
 
def _find_7z():
    for exe in ("7z", "7zz", "7za"):
        if shutil.which(exe):
            return shutil.which(exe)
    for p in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if Path(p).exists():
            return p
    return None
 
 
def _run_7z(sevenz, archive, out_dir, *extra):
    r = subprocess.run([sevenz, "x", str(archive), f"-o{out_dir}", "-y", *extra], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"7-Zip failed on {archive}: {r.stderr.strip() or r.stdout.strip()[-300:]}")
 
def _extract_from_rars(rars, classes, output_dir):
    """Official release: hmdb51_org.rar holds one <class>.rar per class (RAR cannot be read by Python itself, so 7-Zip is used).
    Only the wanted classes are unpacked, one at a time, through a temp folder on the same drive."""
    sevenz = _find_7z()
    if sevenz is None:
        raise RuntimeError("RAR archive(s) found but 7-Zip is not installed. Install 7-Zip (https://www.7-zip.org) and re-run, "
                           "or unpack hmdb51_org.rar by hand so that <class>/*.avi folders exist in the raw folder.")
    for cls in classes:
        dest = output_dir / cls
        if dest.exists() and len(list(dest.glob("*.avi"))) >= 100:
            print(f"[hmdb] {cls}: already extracted ({len(list(dest.glob('*.avi')))} clips), skipping")
            continue
        with tempfile.TemporaryDirectory(dir=output_dir, prefix="_tmp_") as tmp:
            tmp = Path(tmp)
            direct = [r for r in rars if r.stem == cls]
            if direct:                                           # a per-class <cls>.rar lying around
                _run_7z(sevenz, direct[0], tmp / "x")
            else:                                                # outer archive: pull out just this class's inner rar
                for outer in rars:
                    _run_7z(sevenz, outer, tmp / "outer", f"-ir!{cls}.rar")
                    inner = list((tmp / "outer").rglob(f"{cls}.rar"))
                    if inner:
                        _run_7z(sevenz, inner[0], tmp / "x")
                        break
            clips = sorted((tmp / "x").rglob("*.avi")) if (tmp / "x").exists() else []
            if not clips:
                raise RuntimeError(f"could not find the clips of '{cls}' inside the RAR archive(s). Unpack hmdb51_org.rar once by "
                                   f"hand with 7-Zip so that <class>/*.avi folders exist in the raw folder, then re-run.")
            dest.mkdir(parents=True, exist_ok=True)
            for c in clips:
                if not (dest / c.name).exists():
                    shutil.move(str(c), dest / c.name)
        print(f"[hmdb] {cls}: extracted {len(list(dest.glob('*.avi')))} clips from RAR")

def preprocess_hmdb(input_dir, output_dir, hmdb_classes, prune_unwanted=True, dry_run=False, check_labels=False):
    """Extract ONLY the clips of the chosen HMDB-51 classes into output_dir/<class>/<clip>.avi (folder name = label).
    Source can be an unpacked folder tree, .zip file(s) with class folders, or the official RAR release (needs 7-Zip).
    The source archive is NEVER deleted; re-running only adds what is missing and, if prune_unwanted, removes classes
    you dropped from the config. check_labels=True additionally validates the names against the 51 HMDB folder names
    (off by default: a wrong name still stops the run, because no clips are found for it)."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    wanted = list(dict.fromkeys(hmdb_classes))
    if check_labels:
        check_hmdb_labels(wanted)
    print(f"[hmdb start] input={input_dir}  output={output_dir}  classes={wanted}  prune_unwanted={prune_unwanted}  dry_run={dry_run}")
    if not input_dir.exists():
        raise FileNotFoundError(f"HMDB input folder not found: {input_dir}")
    out_resolved = output_dir.resolve()
 
    known = {c: {} for c in wanted}                  # class -> {clip name: (size, source)} for sources we can list
    for f in input_dir.rglob("*.avi"):               # (a) already-unpacked tree  .../<class>/<clip>.avi
        if out_resolved not in f.resolve().parents and f.parent.name in known:
            known[f.parent.name][f.name] = (f.stat().st_size, ("copy", f))
    zips = sorted(input_dir.rglob("*.zip"))
    for z in zips:                                   # (b) zip(s) containing class folders
        with zipfile.ZipFile(z) as zf:
            infos, hits = zf.infolist(), 0
            for info in infos:
                p = Path(info.filename)
                if not info.is_dir() and p.suffix.lower() == ".avi" and p.parent.name in known:
                    known[p.parent.name].setdefault(p.name, (info.file_size, ("zip", z, info.filename)))
                    hits += 1
            kinds = Counter(Path(i.filename).suffix.lower() for i in infos if not i.is_dir())
            print(f"[hmdb zip] {z.name}: {len(infos)} entries, file types {dict(kinds.most_common(4))}, clips of wanted classes: {hits}")
            if hits == 0:
                print(f"[hmdb zip] nothing matched - first entries: {[i.filename for i in infos[:6]]}")
    need_rar = [c for c in wanted if not known[c]]   # (c) classes still missing -> official RAR release
    rars = sorted(input_dir.rglob("*.rar")) if need_rar else []
    print(f"[hmdb found] clips per class in tree/zip sources: { {c: len(v) for c, v in known.items()} } | "
          f"{len(zips)} zip(s), {len(rars)} rar(s) | classes needing RAR: {need_rar}")
    if need_rar and not rars:
        raise FileNotFoundError(f"no clips found for classes {need_rar} in {input_dir} (looked for <class>/*.avi inside folders and .zip files, "
                                f"and for .rar files). Check the spelling in the config (HMDB folder names are short: 'sit', 'fall_floor') "
                                f"and the zip listing printed above.")
    if dry_run:
        for c in wanted:
            print(f"[hmdb] DRY RUN {c}: {len(known[c]) if known[c] else 'extract from RAR with 7-Zip'} clip(s)")
        stale = [d.name for d in output_dir.glob("*") if d.is_dir() and d.name in HMDB51_CLASSES and d.name not in wanted] if output_dir.exists() else []
        print(f"[hmdb] DRY RUN: would prune classes {stale}" if prune_unwanted else "[hmdb] DRY RUN done")
        return {}
 
    output_dir.mkdir(parents=True, exist_ok=True)
    totals = {"extracted": 0, "already_there": 0, "pruned_clips": 0}
    open_zips = {}
    try:
        for c in wanted:
            for name, (size, src) in known[c].items():
                target = output_dir / c / name
                if target.exists() and target.stat().st_size == size:
                    totals["already_there"] += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                if src[0] == "copy":
                    shutil.copyfile(src[1], tmp)
                else:
                    zf = open_zips.setdefault(src[1], zipfile.ZipFile(src[1]))
                    with zf.open(src[2]) as s, open(tmp, "wb") as d:
                        shutil.copyfileobj(s, d, 1 << 20)
                tmp.replace(target)
                totals["extracted"] += 1
            if known[c]:
                print(f"[hmdb] {c}: {len(known[c])} clips available, on disk now: {len(list((output_dir / c).glob('*.avi')))}")
    finally:
        for zf in open_zips.values():
            zf.close()
    if need_rar:
        _extract_from_rars(rars, need_rar, output_dir)
 
    if prune_unwanted:                               # sync: drop classes (and stale clips) that are no longer wanted
        for d in output_dir.glob("*"):
            if not d.is_dir() or d.name not in HMDB51_CLASSES:
                continue
            if d.name not in wanted:
                n = len(list(d.glob("*.avi")))
                shutil.rmtree(d)
                totals["pruned_clips"] += n
                print(f"[hmdb] pruned class folder '{d.name}' ({n} clips) - not in the config any more")
            elif known[d.name]:
                for f in d.glob("*.avi"):
                    if f.name not in known[d.name]:
                        f.unlink()
                        totals["pruned_clips"] += 1
 
    counts = {c: len(list((output_dir / c).glob("*.avi"))) for c in wanted}
    low = {c: n for c, n in counts.items() if n < 100}
    print(f"[hmdb clips per class] {counts}")
    if low:
        print(f"[hmdb WARNING] HMDB-51 has at least ~100 clips per class, these have fewer (partial extraction?): {low}")
    print(f"[hmdb done] {totals} | source archive(s) kept (never deleted)")
    return totals





@hydra.main(version_base=None, config_path="../../configs", config_name="main_config")
def main(cfg: DictConfig) -> None:
    preprocess_rgb(
        input_dir=to_absolute_path(cfg.raw_zip_RGB_FOLDER),
        output_dir=to_absolute_path(cfg.processed_RGB_FOLDER),
        ntu_ids=[c.ntu60 for c in cfg.classes],
        missing_skeleton_file=to_absolute_path(cfg.missing_skeleton_file),
        delete_zip=cfg.delete_zip_after_extract,
        dry_run=cfg.dry_run,
     )
    preprocess_skeleton(
    input_dir=to_absolute_path(cfg.raw_zip_skeleton_FOLDER),
    output_dir=to_absolute_path(cfg.processed_skeleton_FOLDER),
    ntu_ids=[c.ntu60 for c in cfg.classes],
    missing_skeleton_file=to_absolute_path(cfg.missing_skeleton_file),
    dry_run=cfg.dry_run,
    )
    preprocess_hmdb(
    input_dir=to_absolute_path(cfg.raw_zip_HMDB1_FOLDER),
    output_dir=to_absolute_path(cfg.processed_HMDB1_FOLDER),
    hmdb_classes=[c.hmdb for c in cfg.classes],
    dry_run=cfg.dry_run,
)


if __name__ == "__main__":
    main()