"""
Move images from F:\\Datasets\\RLHF\\chosen that have no same-basename counterpart
under F:\\Datasets\\RLHF\\rejected (both walked recursively) to
E:\\AI\\Data\\Images\\Text2Img\\local\\raw\\Unpaired.

Matching: basename stem only, case-insensitive (matches Windows semantics).
Relative subdirectory structure under `chosen/` is preserved at the destination
to avoid collisions when different subfolders share basenames.

Associated sidecars/captions that share the same stem are moved alongside each
unpaired image so the dataset stays self-consistent:
    <stem>.txt              caption
    <stem>-masklabel.png    mask sidecar
    <stem>-condlabel.png    conditioning sidecar

Default is dry-run. Pass --apply to actually move.

Usage:
    python scripts/util/dpo_move_unpaired_chosen.py            # dry-run
    python scripts/util/dpo_move_unpaired_chosen.py --apply    # do it
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

CHOSEN_ROOT = Path(r"F:\Datasets\RLHF\chosen")
REJECTED_ROOT = Path(r"F:\Datasets\RLHF\rejected")
DEST_ROOT = Path(r"E:\AI\Data\Images\Text2Img\local\raw\Unpaired")

IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif"}
SIDECAR_POSTFIXES = ("-masklabel", "-condlabel")


def is_sidecar(stem: str) -> bool:
    return any(stem.endswith(p) for p in SIDECAR_POSTFIXES)


def collect_rejected_stems(root: Path) -> set[str]:
    """Return the set of lowercased basename stems for every image under `root`.
    Sidecars (-masklabel/-condlabel) are excluded from the match set."""
    stems: set[str] = set()
    if not root.exists():
        print(f"warn: rejected root does not exist: {root}", file=sys.stderr)
        return stems
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            if is_sidecar(stem):
                continue
            stems.add(stem.lower())
    return stems


def iter_unpaired_chosen(root: Path, rejected_stems: set[str]):
    """Yield (chosen_image_path, relative_path_from_root) for each image under
    `root` whose stem is not in `rejected_stems`. Sidecars are skipped here —
    they get gathered later as companions of each orphan."""
    if not root.exists():
        print(f"error: chosen root does not exist: {root}", file=sys.stderr)
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTS:
                continue
            if is_sidecar(stem):
                continue
            if stem.lower() in rejected_stems:
                continue
            full = Path(dirpath) / name
            yield full, full.relative_to(root)


def companion_files(image_path: Path) -> list[Path]:
    """Return existing caption/sidecar files for the given image."""
    parent = image_path.parent
    stem = image_path.stem
    candidates = [parent / f"{stem}.txt"]
    for postfix in SIDECAR_POSTFIXES:
        for ext in IMAGE_EXTS:
            candidates.append(parent / f"{stem}{postfix}{ext}")
    return [c for c in candidates if c.exists()]


def move_one(src: Path, dest: Path, apply: bool) -> None:
    if apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f"  skip (dest exists): {dest}")
            return
        shutil.move(str(src), str(dest))
    print(f"  {'move' if apply else 'would move'}: {src}  ->  {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually move files (default: dry-run)")
    parser.add_argument("--chosen", default=str(CHOSEN_ROOT))
    parser.add_argument("--rejected", default=str(REJECTED_ROOT))
    parser.add_argument("--dest", default=str(DEST_ROOT))
    args = parser.parse_args()

    chosen_root = Path(args.chosen)
    rejected_root = Path(args.rejected)
    dest_root = Path(args.dest)

    print(f"chosen   : {chosen_root}")
    print(f"rejected : {rejected_root}")
    print(f"dest     : {dest_root}")
    print(f"mode     : {'APPLY' if args.apply else 'dry-run'}")

    rejected_stems = collect_rejected_stems(rejected_root)
    print(f"rejected stems: {len(rejected_stems)}")

    unpaired_count = 0
    moved_files = 0
    for image_path, rel in iter_unpaired_chosen(chosen_root, rejected_stems):
        unpaired_count += 1
        dest_image = dest_root / rel
        move_one(image_path, dest_image, args.apply)
        moved_files += 1
        for companion in companion_files(image_path):
            dest_companion = dest_root / companion.relative_to(chosen_root)
            move_one(companion, dest_companion, args.apply)
            moved_files += 1

    print(f"\nunpaired images: {unpaired_count}")
    print(f"files {'moved' if args.apply else 'that would move'}: {moved_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
