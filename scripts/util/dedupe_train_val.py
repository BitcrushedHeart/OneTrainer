"""Find and resolve image duplicates inside each person dataset under a root.

Phase 1 — cross-folder (train vs val):
  For each child folder of the root (e.g. F:/Datasets/People/Abbie Walker):
    - phash all images in <person>/train and <person>/val
    - For every val image whose phash is >=85% similar to any train image,
      merge the val caption sidecar into the train caption sidecar (appended
      on a new line, deduped) and delete the val image + its sidecar.

Phase 2 — intra-train (only when train has >60 images):
  Compute both phash and dhash for every train image, cluster pairs where
  EITHER hash hits >=95% similarity, keep the largest image (pixel area,
  file size as tiebreaker), merge the others' captions into it, and delete
  the others.

Duplicates across different person folders are ignored.

Usage:
    venv/Scripts/python.exe scripts/util/dedupe_train_val.py "F:/Datasets/People"
    venv/Scripts/python.exe scripts/util/dedupe_train_val.py "F:/Datasets/People" --dry-run
"""

import argparse
import sys
from pathlib import Path

import imagehash
from PIL import Image, ImageFile, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
HASH_BITS = 64
SIMILARITY_THRESHOLD = 0.85
MAX_HAMMING = int((1.0 - SIMILARITY_THRESHOLD) * HASH_BITS)  # 9 for 85% on 64-bit phash
MASK_SUFFIX = "-masklabel.png"

# Hashing is done against an in-memory centered square crop so that letterboxing,
# borders, or differing aspect ratios don't fool phash/dhash. Files are never
# modified.
LARGE_CROP_PX = 600  # used when image area >= 0.6 MP
SMALL_CROP_PX = 400  # used when image area  < 0.6 MP
SMALL_AREA_THRESHOLD = 600_000  # 0.6 MP

# Phase 2 (intra-train) parameters.
INTRA_TRAIN_SIMILARITY = 0.95
INTRA_TRAIN_MAX_HAMMING = int((1.0 - INTRA_TRAIN_SIMILARITY) * HASH_BITS)  # 3 for 95%
MIN_TRAIN_FOR_INTRA = 60  # only run phase 2 when len(train images) > this


def iter_images(folder: Path):
    if not folder.is_dir():
        return
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.name.lower().endswith(MASK_SUFFIX):
            continue
        yield p


def _centered_crop_for_hash(im: Image.Image) -> Image.Image:
    """Return an in-memory centered square crop, sized by image area.

    >=0.6 MP -> 600x600;  <0.6 MP -> 400x400. Clamped to the image's smaller
    side so very small images aren't upscaled. The original is never modified.
    """
    w, h = im.size
    crop_size = LARGE_CROP_PX if (w * h) >= SMALL_AREA_THRESHOLD else SMALL_CROP_PX
    crop_size = min(crop_size, w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    return im.crop((left, top, left + crop_size, top + crop_size))


def _delete_image_and_sidecar(path: Path, dry_run: bool, reason: str):
    cap = caption_path(path)
    if dry_run:
        print(f"  ! [dry-run] would delete corrupt image {path.name} ({reason})")
        if cap.is_file():
            print(f"  ! [dry-run] would delete sidecar {cap.name}")
        return
    print(f"  ! deleting corrupt image {path.name} ({reason})")
    try:
        path.unlink()
    except OSError as e:
        print(f"    failed to delete {path}: {e}", file=sys.stderr)
    if cap.is_file():
        try:
            cap.unlink()
        except OSError as e:
            print(f"    failed to delete sidecar {cap}: {e}", file=sys.stderr)


def _try_repair(path: Path, dry_run: bool) -> bool:
    """Attempt to recover a damaged image by loading with truncation tolerance
    and re-saving in place. Returns True on success."""
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        try:
            with Image.open(path) as im:
                im.load()
                fmt = (im.format or path.suffix.lstrip(".")).upper()
                if fmt == "JPG":
                    fmt = "JPEG"
                save_kwargs = {"format": fmt}
                save_im = im
                if fmt == "JPEG" and im.mode not in ("RGB", "L", "CMYK"):
                    save_im = im.convert("RGB")
                    save_kwargs["quality"] = 95
                elif fmt == "JPEG":
                    save_kwargs["quality"] = 95
                if dry_run:
                    print(f"  ~ [dry-run] would repair {path.name} (format={fmt})")
                    return True
                tmp = path.with_suffix(path.suffix + ".repair")
                save_im.save(tmp, **save_kwargs)
            tmp.replace(path)
            print(f"  ~ repaired {path.name}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    repair failed for {path.name}: {e}", file=sys.stderr)
            tmp = path.with_suffix(path.suffix + ".repair")
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return False
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev


def phash_image(path: Path, dry_run: bool):
    """phash an image, repairing or deleting it if unreadable.

    Returns the hash, or None if the file was deleted (or unrecoverable in
    dry-run mode)."""
    # 1) Structural verify — catches headerless / truncated-header garbage.
    try:
        with Image.open(path) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
        _delete_image_and_sidecar(path, dry_run, reason=f"verify failed: {e}")
        return None

    # 2) Clean decode + phash on a centered square crop.
    try:
        with Image.open(path) as im:
            im.load()
            return imagehash.phash(_centered_crop_for_hash(im))
    except (UnidentifiedImageError, OSError, ValueError) as e:
        decode_err = e

    # 3) Header was OK but decode failed — try repair.
    if _try_repair(path, dry_run):
        if dry_run:
            # File wasn't actually repaired on disk; skip hashing it this run.
            return None
        try:
            with Image.open(path) as im:
                im.load()
                return imagehash.phash(_centered_crop_for_hash(im))
        except (UnidentifiedImageError, OSError, ValueError) as e:
            _delete_image_and_sidecar(path, dry_run, reason=f"unhashable after repair: {e}")
            return None

    _delete_image_and_sidecar(path, dry_run, reason=f"unrepairable: {decode_err}")
    return None


def hash_folder(folder: Path, dry_run: bool):
    results = []
    for img in iter_images(folder):
        h = phash_image(img, dry_run)
        if h is not None:
            results.append((img, h))
    return results


def caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def merge_captions(train_caption: Path, val_caption: Path, dry_run: bool):
    """Append val caption lines into train caption, deduped, preserving order."""
    if not val_caption.is_file():
        return False  # nothing to merge

    val_text = val_caption.read_text(encoding="utf-8", errors="replace").strip()
    if not val_text:
        return False

    if train_caption.is_file():
        train_text = train_caption.read_text(encoding="utf-8", errors="replace").rstrip()
    else:
        train_text = ""

    existing_lines = [ln for ln in train_text.splitlines() if ln.strip()]
    seen = {ln.strip() for ln in existing_lines}

    new_lines = list(existing_lines)
    added = False
    for ln in val_text.splitlines():
        s = ln.strip()
        if s and s not in seen:
            new_lines.append(ln)
            seen.add(s)
            added = True

    if not added:
        return False

    merged = "\n".join(new_lines) + "\n"
    if dry_run:
        print(f"    [dry-run] would write merged caption -> {train_caption}")
    else:
        train_caption.write_text(merged, encoding="utf-8")
    return True


def process_person(person_dir: Path, dry_run: bool):
    train_dir = person_dir / "train"
    val_dir = person_dir / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        print(f"[skip] {person_dir.name}: missing train/ or val/")
        return 0, 0

    print(f"[scan] {person_dir.name}")
    train_hashes = hash_folder(train_dir, dry_run)
    val_hashes = hash_folder(val_dir, dry_run)

    if not train_hashes or not val_hashes:
        print(f"  train={len(train_hashes)} val={len(val_hashes)} -> nothing to compare")
        return 0, 0

    removed = 0
    merged_count = 0
    for val_img, vh in val_hashes:
        best_match = None
        best_dist = None
        for train_img, th in train_hashes:
            dist = vh - th
            if dist <= MAX_HAMMING and (best_dist is None or dist < best_dist):
                best_match = train_img
                best_dist = dist
                if dist == 0:
                    break

        if best_match is None:
            continue

        sim_pct = (1.0 - best_dist / HASH_BITS) * 100.0
        print(f"  dup: val/{val_img.name}  ~=  train/{best_match.name}  ({sim_pct:.1f}%)")

        train_cap = caption_path(best_match)
        val_cap = caption_path(val_img)

        if merge_captions(train_cap, val_cap, dry_run):
            merged_count += 1
            print(f"    merged caption {val_cap.name} -> {train_cap.name}")

        if dry_run:
            print(f"    [dry-run] would delete {val_img}")
            if val_cap.is_file():
                print(f"    [dry-run] would delete {val_cap}")
        else:
            try:
                val_img.unlink()
            except OSError as e:
                print(f"    ! failed to delete {val_img}: {e}", file=sys.stderr)
                continue
            if val_cap.is_file():
                try:
                    val_cap.unlink()
                except OSError as e:
                    print(f"    ! failed to delete {val_cap}: {e}", file=sys.stderr)
        removed += 1

    print(f"  -> {removed} val image(s) removed, {merged_count} caption(s) merged")
    return removed, merged_count


def _measure_image(path: Path):
    """Compute (phash, dhash, pixel_area, file_size_bytes) for one image.

    Returns None on any failure (corrupt files should already be gone after
    phase 1, so we just skip rather than re-running repair logic)."""
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            crop = _centered_crop_for_hash(im)
            ph = imagehash.phash(crop)
            dh = imagehash.dhash(crop)
        return ph, dh, w * h, path.stat().st_size
    except (UnidentifiedImageError, OSError, ValueError) as e:
        print(f"  ! phase2: skipping {path.name}: {e}", file=sys.stderr)
        return None


def process_person_phase2(person_dir: Path, dry_run: bool):
    train_dir = person_dir / "train"
    if not train_dir.is_dir():
        return 0, 0

    images = list(iter_images(train_dir))
    if len(images) <= MIN_TRAIN_FOR_INTRA:
        return 0, 0

    print(f"[phase2] {person_dir.name} ({len(images)} train images)")

    info = {}
    for img in images:
        m = _measure_image(img)
        if m is not None:
            info[img] = m

    paths = list(info.keys())
    n = len(paths)
    if n < 2:
        print("  -> phase2: nothing to compare")
        return 0, 0

    # Union-find over near-duplicate pairs.
    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        ph_i, dh_i, _, _ = info[paths[i]]
        for j in range(i + 1, n):
            ph_j, dh_j, _, _ = info[paths[j]]
            if (ph_i - ph_j) <= INTRA_TRAIN_MAX_HAMMING or (dh_i - dh_j) <= INTRA_TRAIN_MAX_HAMMING:
                union(paths[i], paths[j])

    clusters: dict[Path, list[Path]] = {}
    for p in paths:
        clusters.setdefault(find(p), []).append(p)

    deleted = 0
    merged_count = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        # Keep the largest by pixel area; break ties by file size, then name.
        members.sort(key=lambda p: (info[p][2], info[p][3], p.name), reverse=True)
        keeper = members[0]
        losers = members[1:]
        kw = info[keeper]
        print(f"  cluster: keep {keeper.name} ({kw[2]}px, {kw[3]}B)")
        keeper_cap = caption_path(keeper)
        for loser in losers:
            lw = info[loser]
            ph_dist = info[keeper][0] - info[loser][0]
            dh_dist = info[keeper][1] - info[loser][1]
            print(
                f"    drop {loser.name} ({lw[2]}px, {lw[3]}B)  "
                f"phash {(1 - ph_dist / HASH_BITS) * 100:.1f}%  dhash {(1 - dh_dist / HASH_BITS) * 100:.1f}%"
            )
            loser_cap = caption_path(loser)
            if merge_captions(keeper_cap, loser_cap, dry_run):
                merged_count += 1
                print(f"      merged caption {loser_cap.name} -> {keeper_cap.name}")
            if dry_run:
                print(f"      [dry-run] would delete {loser}")
                if loser_cap.is_file():
                    print(f"      [dry-run] would delete {loser_cap}")
            else:
                try:
                    loser.unlink()
                except OSError as e:
                    print(f"      ! failed to delete {loser}: {e}", file=sys.stderr)
                    continue
                if loser_cap.is_file():
                    try:
                        loser_cap.unlink()
                    except OSError as e:
                        print(f"      ! failed to delete {loser_cap}: {e}", file=sys.stderr)
            deleted += 1

    print(f"  -> phase2: {deleted} train image(s) removed, {merged_count} caption(s) merged")
    return deleted, merged_count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Root folder containing per-person subfolders")
    ap.add_argument("--dry-run", action="store_true", help="Report actions without modifying files")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"error: root not found or not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    print(f"Root: {root}")
    print(f"Phase 1 (train vs val): {SIMILARITY_THRESHOLD * 100:.0f}% phash  (max hamming = {MAX_HAMMING}/{HASH_BITS})")
    print(
        f"Phase 2 (intra-train, >{MIN_TRAIN_FOR_INTRA} imgs): "
        f"{INTRA_TRAIN_SIMILARITY * 100:.0f}% phash OR dhash  "
        f"(max hamming = {INTRA_TRAIN_MAX_HAMMING}/{HASH_BITS})"
    )
    if args.dry_run:
        print("DRY RUN: no files will be changed")

    total_val_removed = 0
    total_val_merged = 0
    total_train_removed = 0
    total_train_merged = 0
    person_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for person in person_dirs:
        removed, merged = process_person(person, args.dry_run)
        total_val_removed += removed
        total_val_merged += merged

        t_removed, t_merged = process_person_phase2(person, args.dry_run)
        total_train_removed += t_removed
        total_train_merged += t_merged

    print()
    print(
        f"Done. Phase 1: {total_val_removed} val removed, {total_val_merged} captions merged | "
        f"Phase 2: {total_train_removed} train removed, {total_train_merged} captions merged"
    )


if __name__ == "__main__":
    main()
