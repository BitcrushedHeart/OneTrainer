"""Split bundled latent_mask out of an existing image SmartDiskCache into a sibling mask cache.

Usage:
    python scripts/util/_migrate_split_mask_cache.py --image-cache PATH [--mask-cache PATH]
                                                     [--dry-run|--apply] [--sample N]

Default mode is --dry-run. Use --apply to actually write. --sample N limits processing
to the first N entries (used for the smoke test).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import torch

import xxhash

CACHE_VERSION = 3
MASK_POSTFIX = "-masklabel"
MASK_EXTENSION = ".png"
SCHEMA_METHOD = "shape_v1"


def derive_mask_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + MASK_POSTFIX + MASK_EXTENSION


def hash_file(path: str) -> str:
    h = xxhash.xxh64()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def synthetic_hash(filepath: str) -> str:
    return xxhash.xxh64(filepath.encode("utf-8")).hexdigest()


def atomic_write_pt(obj, dest: str) -> None:
    parent = os.path.dirname(dest)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(obj, dest: str) -> None:
    parent = os.path.dirname(dest)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".json.tmp")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def compute_watched_fingerprints(entries: dict) -> dict:
    by_parent: dict[str, set[str]] = {}
    for fp in entries:
        by_parent.setdefault(os.path.dirname(fp), set()).add(os.path.basename(fp))
    fp_map = {}
    for parent, names in by_parent.items():
        count = 0
        mtime_sum = 0.0
        try:
            with os.scandir(parent) as it:
                for e in it:
                    if e.name not in names:
                        continue
                    try:
                        mtime = e.stat().st_mtime
                    except OSError:
                        continue
                    count += 1
                    mtime_sum += mtime
        except OSError:
            return {}
        fp_map[parent] = [count, mtime_sum]
    return fp_map


def migrate(image_cache_dir: str, mask_cache_dir: str, *, apply: bool, sample: int | None) -> dict:
    image_index_path = os.path.join(image_cache_dir, "cache.json")
    if not os.path.isfile(image_index_path):
        raise FileNotFoundError(f"image cache.json not found at {image_index_path}")

    with open(image_index_path, "r", encoding="utf-8") as f:
        image_index = json.load(f)

    image_entries = image_index.get("entries", {})
    if not image_entries:
        return {"entries": 0, "variants_migrated": 0, "synthetic": 0, "image_pts_rewritten": 0, "bytes_saved": 0}

    if apply:
        backup = image_index_path + ".pre-split-bak"
        if not os.path.exists(backup):
            shutil.copy2(image_index_path, backup)
        os.makedirs(mask_cache_dir, exist_ok=True)

    mask_entries: dict = {}
    mask_hash_index: dict = {}
    stats = {
        "entries": 0,
        "variants_migrated": 0,
        "synthetic": 0,
        "image_pts_rewritten": 0,
        "bytes_saved": 0,
        "missing_pts": 0,
        "no_mask_in_pt": 0,
    }

    items = list(image_entries.items())
    if sample is not None:
        items = items[:sample]

    modeltype = image_index.get("modeltype") or _first_modeltype(image_entries)

    for image_path, entry in items:
        stats["entries"] += 1
        mask_path = derive_mask_path(image_path)
        mask_file_exists = os.path.isfile(mask_path)
        if mask_file_exists:
            try:
                mhash = hash_file(mask_path)
                mmtime = os.path.getmtime(mask_path)
            except OSError:
                mhash = synthetic_hash(mask_path)
                mmtime = 0.0
                mask_file_exists = False
        else:
            mhash = synthetic_hash(mask_path)
            mmtime = 0.0
            stats["synthetic"] += 1

        mask_variants: dict = {}

        for res_key, variant in (entry.get("variants") or {}).items():
            image_cache_file = variant["cache_file"]
            mask_cache_file = f"{mhash[:12]}_{res_key}" if res_key != "_" else mhash[:12]
            for v in range(_max_variation(image_cache_dir, image_cache_file)):
                src_pt = os.path.join(image_cache_dir, f"{image_cache_file}_{v + 1}.pt")
                if not os.path.isfile(src_pt):
                    stats["missing_pts"] += 1
                    continue

                data = torch.load(src_pt, weights_only=False, map_location="cpu")
                if "latent_mask" not in data:
                    stats["no_mask_in_pt"] += 1
                    continue

                mask_pt_dest = os.path.join(mask_cache_dir, f"{mask_cache_file}_{v + 1}.pt")
                mask_data = {
                    "latent_mask": data["latent_mask"],
                    "__cache_version": CACHE_VERSION,
                    "__modeltype": entry.get("modeltype", modeltype or ""),
                }
                size_before = os.path.getsize(src_pt)

                if apply:
                    atomic_write_pt(mask_data, mask_pt_dest)
                    rewritten = {k: v for k, v in data.items() if k != "latent_mask"}
                    atomic_write_pt(rewritten, src_pt)
                    stats["bytes_saved"] += size_before - os.path.getsize(src_pt)
                    stats["image_pts_rewritten"] += 1

                stats["variants_migrated"] += 1

            mask_variants[res_key] = {"cache_file": mask_cache_file, "schema_keys": ["latent_mask"]}

        if not mask_variants:
            continue

        mask_entry = {
            "filename": os.path.basename(mask_path),
            "hash": mhash,
            "mtime": mmtime,
            "modeltype": entry.get("modeltype", modeltype or ""),
            "cache_version": CACHE_VERSION,
            "variants": mask_variants,
        }
        mask_entries[mask_path] = mask_entry
        mask_hash_index.setdefault(mhash, []).append(mask_path)

        if apply:
            for variant in entry.get("variants", {}).values():
                keys = variant.get("schema_keys")
                if keys is None:
                    continue
                variant["schema_keys"] = [k for k in keys if k != "latent_mask"]

    if apply and mask_entries:
        mask_index = {
            "version": CACHE_VERSION,
            "entries": mask_entries,
            "hash_index": mask_hash_index,
            "schema": ["latent_mask"],
            "schema_method": SCHEMA_METHOD,
            "last_validated": time.time(),
            "watched_fingerprints": compute_watched_fingerprints(mask_entries),
        }
        atomic_write_json(mask_index, os.path.join(mask_cache_dir, "cache.json"))

    if apply:
        sentinel_name = image_index.get("blank_sentinel")
        if sentinel_name:
            sentinel_path = os.path.join(image_cache_dir, sentinel_name)
            if os.path.isfile(sentinel_path):
                sdata = torch.load(sentinel_path, weights_only=False, map_location="cpu")
                if isinstance(sdata, dict) and "latent_mask" in sdata:
                    atomic_write_pt({k: v for k, v in sdata.items() if k != "latent_mask"}, sentinel_path)

        image_index["watched_fingerprints"] = compute_watched_fingerprints(image_entries)
        image_index["last_validated"] = time.time()
        atomic_write_json(image_index, image_index_path)

    return stats


def _first_modeltype(entries: dict) -> str | None:
    for v in entries.values():
        mt = v.get("modeltype")
        if mt:
            return mt
    return None


def _max_variation(cache_dir: str, cache_file: str) -> int:
    n = 0
    while os.path.exists(os.path.join(cache_dir, f"{cache_file}_{n + 1}.pt")):
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-cache", required=True, help="Path to existing image cache dir.")
    p.add_argument("--mask-cache", help="Path to new mask cache dir. Defaults to sibling 'mask' of image cache.")
    p.add_argument("--apply", action="store_true", help="Actually write. Default is dry-run.")
    p.add_argument("--sample", type=int, default=None, help="Process only the first N entries.")
    args = p.parse_args()

    image_cache = os.path.abspath(args.image_cache)
    mask_cache = (
        os.path.abspath(args.mask_cache) if args.mask_cache else os.path.join(os.path.dirname(image_cache), "mask")
    )

    print(f"image-cache: {image_cache}")
    print(f"mask-cache:  {mask_cache}")
    print(f"mode:        {'APPLY' if args.apply else 'dry-run'}")
    if args.sample is not None:
        print(f"sample:      first {args.sample} entries")

    stats = migrate(image_cache, mask_cache, apply=args.apply, sample=args.sample)

    print()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
