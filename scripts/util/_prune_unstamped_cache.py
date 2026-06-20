"""Prune un-baked (stale/orphan) entries from a SmartDiskCache so it becomes
sourceless-ready.

Sourceless training reads the WHOLE cache index, so any entry that lacks baked
sourceless metadata (e.g. a source file that was deleted, or data written by a
different concept config) blocks the run. This tool removes those entries and,
optionally, deletes the .pt tensors they reference.

Two safety rules:

1. **Cross-referenced keep set.** A sample is kept only if BOTH its image entry
   and its paired text entry are stamped. Sourceless alignment requires every
   surviving image (anchor) entry to map to a stamped text counterpart, so we
   must not keep a half-stamped pair.
2. **Exclusive .pt deletion.** Image entries dedup-share .pt files by name, so a
   .pt is deleted only when it is referenced exclusively by pruned entries —
   never one still referenced by a surviving entry.

cache.json is backed up (timestamped) before any write. Default is a dry-run.

    venv/Scripts/python.exe scripts/util/_prune_unstamped_cache.py --cache-dir F:/workspace/SoReal!/cache
    venv/Scripts/python.exe scripts/util/_prune_unstamped_cache.py --cache-dir F:/workspace/SoReal!/cache --apply --delete-pt
"""

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.util.sourceless_cache_util import entry_has_sourceless_metadata


def _load(cache_json):
    with open(cache_json, encoding="utf-8") as f:
        return json.load(f)


def _entry_meta(entry: dict) -> dict:
    if entry.get("sourceless"):
        return entry["sourceless"]
    for row in (entry.get("sourceless_rows") or {}).values():
        if isinstance(row, dict) and row.get("metadata"):
            return row["metadata"]
    return {}


def _counterpart_text(image_fp: str, entry: dict) -> str:
    linked = (_entry_meta(entry).get("linked_paths") or {}).get("sample_prompt_path")
    if linked:
        return os.path.normpath(linked)
    return os.path.normpath(os.path.splitext(image_fp)[0] + ".txt")


def _variant_cache_files(entry: dict) -> set[str]:
    return {v.get("cache_file") for v in (entry.get("variants") or {}).values() if v.get("cache_file")}


def _deletable_pt(cache_dir: str, index: dict, prune_fps: set[str]) -> list[str]:
    entries = index["entries"]
    surviving_cf: set[str] = set()
    pruned_cf: set[str] = set()
    for fp, entry in entries.items():
        (pruned_cf if fp in prune_fps else surviving_cf).update(_variant_cache_files(entry))
    deletable_cf = pruned_cf - surviving_cf
    sentinel = index.get("blank_sentinel", "blank_sentinel.pt")
    paths = []
    for cf in deletable_cf:
        n = 1
        while True:
            name = f"{cf}_{n}.pt"
            full = os.path.join(cache_dir, name)
            if not os.path.isfile(full):
                break
            if name != sentinel:
                paths.append(full)
            n += 1
    return paths


def _apply_prune(cache_dir, index, prune_fps, pt_to_delete, delete_pt):
    cache_json = os.path.join(cache_dir, "cache.json")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(cache_json, f"{cache_json}.prune-bak-{stamp}")

    for fp in prune_fps:
        index["entries"].pop(fp, None)
    hash_index = index.get("hash_index", {})
    for h in list(hash_index.keys()):
        kept = [fp for fp in hash_index[h] if fp in index["entries"]]
        if kept:
            hash_index[h] = kept
        else:
            del hash_index[h]

    tmp = f"{cache_json}.prune-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, cache_json)

    deleted = 0
    if delete_pt:
        for p in pt_to_delete:
            try:
                os.remove(p)
                deleted += 1
            except OSError as e:  # noqa: PERF203 - per-file tolerance for a one-shot script
                print(f"  warn: could not delete {p}: {e}")
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True, help="cache dir containing image/ and text/ subdirs")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    parser.add_argument("--delete-pt", action="store_true", help="also delete exclusively-referenced .pt tensors")
    args = parser.parse_args()

    image_dir = os.path.join(args.cache_dir, "image")
    text_dir = os.path.join(args.cache_dir, "text")
    image_json = os.path.join(image_dir, "cache.json")
    text_json = os.path.join(text_dir, "cache.json")
    for p in (image_json, text_json):
        if not os.path.isfile(p):
            print(f"error: missing {p}")
            return 1

    image_index = _load(image_json)
    text_index = _load(text_json)
    image_entries = image_index["entries"]
    text_entries = text_index["entries"]

    # Normalized lookups for the text side.
    text_norm_to_key = {os.path.normpath(k): k for k in text_entries}
    text_stamped_norm = {os.path.normpath(k) for k, e in text_entries.items() if entry_has_sourceless_metadata(e)}

    # Cross-referenced keep set: image stamped AND counterpart text stamped+present.
    good_image = {}
    for fp, entry in image_entries.items():
        if not entry_has_sourceless_metadata(entry):
            continue
        if _counterpart_text(fp, entry) in text_stamped_norm:
            good_image[fp] = entry
    keep_text_keys = {
        text_norm_to_key[n] for n in (_counterpart_text(fp, e) for fp, e in good_image.items()) if n in text_norm_to_key
    }

    prune_image = set(image_entries) - set(good_image)
    prune_text = set(text_entries) - keep_text_keys

    image_pt = _deletable_pt(image_dir, image_index, prune_image)
    text_pt = _deletable_pt(text_dir, text_index, prune_text)

    print("IMAGE cache:")
    print(f"  entries:        {len(image_entries)}")
    print(f"  keep (paired):  {len(good_image)}")
    print(f"  prune:          {len(prune_image)}")
    print(f"  .pt to delete:  {len(image_pt)}")
    print("TEXT cache:")
    print(f"  entries:        {len(text_entries)}")
    print(f"  keep:           {len(keep_text_keys)}")
    print(f"  prune:          {len(prune_text)}")
    print(f"  .pt to delete:  {len(text_pt)}")

    if not args.apply:
        print("\ndry-run only — nothing written. Re-run with --apply [--delete-pt] to execute.")
        return 0

    di = _apply_prune(image_dir, image_index, prune_image, image_pt, args.delete_pt)
    dt = _apply_prune(text_dir, text_index, prune_text, text_pt, args.delete_pt)
    print(f"\napplied. image entries now {len(image_index['entries'])}, text {len(text_index['entries'])}.")
    if args.delete_pt:
        print(f"deleted {di} image .pt and {dt} text .pt files.")
    print("cache.json backed up (.prune-bak-*). Re-run the verifier to confirm 100% coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
