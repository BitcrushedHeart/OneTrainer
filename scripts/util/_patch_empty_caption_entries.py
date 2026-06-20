"""Backfill text-cache entries for caption-less images, no GPU/encoder needed.

An image with no caption .txt gets an image-cache entry but, under the old
behaviour, no text-cache entry — which leaves it unpaired and trips the
sourceless alignment guard. Sourceless serving already treats a missing text
row as the zeroed sentinel embedding, so we make that explicit: for each
unpaired image anchor we create a text entry whose .pt is a copy of the text
``blank_sentinel.pt`` (the pre-existing zeroed empty-caption tensor) and whose
prompt is "".

This avoids re-running the cache flow (no VAE/text-encoder load). The proper
long-term fix is ``tolerate_missing_source=True`` on the text cache so future
runs build these entries automatically.

cache.json (text) is backed up before writing. Default is a dry-run.

    venv/Scripts/python.exe scripts/util/_patch_empty_caption_entries.py --cache-dir F:/workspace/SoReal!/cache
    venv/Scripts/python.exe scripts/util/_patch_empty_caption_entries.py --cache-dir F:/workspace/SoReal!/cache --apply
"""

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import xxhash

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.util.sourceless_cache_util import entry_has_sourceless_metadata

TEXT_SCHEMA_KEYS = ["text_encoder_hidden_state", "tokens", "tokens_mask"]
NO_RESOLUTION_KEY = "_"


def _entry_meta(entry: dict) -> dict:
    if entry.get("sourceless"):
        return entry["sourceless"]
    for row in (entry.get("sourceless_rows") or {}).values():
        if isinstance(row, dict) and row.get("metadata"):
            return row["metadata"]
    return {}


def _anchor_concept(entry: dict, source_index) -> dict | None:
    row = (entry.get("sourceless_rows") or {}).get(str(source_index)) or {}
    for rv in (row.get("runtime_values") or entry.get("sourceless_runtime_values") or {}).values():
        if isinstance(rv, dict) and isinstance(rv.get("concept"), dict):
            return rv["concept"]
    return None


def _counterpart_text(image_fp: str, entry: dict) -> str:
    linked = (_entry_meta(entry).get("linked_paths") or {}).get("sample_prompt_path")
    if linked:
        return os.path.normpath(linked)
    return os.path.normpath(os.path.splitext(image_fp)[0] + ".txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = parser.parse_args()

    image_dir = os.path.join(args.cache_dir, "image")
    text_dir = os.path.join(args.cache_dir, "text")
    image_json = os.path.join(image_dir, "cache.json")
    text_json = os.path.join(text_dir, "cache.json")
    sentinel = os.path.join(text_dir, "blank_sentinel.pt")
    if not os.path.isfile(sentinel):
        print(f"error: text blank_sentinel.pt missing at {sentinel}")
        return 1

    with open(image_json, encoding="utf-8") as f:
        image_entries = json.load(f)["entries"]
    with open(text_json, encoding="utf-8") as f:
        text_index = json.load(f)
    text_entries = text_index["entries"]
    text_norm = {os.path.normpath(k) for k in text_entries}

    todo = []  # (text_key, source_index, concept, image_fp)
    for fp, entry in image_entries.items():
        if not entry_has_sourceless_metadata(entry):
            continue
        ctp = _counterpart_text(fp, entry)
        if ctp in text_norm:
            continue
        meta = _entry_meta(entry)
        source_index = meta.get("source_index")
        todo.append((ctp, source_index, _anchor_concept(entry, source_index), fp, dict(meta)))

    print(f"unpaired image anchors needing empty-caption text entries: {len(todo)}")
    for ctp, si, concept, _fp, _meta in todo:
        print(f"  {ctp}  (source_index={si}, concept={'yes' if concept else 'MISSING'})")

    if not todo:
        print("nothing to do — every image anchor already has a text counterpart.")
        return 0

    if not args.apply:
        print("\ndry-run only — nothing written. Re-run with --apply to create the entries.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(text_json, f"{text_json}.empty-caption-bak-{stamp}")

    created = 0
    for text_key, source_index, concept, _image_fp, meta in todo:
        cache_file = xxhash.xxh64(f"emptycap::{text_key}".encode()).hexdigest()[:12]
        shutil.copy2(sentinel, os.path.join(text_dir, f"{cache_file}_1.pt"))

        meta = dict(meta)
        meta["source_path_in_name"] = "sample_prompt_path"
        runtime = {"0": {"prompt": ""}}
        if isinstance(concept, dict):
            runtime["0"]["concept"] = concept

        text_entries[os.path.normpath(text_key)] = {
            "filename": os.path.basename(text_key),
            "hash": xxhash.xxh64(text_key.encode("utf-8")).hexdigest(),
            "mtime": 0.0,
            "modeltype": "Z_IMAGE",
            "variants": {NO_RESOLUTION_KEY: {"cache_file": cache_file, "schema_keys": list(TEXT_SCHEMA_KEYS)}},
            "cache_version": 3,
            "sidecar_mtimes": {},
            "sidecar_hashes": {},
            "sourceless": meta,
            "sourceless_rows": {str(source_index): {"metadata": meta, "runtime_values": runtime}},
            "sourceless_runtime_values": runtime,
        }
        created += 1

    tmp = f"{text_json}.empty-caption-tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(text_index, f, indent=2)
    os.replace(tmp, text_json)

    print(f"\ncreated {created} empty-caption text entries (zeroed sentinel .pt). text/cache.json backed up.")
    print("Run _check_runpod_ready.py to confirm alignment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
