"""Resurrect SmartDiskCache index entries for an existing text cache.

When a normal (non-trust) validation run decides entries are stale, it deletes
them from cache.json *before* rebuilding (SmartDiskCache.__refresh_cache). If
that run is stopped early, the .pt files are still on disk but the index no
longer references them, so every later run re-encodes those files from
scratch.

Text-cache .pt files are content-addressed: ``{xxh64(file)[:12]}_{v+1}.pt``
with no resolution suffix (variant key ``"_"``). So as long as the source
.txt content is unchanged (a touch-only mtime change), we can recompute each
file's hash, find its .pt on disk, and rebuild the index entry directly --
no text encoder involved.

This also merges back any entries that survive in cache.json.bak but were
dropped from cache.json (the .bak is one flush older).

Usage (dry-run first; stop training before --apply):
    venv/Scripts/python.exe scripts/util/_rescue_text_cache_index.py
        <text_cache_dir> --source-root <dataset_dir> [--source-root ...]
        [--pattern *.txt] [--apply]

Only counts are printed; no dataset content is dumped.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys
from collections import Counter

import xxhash

CACHE_VERSION = 3
NO_RESOLUTION_KEY = "_"


def hash_file(filepath: str) -> str:
    h = xxhash.xxh64()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: could not read {os.path.basename(path)}: {e}")
        return None


def scan_pt_files(cache_dir: str) -> set[str]:
    found = set()
    with os.scandir(os.path.realpath(cache_dir)) as it:
        for e in it:
            if e.name.endswith(".pt") and e.is_file(follow_symlinks=False):
                found.add(e.name)
    return found


def majority(values) -> object | None:
    counts = Counter(v for v in values if v is not None)
    return counts.most_common(1)[0][0] if counts else None


def entry_pt_names(entry: dict) -> list[str]:
    """All .pt names referenced by an entry's variants (variation 1 only --
    enough to prove the cache files exist)."""
    names = []
    for variant in entry.get("variants", {}).values():
        cf = variant.get("cache_file")
        if cf:
            names.append(f"{cf}_1.pt")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", help="text cache directory (contains cache.json and *.pt)")
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        dest="source_roots",
        help="dataset directory to scan recursively for caption files (repeatable)",
    )
    parser.add_argument("--pattern", default=".txt", help="caption file extension (default: .txt)")
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    args = parser.parse_args()

    cache_json_path = os.path.join(args.cache_dir, "cache.json")
    index = load_index(cache_json_path)
    if index is None:
        print(f"error: no readable cache.json in {args.cache_dir}")
        return 1
    entries: dict = index.setdefault("entries", {})
    hash_index: dict = index.setdefault("hash_index", {})

    pt_files = scan_pt_files(args.cache_dir)
    print(f"index entries: {len(entries)}, .pt files on disk: {len(pt_files)}")

    # Template metadata from surviving entries.
    modeltype = majority(e.get("modeltype") for e in entries.values())
    schema_keys = majority(
        tuple(v.get("schema_keys", []))
        for e in entries.values()
        for v in e.get("variants", {}).values()
        if v.get("schema_keys")
    )
    if schema_keys is None or modeltype is None:
        print("error: no surviving entries to copy modeltype/schema_keys from -- pass me to Claude, needs a fallback")
        return 1
    schema_keys = list(schema_keys)
    print(f"template: modeltype={modeltype}, schema_keys={schema_keys}")

    restored_from_bak = 0
    bak = load_index(cache_json_path + ".bak")
    if bak:
        for fp, entry in bak.get("entries", {}).items():
            if fp in entries or entry.get("modeltype") != modeltype:
                continue
            pts = entry_pt_names(entry)
            if pts and all(name in pt_files for name in pts):
                entries[fp] = entry
                h = entry.get("hash")
                if h:
                    hash_index.setdefault(h, [])
                    if fp not in hash_index[h]:
                        hash_index[h].append(fp)
                restored_from_bak += 1
    print(f"restored from cache.json.bak: {restored_from_bak}")

    # Scan dataset for caption files.
    ext = args.pattern if args.pattern.startswith(".") else f".{args.pattern}"
    source_files = []
    for root in args.source_roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            source_files.extend(
                os.path.normpath(os.path.abspath(os.path.join(dirpath, name)))
                for name in filenames
                if name.lower().endswith(ext.lower())
            )
    print(f"caption files found under source roots: {len(source_files)}")
    if not source_files:
        print("error: nothing to do")
        return 1

    # Casing/path sanity: entry keys must match what os.path.normpath of the
    # concept paths produces at runtime.
    already = sum(1 for fp in source_files if fp in entries)
    casefold_keys = {fp.casefold() for fp in entries}
    casefold_hits = sum(1 for fp in source_files if fp not in entries and fp.casefold() in casefold_keys)
    if casefold_hits:
        print(
            f"warning: {casefold_hits} files match existing entries only when ignoring case --"
            " your --source-root casing differs from the concept paths; fix the casing and rerun"
        )

    pt_pattern = re.compile(r"^(?P<hash12>[0-9a-f]{12})_(?P<var>\d+)\.pt$")
    variations_by_hash12: dict[str, int] = {}
    for name in pt_files:
        m = pt_pattern.match(name)
        if m:
            h12 = m.group("hash12")
            variations_by_hash12[h12] = max(variations_by_hash12.get(h12, 0), int(m.group("var")))

    resurrected = 0
    no_pt_match = 0
    hash_errors = 0
    for fp in source_files:
        if fp in entries:
            continue
        try:
            full_hash = hash_file(fp)
            mtime = os.path.getmtime(fp)
        except OSError:
            hash_errors += 1
            continue
        hash12 = full_hash[:12]
        if hash12 not in variations_by_hash12:
            no_pt_match += 1
            continue
        entries[fp] = {
            "filename": os.path.basename(fp),
            "hash": full_hash,
            "mtime": mtime,
            "modeltype": modeltype,
            "variants": {NO_RESOLUTION_KEY: {"cache_file": hash12, "schema_keys": schema_keys}},
            "cache_version": CACHE_VERSION,
            "sidecar_mtimes": {},
            "sidecar_hashes": {},
        }
        hash_index.setdefault(full_hash, [])
        if fp not in hash_index[full_hash]:
            hash_index[full_hash].append(fp)
        resurrected += 1

    print()
    print(f"already indexed:        {already}")
    print(f"resurrected:            {resurrected}  (existing .pt re-linked, no re-encode)")
    print(f"no matching .pt:        {no_pt_match}  (content changed or never cached -- these WILL re-encode)")
    if hash_errors:
        print(f"unreadable files:       {hash_errors}")

    if not args.apply:
        print()
        print("dry-run only -- nothing written. Re-run with --apply to save.")
        return 0

    if resurrected == 0 and restored_from_bak == 0:
        print("nothing to write.")
        return 0

    # Full validation pops this token anyway, but drop it so a fast-validate
    # pass can't skip over the new entries' first real registration.
    index.pop("last_validated", None)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{cache_json_path}.rescue-bak-{stamp}"
    shutil.copy2(cache_json_path, backup_path)
    tmp_path = cache_json_path + ".rescue-tmp"
    with open(tmp_path, "w") as f:
        json.dump(index, f, separators=(",", ":"))
    os.replace(tmp_path, cache_json_path)
    print()
    print(f"wrote {cache_json_path} ({len(entries)} entries); original backed up to {os.path.basename(backup_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
