"""Shrink a SmartDiskCache cache.json by interning its variations-group values.

Every stamped sourceless metadata block used to carry the raw ``group_values``
list -- a full copy of each variations-group input. For a text cache that
includes the concept's ``text`` config block (tag-dropout whitelist and all),
duplicated into the entry-level ``sourceless`` block *and* into every
``sourceless_rows`` row. Nothing ever read those values back: all three
consumers in SmartDiskCache reduce them to ``sha256(json.dumps(...))`` and
compare that for equality. This rewrites each block to carry only the derived
``group_key``.

The conversion is information-preserving for every consumer -- the key is
byte-identical to what the old read path computed -- so sourceless training
reads exactly the same groups, balancing and variation counts as before.
SmartDiskCache also still understands un-migrated caches (it falls back to
hashing ``group_values`` when ``group_key`` is absent), so running this is an
optimization, never a prerequisite.

Stop training before running this: the trainer holds the index in memory and
rewrites it on its own schedule.

Usage:
    python scripts/util/_shrink_cache_index.py <cache_dir> [<cache_dir> ...]
    python scripts/util/_shrink_cache_index.py <cache_dir> --apply
"""

import argparse
import hashlib
import json
import os
import shutil
import sys


def string_key(data) -> str:
    """Byte-identical to SmartDiskCache.__string_key."""
    json_data = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"), indent=None)
    return hashlib.sha256(json_data.encode("utf-8")).hexdigest()


def legacy_group_key(meta: dict) -> str:
    """The key the old read path derived from a stamped block.

    Mirrors _init_sourceless_groups_from_records / _sourceless_row_group_key,
    including the "missing field means [''] " default. An empty list is hashed
    as-is -- it is not the same as a missing field.
    """
    group_values = meta.get("group_values")
    if group_values is None:
        group_values = [""]
    return string_key(group_values)


def convert_block(meta) -> bool:
    """Replace group_values with the derived group_key. True if changed."""
    if not isinstance(meta, dict) or "group_values" not in meta:
        return False
    key = meta.get("group_key")
    if not isinstance(key, str):
        key = legacy_group_key(meta)
    # Verify before discarding the source of truth: the migrated block must
    # hash to exactly what the old path would have produced.
    assert key == legacy_group_key(meta), "group key mismatch - refusing to convert"
    meta["group_key"] = key
    del meta["group_values"]
    return True


def process(cache_dir: str, apply: bool) -> int:
    path = os.path.join(cache_dir, "cache.json")
    if not os.path.isfile(path):
        print(f"  no cache.json in {cache_dir}")
        return 1

    before = os.path.getsize(path)
    with open(path, "rb") as f:
        index = json.loads(f.read())

    entries = index.get("entries", {})
    converted = 0
    # Every block is visited by identity-safe means: `row["metadata"]` is
    # often the *same dict object* as `entry["sourceless"]`, so convert_block
    # is written to be idempotent rather than assuming distinct objects.
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        if convert_block(entry.get("sourceless")):
            converted += 1
        rows = entry.get("sourceless_rows")
        if isinstance(rows, dict):
            for row in rows.values():
                if isinstance(row, dict) and convert_block(row.get("metadata")):
                    converted += 1

    payload = json.dumps(index, indent=None).encode("utf-8")
    after = len(payload)
    pct = 100 * (1 - after / before) if before else 0.0
    print(f"  entries {len(entries)}, blocks converted {converted}")
    print(f"  {before / 1024**3:.2f} GB -> {after / 1024**3:.2f} GB  ({pct:.1f}% smaller)")

    if not apply:
        print("  dry run - pass --apply to write")
        return 0

    tmp = path + ".shrink.tmp"
    bak = path + ".preshrink.bak"
    with open(tmp, "wb") as f:
        f.write(payload)
    # Verify the rewritten file parses and still carries every entry before
    # letting it replace the original.
    with open(tmp, "rb") as f:
        check = json.loads(f.read())
    if len(check.get("entries", {})) != len(entries):
        os.remove(tmp)
        print("  ERROR: entry count changed, aborted")
        return 1
    shutil.copy2(path, bak)
    os.replace(tmp, path)
    print(f"  written (previous index kept at {os.path.basename(bak)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "cache_dirs", nargs="+", help="directories containing a cache.json (e.g. <cache>/image <cache>/text)"
    )
    parser.add_argument("--apply", action="store_true", help="write the shrunk index (default: dry run)")
    args = parser.parse_args()

    rc = 0
    for cache_dir in args.cache_dirs:
        print(cache_dir)
        rc |= process(cache_dir, args.apply)
    return rc


if __name__ == "__main__":
    sys.exit(main())
