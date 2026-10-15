"""Count cache entries that train on another pair's tensors (read-only).

SmartDiskCache used to dedup cache files by the chosen image's content hash
alone, so every DPO pair sharing a chosen image (cross-concept twins, or one
chosen image with several rejected images inside a concept) pointed at ONE
.pt - the one the first pair built - and the others trained on that pair's
rejected latent and mask. mgds now keys files on chosen + sidecars and flags
such entries for a rebuild on the next normal cache pass; this script shows
what an index looks like before that pass, from cache.json alone.

    venv/Scripts/python.exe scripts/util/dpo_cache_dedup_check.py "F:/workspace/SoReal!/SourcelessDPO/image"

Per concept folder (the path component before "chosen"): how many entries sit
in a mixed group (a file shared by entries whose sidecar hashes differ), how
many of those train on another entry's tensors (the group minus the entry
whose path the .pt stamps as its builder), and how many files that is. The
holder is read from each shared .pt's ``image_path`` key with torch.load's
mmap (no tensor is read). Prints counts and pair FILE NAMES only; never opens
an image or a caption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from mgds.pipelineModules.SmartDiskCache import SmartDiskCache  # noqa: E402


def concept_folder(path: str) -> str:
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    lowered = [p.lower() for p in parts]
    if "chosen" in lowered:
        index = lowered.index("chosen")
        return parts[index - 1] if index > 0 else parts[0]
    return parts[-2] if len(parts) >= 2 else path


def sidecar_signature(entry: dict) -> tuple[str, ...]:
    return tuple(sorted((entry.get("sidecar_hashes") or {}).values()))


def peek_holder(cache_dir: str, cache_file: str) -> str | None:
    import torch

    try:
        data = torch.load(
            os.path.join(cache_dir, f"{cache_file}_1.pt"), weights_only=False, map_location="cpu", mmap=True
        )
    except Exception:  # noqa: BLE001 - unreadable file: holder unknown
        return None
    value = data.get("image_path") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


def same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", help="cache directory (holding cache.json) or the cache.json itself")
    parser.add_argument("--examples", type=int, default=3, help="example groups to print per concept")
    parser.add_argument("--no-peek", action="store_true", help="do not open .pt files to identify holders")
    args = parser.parse_args()

    cache_dir = os.path.dirname(args.cache) if args.cache.lower().endswith(".json") else args.cache
    index_path = os.path.join(cache_dir, "cache.json")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    entries = index.get("entries", {})

    by_file: dict[str, list[str]] = defaultdict(list)
    for filepath, entry in entries.items():
        for cache_file in SmartDiskCache._iter_variant_cache_files(entry):
            by_file[cache_file].append(filepath)

    mixed_groups = {
        cache_file: paths
        for cache_file, paths in by_file.items()
        if len(paths) > 1 and len({sidecar_signature(entries[p]) for p in paths}) > 1
    }
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "entries": 0,
            "in_mixed_group": 0,
            "on_other_pair": 0,
            "files": set(),
            "holder_unknown": 0,
            "examples": [],
        }
    )
    for filepath in entries:
        stats[concept_folder(filepath)]["entries"] += 1

    largest = 0
    for cache_file, paths in mixed_groups.items():
        largest = max(largest, len(paths))
        holder = None if args.no_peek else peek_holder(cache_dir, cache_file)
        holder_path = next((p for p in paths if holder and same_path(p, holder)), None)
        for p in paths:
            concept = concept_folder(p)
            stat = stats[concept]
            stat["in_mixed_group"] += 1
            stat["files"].add(cache_file)
            if holder_path is None:
                stat["holder_unknown"] += 1
            if holder_path is None or p != holder_path:
                stat["on_other_pair"] += 1
            if len(stat["examples"]) < args.examples and holder_path is not None and p != holder_path:
                stat["examples"].append(
                    f"{os.path.basename(p)} -> {cache_file} (holds {concept_folder(holder_path)}/"
                    f"{os.path.basename(holder_path)}; {len(paths)} entries on this file)"
                )

    print(f"{index_path}: {len(entries)} entries, {len(by_file)} cache files")
    print(
        f"mixed groups (one file, differing rejected/mask): {len(mixed_groups)} files, largest holds {largest} entries"
    )
    if any(s["holder_unknown"] for s in stats.values()):
        print("(holder unknown for some files: counted as training on another pair)")
    print()
    width = max((len(c) for c in stats), default=10)
    print(f"{'concept':<{width}}  {'entries':>8}  {'in mixed':>9}  {'on other pair':>14}  {'files':>6}")
    for concept in sorted(stats, key=lambda c: -stats[c]["on_other_pair"]):
        s = stats[concept]
        print(
            f"{concept:<{width}}  {s['entries']:>8}  {s['in_mixed_group']:>9}  {s['on_other_pair']:>14}  {len(s['files']):>6}"
        )
    print()
    for concept in sorted(stats, key=lambda c: -stats[c]["on_other_pair"]):
        for example in stats[concept]["examples"]:
            print(f"  {concept}: {example}")
    victims = sum(s["on_other_pair"] for s in stats.values())
    print()
    print(
        f"{victims} entries train on another pair's tensors; a normal cache pass rebuilds each into its own file "
        f"(about {victims} new .pt files), the holders keep theirs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
