"""Confirm a cache is ready for sourceless training / the RunPod wizard.

Two checks the per-cache verifier can't do alone:

1. **Wizard pre-flight** — the exact `sourceless_cache_problems` gate the RunPod
   wizard runs before uploading.
2. **Cross-cache alignment** — what the in-training hard guard
   (`SmartDiskCache.__init_sourceless`) enforces: every image (anchor) entry
   must map to a *stamped* text counterpart. A cache can be 100% stamped on
   each side yet still fail here if image/text counts diverge.

Misaligned image anchors are listed, with whether their caption .txt exists on
disk (a missing .txt is the usual cause — fix by enabling
`tolerate_missing_source` on the text cache so it builds an empty-caption entry).

    venv/Scripts/python.exe scripts/util/_check_runpod_ready.py --cache-dir F:/workspace/SoReal!/cache
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.util.sourceless_cache_util import entry_has_sourceless_metadata, sourceless_cache_problems


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    problems = sourceless_cache_problems(args.cache_dir)
    print("1) Wizard pre-flight (per-cache metadata coverage):")
    if problems:
        for p in problems:
            print(f"   - {p}")
    else:
        print("   OK — every entry stamped.")

    image_json = os.path.join(args.cache_dir, "image", "cache.json")
    text_json = os.path.join(args.cache_dir, "text", "cache.json")
    misaligned = []
    if os.path.isfile(image_json) and os.path.isfile(text_json):
        with open(image_json, encoding="utf-8") as f:
            image_entries = json.load(f)["entries"]
        with open(text_json, encoding="utf-8") as f:
            text_entries = json.load(f)["entries"]
        text_stamped_norm = {os.path.normpath(k) for k, e in text_entries.items() if entry_has_sourceless_metadata(e)}
        for fp, entry in image_entries.items():
            if not entry_has_sourceless_metadata(entry):
                continue
            if _counterpart_text(fp, entry) not in text_stamped_norm:
                misaligned.append(fp)

    print("\n2) Cross-cache alignment (image anchor -> stamped text counterpart):")
    if misaligned:
        print(f"   {len(misaligned)} image entries have no stamped text counterpart:")
        for fp in misaligned[: args.show]:
            txt = _counterpart_text(fp, {})
            exists = "caption EXISTS" if os.path.isfile(txt) else "no caption file"
            print(f"     {fp}  ({exists})")
        if len(misaligned) > args.show:
            print(f"     ... and {len(misaligned) - args.show} more")
    else:
        print("   OK — every image anchor maps to a stamped text entry.")

    print()
    if not problems and not misaligned:
        print("READY: cache will pass the RunPod wizard pre-flight and the in-training sourceless guard.")
        return 0
    print("NOT READY — resolve the items above (re-cache after enabling tolerate_missing_source, or prune).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
