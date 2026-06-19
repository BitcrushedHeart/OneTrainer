"""One-shot script: scan the RLHF chosen/rejected folders for paired images
and dump their source pixel dimensions to a JSON file that the unit tests can
parametrize over.

PIL ``Image.open(...).size`` reads only the header so the walk is cheap even
for thousands of files. Pair name is the file stem (e.g. ``pair_0092``); only
stems present in both ``chosen`` and ``rejected`` are emitted.

Usage:
    venv\\Scripts\\python.exe scripts\\util\\dump_rlhf_aspect_ratios.py

Override paths if your dataset lives elsewhere:
    venv\\Scripts\\python.exe scripts\\util\\dump_rlhf_aspect_ratios.py \\
        --chosen   F:/Datasets/RLHF/chosen/train \\
        --rejected F:/Datasets/RLHF/rejected/train \\
        --out      tests/data/rlhf_aspect_ratios.json
"""

import argparse
import json
from pathlib import Path

from PIL import Image

DEFAULT_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".avif")


def _index_dir(root: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for ext in extensions:
        for path in root.glob(f"*{ext}"):
            files[path.stem] = path
        for path in root.glob(f"*{ext.upper()}"):
            files[path.stem] = path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chosen", default="F:/Datasets/RLHF/chosen/train")
    parser.add_argument("--rejected", default="F:/Datasets/RLHF/rejected/train")
    parser.add_argument("--out", default="tests/data/rlhf_aspect_ratios.json")
    args = parser.parse_args()

    chosen_dir = Path(args.chosen)
    rejected_dir = Path(args.rejected)
    out_path = Path(args.out)

    if not chosen_dir.is_dir():
        raise SystemExit(f"chosen directory not found: {chosen_dir}")
    if not rejected_dir.is_dir():
        raise SystemExit(f"rejected directory not found: {rejected_dir}")

    chosen = _index_dir(chosen_dir, DEFAULT_EXTENSIONS)
    rejected = _index_dir(rejected_dir, DEFAULT_EXTENSIONS)

    common = sorted(set(chosen.keys()) & set(rejected.keys()))
    if not common:
        raise SystemExit("no common pair stems found between chosen and rejected")

    entries: dict[str, dict[str, list[int]]] = {}
    for stem in common:
        with Image.open(chosen[stem]) as ci, Image.open(rejected[stem]) as ri:
            entries[stem] = {
                "chosen": list(ci.size),  # (w, h)
                "rejected": list(ri.size),
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} pairs to {out_path}")
    print(f"  (chosen-only: {len(set(chosen) - set(rejected))}, rejected-only: {len(set(rejected) - set(chosen))})")


if __name__ == "__main__":
    main()
