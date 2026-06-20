"""Report sourceless-metadata coverage of a SmartDiskCache (read-only).

Run this before shipping a cache to a cloud GPU for sourceless training: it
confirms that every entry in ``<cache>/image/cache.json`` and
``<cache>/text/cache.json`` carries the baked sourceless metadata. Reads only
the JSON index — no ``.pt`` tensors are opened — so it is fast even on a
multi-hundred-GB cache. Exits non-zero when anything is unstamped, so it can
gate a deploy/upload script.

    venv/Scripts/python.exe scripts/util/_verify_sourceless_cache.py --cache-dir F:/workspace/SoReal!/cache
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.util.sourceless_cache_util import sourceless_cache_coverage, sourceless_cache_problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="cache directory (the parent of the image/ and text/ subdirs, or a single-index VAE cache)",
    )
    parser.add_argument("--show", type=int, default=10, help="max missing paths to list per index (default: 10)")
    args = parser.parse_args()

    coverages = sourceless_cache_coverage(args.cache_dir)
    any_index = False
    for cov in coverages:
        if not cov.exists:
            continue
        any_index = True
        if cov.error:
            print(f"  {cov.path}: ERROR — {cov.error}")
            continue
        print(f"  {cov.path}: {cov.stamped}/{cov.total} stamped, {len(cov.missing)} missing")
        for missing_path in cov.missing[: args.show]:
            print(f"      missing: {missing_path}")
        if len(cov.missing) > args.show:
            print(f"      ... and {len(cov.missing) - args.show} more")

    if not any_index:
        print(f"  no cache.json found under '{args.cache_dir}'")

    problems = sourceless_cache_problems(args.cache_dir)
    print()
    if problems:
        print("NOT READY for sourceless training:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("READY: every cache entry carries baked sourceless metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
