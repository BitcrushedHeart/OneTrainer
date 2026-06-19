"""Summarize a soreal_weight_delta.py JSON by tensor suffix class.

Reports, per suffix (.dora_scale, .initial_norm, .oft_R.weight, .scaled_oft):
  - identical count, changed count
  - mean/max relative L2 drift
  - top absolute differences
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    rows = data["all_rows"]

    by_suffix = defaultdict(list)
    for r in rows:
        suf = r["name"].rsplit(".", 1)[-1] if "." in r["name"] else r["name"]
        # normalise oft_R.weight
        if r["name"].endswith(".oft_R.weight"):
            suf = "oft_R.weight"
        by_suffix[suf].append(r)

    print(f"Total tensors: {len(rows)}  identical: {data['identical_count']}  changed: {data['changed_count']}")
    print()

    for suf, items in sorted(by_suffix.items()):
        changed = [x for x in items if not x.get("identical") and not x.get("shape_mismatch")]
        identical = [x for x in items if x.get("identical")]
        shape_mismatch = [x for x in items if x.get("shape_mismatch")]
        print(
            f"=== .{suf} (n={len(items)}) — identical={len(identical)}  changed={len(changed)}  shape_mismatch={len(shape_mismatch)} ==="
        )
        if changed:
            rel = sorted(changed, key=lambda r: r["rel_l2"], reverse=True)
            print("  top rel_l2 drift:")
            for r in rel[: args.top]:
                print(
                    f"    {r['rel_l2']:.4g}  delta_max={r['delta_max']:.4g}  prev_l2={r['prev_l2']:.4g}  -> {r['name']}"
                )
            dmax = sorted(changed, key=lambda r: r["delta_max"], reverse=True)
            print("  top delta_max:")
            for r in dmax[: args.top]:
                print(f"    {r['delta_max']:.4g}  rel_l2={r['rel_l2']:.4g}  -> {r['name']}")
        print()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
