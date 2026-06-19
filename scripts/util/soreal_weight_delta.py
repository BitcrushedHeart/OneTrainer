"""Tensor-wise delta between two safetensors checkpoints.

Computes for each shared tensor:
  * whether bytes are identical (merge skipped it)
  * abs-delta max, mean, l2
  * relative delta (delta_l2 / prev_l2)
  * dtype on each side
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from safetensors import safe_open


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--curr", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    prev_path = Path(args.prev)
    curr_path = Path(args.curr)
    print(f"[delta] {prev_path.name} vs {curr_path.name}", flush=True)

    rows = []
    dtype_hist_prev: dict[str, int] = {}
    dtype_hist_curr: dict[str, int] = {}
    identical_count = 0
    changed_count = 0

    with (
        safe_open(str(prev_path), framework="pt", device="cpu") as fa,
        safe_open(str(curr_path), framework="pt", device="cpu") as fb,
    ):
        keys = sorted(set(fa.keys()) & set(fb.keys()))
        for i, k in enumerate(keys):
            a = fa.get_tensor(k)
            b = fb.get_tensor(k)
            dtype_hist_prev[str(a.dtype)] = dtype_hist_prev.get(str(a.dtype), 0) + 1
            dtype_hist_curr[str(b.dtype)] = dtype_hist_curr.get(str(b.dtype), 0) + 1

            if a.shape != b.shape:
                rows.append(
                    {
                        "name": k,
                        "shape_mismatch": True,
                        "prev_shape": list(a.shape),
                        "curr_shape": list(b.shape),
                    }
                )
                continue

            # exact byte identity short-circuit
            if a.dtype == b.dtype and torch.equal(a, b):
                identical_count += 1
                rows.append(
                    {
                        "name": k,
                        "identical": True,
                        "dtype": str(a.dtype),
                        "numel": int(a.numel()),
                    }
                )
            else:
                changed_count += 1
                a32 = a.to(torch.float32)
                b32 = b.to(torch.float32)
                d = b32 - a32
                abs_d = d.abs()
                prev_l2 = a32.norm(p=2).item() or 1e-12
                rows.append(
                    {
                        "name": k,
                        "identical": False,
                        "dtype_prev": str(a.dtype),
                        "dtype_curr": str(b.dtype),
                        "numel": int(a.numel()),
                        "delta_max": float(abs_d.max().item()),
                        "delta_mean": float(abs_d.mean().item()),
                        "delta_l2": float(d.norm(p=2).item()),
                        "prev_l2": float(prev_l2),
                        "rel_l2": float(d.norm(p=2).item() / prev_l2),
                        "delta_nan": int(torch.isnan(d).sum().item()),
                        "delta_inf": int(torch.isinf(d).sum().item()),
                    }
                )
            if (i + 1) % 200 == 0 or (i + 1) == len(keys):
                print(f"  ... {i + 1}/{len(keys)}  changed={changed_count} identical={identical_count}", flush=True)
            del a, b

    # sort changed tensors by relative l2 drift
    changed = [r for r in rows if not r.get("identical") and not r.get("shape_mismatch")]
    changed.sort(key=lambda r: r["rel_l2"], reverse=True)

    out = {
        "prev": str(prev_path),
        "curr": str(curr_path),
        "total_shared": len(rows),
        "identical_count": identical_count,
        "changed_count": changed_count,
        "dtype_hist_prev": dtype_hist_prev,
        "dtype_hist_curr": dtype_hist_curr,
        "top_changed_by_rel_l2": changed[:40],
        "top_changed_by_delta_max": sorted(changed, key=lambda r: r["delta_max"], reverse=True)[:40],
        "all_rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[done] total_shared={len(rows)} identical={identical_count} changed={changed_count}")
    print(f"  dtype prev: {dtype_hist_prev}")
    print(f"  dtype curr: {dtype_hist_curr}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
