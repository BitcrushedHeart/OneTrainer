"""Quick structural inspection of an OFT adapter safetensors file.

Prints metadata, key buckets, dtypes, and highlights potential issues:
  - initial_norm tensors with zero/very-small rows
  - dora_scale / initial_norm row-wise ratio extremes
  - any NaN/Inf
  - any exactly-zero tensors
  - oft_R.weight magnitude per tensor (should be modest)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from safetensors import safe_open


def row_norm(t: torch.Tensor) -> torch.Tensor:
    """Return a 1D tensor of per-row norms, flattening trailing dims."""
    if t.ndim == 0:
        return t.abs().unsqueeze(0)
    return t.float().reshape(t.shape[0], -1).norm(dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    path = Path(args.inp)
    print(f"[inspect] {path}", flush=True)
    print(f"[inspect] size: {path.stat().st_size / 1e6:.1f} MB", flush=True)

    info: dict = {"path": str(path), "size_bytes": path.stat().st_size}
    key_suffix_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    nan_inf_keys: list[dict] = []
    zero_norm_issues: list[dict] = []
    dora_ratio_issues: list[dict] = []
    per_tensor_rows: list[dict] = []

    # Stash row-norms of initial_norm and dora_scale so we can pair them.
    initial_norm_rows: dict[str, torch.Tensor] = {}
    dora_scale_rows: dict[str, torch.Tensor] = {}

    with safe_open(str(path), framework="pt", device="cpu") as f:
        try:
            metadata = dict(f.metadata() or {})
        except Exception:
            metadata = {}
        info["metadata"] = metadata

        keys = list(f.keys())
        info["num_tensors"] = len(keys)

        for k in keys:
            # derive suffix
            parts = k.rsplit(".", 1)
            suffix = parts[-1] if len(parts) == 2 else k
            key_suffix_counts[suffix] += 1

            t = f.get_tensor(k)
            dtype_counts[str(t.dtype)] += 1

            t32 = t.float()
            nan = int(torch.isnan(t32).sum().item())
            inf = int(torch.isinf(t32).sum().item())
            if nan or inf:
                nan_inf_keys.append({"name": k, "nan": nan, "inf": inf, "shape": list(t.shape)})

            # per-tensor row info for a few tensor classes we care about
            if k.endswith(".initial_norm") or k.endswith(".dora_scale"):
                rows = row_norm(t)  # for these, row IS the norm already, flatten is fine
                flat = t.float().flatten()
                stem = k[: -len(".initial_norm")] if k.endswith(".initial_norm") else k[: -len(".dora_scale")]
                if k.endswith(".initial_norm"):
                    initial_norm_rows[stem] = flat
                else:
                    dora_scale_rows[stem] = flat
                zero_rows = int((rows == 0).sum().item())
                per_tensor_rows.append(
                    {
                        "name": k,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "zero_rows": zero_rows,
                        "min_abs": float(flat.abs().min().item()),
                        "max_abs": float(flat.abs().max().item()),
                        "mean_abs": float(flat.abs().mean().item()),
                    }
                )
                if zero_rows > 0 and k.endswith(".initial_norm"):
                    zero_norm_issues.append(
                        {
                            "name": k,
                            "zero_rows": zero_rows,
                            "total_rows": int(t.shape[0]),
                        }
                    )
            elif k.endswith(".oft_R.weight"):
                per_tensor_rows.append(
                    {
                        "name": k,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                        "max_abs": float(t32.abs().max().item()),
                        "mean_abs": float(t32.abs().mean().item()),
                        "std": float(t32.std(unbiased=False).item()),
                    }
                )

            del t

    # Compute dora_scale / initial_norm ratio per paired stem to surface instability
    for stem, inorm in initial_norm_rows.items():
        if stem not in dora_scale_rows:
            continue
        dsc = dora_scale_rows[stem]
        if dsc.shape != inorm.shape:
            dora_ratio_issues.append(
                {
                    "stem": stem,
                    "shape_mismatch": True,
                    "initial_shape": list(inorm.shape),
                    "dora_shape": list(dsc.shape),
                }
            )
            continue
        # clamp denominator per LoRAModule.py:550
        denom = inorm.clamp(min=1e-8)
        ratio = dsc / denom
        r_abs = ratio.abs()
        abnormal = ((r_abs > 100.0) | (~torch.isfinite(ratio))).nonzero(as_tuple=False).flatten().tolist()
        if abnormal or r_abs.max().item() > 50.0:
            dora_ratio_issues.append(
                {
                    "stem": stem,
                    "max_abs_ratio": float(r_abs.max().item()),
                    "min_abs_ratio": float(r_abs.min().item()),
                    "num_rows": int(inorm.numel()),
                    "abnormal_rows_count": len(abnormal),
                    "abnormal_rows_sample": abnormal[:10],
                    "sample_initial_norm": [float(inorm[i].item()) for i in abnormal[:5]],
                    "sample_dora_scale": [float(dsc[i].item()) for i in abnormal[:5]],
                    "sample_ratio": [float(ratio[i].item()) for i in abnormal[:5]],
                }
            )

    # Report
    print()
    print("=== Metadata ===")
    for k, v in sorted(metadata.items())[:30]:
        vs = v if len(str(v)) < 200 else str(v)[:200] + "..."
        print(f"  {k}: {vs}")

    print()
    print("=== Dtype histogram ===")
    for d, c in dtype_counts.most_common():
        print(f"  {d}: {c}")

    print()
    print(f"=== Key suffixes (top {args.top}) ===")
    for suf, c in key_suffix_counts.most_common(args.top):
        print(f"  .{suf}: {c}")

    print()
    print(f"=== NaN/Inf tensors: {len(nan_inf_keys)} ===")
    for row in nan_inf_keys[: args.top]:
        print(f"  {row}")

    print()
    print(f"=== initial_norm tensors with zero rows: {len(zero_norm_issues)} ===")
    for row in zero_norm_issues[: args.top]:
        print(f"  {row}")

    print()
    print(f"=== dora_scale/initial_norm ratio anomalies: {len(dora_ratio_issues)} ===")
    for row in dora_ratio_issues[: args.top]:
        print(f"  {row}")

    # Surface biggest oft_R and biggest initial_norm/dora_scale
    def top(rows, key, n=10, reverse=True):
        return sorted([r for r in rows if key in r], key=lambda r: r[key], reverse=reverse)[:n]

    print()
    print("=== initial_norm / dora_scale: top max_abs ===")
    rel_rows = [r for r in per_tensor_rows if r["name"].endswith(".initial_norm") or r["name"].endswith(".dora_scale")]
    for r in top(rel_rows, "max_abs", n=15):
        print(
            f"  {r['name']}  max_abs={r['max_abs']:.4g}  min_abs={r['min_abs']:.4g}  shape={r['shape']}  dtype={r['dtype']}  zero_rows={r.get('zero_rows')}"
        )

    print()
    print("=== initial_norm / dora_scale: top min_abs near zero ===")
    for r in sorted(rel_rows, key=lambda r: r["min_abs"])[:15]:
        print(
            f"  {r['name']}  min_abs={r['min_abs']:.4g}  max_abs={r['max_abs']:.4g}  shape={r['shape']}  zero_rows={r.get('zero_rows')}"
        )

    print()
    print("=== oft_R.weight: top max_abs ===")
    r_rows = [r for r in per_tensor_rows if r["name"].endswith(".oft_R.weight")]
    for r in top(r_rows, "max_abs", n=10):
        print(
            f"  {r['name']}  max_abs={r['max_abs']:.4g}  mean_abs={r['mean_abs']:.4g}  std={r['std']:.4g}  dtype={r['dtype']}"
        )

    if args.out_json:
        out = {
            "info": info,
            "dtype_counts": dict(dtype_counts),
            "key_suffix_counts": dict(key_suffix_counts),
            "nan_inf_keys": nan_inf_keys,
            "zero_norm_issues": zero_norm_issues,
            "dora_ratio_issues": dora_ratio_issues,
            "per_tensor_rows": per_tensor_rows,
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"\n[done] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
