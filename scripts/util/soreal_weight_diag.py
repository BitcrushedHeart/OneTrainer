"""SoReal weight diagnostic.

Loads two safetensors checkpoints tensor-by-tensor (no full model assembly),
computes numerical statistics, flags NaN/Inf, and produces a comparison
report. Designed to diagnose why v0.96 is NaN'ing in training while v0.95
was stable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from safetensors import safe_open

FINITE_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    """Compute numerical stats for a single tensor.

    Promotes to float32 for reductions to avoid bf16/fp16 overflow during
    the reduction itself (a genuine concern for SDXL unet tensors).
    """
    if t.dtype not in FINITE_DTYPES:
        return {
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
            "skipped": True,
        }

    t32 = t.detach().to(torch.float32)
    nan_mask = torch.isnan(t32)
    inf_mask = torch.isinf(t32)
    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())

    finite = t32[~(nan_mask | inf_mask)]
    if finite.numel() == 0:
        return {
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
            "nan": nan_count,
            "inf": inf_count,
            "finite_numel": 0,
        }

    abs_finite = finite.abs()
    return {
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "nan": nan_count,
        "inf": inf_count,
        "finite_numel": int(finite.numel()),
        "max_abs": float(abs_finite.max().item()),
        "mean_abs": float(abs_finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "l2_norm": float(finite.norm(p=2).item()),
    }


def scan_checkpoint(path: Path) -> dict[str, Any]:
    print(f"[scan] {path.name} ({path.stat().st_size / 1e9:.2f} GB)", flush=True)
    per_tensor: dict[str, dict[str, Any]] = {}
    bad_tensors: list[str] = []
    metadata: dict[str, str] | None = None

    with safe_open(str(path), framework="pt", device="cpu") as f:
        try:
            metadata = f.metadata() or {}
        except Exception:
            metadata = {}
        keys = list(f.keys())
        for i, key in enumerate(keys):
            t = f.get_tensor(key)
            stats = tensor_stats(t)
            per_tensor[key] = stats
            if stats.get("nan", 0) or stats.get("inf", 0):
                bad_tensors.append(key)
            if (i + 1) % 200 == 0 or (i + 1) == len(keys):
                print(f"  ... {i + 1}/{len(keys)} tensors", flush=True)
            del t

    return {
        "path": str(path),
        "num_tensors": len(per_tensor),
        "metadata": metadata,
        "tensors": per_tensor,
        "bad_tensors": bad_tensors,
    }


def classify(name: str) -> str:
    """Rough bucket so we can see which subsystem is drifting."""
    n = name.lower()
    if n.startswith("model.diffusion_model") or ".unet." in n or n.startswith("unet."):
        if "attn" in n or "attention" in n:
            return "unet.attn"
        if "ff" in n or "mlp" in n or "proj" in n:
            return "unet.ff/proj"
        if "conv" in n:
            return "unet.conv"
        if "norm" in n or "ln_" in n:
            return "unet.norm"
        return "unet.other"
    if "text_encoder" in n or n.startswith("conditioner") or "clip" in n:
        if "attn" in n:
            return "te.attn"
        if "mlp" in n or "fc" in n:
            return "te.mlp"
        if "norm" in n or "ln_" in n:
            return "te.norm"
        return "te.other"
    if "first_stage" in n or "vae" in n or "decoder" in n or "encoder" in n:
        return "vae"
    return "other"


def compare(prev: dict[str, Any], curr: dict[str, Any], top_n: int = 30) -> dict[str, Any]:
    shared = sorted(set(prev["tensors"]) & set(curr["tensors"]))
    only_prev = sorted(set(prev["tensors"]) - set(curr["tensors"]))
    only_curr = sorted(set(curr["tensors"]) - set(prev["tensors"]))

    rows: list[dict[str, Any]] = []
    bucket_agg: dict[str, dict[str, float]] = {}

    for key in shared:
        a = prev["tensors"][key]
        b = curr["tensors"][key]
        if a.get("skipped") or b.get("skipped"):
            continue
        if "max_abs" not in a or "max_abs" not in b:
            continue

        # avoid div-by-zero
        def ratio(x: float, y: float) -> float:
            if y == 0 and x == 0:
                return 1.0
            if y == 0:
                return math.inf
            return x / y

        r_max = ratio(b["max_abs"], a["max_abs"])
        r_std = ratio(b["std"], a["std"]) if a.get("std") else float("nan")
        r_norm = ratio(b["l2_norm"], a["l2_norm"])

        bucket = classify(key)
        agg = bucket_agg.setdefault(
            bucket,
            {
                "count": 0,
                "max_abs_ratio_max": 0.0,
                "max_abs_ratio_sum": 0.0,
                "norm_ratio_sum": 0.0,
                "std_ratio_sum": 0.0,
                "nan_curr": 0,
                "inf_curr": 0,
                "nan_prev": 0,
                "inf_prev": 0,
            },
        )
        agg["count"] += 1
        agg["max_abs_ratio_sum"] += r_max if math.isfinite(r_max) else 0.0
        agg["max_abs_ratio_max"] = max(
            agg["max_abs_ratio_max"], r_max if math.isfinite(r_max) else agg["max_abs_ratio_max"]
        )
        agg["norm_ratio_sum"] += r_norm if math.isfinite(r_norm) else 0.0
        agg["std_ratio_sum"] += r_std if math.isfinite(r_std) else 0.0
        agg["nan_curr"] += int(b.get("nan", 0))
        agg["inf_curr"] += int(b.get("inf", 0))
        agg["nan_prev"] += int(a.get("nan", 0))
        agg["inf_prev"] += int(a.get("inf", 0))

        rows.append(
            {
                "name": key,
                "bucket": bucket,
                "dtype_prev": a["dtype"],
                "dtype_curr": b["dtype"],
                "prev_max_abs": a["max_abs"],
                "curr_max_abs": b["max_abs"],
                "max_abs_ratio": r_max,
                "prev_std": a.get("std"),
                "curr_std": b.get("std"),
                "std_ratio": r_std,
                "prev_l2": a["l2_norm"],
                "curr_l2": b["l2_norm"],
                "l2_ratio": r_norm,
                "prev_nan": a.get("nan", 0),
                "curr_nan": b.get("nan", 0),
                "prev_inf": a.get("inf", 0),
                "curr_inf": b.get("inf", 0),
            }
        )

    # rank by magnitude of drift
    rows.sort(
        key=lambda r: (
            0 if math.isfinite(r["max_abs_ratio"]) else -1,
            abs(math.log(max(r["max_abs_ratio"], 1e-12))) if r["max_abs_ratio"] > 0 else 0,
        ),
        reverse=True,
    )
    top_drift = rows[:top_n]

    # biggest absolute max in curr (irrespective of ratio)
    by_curr_max = sorted(rows, key=lambda r: r["curr_max_abs"], reverse=True)[:top_n]

    # aggregate per-bucket means
    for b, agg in bucket_agg.items():
        c = max(agg["count"], 1)
        agg["max_abs_ratio_mean"] = agg["max_abs_ratio_sum"] / c
        agg["norm_ratio_mean"] = agg["norm_ratio_sum"] / c
        agg["std_ratio_mean"] = agg["std_ratio_sum"] / c

    return {
        "shared_count": len(shared),
        "only_prev_count": len(only_prev),
        "only_curr_count": len(only_curr),
        "only_prev_sample": only_prev[:20],
        "only_curr_sample": only_curr[:20],
        "top_drift": top_drift,
        "top_curr_max": by_curr_max,
        "buckets": bucket_agg,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--curr", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    prev_scan = scan_checkpoint(Path(args.prev))
    curr_scan = scan_checkpoint(Path(args.curr))
    diff = compare(prev_scan, curr_scan)

    # trim per-tensor to summary (keep bad + top-drift full, drop the rest to keep file size sane)
    keep_keys: set[str] = set(prev_scan["bad_tensors"]) | set(curr_scan["bad_tensors"])
    for row in diff["top_drift"] + diff["top_curr_max"]:
        keep_keys.add(row["name"])

    prev_summary = {k: v for k, v in prev_scan["tensors"].items() if k in keep_keys}
    curr_summary = {k: v for k, v in curr_scan["tensors"].items() if k in keep_keys}

    out = {
        "prev": {
            "path": prev_scan["path"],
            "num_tensors": prev_scan["num_tensors"],
            "metadata": prev_scan["metadata"],
            "bad_tensors": prev_scan["bad_tensors"],
            "tensors_sampled": prev_summary,
        },
        "curr": {
            "path": curr_scan["path"],
            "num_tensors": curr_scan["num_tensors"],
            "metadata": curr_scan["metadata"],
            "bad_tensors": curr_scan["bad_tensors"],
            "tensors_sampled": curr_summary,
        },
        "diff": diff,
    }

    Path(args.out_json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    print(f"  prev bad: {len(prev_scan['bad_tensors'])}  curr bad: {len(curr_scan['bad_tensors'])}")
    print(
        f"  shared tensors: {diff['shared_count']}  only_prev: {diff['only_prev_count']}  only_curr: {diff['only_curr_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
