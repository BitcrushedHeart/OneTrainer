"""Inspect + compare two OFT/DoRA-OFT adapter .safetensors files.

Reports rotation magnitude (oft_R std => effective rotation), block geometry,
initial_norm + dora_scale health, NaN/Inf, and a PATH-SAFE diff of the two
embedded ot_config training configs. Pure header/tensor read; no model load.

Usage: python scripts/util/_oft_compare_adapters.py <fileA> <fileB> <out.json>
"""

import json
import math
import sys
from collections import Counter

import torch

from safetensors import safe_open


def tail(key: str) -> str:
    return key.split(".")[-1]


def agg_stats(tensors) -> dict:
    """Aggregate stats over an iterable of 1-D-flattened float tensors."""
    mins, maxs, absmax, nan, inf, nz, near0, total = [], [], 0.0, 0, 0, 0, 0, 0
    sq_sum, sum_ = 0.0, 0.0
    for t in tensors:
        tf = t.float().flatten()
        mins.append(float(tf.min()))
        maxs.append(float(tf.max()))
        absmax = max(absmax, float(tf.abs().max()))
        nan += int(torch.isnan(tf).sum())
        inf += int(torch.isinf(tf).sum())
        nz += int((tf == 0).sum())
        near0 += int((tf.abs() < 1e-6).sum())
        total += int(tf.numel())
        sum_ += float(tf.sum())
        sq_sum += float((tf * tf).sum())
    mean = sum_ / total if total else 0.0
    std = math.sqrt(max(sq_sum / total - mean * mean, 0.0)) if total else 0.0
    return {
        "n_layers": len(mins),
        "global_min": min(mins) if mins else None,
        "global_max": max(maxs) if maxs else None,
        "abs_max": absmax,
        "mean": mean,
        "std": std,
        "total_elems": total,
        "n_exact_zero": nz,
        "n_near_zero_1e-6": near0,
        "n_nan": nan,
        "n_inf": inf,
    }


def block_size_from_nelem(n_elements: int):
    # n_elements = b*(b-1)/2  ->  b = (1 + sqrt(1+8n))/2
    b = (1 + math.sqrt(1 + 8 * n_elements)) / 2
    return int(round(b)) if abs(b - round(b)) < 1e-6 else None


PATH_HINTS = ("\\", "/", ":", ".safetensors", ".json", ".pt", ".ckpt", ".png", ".jpg", ".txt")


def safe_scalar(v):
    if isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        if len(v) > 60 or any(h in v for h in PATH_HINTS):
            return "<redacted str>"
        return v
    return None  # skip lists/dicts at this level


def flat_scalars(cfg: dict, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat_scalars(v, key + "."))
        else:
            s = safe_scalar(v)
            if s is not None:
                out[key] = s
    return out


def inspect(path: str) -> dict:
    out = {"path": path, "metadata_keys": [], "key_tails": {}, "oft_geoms": []}
    with safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        out["metadata_keys"] = list(md.keys())
        out["_ot_config_raw"] = md.get("ot_config")
        keys = list(f.keys())
        out["key_tails"] = dict(Counter(tail(k) for k in keys))
        out["total_keys"] = len(keys)

        oft_keys = [k for k in keys if k.endswith("oft_R.weight")]
        norm_keys = [k for k in keys if k.endswith("initial_norm")]
        dora_keys = [k for k in keys if k.endswith("dora_scale")]

        # distinct oft_R geometries -> block_size / in_features
        geoms = {}
        for k in oft_keys:
            shp = tuple(f.get_slice(k).get_shape())
            geoms.setdefault(shp, k)
        for shp, k in geoms.items():
            r, n_elements = (shp[0], shp[1]) if len(shp) == 2 else (None, None)
            bs = block_size_from_nelem(n_elements) if n_elements else None
            out["oft_geoms"].append(
                {
                    "shape": list(shp),
                    "example_key": k,
                    "num_blocks_r": r,
                    "n_elements": n_elements,
                    "block_size": bs,
                    "in_features": (r * bs) if (r and bs) else None,
                }
            )

        out["oft_R"] = agg_stats(f.get_tensor(k) for k in oft_keys)
        out["initial_norm"] = agg_stats(f.get_tensor(k) for k in norm_keys) if norm_keys else None
        out["dora_scale"] = agg_stats(f.get_tensor(k) for k in dora_keys) if dora_keys else None
    return out


def config_diff(a_raw, b_raw):
    if not a_raw or not b_raw:
        return {"available": False}
    try:
        ca, cb = flat_scalars(json.loads(a_raw)), flat_scalars(json.loads(b_raw))
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)}
    diffs = {}
    for k in sorted(set(ca) | set(cb)):
        va, vb = ca.get(k, "<absent>"), cb.get(k, "<absent>")
        if va != vb:
            diffs[k] = {"A": va, "B": vb}
    return {"available": True, "differing_scalars": diffs}


def main():
    a, b, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    ra, rb = inspect(a), inspect(b)
    diff = config_diff(ra.pop("_ot_config_raw", None), rb.pop("_ot_config_raw", None))
    report = {"A": ra, "B": rb, "config_diff": diff}
    with open(outp, "w") as f:
        json.dump(report, f, indent=2)
    print("WROTE", outp)

    for label, r in (("A", ra), ("B", rb)):
        print(f"\n=== {label}: {r['path']}")
        for g in r["oft_geoms"]:
            print(
                f"  oft_R geom shape={g['shape']} block_size={g['block_size']} "
                f"num_blocks={g['num_blocks_r']} in_features={g['in_features']}"
            )
        o = r["oft_R"]
        print(
            f"  oft_R: layers={o['n_layers']} std={o['std']:.5g} abs_max={o['abs_max']:.5g} "
            f"nan={o['n_nan']} inf={o['n_inf']}"
        )
        n = r["initial_norm"]
        if n:
            print(
                f"  initial_norm: min={n['global_min']:.4g} max={n['global_max']:.4g} "
                f"near0={n['n_near_zero_1e-6']} zero={n['n_exact_zero']} nan={n['n_nan']}"
            )
        d = r["dora_scale"]
        if d:
            print(
                f"  dora_scale: min={d['global_min']:.4g} max={d['global_max']:.4g} std={d['std']:.4g} "
                f"near0={d['n_near_zero_1e-6']} zero={d['n_exact_zero']} nan={d['n_nan']}"
            )

    print("\n=== CONFIG DIFF (A=corrupt, B=good; path-like values redacted) ===")
    if diff.get("available"):
        for k, v in diff["differing_scalars"].items():
            print(f"  {k}: A={v['A']!r}  B={v['B']!r}")
    else:
        print("  unavailable:", diff)


if __name__ == "__main__":
    main()
