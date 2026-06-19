"""Detect the exact cells where (initial_norm == 0) AND (dora_scale == 0) will
produce NaN under `dora_scale / initial_norm` at inference time (per the
ComfyUI-OFTv2 loader, which does not clamp like OneTrainer does).

Also checks: rows where one is zero and the other isn't (Inf / 0, still bad).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()

    path = Path(args.inp)
    print(f"[nan-check] {path}")
    total_rows_dead = 0
    total_tensors_with_issues = 0
    tensors_checked = 0

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        inorm_keys = [k for k in keys if k.endswith(".initial_norm")]

        for ik in sorted(inorm_keys):
            stem = ik[: -len(".initial_norm")]
            dk = stem + ".dora_scale"
            if dk not in keys:
                continue
            inorm = f.get_tensor(ik).float()
            dscale = f.get_tensor(dk).float()
            if inorm.shape != dscale.shape:
                print(
                    f"  [WARN] {stem}: shape mismatch initial_norm={list(inorm.shape)} dora_scale={list(dscale.shape)}"
                )
                continue

            tensors_checked += 1

            inorm_zero = inorm == 0
            dscale_zero = dscale == 0
            # Simulated ComfyUI behavior: scale = dscale / inorm (no clamp)
            # Cases:
            #   inorm==0, dscale==0 -> 0/0 = NaN
            #   inorm==0, dscale!=0 -> x/0 = ±Inf
            #   inorm!=0, dscale==0 -> 0
            #   otherwise -> finite
            nan_mask = inorm_zero & dscale_zero
            inf_mask = inorm_zero & ~dscale_zero
            problem_mask = nan_mask | inf_mask

            if problem_mask.any():
                total_tensors_with_issues += 1
                nan_idx = nan_mask.nonzero(as_tuple=False).flatten().tolist()
                inf_idx = inf_mask.nonzero(as_tuple=False).flatten().tolist()
                if nan_idx:
                    print(
                        f"  [NaN] {stem}  rows {nan_idx[:10]}{'...' if len(nan_idx) > 10 else ''}  (0/0 under ComfyUI-OFTv2)"
                    )
                    total_rows_dead += len(nan_idx)
                if inf_idx:
                    sample_dscale = [float(dscale.flatten()[i].item()) for i in inf_idx[:5]]
                    print(
                        f"  [Inf] {stem}  rows {inf_idx[:10]}{'...' if len(inf_idx) > 10 else ''}  dscale_sample={sample_dscale} (x/0)"
                    )
                    total_rows_dead += len(inf_idx)

    print()
    print(f"[summary] tensors checked: {tensors_checked}")
    print(f"[summary] tensors with NaN/Inf-producing rows: {total_tensors_with_issues}")
    print(f"[summary] total rows that will produce NaN/Inf at inference: {total_rows_dead}")
    return 0 if total_rows_dead == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
