"""Patch zero-norm rows in a DoRA-OFT adapter safetensors.

Scans `initial_norm` tensors for rows == 0 and replaces them with 1.0 so that
downstream merge math (W_merged = (W_base + diff) * (dora_scale / initial_norm))
doesn't produce 0/0 = NaN at inference time. dora_scale for those rows is left
as-is (typically 0), which makes the effective scale 0/1 = 0 — i.e. the channel
stays dead rather than going NaN.

Writes to <input>.patched.safetensors next to the source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", help="Input safetensors path (positional).")
    ap.add_argument("--in", dest="inp_flag", default=None, help="Input safetensors path (alternative to positional).")
    ap.add_argument("--out", dest="outp", default=None, help="Output path; defaults to <in>.patched.safetensors")
    ap.add_argument("--eps", type=float, default=1.0, help="Replacement value for zero-norm rows (default 1.0).")
    args = ap.parse_args()

    inp = args.inp or args.inp_flag
    if not inp:
        ap.error("input path required (positional or --in)")
    args.inp = inp

    src = Path(args.inp)
    dst = Path(args.outp) if args.outp else src.with_suffix(".patched.safetensors")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    patched: list[tuple[str, list[int]]] = []

    with safe_open(str(src), framework="pt", device="cpu") as f:
        try:
            metadata = dict(f.metadata() or {})
        except Exception:
            metadata = {}
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith(".initial_norm") and t.ndim == 2:
                t32 = t.float()
                row_norms = t32.abs().sum(dim=1)  # initial_norm is already a per-row scalar
                zero_mask = row_norms == 0
                if zero_mask.any():
                    zero_idx = zero_mask.nonzero(as_tuple=False).flatten().tolist()
                    t = t.clone()
                    t[zero_idx, :] = args.eps
                    patched.append((k, zero_idx))
            tensors[k] = t

    if not patched:
        print(f"[patch] no zero-norm rows found in {src}; nothing to patch.")
        return 0

    save_file(tensors, str(dst), metadata=metadata)
    print(f"[patch] wrote {dst}")
    for k, idx in patched:
        print(f"  {k}: set rows {idx} to {args.eps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
