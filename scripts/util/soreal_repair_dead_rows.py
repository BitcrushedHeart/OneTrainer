"""Repair all-zero weight rows in a merged checkpoint by sourcing them from
a reference checkpoint of the same shape.

Intended for models where an earlier merge step (e.g. ComfyUI's fp16 LoRA
apply) underflowed a small-norm row to exactly zero. Iterates every 2D+
float tensor in --current, finds rows with zero L2 norm, and copies the
matching rows from --reference. Preserves dtype, metadata, and all other
tensors byte-for-byte.

Writes <current>.repaired.safetensors next to the source unless --out is
given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import save_file

FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True, help="Damaged checkpoint to repair")
    ap.add_argument("--reference", required=True, help="Reference checkpoint to pull clean rows from")
    ap.add_argument("--out", default=None, help="Output path; defaults to <current>.repaired.safetensors")
    ap.add_argument("--dry-run", action="store_true", help="Report dead rows but don't write output")
    args = ap.parse_args()

    cur = Path(args.current)
    ref = Path(args.reference)
    dst = Path(args.out) if args.out else cur.with_suffix(".repaired.safetensors")

    print(f"[repair] current   : {cur}")
    print(f"[repair] reference : {ref}")
    if not cur.is_file() or not ref.is_file():
        print("[repair] error: missing input file", file=sys.stderr)
        return 2

    repairs: list[tuple[str, list[int]]] = []
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}

    with (
        safe_open(str(cur), framework="pt", device="cpu") as fc,
        safe_open(str(ref), framework="pt", device="cpu") as fr,
    ):
        try:
            metadata = dict(fc.metadata() or {})
        except Exception:
            metadata = {}

        cur_keys = set(fc.keys())
        ref_keys = set(fr.keys())

        for k in sorted(cur_keys):
            t = fc.get_tensor(k)
            if t.dtype not in FLOAT_DTYPES or t.ndim < 2:
                tensors[k] = t
                continue

            row_norms = t.float().reshape(t.shape[0], -1).norm(dim=1)
            dead = (row_norms == 0).nonzero(as_tuple=False).flatten().tolist()
            if not dead:
                tensors[k] = t
                continue

            if k not in ref_keys:
                print(f"[repair] {k}: dead rows {dead} but NOT in reference — leaving as-is")
                tensors[k] = t
                continue

            r = fr.get_tensor(k)
            if r.shape != t.shape or r.dtype != t.dtype:
                print(
                    f"[repair] {k}: shape/dtype mismatch (cur={tuple(t.shape)}/{t.dtype} "
                    f"ref={tuple(r.shape)}/{r.dtype}) — leaving as-is"
                )
                tensors[k] = t
                continue

            # Verify reference rows are finite & non-zero before copying
            r_row_norms = r.float().reshape(r.shape[0], -1).norm(dim=1)
            bad_in_ref = [i for i in dead if (not torch.isfinite(r_row_norms[i])) or r_row_norms[i].item() == 0]
            if bad_in_ref:
                print(f"[repair] {k}: reference rows {bad_in_ref} are also zero/non-finite — leaving as-is")
                tensors[k] = t
                continue

            patched = t.clone()
            for i in dead:
                patched[i] = r[i]
            tensors[k] = patched
            repairs.append((k, dead))

    if not repairs:
        print("[repair] no repairable dead rows found.")
        return 0

    print(f"[repair] repaired {len(repairs)} tensor(s):")
    for k, idx in repairs:
        print(f"  {k}: rows {idx}")

    if args.dry_run:
        print("[repair] dry-run — not writing output.")
        return 0

    save_file(tensors, str(dst), metadata=metadata)
    print(f"[repair] wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
