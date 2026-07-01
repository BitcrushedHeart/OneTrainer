"""Deviation-from-identity for an OFT / DoRA-OFT adapter, in the COFT framework.

Rebuilds each rotation block R via the SAME 5-term Cayley-Neumann path used in
training/inference (dividing oft_R by 2*sqrt(block-1) iff the adapter carries the
scaled_oft flag), then reports, per block, aggregated over all layers:
  ||R - I||_F  -- the COFT "deviation from identity" (the quantity COFT bounds by eps)
  ||Q||_F      -- Frobenius norm of the skew generator (R ~= I + 2Q, so ||R-I|| ~= 2||Q||)

Because it accounts for scaled_oft, this is the apples-to-apples rotation strength
across runs -- the raw oft_R std is NOT comparable when scaled_oft differs.

Usage: python _oft_deviation.py <adapter.safetensors> [<adapter2> ...]
"""

import math
import os
import sys

import torch

from safetensors import safe_open


def block_size_from_nelem(n):
    return int(round((1 + math.sqrt(1 + 8 * n)) / 2))


def cayley_neumann_5(Q):
    # R = I + 2Q + 2Q^2 + 2Q^3 + Q^4  (matches oft_utils._cayley_batch, 5 terms)
    b, n, _ = Q.shape
    R = torch.eye(n, dtype=Q.dtype).repeat(b, 1, 1)
    R = R + 2.0 * Q
    q2 = torch.bmm(Q, Q)
    R = R + 2.0 * q2
    q3 = torch.bmm(q2, Q)
    R = R + 2.0 * q3
    q4 = torch.bmm(q3, Q)
    R = R + q4
    return R


def analyze(path):
    print(f"\n################ {path}")
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        oft_keys = sorted(k for k in keys if k.endswith("oft_R.weight"))
        if not oft_keys:
            print("  no oft_R.weight keys -- not an OFT adapter")
            return

        dev_all, qf_all = [], []
        worst = (0.0, None)
        scaled_any = False
        bs0 = None
        for wk in oft_keys:
            prefix = wk[: -len("weight")]  # "<...>.oft_R."
            scaled = (prefix + "scaled_oft") in keys
            scaled_any = scaled_any or scaled
            w = f.get_tensor(wk).to(torch.float64)  # (r, n_elements)
            r, n_elem = w.shape
            bs = block_size_from_nelem(n_elem)
            bs0 = bs0 or bs
            if scaled:
                w = w / (2.0 * math.sqrt(bs - 1))
            rows, cols = torch.triu_indices(bs, bs, 1)
            mat = torch.zeros(r, bs, bs, dtype=w.dtype)
            mat[:, rows, cols] = w
            skew = mat - mat.transpose(-2, -1)
            rot = cayley_neumann_5(skew)
            eye = torch.eye(bs, dtype=skew.dtype).unsqueeze(0)
            dev = (rot - eye).norm(dim=(1, 2))  # ||R-I||_F per block
            qf = skew.norm(dim=(1, 2))  # ||Q||_F per block
            dev_all.append(dev)
            qf_all.append(qf)
            if float(dev.max()) > worst[0]:
                worst = (float(dev.max()), wk.rsplit(".oft_R", 1)[0])

        dev = torch.cat(dev_all)
        qf = torch.cat(qf_all)
        print(
            f"  oft_R layers={len(oft_keys)}  blocks={dev.numel()}  block_size={bs0}  "
            f"scaled_oft={'ON' if scaled_any else 'OFF'}"
        )
        print(
            f"  ||R - I||_F per block:  mean={float(dev.mean()):.4g}  "
            f"median={float(dev.median()):.4g}  max={float(dev.max()):.4g}"
        )
        print(
            f"  ||Q||_F     per block:  mean={float(qf.mean()):.4g}  "
            f"median={float(qf.median()):.4g}  max={float(qf.max()):.4g}   (||Q||_2 <= ||Q||_F)"
        )
        print(f"  largest-rotation layer: {worst[1]}  (||R-I||_F={worst[0]:.4g})")


def main():
    for p in sys.argv[1:]:
        if not os.path.exists(p):
            print(f"\n################ {p}\n  FILE NOT FOUND -- skipping")
            continue
        analyze(p)


if __name__ == "__main__":
    main()
