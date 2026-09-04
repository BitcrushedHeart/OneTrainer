"""Fit BLOCK-DIAGONAL orthogonal rotations to the distillation delta.

Follow-up to _distill_rotation_diag.py. That established (V0.95 Base vs
Distilled, late blocks): the delta is tiny (~1-2%) and spectrum-preserving,
hence *some* dense two-sided rotation pair explains ~99.95% of it (near-
tautological given equal spectra). The non-tautological question: how much
survives OneTrainer's OFT constraints, i.e. BLOCK-DIAGONAL rotation blocks,
and one-sided (current impl) vs two-sided (would need infra change)?

Per matrix, reports fraction of delta energy explained (1 - resid^2/||d||^2) by:
  rot1s_blk : R_L block-diag, left only        -- what OneTrainer OFT can express
  rot2s_blk : R_L, R_R both block-diag         -- OFT with a two-sided extension
  rot2s_dense: closed form 1-(||sig_d-sig_b||/||d||)^2  -- dense ceiling (von Neumann)

Solver: alternating blockwise Procrustes from identity init. Left step
decomposes over row blocks (min ||R_i X_i - Wd_i|| -> SVD of Wd_i X_i^T);
right step over column blocks (min ||Y_j R_j - Wd_j|| -> SVD of Y_j^T Wd_j).
Each step is exactly optimal given the other side, so the residual decreases
monotonically. Near-identity target => no local-minimum drama expected.

All fp32 on GPU (weights are bf16; fp32 rounding is far below bf16
quantization noise). TF32 disabled -- its ~1e-3 relative matmul error is NOT
below the signal here.
"""

import argparse
import re
import time

import torch

from safetensors import safe_open

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Base_V0.95.safetensors"
DEFAULT_DIST = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Distilled_V0.95.safetensors"

FAMILIES = {
    "qkv": ("attention.to_q", "attention.to_k", "attention.to_v"),
    "mlp": ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
}
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)\.weight$")


def blockwise_left(x: torch.Tensor, wd: torch.Tensor, b: int) -> torch.Tensor:
    """Optimal block-diag orthogonal R minimizing ||R x - wd||; returns R x."""
    out = torch.empty_like(wd)
    for i in range(0, x.shape[0], b):
        u, _, vh = torch.linalg.svd(wd[i : i + b] @ x[i : i + b].T)
        out[i : i + b] = (u @ vh) @ x[i : i + b]
    return out


def blockwise_right(y: torch.Tensor, wd: torch.Tensor, b: int) -> torch.Tensor:
    """Optimal block-diag orthogonal R minimizing ||y R - wd||; returns y R."""
    out = torch.empty_like(wd)
    for j in range(0, y.shape[1], b):
        u, _, vh = torch.linalg.svd(y[:, j : j + b].T @ wd[:, j : j + b])
        out[:, j : j + b] = y[:, j : j + b] @ (u @ vh)
    return out


def expl(fit: torch.Tensor, wd: torch.Tensor, ndelta2: float) -> float:
    return 1.0 - float(((fit - wd) ** 2).sum()) / ndelta2


def singvals(w: torch.Tensor) -> torch.Tensor:
    m, n = w.shape
    g = w @ w.T if m <= n else w.T @ w
    return torch.linalg.eigvalsh(g).clamp_min(0).sqrt().flip(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--distilled", default=DEFAULT_DIST)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--iters", type=int, default=12, help="alternating left/right sweeps for the two-sided fit")
    ap.add_argument("--early-end", type=int, default=6)
    ap.add_argument("--mid-end", type=int, default=21)
    ap.add_argument("--band", default="late", choices=("early", "mid", "late", "all"))
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False

    stem_to_fam = {stem: fam for fam, stems in FAMILIES.items() for stem in stems}

    def band_of(i: int) -> str:
        return "early" if i <= args.early_end else ("mid" if i <= args.mid_end else "late")

    agg = {fam: {"w": 0.0, "r1b": 0.0, "r2b": 0.0, "r2d": 0.0, "n": 0} for fam in FAMILIES}
    t0 = time.time()

    with (
        safe_open(args.base, framework="pt", device="cpu") as fb,
        safe_open(args.distilled, framework="pt", device="cpu") as fd,
    ):
        keys = sorted(
            (int(m.group(1)), m.group(2), k)
            for k in fb.keys()
            if (m := LAYER_RE.match(k))
            and m.group(2) in stem_to_fam
            and (args.band == "all" or band_of(int(m.group(1))) == args.band)
        )
        print(f"block-fit: {len(keys)} matrices, block={args.block}, device={args.device}, band={args.band}\n")
        print(f"{'layer':>5} {'fam':>4} {'matrix':>18} {'rot1s_blk':>10} {'rot2s_blk':>10} {'rot2s_dense':>12}", flush=True)

        for i, stem, k in keys:
            wb = fb.get_tensor(k).to(args.device, torch.float32)
            wd = fd.get_tensor(k).to(args.device, torch.float32)
            if wb.shape[0] % args.block or wb.shape[1] % args.block:
                print(f"{i:>5} {stem_to_fam[stem]:>4} {stem:>18}  SKIP shape {tuple(wb.shape)} % {args.block} != 0")
                continue
            ndelta2 = float(((wd - wb) ** 2).sum())

            # one-sided, block-diag (what OneTrainer OFT expresses today)
            r1b = expl(blockwise_left(wb, wd, args.block), wd, ndelta2)

            # two-sided, block-diag: alternate exact left/right block solves
            fit = wb
            for _ in range(args.iters):
                fit = blockwise_left(fit, wd, args.block)
                fit = blockwise_right(fit, wd, args.block)
            r2b = expl(fit, wd, ndelta2)

            # dense two-sided ceiling, closed form (von Neumann)
            sb, sd = singvals(wb), singvals(wd)
            r2d = 1.0 - float(((sd - sb) ** 2).sum()) / ndelta2

            fam = stem_to_fam[stem]
            print(f"{i:>5} {fam:>4} {stem:>18} {r1b:>10.4f} {r2b:>10.4f} {r2d:>12.4f}", flush=True)
            a = agg[fam]
            a["w"] += ndelta2
            a["r1b"] += ndelta2 * r1b
            a["r2b"] += ndelta2 * r2b
            a["r2d"] += ndelta2 * r2d
            a["n"] += 1
            del wb, wd, fit
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    print("\n=== per-family (energy-weighted) ===")
    print(f"{'family':>6} {'matrices':>9} {'rot1s_blk':>10} {'rot2s_blk':>10} {'rot2s_dense':>12}")
    for fam, a in agg.items():
        if a["n"]:
            w = a["w"]
            print(f"{fam:>6} {a['n']:>9} {a['r1b'] / w:>10.4f} {a['r2b'] / w:>10.4f} {a['r2d'] / w:>12.4f}")

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
