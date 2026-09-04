"""Diagnose how *rotation-like* the SoReal distillation delta is, per late-block matrix.

Question: could the distillation (Distilled - Base) be re-expressed as a pure
orthogonal rotation of the base weights (OFT-style), rather than an additive
delta that also moves singular values (gain)? A rotation preserves the entire
singular-value spectrum; step-distillation usually needs gain changes. This
measures, per weight matrix, whether the change is spectrum-preserving (rotation
would work) or gain-dominated (rotation-only would be too weak).

Per matrix W_base, W_dist (same shape m x n), all math on the smaller side
(<= 3840 here) so every op is a <=3840-square SVD/eigh:
  rel_delta  = ||d||_F / ||W_base||_F                      how much moved
  spec_shift = ||sig_d - sig_b|| / ||sig_b||               gain change (0 = spectrum unchanged)
  nuc_ratio  = sum(sig_d) / sum(sig_b)                     overall gain in/deflation
  rot_expl   = 1 - ||R W_base - W_dist||_F^2 / ||d||_F^2   fraction of the delta a single
               best-fit orthogonal transform explains (Procrustes; the generous,
               best-single-side bound). ~1 = rotation carries it; ~0 = it can't.

Read: high rot_expl AND low spec_shift -> the distillation IS mostly a rotation,
so the "strip magnitudes -> OFT" idea is alive. Low rot_expl OR high spec_shift
-> gain moved, rotation-only would discard the essential part.
"""

import argparse
import re
from collections import defaultdict

import torch

from safetensors import safe_open

DEFAULT_BASE = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Base_V0.95.safetensors"
DEFAULT_DIST = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul\SoReal!_Distilled_V0.95.safetensors"

# family -> the per-layer weight-key stems it covers
FAMILIES = {
    "qkv": ("attention.to_q", "attention.to_k", "attention.to_v"),
    "mlp": ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
    "out": ("attention.to_out.0",),  # bonus: the V22 magnitude-collapse layer
}
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)\.weight$")


def singvals(w: torch.Tensor) -> torch.Tensor:
    """Singular values of w via the smaller-side Gram (eigh), sorted descending."""
    m, n = w.shape
    g = w @ w.T if m <= n else w.T @ w
    ev = torch.linalg.eigvalsh(g)  # ascending, real, >= 0 up to fp noise
    return ev.clamp_min(0).sqrt().flip(0)


def procrustes_resid2(wb: torch.Tensor, wd: torch.Tensor) -> float:
    """Squared residual of the best-fit orthogonal transform, computed DIRECTLY.

    Forms R = U V^T from the SVD of the smaller-side cross-product, then measures
    ||R wb - wd||^2 (or ||wb R - wd||^2) directly -- no ||wb||^2+||wd||^2-2||.||_*
    cancellation, which annihilates a ~1% delta sitting on top of a large weight.
    Done in float64 so the residual survives.
    """
    wb = wb.cpu().double()  # fp64 SVD on CPU: predictable + no GPU contention (GeForce fp64 is 1/64 rate)
    wd = wd.cpu().double()
    m, n = wb.shape
    if m <= n:  # left problem: min_R ||R wb - wd||, R is m x m
        u, _, vh = torch.linalg.svd(wd @ wb.T)  # cross = wd wb^T
        resid = (u @ vh) @ wb - wd
    else:  # right problem: min_R ||wb R - wd||, R is n x n
        u, _, vh = torch.linalg.svd(wb.T @ wd)  # cross = wb^T wd
        resid = wb @ (u @ vh) - wd
    return float((resid**2).sum())


def analyze(wb: torch.Tensor, wd: torch.Tensor) -> dict:
    delta = wd - wb
    nb = float(wb.norm())
    nd = float(wd.norm())
    ndelta2 = float((delta**2).sum())
    sb, sd = singvals(wb), singvals(wd)
    spec_shift = float((sd - sb).norm() / sb.norm())
    nuc_ratio = float(sd.sum() / sb.sum())
    resid2 = procrustes_resid2(wb, wd)
    rot_expl = 1.0 - resid2 / ndelta2 if ndelta2 > 0 else 1.0
    return {
        "rel_delta": (ndelta2**0.5) / nb if nb > 0 else 0.0,
        "spec_shift": spec_shift,
        "nuc_ratio": nuc_ratio,
        "rot_expl": rot_expl,
        "ndelta2": ndelta2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--distilled", default=DEFAULT_DIST)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--early-end", type=int, default=6)
    ap.add_argument("--mid-end", type=int, default=21)
    ap.add_argument("--band", default="late", choices=("early", "mid", "late", "all"))
    args = ap.parse_args()

    stem_to_fam = {stem: fam for fam, stems in FAMILIES.items() for stem in stems}

    def band_of(i: int) -> str:
        return "early" if i <= args.early_end else ("mid" if i <= args.mid_end else "late")

    rows: list[tuple] = []
    agg: dict[str, dict] = defaultdict(lambda: {"w": 0.0, "spec": 0.0, "rot": 0.0, "reld": 0.0, "n": 0})

    with (
        safe_open(args.base, framework="pt", device="cpu") as fb,
        safe_open(args.distilled, framework="pt", device="cpu") as fd,
    ):
        keys = []
        for k in fb.keys():
            mobj = LAYER_RE.match(k)
            if not mobj:
                continue
            i, stem = int(mobj.group(1)), mobj.group(2)
            if stem in stem_to_fam and (args.band == "all" or band_of(i) == args.band):
                keys.append((i, stem, k))
        keys.sort()
        print(f"analyzing {len(keys)} matrices on {args.device}  (band={args.band})\n", flush=True)
        print(
            f"{'layer':>5} {'fam':>4} {'matrix':>18} {'rel_delta':>10} {'spec_shift':>11} {'nuc_ratio':>10} {'rot_expl':>9}",
            flush=True,
        )

        for i, stem, k in keys:
            wb = fb.get_tensor(k).to(args.device, torch.float32)
            wd = fd.get_tensor(k).to(args.device, torch.float32)
            try:
                r = analyze(wb, wd)
            except torch.cuda.OutOfMemoryError:
                wb, wd = wb.cpu(), wd.cpu()
                r = analyze(wb, wd)
            fam = stem_to_fam[stem]
            rows.append((i, fam, stem, r))
            print(
                f"{i:>5} {fam:>4} {stem:>18} {r['rel_delta']:>10.4f} {r['spec_shift']:>11.4f} "
                f"{r['nuc_ratio']:>10.4f} {r['rot_expl']:>9.4f}",
                flush=True,
            )
            a = agg[fam]
            w = r["ndelta2"]  # energy-weight the aggregates by how much the tensor moved
            a["w"] += w
            a["spec"] += w * r["spec_shift"]
            a["rot"] += w * r["rot_expl"]
            a["reld"] += w * r["rel_delta"]
            a["n"] += 1
            del wb, wd
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    print("\n=== per-family (energy-weighted) ===")
    print(f"{'family':>6} {'matrices':>9} {'rel_delta':>10} {'spec_shift':>11} {'rot_expl':>9}")
    for fam in sorted(agg):
        a = agg[fam]
        w = a["w"] or 1.0
        print(f"{fam:>6} {a['n']:>9} {a['reld'] / w:>10.4f} {a['spec'] / w:>11.4f} {a['rot'] / w:>9.4f}")

    print(
        "\nread: high rot_expl + low spec_shift => rotation-like (OFT idea alive); "
        "low rot_expl or high spec_shift => gain moved, rotation-only too weak."
    )


if __name__ == "__main__":
    main()
