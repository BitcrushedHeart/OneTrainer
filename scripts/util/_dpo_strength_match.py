"""Find the DPO-OFT merge *strength* that makes Base / Lightning best match Turbo.

Premise: the DPO adapter (OFT_2) looks great baked into Turbo at strength 1.
Base, Turbo and Lightning are byte-identical except the late-block rotations of
qkv_late / mlp_late (blocks 22-29). We want a single scalar merge strength for
Base and for Lightning that lands their *merged* late-block weights as close as
possible to Turbo-at-strength-1.

Why a closed form exists: OneTrainer's merge `strength` is a LINEAR blend of the
final weight (oft_merge.py):  M_X(s) = (1-s)*W_X + s*(W_X @ R^T)  =  W_X + s*C_X,
with  C_X = W_X @ (R^T - I).  The bake side/convention is W_new = W @ R^T,
block-diagonal on the input dim (LoRAModule.apply_to_module, line 807-810).

Target (Turbo at s=1):  T = W_turbo @ R^T.
Per matrix, minimize ||W_X + s*C_X - T||_F over s  ->  a scalar least-squares:
    s* = <C_X, T0_X> / <C_X, C_X>,   T0_X = T - W_X.
Pool across matrices by summing numerators and denominators (proper pooled LS).

Also reported:
  base_diff  = ||W_turbo - W_X|| / ||W_X||   how far this model's late block is from Turbo
  closure    = 1 - ||s* C_X - T0_X||^2 / ||T0_X||^2   fraction of the *squared* gap
               to Turbo that the best scalar strength actually removes. The rest is
               the irreducible two-sided distillation delta a scalar can't cross.

Read: s* tells you the strength; closure tells you whether strength is even the
right tool here (high closure -> yes; low closure -> the gap is distillation, not
intensity, and no scalar will get you to Turbo).
"""

import argparse
import re
from collections import defaultdict

import torch

from safetensors import safe_open

torch.set_num_threads(1)  # batched linalg segfaults under Windows MKL with default threading

DIR = r"E:\AI\Data\Models\StableDiffusion\SoReal!\03Jul"
TURBO = DIR + r"\SoReal!_Turbo_V0.95.safetensors"
BASE = DIR + r"\SoReal!_Base_V0.95.safetensors"
LIGHT = DIR + r"\SoReal!_Lightning_V0.95.safetensors"
ADAPTER = r"E:\AI\Data\Models\Lora\DPO\SoReal_DPO_R2_5.safetensors"

# adapter stem (transformer.layers.N.<sub>.oft_R.weight) -> base key (layers.N.<sub>.weight)
FAMILIES = {
    "qkv": ("attention.to_q", "attention.to_k", "attention.to_v"),
    "out": ("attention.to_out.0",),
    "mlp": ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
}
STEM_TO_FAM = {s: f for f, ss in FAMILIES.items() for s in ss}
ADK = re.compile(r"^transformer\.layers\.(\d+)\.(.+)\.oft_R\.weight$")

BLOCK = 128
NEUMANN_TERMS = 5
CLIP = 0.99

# NOTE: modules/util/oft_align.py is the productionized twin of the reconstruct +
# least-squares math below (used at merge time to auto-align to the Turbo sentinel).
# This script stays self-contained so it runs without importing the model package.


def cayley_neumann(vec: torch.Tensor, block_size: int, clip: float) -> torch.Tensor:
    """Reproduce OFTRotationModule._cayley_batch: skew -> true-spectral clip -> Cayley-Neumann(5).

    vec: (r, n_elements) upper-tri params. Returns R: (r, block, block) orthogonal.
    Faithful to oft_utils.py except the spectral norm is the EXACT 2-norm (deterministic)
    rather than the 1-step power iteration from non-persistent random state.
    """
    r = vec.shape[0]
    rows, cols = torch.triu_indices(block_size, block_size, 1, device=vec.device)
    Q = torch.zeros(r, block_size, block_size, device=vec.device, dtype=vec.dtype)
    bidx = torch.arange(r, device=vec.device)[:, None]
    Q = Q.index_put((bidx, rows, cols), vec)
    Q = Q - Q.transpose(-2, -1)  # skew-symmetric

    if clip is not None:
        # exact spectral norm per block via symmetric eigh: for skew Q, Q^T Q = -Q@Q (SPD),
        # so sigma_max = sqrt(lambda_max(-Q@Q)). Avoids batched SVD (segfaults under Win MKL).
        ev = torch.linalg.eigvalsh(-torch.bmm(Q, Q))  # ascending, >= 0
        sigma = ev[:, -1].clamp_min(0).sqrt().reshape(r, 1, 1)
        Q = Q * (clip / torch.clamp(sigma, min=clip))

    R = torch.eye(block_size, device=vec.device, dtype=vec.dtype).repeat(r, 1, 1)
    R.add_(Q, alpha=2.0)
    Q2 = torch.bmm(Q, Q)
    R.add_(Q2, alpha=2.0)
    Qp = Q2
    for _ in range(3, NEUMANN_TERMS - 1):
        Qp = torch.bmm(Qp, Q)
        R.add_(Qp, alpha=2.0)
    Qp = torch.bmm(Qp, Q)
    R.add_(Qp)  # last term coeff 1 (matches oft_utils)
    return R


def merged_full(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """W @ R^T, block-diagonal on the input dim. Matches apply_to_module einsum 'orc,rkc->ork'."""
    out, inf = W.shape
    rank = R.shape[0]
    Wr = W.reshape(out, rank, R.shape[1])
    Mr = torch.einsum("orc,rkc->ork", Wr, R)
    return Mr.reshape(out, inf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--late-start", type=int, default=22)
    ap.add_argument("--sanity-layer", type=int, default=10, help="a shared mid block; should give s*~1, gap~0")
    ap.add_argument("--per-matrix", action="store_true", help="print every matrix, not just families")
    args = ap.parse_args()
    dev = args.device

    fa = safe_open(ADAPTER, framework="pt", device="cpu")
    fb = safe_open(BASE, framework="pt", device="cpu")
    ft = safe_open(TURBO, framework="pt", device="cpu")
    fl = safe_open(LIGHT, framework="pt", device="cpu")

    # collect adapter oft_R keys for late blocks + one sanity layer
    items = []  # (layer, fam, stem, adapter_key, base_key)
    for k in fa.keys():  # noqa: SIM118
        m = ADK.match(k)
        if not m:
            continue
        i, stem = int(m.group(1)), m.group(2)
        if stem not in STEM_TO_FAM:
            continue
        if i >= args.late_start or i == args.sanity_layer:
            items.append((i, STEM_TO_FAM[stem], stem, k, f"layers.{i}.{stem}.weight"))
    items.sort()

    # pooled accumulators: (num, den, gap2, resid2, basew2)  keyed by ('base'|'light', fam-or-scope)
    pool = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # flat per-matrix records for the granularity ladder: (tag, layer, fam, num, den, gap2)
    records: list[tuple] = []

    hdr = f"{'lyr':>3} {'fam':>4} {'matrix':>18} {'baseΔ%':>7} | {'s_base':>8} {'clos_b%':>8} | {'s_light':>8} {'clos_l%':>8}"
    if args.per_matrix:
        print(hdr, flush=True)

    # Phase 1: read every tensor up front (mmap only, no linalg). Interleaving
    # safetensors mmap reads with LAPACK/einsum segfaults on this Windows torch
    # build, so we fully separate I/O from math.
    preloaded = []  # (i, fam, stem, vec_cpu, Wb_cpu, Wt_cpu, Wl_cpu)
    for i, fam, stem, ak, bk in items:
        vec = fa.get_tensor(ak).to(torch.float32).clone()
        preloaded.append(
            (
                i,
                fam,
                stem,
                vec,
                fb.get_tensor(bk).to(torch.float32).clone(),
                ft.get_tensor(bk).to(torch.float32).clone(),
                fl.get_tensor(bk).to(torch.float32).clone(),
            )
        )
    del fa, fb, ft, fl  # close mmaps before any math

    # Phase 2: pure math (GPU if available). Sums accumulate in fp64 via dtype=,
    # without materializing a full double copy of the big matrices.
    for i, fam, stem, vec, Wb, Wt, Wl in preloaded:
        vec = vec.to(dev)
        R = cayley_neumann(vec, BLOCK, CLIP)
        Wb, Wt, Wl = Wb.to(dev), Wt.to(dev), Wl.to(dev)
        if Wb.shape[1] != R.shape[0] * BLOCK:
            print(f"  skip layers.{i}.{stem}: in_features {Wb.shape[1]} != rank*block {R.shape[0] * BLOCK}")
            continue

        T = merged_full(Wt, R)  # Turbo @ R^T  (target, strength 1)
        scope = "late" if i >= args.late_start else "sanity"

        row = {}
        for tag, W in (("base", Wb), ("light", Wl)):
            C = merged_full(W, R) - W  # = W @ (R^T - I)
            T0 = T - W
            num = torch.sum(C * T0, dtype=torch.float64).item()
            den = torch.sum(C * C, dtype=torch.float64).item()
            gap2 = torch.sum(T0 * T0, dtype=torch.float64).item()
            s = num / den if den > 0 else float("nan")
            resid2 = gap2 - (num * num / den if den > 0 else 0.0)
            basew2 = torch.sum(W * W, dtype=torch.float64).item()
            row[tag] = (s, gap2, resid2, basew2)
            records.append((tag, i, fam, num, den, gap2))
            for key in ((tag, scope), (tag, f"{scope}:{fam}")):
                p = pool[key]
                p[0] += num
                p[1] += den
                p[2] += gap2
                p[3] += resid2
                p[4] += basew2

        if args.per_matrix:
            sb, gb, rb, wb2 = row["base"]
            sl, gl, rl, wl2 = row["light"]
            bd = (gb**0.5 / wb2**0.5 * 100) if wb2 > 0 else 0.0
            cb = (1 - rb / gb) * 100 if gb > 0 else 100.0
            cl = (1 - rl / gl) * 100 if gl > 0 else 100.0
            print(
                f"{i:>3} {fam:>4} {stem:>18} {bd:>7.3f} | {sb:>8.3f} {cb:>8.1f} | {sl:>8.3f} {cl:>8.1f}",
                flush=True,
            )
        del vec, R, Wb, Wt, Wl, T
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    def show(scope_label: str, scope_key: str) -> None:
        print(f"\n=== {scope_label} (pooled least-squares strength; Turbo := 1.000) ===")
        print(f"{'model':>7} {'scope':>14} {'strength':>9} {'gap_closed':>11} {'baseΔ_toTurbo':>14}")
        for tag in ("base", "light"):
            for key in sorted(pool):
                if key[0] != tag or not key[1].startswith(scope_key):
                    continue
                num_den = pool[key]
                # recompute pooled num from den*s? we stored num,den as [0],[1]
                s = num_den[0] / num_den[1] if num_den[1] > 0 else float("nan")
                closed = (1 - num_den[3] / num_den[2]) * 100 if num_den[2] > 0 else 100.0
                bd = (num_den[2] ** 0.5 / num_den[4] ** 0.5 * 100) if num_den[4] > 0 else 0.0
                print(f"{tag:>7} {key[1]:>14} {s:>9.3f} {closed:>10.1f}% {bd:>13.3f}%")

    show("LATE blocks 22-29", "late")
    show(f"SANITY shared layer {args.sanity_layer}", "sanity")

    # ------- granularity ladder: how much of the Turbo gap a scalar-strength scheme
    # ------- can close as we allow more independent knobs. Only the differing late
    # ------- families (qkv+mlp) matter; grouping key coarsens from one to per-matrix.
    # one row per matrix; add a unique index for the per-matrix granularity
    late_recs = [
        (*r, idx) for idx, r in enumerate(r for r in records if r[1] >= args.late_start and r[2] in ("qkv", "mlp"))
    ]

    def ladder2(tag: str, group_of) -> tuple[float, float, float]:
        groups: dict = defaultdict(lambda: [0.0, 0.0])
        gap_tot = 0.0
        for t, i, fam, num, den, gap2, idx in late_recs:
            if t != tag:
                continue
            g = groups[group_of(i, fam, idx)]
            g[0] += num
            g[1] += den
            gap_tot += gap2
        explained = sum((num * num) / den if den > 0 else 0.0 for num, den in groups.values())
        strengths = [num / den if den > 0 else float("nan") for num, den in groups.values()]
        closed = (explained / gap_tot) * 100 if gap_tot > 0 else 0.0
        return closed, min(strengths), max(strengths)

    grans = [
        ("global", 1, lambda i, fam, idx: 0),
        ("per-family", 2, lambda i, fam, idx: fam),
        ("per-layer", 8, lambda i, fam, idx: i),
        ("per-layer x family", 16, lambda i, fam, idx: (i, fam)),
        ("per-matrix", 56, lambda i, fam, idx: idx),
    ]
    print("\n=== GRANULARITY LADDER: max Turbo-gap closed on qkv_late+mlp_late (energy-weighted) ===")
    print(
        f"{'granularity':>20} {'knobs':>5} | {'base clos%':>10} {'base s-range':>16} | {'light clos%':>11} {'light s-range':>16}"
    )
    for label, knobs, gof in grans:
        cb, blo, bhi = ladder2("base", gof)
        cl, llo, lhi = ladder2("light", gof)
        print(
            f"{label:>20} {knobs:>5} | {cb:>10.1f} {f'{blo:.3f}-{bhi:.3f}':>16} | {cl:>11.1f} {f'{llo:.3f}-{lhi:.3f}':>16}"
        )

    # The actually-differing late blocks only: qkv_late + mlp_late (exclude `out`,
    # which is identical across models and just drags the pooled strength toward 1.0).
    print("\n=== LATE, ONLY the differing families (qkv_late + mlp_late; `out` excluded) ===")
    print(f"{'model':>7} {'strength':>9} {'gap_closed':>11} {'baseΔ_toTurbo':>14}")
    for tag in ("base", "light"):
        acc = [0.0, 0.0, 0.0, 0.0, 0.0]
        for fam in ("qkv", "mlp"):
            p = pool.get((tag, f"late:{fam}"))
            if p:
                acc = [a + b for a, b in zip(acc, p, strict=True)]
        s = acc[0] / acc[1] if acc[1] > 0 else float("nan")
        closed = (1 - acc[3] / acc[2]) * 100 if acc[2] > 0 else 100.0
        bd = (acc[2] ** 0.5 / acc[4] ** 0.5 * 100) if acc[4] > 0 else 0.0
        print(f"{tag:>7} {s:>9.3f} {closed:>10.1f}% {bd:>13.3f}%")

    print(
        "\nlayman: 'strength' is the DPO merge slider (Turbo uses 1.0). s>1 = push DPO harder,\n"
        "s<1 = ease off, s<0 = rotate the opposite way. gap_closed = how much of the visual\n"
        "difference-from-Turbo that strength can actually remove; if it's low, the leftover is\n"
        "the distillation itself (few-step behaviour), which no strength setting can turn into Turbo."
    )


if __name__ == "__main__":
    main()
