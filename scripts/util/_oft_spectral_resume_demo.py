"""Demo: the OFT spectral-clip resume glitch, its re-convergence curve, and the fix.

u_state/v_state (power-iteration state for the sigma estimate behind
oft_clipped_norm) are non-persistent buffers, so resuming from a backup
restarts them from random vectors. This script measures, on a synthetic
trained adapter:

  1. how wrong the clip is immediately after a simulated resume
     (estimated sigma vs true sigma, applied scale vs correct scale),
  2. how many forward passes the one-step-per-forward power iteration
     needs to bring the effective rotation back to the pre-resume one,
  3. that warm_start_spectral_state() closes the gap before the first step.

Run: venv/Scripts/python.exe scripts/util/_oft_spectral_resume_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.module.oft_utils import OFTRotationModule

import torch

torch.manual_seed(0)

R_BLOCKS = 12
BLOCK_SIZE = 320
CLIP = 0.95
CONVERGE_STEPS = 400  # forward passes used to reach the "trained" reference state
RESUME_TRACK_STEPS = [0, 1, 5, 10, 25, 50, 100, 200, 400]


def build_module() -> OFTRotationModule:
    n_elements = BLOCK_SIZE * (BLOCK_SIZE - 1) // 2
    m = OFTRotationModule(
        r=R_BLOCKS,
        n_elements=n_elements,
        block_size=BLOCK_SIZE,
        in_features=R_BLOCKS * BLOCK_SIZE,
        use_cayley_neumann=True,
        num_cayley_neumann_terms=5,
        oft_clipped_norm=CLIP,
    )
    # Synthetic "trained" adapter: per-block true sigma spread across ~0.5..2.0
    # so some blocks sit under the clip and some well over it.
    with torch.no_grad():
        w = torch.randn(R_BLOCKS, n_elements)
        skew = m._pytorch_skew_symmetric(w, BLOCK_SIZE)
        sigma = torch.linalg.svdvals(skew)[:, 0]
        targets = torch.linspace(0.5, 2.0, R_BLOCKS)
        m.weight.copy_(w * (targets / sigma).unsqueeze(-1))
    return m


def true_sigma(m: OFTRotationModule) -> torch.Tensor:
    with torch.no_grad():
        skew = m._pytorch_skew_symmetric(m.weight, BLOCK_SIZE)
        return torch.linalg.svdvals(skew)[:, 0]


def estimated_sigma(m: OFTRotationModule) -> torch.Tensor:
    """The sigma the clip actually uses, from the current u/v state (no update)."""
    with torch.no_grad():
        skew = m._pytorch_skew_symmetric(m.weight, BLOCK_SIZE)
        u = m.u_state.unsqueeze(-1)
        v = m.v_state.unsqueeze(-1)
        v_raw = torch.bmm(skew.mT, u)
        v = v_raw / torch.linalg.vector_norm(v_raw, dim=1, keepdim=True).clamp_min(1e-8)
        u_raw = torch.bmm(skew, v)
        u = u_raw / torch.linalg.vector_norm(u_raw, dim=1, keepdim=True).clamp_min(1e-8)
        return torch.sum(u * torch.bmm(skew, v), dim=1).squeeze(-1)


def rotation(m: OFTRotationModule) -> torch.Tensor:
    m.eval()  # eval: use the stored u/v without advancing them
    with torch.no_grad():
        return m._cayley_batch(m.weight, BLOCK_SIZE, m.use_cayley_neumann, m.num_cayley_neumann_terms)


def advance_one_forward(m: OFTRotationModule):
    m.train()  # train: _spectral_norm writes the updated u/v back
    with torch.no_grad():
        m._cayley_batch(m.weight, BLOCK_SIZE, m.use_cayley_neumann, m.num_cayley_neumann_terms)


def randomize_state(m: OFTRotationModule):
    """What a resume-from-backup does today: fresh random power-iteration state."""
    with torch.no_grad():
        for buf in (m.u_state, m.v_state):
            fresh = torch.randn_like(buf)
            buf.copy_(fresh / fresh.norm(dim=1, keepdim=True).clamp_min(1e-12))


def report(m, sig_true, r_ref, label):
    sig_est = estimated_sigma(m)
    clipped = sig_true > CLIP
    scale_true = (CLIP / sig_true.clamp(min=CLIP)).clamp(max=1.0)
    scale_est = (CLIP / sig_est.clamp(min=CLIP)).clamp(max=1.0)
    r_now = rotation(m)
    rel_err = (r_now - r_ref).norm() / r_ref.norm()
    unclipped = int((clipped & (scale_est > 0.999)).sum())
    print(
        f"  {label:>10}   sigma est/true (clipped blocks): "
        f"{sig_est[clipped].mean():.3f}/{sig_true[clipped].mean():.3f}   "
        f"scale est/correct: {scale_est[clipped].mean():.3f}/{scale_true[clipped].mean():.3f}   "
        f"blocks wrongly unclipped: {unclipped}/{int(clipped.sum())}   "
        f"rel ||R - R_ref||: {rel_err:.2e}"
    )
    return rel_err


def main():
    m = build_module()
    sig_true = true_sigma(m)
    print(f"{R_BLOCKS} blocks of {BLOCK_SIZE}, clip={CLIP}")
    print(f"true sigma per block: {[f'{s:.2f}' for s in sig_true.tolist()]}")
    print(f"blocks over the clip: {int((sig_true > CLIP).sum())}/{R_BLOCKS}\n")

    # Converge u/v as a long training run would, then freeze the reference rotation.
    for _ in range(CONVERGE_STEPS):
        advance_one_forward(m)
    r_ref = rotation(m)

    print("Pre-resume (converged training state):")
    report(m, sig_true, r_ref, "reference")

    print("\nSimulated resume-from-backup (random u/v), tracking forward passes:")
    randomize_state(m)
    step = 0
    for target in RESUME_TRACK_STEPS:
        while step < target:
            advance_one_forward(m)
            step += 1
        report(m, sig_true, r_ref, f"step {step}")

    print("\nSame resume, but with warm_start_spectral_state() (the fix):")
    randomize_state(m)
    m.warm_start_spectral_state()
    rel = report(m, sig_true, r_ref, "step 0")
    ok = rel < 1e-4
    print(f"\nwarm start restores the pre-resume rotation immediately: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
