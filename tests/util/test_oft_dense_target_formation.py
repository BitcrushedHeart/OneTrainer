"""Tests for the education-anchored teacher target in
oft_dense_fit.dense_warmstart_for_tensor:

    T = W_v + R_L (M_b - W_b) R_R

Only the education delta rides the fitted dense transport. The old form
``T = R_L M_b R_R`` differs by exactly the transport fit's residual -- on real
lineups that residual is distillation residue (2-3% of the distill delta,
which is up to several times the education itself), so it polluted the target
with signal the student should never chase. The two forms coincide when the
transport is exact.
"""

from modules.util.oft_align import AdapterOFTConfig
from modules.util.oft_conjugate import clipped_skew_from_vec
from modules.util.oft_dense_fit import dense_warmstart_for_tensor, neumann5_rotation, student_merged

import torch

CFG = AdapterOFTConfig(block_size=4, clip=0.99, scaled=False)


def _dense_rotation(n: int, angle: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    skew = torch.randn(n, n, generator=gen, dtype=torch.float64)
    skew = skew - skew.T
    skew = skew * (angle / torch.linalg.matrix_norm(skew, ord=2))
    return torch.matrix_exp(skew)


def _adapter(
    out_f: int, in_f: int, theta: float, seed: int, mult_std: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """(oft_R vec, dora multiplier) for b=4 blocks at a trained-scale angle."""
    gen = torch.Generator().manual_seed(seed)
    r = in_f // 4
    vec = theta * torch.randn(r, 6, generator=gen, dtype=torch.float64)  # b=4 -> 6 elements
    mult = 1.0 + mult_std * torch.randn(out_f, generator=gen, dtype=torch.float64)
    return vec, mult


def _merged_base(w_base: torch.Tensor, vec: torch.Tensor, mult: torch.Tensor) -> torch.Tensor:
    q_b, _ = clipped_skew_from_vec(vec, CFG)
    return student_merged(w_base, neumann5_rotation(q_b), mult)


def test_target_carries_exactly_the_transported_education():
    """On a NOISY transport (the real regime) the target minus W_v must be the
    transported education R_L (M_b - W_b) R_R and nothing else -- no dense-fit
    residual leakage -- and its norm must equal ||M_b - W_b|| by orthogonal
    invariance (so the reported floor denominator IS the education norm)."""
    gen = torch.Generator().manual_seed(0)
    out_f, in_f = 10, 8
    w_base = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    r_left = _dense_rotation(out_f, 0.1, seed=1)
    r_right = _dense_rotation(in_f, 0.1, seed=2)
    noise = 0.05 * torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    w_variant = r_left @ w_base @ r_right + noise

    vec, mult = _adapter(out_f, in_f, theta=1e-2, seed=3)
    m_b = _merged_base(w_base, vec, mult)

    ws = dense_warmstart_for_tensor(vec, mult, w_base, w_variant, CFG)
    transported = ws.dense.r_left @ (m_b - w_base) @ ws.dense.r_right
    assert torch.allclose(ws.target - w_variant, transported, atol=1e-10)

    education_norm = float((m_b - w_base).norm())
    target_offset_norm = float((ws.target - w_variant).norm())
    assert abs(target_offset_norm - education_norm) < 1e-6 * education_norm


def test_coincides_with_old_form_on_exact_transport():
    """When the variant IS an exact dense transport of Base, the fit residual
    vanishes and T = W_v + R_L (M_b - W_b) R_R equals the old R_L M_b R_R."""
    gen = torch.Generator().manual_seed(10)
    out_f, in_f = 10, 8
    w_base = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    r_left = _dense_rotation(out_f, 0.15, seed=11)
    r_right = _dense_rotation(in_f, 0.15, seed=12)
    w_variant = r_left @ w_base @ r_right

    vec, mult = _adapter(out_f, in_f, theta=1e-2, seed=13)
    m_b = _merged_base(w_base, vec, mult)

    ws = dense_warmstart_for_tensor(vec, mult, w_base, w_variant, CFG, dense_iters=300, dense_tol=0.0)
    old_form = ws.dense.r_left @ m_b @ ws.dense.r_right
    delta = float((w_variant - w_base).norm())
    assert float((ws.target - old_form).norm()) < 1e-6 * delta


def test_noisy_transport_target_beats_old_form_against_truth():
    """With transport noise and a small education, the old form's error against
    the TRUE education-transported target is dominated by the dense-fit
    residual; the education-anchored form only pays the (much smaller)
    rotation-recovery error scaled by the education. The education is kept
    small relative to the noise (the lineup's merge/distill regime) so the
    separation is structural, not threshold tuning."""
    gen = torch.Generator().manual_seed(20)
    out_f, in_f = 10, 8
    w_base = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    r_left = _dense_rotation(out_f, 0.1, seed=21)
    r_right = _dense_rotation(in_f, 0.1, seed=22)
    delta_clean = r_left @ w_base @ r_right - w_base
    noise = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    noise = noise * (0.2 * float(delta_clean.norm()) / float(noise.norm()))
    w_variant = w_base + delta_clean + noise

    vec, mult = _adapter(out_f, in_f, theta=5e-3, seed=23, mult_std=0.01)
    m_b = _merged_base(w_base, vec, mult)
    education = m_b - w_base
    truth = w_variant + r_left @ education @ r_right  # perfect education transport onto the actual W_v

    ws = dense_warmstart_for_tensor(vec, mult, w_base, w_variant, CFG, dense_iters=300, dense_tol=0.0)
    old_form = ws.dense.r_left @ m_b @ ws.dense.r_right
    err_new = float((ws.target - truth).norm())
    err_old = float((old_form - truth).norm())
    assert err_new < 0.3 * err_old
