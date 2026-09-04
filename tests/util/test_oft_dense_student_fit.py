"""Tests for oft_dense_fit.fit_student_als -- fitting the deployed student form

    D(B, m) = rowscale(m) * (W_v @ blockdiag(B)^T)

to a dense target T by alternating least squares. The block half-step is an
orthogonal Procrustes problem in a two-sided-scaled layout; its closed form
(B_j = polar(T_j^T A_j) with A = rowscale(m) W_v) is verified against exact
brute force (2x2 blocks: the orthogonal group is a 1-parameter grid) and
against autograd descent on larger blocks. The row half-step is closed-form
least squares. Each half-step is optimal given the other, so the objective
history must decrease monotonically, and the fit must exactly recover a target
that lies inside the parameterization.
"""

from modules.util.oft_dense_fit import fit_student_als, student_merged

import torch


def _objective(w_variant, target, r_blocks, mult) -> float:
    return float((student_merged(w_variant, r_blocks, mult) - target).norm())


def test_block_step_matches_2x2_brute_force():
    """One ALS iteration with DoRA off runs exactly one block half-step at
    m = 1. For 2x2 blocks, grid over rotation angle + reflection branch is an
    exact brute force per block (blocks are independent given m)."""
    gen = torch.Generator().manual_seed(0)
    out_f, in_f, block = 6, 8, 2
    w_variant = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    target = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)

    fit = fit_student_als(w_variant, target, block, dora=False, iters=1)
    got = _objective(w_variant, target, fit.r_blocks, fit.multiplier)

    theta = torch.linspace(0, 2 * torch.pi, 20000, dtype=torch.float64)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    refl = rot.clone()
    refl[:, :, 1] = -refl[:, :, 1]
    candidates = torch.cat([rot, refl])  # (N, 2, 2)

    best_sq = 0.0
    for j in range(in_f // block):
        cols = slice(j * block, (j + 1) * block)
        # ||W_j C^T - T_j||^2 over the candidate grid
        fits = torch.einsum("of,ncf->noc", w_variant[:, cols], candidates)
        errs = (fits - target[:, cols]).square().sum(dim=(-2, -1))
        best_sq += float(errs.min())
    assert got**2 <= best_sq + 1e-9


def test_block_step_beats_autograd_descent():
    """Larger blocks: Adam descent on a matrix_exp(skew) parameterization of
    the same objective (m fixed at 1) must not find a better orthogonal block
    than the closed-form Procrustes step."""
    gen = torch.Generator().manual_seed(1)
    out_f, in_f, block = 10, 8, 4
    w_variant = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    target = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)

    fit = fit_student_als(w_variant, target, block, dora=False, iters=1)
    closed = _objective(w_variant, target, fit.r_blocks, fit.multiplier)

    r = in_f // block
    skew_p = torch.zeros(r, block, block, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([skew_p], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        skew = skew_p - skew_p.transpose(-2, -1)
        blocks = torch.matrix_exp(skew)
        d = torch.einsum("orc,rkc->ork", w_variant.reshape(out_f, r, block), blocks).reshape(out_f, in_f)
        loss = (d - target).square().sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        skew = skew_p - skew_p.transpose(-2, -1)
        autograd_obj = float(
            (
                torch.einsum("orc,rkc->ork", w_variant.reshape(out_f, r, block), torch.matrix_exp(skew)).reshape(
                    out_f, in_f
                )
                - target
            ).norm()
        )
    # matrix_exp only reaches SO(block); the closed form searches all of O(block)
    assert closed <= autograd_obj + 1e-6


def test_exact_recovery_of_representable_target():
    gen = torch.Generator().manual_seed(2)
    out_f, in_f, block = 12, 8, 4
    w_variant = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    r = in_f // block
    skew = torch.randn(r, block, block, generator=gen, dtype=torch.float64) * 0.1
    skew = skew - skew.transpose(-2, -1)
    b_true = torch.matrix_exp(skew)
    m_true = 1.0 + 0.1 * torch.randn(out_f, generator=gen, dtype=torch.float64)
    target = student_merged(w_variant, b_true, m_true)

    fit = fit_student_als(w_variant, target, block, dora=True, iters=8)
    assert _objective(w_variant, target, fit.r_blocks, fit.multiplier) < 1e-8
    assert torch.allclose(fit.multiplier, m_true, atol=1e-6)
    assert torch.allclose(fit.r_blocks, b_true, atol=1e-6)


def test_objective_history_monotonic_on_dense_target():
    """A dense (not block-representable) target: ALS converges to a strictly
    positive floor, decreasing monotonically along the recorded history."""
    gen = torch.Generator().manual_seed(3)
    out_f, in_f, block = 10, 12, 4
    w_variant = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    dense_rot = torch.matrix_exp(
        (lambda s: (s - s.T) * 0.2)(torch.randn(in_f, in_f, generator=gen, dtype=torch.float64))
    )
    target = w_variant @ dense_rot

    fit = fit_student_als(w_variant, target, block, dora=True, iters=6)
    hist = fit.objective_history
    assert len(hist) == 6
    for a, b in zip(hist, hist[1:], strict=False):
        assert b <= a + 1e-12
    final = _objective(w_variant, target, fit.r_blocks, fit.multiplier)
    assert abs(final - hist[-1]) < 1e-9
    assert final > 0.0  # cross-block content is unreachable: a real floor


def test_dora_false_keeps_multiplier_ones():
    gen = torch.Generator().manual_seed(4)
    w_variant = torch.randn(8, 8, generator=gen, dtype=torch.float64)
    target = torch.randn(8, 8, generator=gen, dtype=torch.float64)
    fit = fit_student_als(w_variant, target, 4, dora=False, iters=3)
    assert torch.equal(fit.multiplier, torch.ones(8, dtype=torch.float64))


def test_zero_row_multiplier_defaults_to_one():
    gen = torch.Generator().manual_seed(5)
    w_variant = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    w_variant[2] = 0.0
    target = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    fit = fit_student_als(w_variant, target, 4, dora=True, iters=3)
    assert float(fit.multiplier[2]) == 1.0
    assert torch.isfinite(fit.multiplier).all()
