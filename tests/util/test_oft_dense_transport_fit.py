"""Tests for oft_dense_fit.dense_transport_fit -- recovering the DENSE two-sided
rotation W_v ~= R_L W_b R_R.

The fit must be robust where naive SVD column alignment is not: clustered /
degenerate singular values (real weight matrices have clusters of small
singular values, and per-column U_v U_b^T alignment picks arbitrary bases
inside a cluster). The alternating dense Procrustes refinement optimizes the
fit residual directly, so these tests pin: exact recovery on generic dense
transports (square, tall, wide), exact FIT on degenerate-spectrum transports,
identity-biased completion of the unidentified subspace on non-square tensors
(measured through the transported-education error, which is what the warm
start actually consumes), and monotonic residual decrease.
"""

from modules.util.oft_dense_fit import dense_transport_fit

import torch

import pytest


def _dense_rotation(n: int, angle: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    skew = torch.randn(n, n, generator=gen, dtype=torch.float64)
    skew = skew - skew.T
    skew = skew * (angle / torch.linalg.matrix_norm(skew, ord=2))
    return torch.matrix_exp(skew)


def _gram_dev(r: torch.Tensor) -> float:
    eye = torch.eye(r.shape[-1], dtype=r.dtype)
    return float((r @ r.T - eye).norm())


@pytest.mark.parametrize("shape", [(12, 12), (18, 8), (8, 18)])
def test_exact_dense_transport_recovery(shape):
    out_f, in_f = shape
    gen = torch.Generator().manual_seed(10)
    w_base = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    r_left = _dense_rotation(out_f, 0.15, seed=11)
    r_right = _dense_rotation(in_f, 0.15, seed=12)
    w_variant = r_left @ w_base @ r_right

    fit = dense_transport_fit(w_base, w_variant, iters=300, tol=0.0)
    assert fit.rel_residual < 1e-8
    assert _gram_dev(fit.r_left) < 1e-8
    assert _gram_dev(fit.r_right) < 1e-8

    # The warm start consumes R_L @ (education) @ R_R. On the identified
    # subspaces the fitted rotations must match the truth; on the unidentified
    # null space (non-square case) the completion is identity, which can differ
    # from the true rotation by at most the transport angle -- so the
    # transported education is correct to within that angle.
    gen2 = torch.Generator().manual_seed(13)
    education = 1e-2 * torch.randn(out_f, in_f, generator=gen2, dtype=torch.float64)
    moved = fit.r_left @ education @ fit.r_right
    truth = r_left @ education @ r_right
    rel = float((moved - truth).norm() / education.norm())
    if out_f == in_f:
        assert rel < 1e-8  # fully identified: exact
    else:
        assert rel < 2.5 * 0.15


def test_clustered_spectrum_fit_is_exact():
    """Exactly repeated singular values: (R_L, R_R) are not unique (gauge
    freedom inside each cluster), but the FIT must still reach machine zero --
    this is precisely where naive column-wise SVD alignment breaks."""
    gen = torch.Generator().manual_seed(20)
    u = torch.linalg.qr(torch.randn(12, 12, generator=gen, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(12, 12, generator=gen, dtype=torch.float64))[0]
    s = torch.tensor([5.0] * 4 + [1.0] * 4 + [0.01] * 4, dtype=torch.float64)
    w_base = (u * s) @ v.T
    r_left = _dense_rotation(12, 0.2, seed=21)
    r_right = _dense_rotation(12, 0.2, seed=22)
    w_variant = r_left @ w_base @ r_right

    fit = dense_transport_fit(w_base, w_variant, iters=500, tol=0.0)
    assert fit.rel_residual < 1e-7
    assert _gram_dev(fit.r_left) < 1e-8
    assert _gram_dev(fit.r_right) < 1e-8


def test_near_clustered_spectrum_recovers_education_transport():
    """Near-degenerate (1e-6-split) clusters: the transport is identifiable but
    ill-conditioned for naive alignment. The alternating fit plus identity-
    biased near-ties must still transport a small education correctly to
    within the education's own scale."""
    gen = torch.Generator().manual_seed(30)
    u = torch.linalg.qr(torch.randn(10, 10, generator=gen, dtype=torch.float64))[0]
    v = torch.linalg.qr(torch.randn(10, 10, generator=gen, dtype=torch.float64))[0]
    s = torch.tensor([3.0, 3.0 + 1e-6, 3.0 + 2e-6, 1.0, 1.0 + 1e-6, 0.5, 0.1, 0.1 + 1e-6, 0.02, 0.02 + 1e-6])
    w_base = (u * s.to(torch.float64)) @ v.T
    r_left = _dense_rotation(10, 0.05, seed=31)
    r_right = _dense_rotation(10, 0.05, seed=32)
    w_variant = r_left @ w_base @ r_right

    fit = dense_transport_fit(w_base, w_variant, iters=500, tol=0.0)
    assert fit.rel_residual < 1e-7

    gen2 = torch.Generator().manual_seed(33)
    education = 1e-3 * torch.randn(10, 10, generator=gen2, dtype=torch.float64)
    moved = fit.r_left @ education @ fit.r_right
    truth = r_left @ education @ r_right
    # Gauge freedom inside a 1e-6-split cluster can rotate the fitted factors
    # relative to the truth, but only where the data cannot tell the
    # difference; the transported education must stay well below the naive
    # no-transport error ||truth - education|| (~2*angle).
    naive = float((truth - education).norm() / education.norm())
    rel = float((moved - truth).norm() / education.norm())
    assert rel < 0.25 * naive


def test_residual_history_monotonic_on_noisy_transport():
    """Real transports are not exact; each half-step is optimal given the
    other side, so the recorded residual history must never increase."""
    gen = torch.Generator().manual_seed(40)
    w_base = torch.randn(14, 9, generator=gen, dtype=torch.float64)
    r_left = _dense_rotation(14, 0.1, seed=41)
    r_right = _dense_rotation(9, 0.1, seed=42)
    noise = 0.02 * torch.randn(14, 9, generator=gen, dtype=torch.float64)
    w_variant = r_left @ w_base @ r_right + noise

    fit = dense_transport_fit(w_base, w_variant, iters=50, tol=0.0)
    hist = fit.residual_history
    assert len(hist) >= 2
    for a, b in zip(hist, hist[1:], strict=False):
        assert b <= a + 1e-12
    assert 0.0 < fit.rel_residual < 1.0  # noise floor: neither exact nor useless


def test_identity_transport_returns_identity():
    gen = torch.Generator().manual_seed(50)
    w = torch.randn(8, 8, generator=gen, dtype=torch.float64)
    fit = dense_transport_fit(w, w.clone(), iters=20, tol=0.0)
    eye = torch.eye(8, dtype=torch.float64)
    assert float((fit.r_left - eye).norm()) < 1e-9
    assert float((fit.r_right - eye).norm()) < 1e-9
    assert fit.rel_residual == 0.0
