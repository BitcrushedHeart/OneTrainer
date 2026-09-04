"""Tests for modules/util/oft_polar.py -- batched Newton-Schulz polar iteration.

The polar factor is the solution of the orthogonal Procrustes problem, and the
repo needs it batched WITHOUT batched LAPACK (Windows-MKL segfault constraint):
Newton-Schulz is pure matmul. These tests pin the iteration against the SVD
polar ground truth, check the orthogonality guard, and check the per-matrix
SVD fallback on matrices Newton-Schulz cannot handle (rank deficiency).
"""

from modules.util.oft_polar import newton_schulz_polar, polar_orthogonal

import torch


def _svd_polar(m: torch.Tensor) -> torch.Tensor:
    """Ground-truth polar via one non-batched SVD per matrix."""
    out = torch.empty_like(m)
    for i in range(m.shape[0]):
        u, _, vh = torch.linalg.svd(m[i])
        out[i] = u @ vh
    return out


def test_newton_schulz_matches_svd_polar():
    gen = torch.Generator().manual_seed(0)
    m = torch.randn(6, 8, 8, generator=gen, dtype=torch.float64)
    x, err = newton_schulz_polar(m, iters=100)
    assert torch.allclose(x, _svd_polar(m), atol=1e-9)
    assert float(err.max()) < 1e-9


def test_newton_schulz_orthogonality_and_err_report():
    gen = torch.Generator().manual_seed(1)
    m = torch.randn(4, 12, 12, generator=gen, dtype=torch.float64)
    x, err = newton_schulz_polar(m, iters=100)
    eye = torch.eye(12, dtype=torch.float64)
    gram_dev = (x @ x.transpose(-2, -1) - eye).flatten(1).norm(dim=1)
    assert float(gram_dev.max()) < 1e-9
    # err is exactly the per-matrix Gram deviation the guard uses
    assert torch.allclose(err, gram_dev / 12.0**0.5, atol=1e-12)


def test_newton_schulz_ill_conditioned_converges():
    """Singular-value spread over 6 decades: slow start, but still converges."""
    gen = torch.Generator().manual_seed(2)
    q1 = torch.linalg.qr(torch.randn(10, 10, generator=gen, dtype=torch.float64))[0]
    q2 = torch.linalg.qr(torch.randn(10, 10, generator=gen, dtype=torch.float64))[0]
    s = torch.logspace(0, -6, 10, dtype=torch.float64)
    m = (q1 * s) @ q2.T
    x, err = newton_schulz_polar(m.unsqueeze(0), iters=200)
    assert float(err.max()) < 1e-8
    assert torch.allclose(x[0], q1 @ q2.T, atol=1e-7)


def test_rank_deficient_falls_back_to_svd():
    """NS converges to a partial isometry on singular input -- the fallback
    must detect the failed orthogonality and hand those matrices to per-matrix
    SVD, which always returns a (completed) orthogonal factor."""
    gen = torch.Generator().manual_seed(3)
    good = torch.randn(5, 5, generator=gen, dtype=torch.float64)
    u = torch.randn(5, 1, generator=gen, dtype=torch.float64)
    v = torch.randn(5, 1, generator=gen, dtype=torch.float64)
    bad = u @ v.T  # rank 1
    m = torch.stack([good, bad])

    # 40 iterations: enough for the good matrix, but before fp roundoff noise
    # (amplified 1.5x per sweep in the zero-sigma subspace) fills the rank gap
    # with an arbitrary noise-seeded completion.
    _, err = newton_schulz_polar(m, iters=40)
    assert float(err[0]) < 1e-9
    assert float(err[1]) > 1e-2  # NS alone cannot orthogonalize rank-1

    x = polar_orthogonal(m, iters=40)
    eye = torch.eye(5, dtype=torch.float64)
    gram_dev = (x @ x.transpose(-2, -1) - eye).flatten(1).norm(dim=1)
    assert float(gram_dev.max()) < 1e-9
    assert torch.allclose(x[0], _svd_polar(good.unsqueeze(0))[0], atol=1e-9)


def test_polar_solves_procrustes_2x2_brute_force():
    """polar(M) maximizes tr(R^T M) over ALL orthogonal R. For 2x2 the
    orthogonal group is one angle (+ reflection branch), so a dense grid is an
    exact brute force."""
    gen = torch.Generator().manual_seed(4)
    m = torch.randn(3, 2, 2, generator=gen, dtype=torch.float64)
    x = polar_orthogonal(m, iters=100)

    theta = torch.linspace(0, 2 * torch.pi, 20000, dtype=torch.float64)
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (N, 2, 2)
    refl = rot.clone()
    refl[:, :, 1] = -refl[:, :, 1]  # flip second column: det -1 branch
    candidates = torch.cat([rot, refl])

    for i in range(m.shape[0]):
        best = (candidates * m[i]).sum(dim=(-2, -1)).max()
        got = (x[i] * m[i]).sum()
        assert got >= best - 1e-6, f"matrix {i}: polar trace {got} below brute-force {best}"


def test_float32_accuracy():
    gen = torch.Generator().manual_seed(5)
    m = torch.randn(8, 16, 16, generator=gen, dtype=torch.float32)
    x = polar_orthogonal(m, iters=100)
    ref = _svd_polar(m.to(torch.float64)).to(torch.float32)
    assert torch.allclose(x, ref, atol=5e-5)
