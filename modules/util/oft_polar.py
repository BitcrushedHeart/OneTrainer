"""Batched orthogonal polar factors via Newton-Schulz iteration.

The orthogonal Procrustes solution ``argmax_{R orthogonal} tr(R^T M)`` is the
orthogonal polar factor of ``M``. This repo needs it BATCHED over stacks of
adapter blocks, but batched LAPACK (svd/eigh) segfaults on CPU under Windows
MKL with default threading (see scripts/util/_oft_adapter_retarget.py and
CLAUDE.md). Newton-Schulz is pure batched matmul -- safe everywhere and fast
on GPU:

    X_0     = M / s              (s >= sigma_max(M), so all sigma <= 1)
    X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k

which drives every singular value to 1 while preserving the singular vectors,
converging to ``U V^T`` of ``M = U S V^T`` -- provided sigma_min > 0. On
rank-deficient input the zero singular values stay 0 (the limit is a partial
isometry, not orthogonal), so results are validated by their Gram deviation
and failed matrices fall back to one NON-batched SVD each, which is stable on
CPU and returns LAPACK's orthogonal completion.

Keep TF32 disabled when calling this on CUDA for fitting work: the ~1e-3
TF32 matmul error is not below the signal these fits chase.
"""

from __future__ import annotations

import torch


def gram_deviation(x: torch.Tensor) -> torch.Tensor:
    """Per-matrix ||X X^T - I||_F / sqrt(n) for a (..., n, n) stack."""
    n = x.shape[-1]
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    return (x @ x.transpose(-2, -1) - eye).flatten(start_dim=-2).norm(dim=-1) / n**0.5


def newton_schulz_polar(m: torch.Tensor, iters: int = 60, tol: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched orthogonal polar factor of a (..., n, n) stack, matmul only.

    Returns ``(X, err)`` where ``err`` is the per-matrix Gram deviation
    (``gram_deviation``); callers decide what deviation they tolerate.
    Normalization uses ``sqrt(||M||_1 ||M||_inf) >= sigma_max`` so the
    iteration starts inside its convergence region; the loop exits early once
    every matrix in the stack is within ``tol``.
    """
    if m.shape[-1] != m.shape[-2]:
        raise ValueError(f"newton_schulz_polar needs square matrices, got {tuple(m.shape)}")
    abs_m = m.abs()
    norm_1 = abs_m.sum(dim=-2).amax(dim=-1)
    norm_inf = abs_m.sum(dim=-1).amax(dim=-1)
    scale = (norm_1 * norm_inf).sqrt().clamp_min(torch.finfo(m.dtype).tiny)
    x = m / scale[..., None, None]
    for _ in range(iters):
        x = 1.5 * x - 0.5 * (x @ x.transpose(-2, -1) @ x)
        if float(gram_deviation(x).max()) <= tol:
            break
    return x, gram_deviation(x)


def polar_orthogonal(
    m: torch.Tensor,
    iters: int = 60,
    tol: float = 1e-12,
    fallback_tol: float = 1e-5,
) -> torch.Tensor:
    """Batched polar with a per-matrix single-SVD fallback where NS fails.

    Matrices whose Newton-Schulz result deviates from orthogonality by more
    than ``fallback_tol`` (rank deficiency, extreme conditioning) are redone
    one at a time with ``torch.linalg.svd`` -- non-batched LAPACK, which does
    not hit the Windows-MKL batched-linalg segfault.
    """
    x, err = newton_schulz_polar(m, iters=iters, tol=tol)
    bad = err > fallback_tol
    if bool(bad.any()):
        flat_m = m.reshape(-1, m.shape[-2], m.shape[-1])
        flat_x = x.reshape(-1, m.shape[-2], m.shape[-1])
        for i in torch.nonzero(bad.reshape(-1)).flatten().tolist():
            u, _, vh = torch.linalg.svd(flat_m[i])
            flat_x[i] = u @ vh
        x = flat_x.reshape(m.shape)
    return x
