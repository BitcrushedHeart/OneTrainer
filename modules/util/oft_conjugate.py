"""Conjugate a Base-trained OFT adapter's skew generators into a variant's frame.

The SoReal! distillation delta on the late qkv/mlp tensors is ~99.95% a
two-sided rotation, ``W_v ~= blockdiag(R_L) @ W_b @ blockdiag(R_R)``
(scripts/util/_distill_rotation_diag.py / _distill_rotation_blockfit.py). An
OFT adapter trained on Base encodes its education as a rotation in *Base's*
input frame; transplanting the stored skew vector verbatim applies that
rotation in the variant's (rotated) frame. The frame-correct adapter is the
conjugate of the generator by the input-side transport:

    Q_v = R_R^T Q_b R_R        (per input-side block)

This is exact in the parameterization: the module's Cayley-Neumann-5 map
``R(Q) = I + 2Q + 2Q^2 + 2Q^3 + Q^4`` is a polynomial in Q, and orthogonal
conjugation commutes with polynomials, so ``R(Q_v) = R_R^T R(Q_b) R_R`` holds
term by term -- no Cayley inverse, and the Neumann truncation cancels. With the
bake convention ``M = W @ blockdiag(R(Q))^T`` (OFTModule.apply_to_module) the
merged variant then satisfies ``M_v = R_L M_b R_R``: the education transports
exactly, and R_L cancels out of the adapter entirely. Conjugation preserves
each block's spectrum, so the 0.99 spectral clip behaves identically before
and after.

DoRA multipliers do NOT conjugate: they are per-output-row and a nontrivial
R_L mixes rows within a block. ``transport_dora_multiplier`` offers the two
sanctioned policies (verbatim copy when R_L ~= I, or the retarget-style scalar
map ``m -> 1 + s(m - 1)``); the activation-matching polish absorbs whatever a
policy cannot express.

Pure tensor math -- no file I/O. Callers on this Windows box must obey the MKL
discipline (CLAUDE.md): materialize every safetensors tensor and close the
files BEFORE calling anything here, and set ``torch.set_num_threads(1)`` in
scripts before batched linalg.
"""

from __future__ import annotations

from dataclasses import dataclass

from modules.util.oft_align import AdapterOFTConfig, block_size_from_elements

import torch

# Preferred left-block widths, largest first. The left partition only shapes the
# alternating solve and the DoRA diagnostic (R_L cancels out of the conjugation),
# so any divisor works; larger blocks explain more of the delta.
DEFAULT_LEFT_BLOCK_CANDIDATES = (512, 480, 256, 128, 64, 32, 16, 8, 4)


@dataclass
class BlockRotationFit:
    """Accumulated per-block rotations fitting W_v ~= blockdiag(r_left) @ W_b @ blockdiag(r_right)."""

    r_left: torch.Tensor  # (out // block_left, block_left, block_left)
    r_right: torch.Tensor  # (in // block_right, block_right, block_right)
    rel_residual: float  # ||fit - W_v||_F / ||W_v - W_b||_F -- fraction of the delta unexplained
    left_identity_dev: float  # max_i ||r_left[i] - I||_F / sqrt(2*b); ~1 for an unrelated rotation
    right_identity_dev: float


def pick_divisor_block(
    dim: int,
    preferred: int,
    candidates: tuple[int, ...] = DEFAULT_LEFT_BLOCK_CANDIDATES,
) -> int:
    """``preferred`` if it divides ``dim``, else the largest dividing candidate."""
    if preferred > 0 and dim % preferred == 0:
        return preferred
    for c in candidates:
        if 0 < c <= dim and dim % c == 0:
            return c
    raise ValueError(f"no block width in {candidates} divides dim {dim} (preferred {preferred})")


def apply_block_rotations(
    w: torch.Tensor,
    r_left: torch.Tensor | None = None,
    r_right: torch.Tensor | None = None,
) -> torch.Tensor:
    """``blockdiag(r_left) @ w @ blockdiag(r_right)`` without materializing the block diagonals."""
    out_features, in_features = w.shape
    result = w
    if r_left is not None:
        bl = r_left.shape[-1]
        blocks = result.reshape(out_features // bl, bl, in_features)
        result = torch.einsum("ipq,iqf->ipf", r_left, blocks).reshape(out_features, in_features)
    if r_right is not None:
        br = r_right.shape[-1]
        blocks = result.reshape(out_features, in_features // br, br)
        result = torch.einsum("ojb,jbc->ojc", blocks, r_right).reshape(out_features, in_features)
    return result


def _max_identity_dev(r_blocks: torch.Tensor) -> float:
    b = r_blocks.shape[-1]
    eye = torch.eye(b, dtype=r_blocks.dtype, device=r_blocks.device)
    dev = (r_blocks - eye).flatten(1).norm(dim=1) / (2.0 * b) ** 0.5
    return float(dev.max())


def rotation_agreement(ra: torch.Tensor, rb: torch.Tensor) -> float:
    """max_i ||ra[i] @ rb[i]^T - I||_F / sqrt(2b): 0 for equal rotations, ~1 for unrelated."""
    return _max_identity_dev(torch.bmm(ra, rb.transpose(-2, -1)))


def blockwise_procrustes_rotations(
    w_base: torch.Tensor,
    w_variant: torch.Tensor,
    block_right: int,
    block_left: int,
    iters: int = 100,
    tol: float = 1e-9,
) -> BlockRotationFit:
    """Alternating blockwise Procrustes fit of the two-sided transport.

    Lifted from scripts/util/_distill_rotation_blockfit.py's blockwise_left /
    blockwise_right, which apply the optimal per-block rotation and discard it;
    here each sweep's rotation is accumulated so the total per-block R_L / R_R
    stacks come back to the caller. Each half-step is exactly optimal given the
    other side, so the residual decreases monotonically -- linearly at ~0.6 per
    sweep on near-identity transports, reaching machine precision within ~50-200
    sweeps on exact transports. ``iters`` caps the sweeps; the loop exits early
    once a sweep improves the residual by less than ``tol`` of the delta norm
    (on real data the fit plateaus at the non-rotational noise floor and exits
    long before the cap).

    ``block_right`` must be the ADAPTER'S actual partition for this tensor
    (block_size_from_elements of the oft_R row width), never the requested
    ``ot_config`` value.
    """
    if w_base.shape != w_variant.shape:
        raise ValueError(f"shape mismatch: base {tuple(w_base.shape)} vs variant {tuple(w_variant.shape)}")
    out_features, in_features = w_base.shape
    if out_features % block_left:
        raise ValueError(f"block_left {block_left} does not divide out_features {out_features}")
    if in_features % block_right:
        raise ValueError(f"block_right {block_right} does not divide in_features {in_features}")

    n_left = out_features // block_left
    n_right = in_features // block_right
    dtype, device = w_base.dtype, w_base.device
    r_left = torch.eye(block_left, dtype=dtype, device=device).repeat(n_left, 1, 1)
    r_right = torch.eye(block_right, dtype=dtype, device=device).repeat(n_right, 1, 1)

    delta = float((w_variant - w_base).norm())
    fit = w_base.clone()
    resid = delta
    prev_resid = float("inf")
    for _ in range(iters):
        for i in range(n_left):
            rows = slice(i * block_left, (i + 1) * block_left)
            u, _, vh = torch.linalg.svd(w_variant[rows] @ fit[rows].T)
            step = u @ vh
            fit[rows] = step @ fit[rows]
            r_left[i] = step @ r_left[i]  # new rotation composes on the left of the accumulate
        for j in range(n_right):
            cols = slice(j * block_right, (j + 1) * block_right)
            u, _, vh = torch.linalg.svd(fit[:, cols].T @ w_variant[:, cols])
            step = u @ vh
            fit[:, cols] = fit[:, cols] @ step
            r_right[j] = r_right[j] @ step  # composes on the right
        resid = float((fit - w_variant).norm())
        if prev_resid - resid <= tol * delta:
            break
        prev_resid = resid
    return BlockRotationFit(
        r_left=r_left,
        r_right=r_right,
        rel_residual=resid / delta if delta > 0.0 else 0.0,
        left_identity_dev=_max_identity_dev(r_left),
        right_identity_dev=_max_identity_dev(r_right),
    )


def clipped_skew_from_vec(vec: torch.Tensor, cfg: AdapterOFTConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """(post-clip skew Q (r, b, b), PRE-clip per-block sigma (r,)).

    The first half of oft_align.reconstruct_oft_rotation, factored out so the
    generator itself is available: un-scale, scatter the strict upper triangle,
    antisymmetrize, exact eigvalsh spectral norm, apply the clip rescale.
    Block size derives from ``vec.shape[-1]`` (ot_config holds the
    pre-adjustment request -- see block_size_from_elements).
    """
    bs = block_size_from_elements(vec.shape[-1])
    scaling = 2.0 * (bs - 1) ** 0.5 if cfg.scaled else 1.0
    v = vec / scaling
    r = v.shape[0]
    rows, cols = torch.triu_indices(bs, bs, 1, device=v.device)
    q = torch.zeros(r, bs, bs, device=v.device, dtype=v.dtype)
    bidx = torch.arange(r, device=v.device)[:, None]
    q = q.index_put((bidx, rows, cols), v)
    q = q - q.transpose(-2, -1)

    ev = torch.linalg.eigvalsh(-torch.bmm(q, q))  # sigma^2 of skew q, ascending
    sigma = ev[:, -1].clamp_min(0).sqrt()
    if cfg.clip is not None:
        q = q * (cfg.clip / torch.clamp(sigma, min=cfg.clip)).reshape(r, 1, 1)
    return q, sigma


def conjugate_skew_vec(vec: torch.Tensor, r_right: torch.Tensor, cfg: AdapterOFTConfig) -> torch.Tensor:
    """Re-encode ``Q_v = r_right^T Q r_right`` as a stored oft_R skew vector.

    Operates on the POST-clip generator: conjugation preserves the spectrum, so
    an under-clip block round-trips exactly, while an over-clip block has its
    clip baked into the written vector (idempotent on reload: sigma lands on the
    clip, so the module's rescale multiplies by 1 -- no double shrink).
    """
    q, _ = clipped_skew_from_vec(vec, cfg)
    q_v = torch.bmm(r_right.transpose(-2, -1), torch.bmm(q, r_right))
    # Kill fp roundoff and any residual non-orthogonality of the fitted r_right.
    q_v = 0.5 * (q_v - q_v.transpose(-2, -1))
    bs = q_v.shape[-1]
    rows, cols = torch.triu_indices(bs, bs, 1, device=q_v.device)
    scaling = 2.0 * (bs - 1) ** 0.5 if cfg.scaled else 1.0
    return q_v[:, rows, cols] * scaling


def transport_dora_multiplier(mult: torch.Tensor, policy: str, s: float | None = None) -> torch.Tensor:
    """Transport a per-output-row DoRA multiplier under the named policy.

    ``copy``: verbatim -- correct when R_L ~= I (check left_identity_dev).
    ``scale``: the retarget-style family map ``m -> 1 + s(m - 1)``.
    """
    if policy == "copy":
        return mult.clone()
    if policy == "scale":
        if s is None:
            raise ValueError("transport_dora_multiplier(policy='scale') requires s")
        return 1.0 + s * (mult - 1.0)
    raise ValueError(f"unknown dora transport policy {policy!r} (expected 'copy' or 'scale')")
