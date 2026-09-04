"""Math tests for modules/util/oft_conjugate.py.

The conjugation identity under test: with the bake convention
``M = W @ blockdiag(R_oft)^T`` and a two-sided block-diagonal transport
``W_v = blockdiag(R_L) @ W_b @ blockdiag(R_R)``, the variant-frame adapter is
``Q_v = R_R^T Q_b R_R`` (per input-side block). Cayley-Neumann-5 is a
polynomial in Q and orthogonal conjugation commutes with polynomials, so this
is exact in the parameterization regardless of the Neumann truncation --
``M_v = R_L M_b R_R`` holds for ANY orthogonal R_R satisfying the transport,
which makes the merged-equation assert immune to per-block fit ambiguity.

Skew vectors are built at theta ~ 1e-2 so the O(theta^5) Neumann truncation
drift (~1e-10) sits far below every fp32 tolerance here.
"""

from modules.util.oft_align import (
    AdapterOFTConfig,
    merged_full,
    reconstruct_oft_rotation,
)
from modules.util.oft_conjugate import (
    apply_block_rotations,
    blockwise_procrustes_rotations,
    clipped_skew_from_vec,
    conjugate_skew_vec,
    pick_divisor_block,
    rotation_agreement,
    transport_dora_multiplier,
)

import torch

import pytest

OUT, IN = 12, 16
BLOCK_RIGHT = 8  # adapter partition: r = IN // BLOCK_RIGHT = 2 blocks
BLOCK_LEFT = 4  # R_L partition: OUT // BLOCK_LEFT = 3 blocks
R_BLOCKS = IN // BLOCK_RIGHT
N_ELEMENTS = BLOCK_RIGHT * (BLOCK_RIGHT - 1) // 2

CFG = AdapterOFTConfig(block_size=BLOCK_RIGHT, clip=0.99, scaled=False)


def _skew_vec(seed: int, scale: float = 1e-2, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(R_BLOCKS, N_ELEMENTS, generator=gen, dtype=dtype) * scale


def _weights(seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(OUT, IN, generator=gen, dtype=dtype)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.float64, 1e-10, 1e-12),
    ],
)
def test_conjugation_identity_exact_with_ground_truth(make_orthogonal_blocks, dtype, rtol, atol):
    """With known transport rotations the merged equation is machine-exact:
    M_v = R_L M_b R_R, independent of any fitting. This is the core claim --
    the Neumann-5 polynomial commutes with orthogonal conjugation term by term."""
    w_base = _weights(0, dtype)
    r_left = make_orthogonal_blocks(OUT // BLOCK_LEFT, BLOCK_LEFT, seed=1, angle=0.3, dtype=dtype)
    r_right = make_orthogonal_blocks(R_BLOCKS, BLOCK_RIGHT, seed=2, angle=0.3, dtype=dtype)
    w_variant = apply_block_rotations(w_base, r_left=r_left, r_right=r_right)

    vec = _skew_vec(3, dtype=dtype)
    vec_v = conjugate_skew_vec(vec, r_right, CFG)

    merged_variant = merged_full(w_variant, reconstruct_oft_rotation(vec_v, CFG))
    merged_base = merged_full(w_base, reconstruct_oft_rotation(vec, CFG))
    expected = apply_block_rotations(merged_base, r_left=r_left, r_right=r_right)
    torch.testing.assert_close(merged_variant, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "resid_max", "rtol", "atol"),
    [
        (torch.float32, 1e-4, 1e-3, 2e-4),
        (torch.float64, 1e-10, 1e-7, 1e-9),
    ],
)
@pytest.mark.parametrize("angle", [0.05, 0.3])
def test_procrustes_recovers_exact_transport(make_orthogonal_blocks, dtype, resid_max, rtol, atol, angle):
    """Exact transport: the solver drives the residual to (near) machine floor,
    and the merged equation holds in the FITTED rotations to fit precision.
    tol=0 disables the early exit so the full 200 sweeps run."""
    w_base = _weights(0, dtype)
    r_left = make_orthogonal_blocks(OUT // BLOCK_LEFT, BLOCK_LEFT, seed=1, angle=angle, dtype=dtype)
    r_right = make_orthogonal_blocks(R_BLOCKS, BLOCK_RIGHT, seed=2, angle=angle, dtype=dtype)
    w_variant = apply_block_rotations(w_base, r_left=r_left, r_right=r_right)

    fit = blockwise_procrustes_rotations(w_base, w_variant, BLOCK_RIGHT, BLOCK_LEFT, iters=200, tol=0.0)
    assert fit.rel_residual < resid_max

    vec = _skew_vec(3, dtype=dtype)
    vec_v = conjugate_skew_vec(vec, fit.r_right, CFG)

    merged_variant = merged_full(w_variant, reconstruct_oft_rotation(vec_v, CFG))
    merged_base = merged_full(w_base, reconstruct_oft_rotation(vec, CFG))
    expected = apply_block_rotations(merged_base, r_left=fit.r_left, r_right=fit.r_right)
    torch.testing.assert_close(merged_variant, expected, rtol=rtol, atol=atol)


def test_conjugated_vec_is_skew_and_sigma_preserved(make_orthogonal_blocks):
    """Re-extracted Q_v is skew and per-block sigma is unchanged by conjugation."""
    vec = _skew_vec(4)
    r_right = make_orthogonal_blocks(R_BLOCKS, BLOCK_RIGHT, seed=5, angle=0.3)
    vec_v = conjugate_skew_vec(vec, r_right, CFG)

    q_v, sigma_v = clipped_skew_from_vec(vec_v, CFG)
    _, sigma = clipped_skew_from_vec(vec, CFG)

    assert float((q_v + q_v.transpose(-2, -1)).abs().max()) < 1e-6
    torch.testing.assert_close(sigma_v, sigma, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("scaled", [False, True])
def test_roundtrip_under_clip_identity_conjugation(scaled):
    """Identity R_R: the stored vector round-trips (scatter/extract is exact)."""
    cfg = AdapterOFTConfig(block_size=BLOCK_RIGHT, clip=0.99, scaled=scaled)
    vec = _skew_vec(6)  # sigma << clip
    identity = torch.eye(BLOCK_RIGHT).repeat(R_BLOCKS, 1, 1)
    vec_v = conjugate_skew_vec(vec, identity, cfg)
    torch.testing.assert_close(vec_v, vec, rtol=1e-6, atol=1e-7)


def test_clip_path_bakes_clip_idempotently(make_orthogonal_blocks):
    """Over-clip blocks: conjugation writes the clipped generator; sigma lands on
    the clip and a second identity conjugation is a no-op (no double shrink)."""
    vec = _skew_vec(7, scale=1.0)  # sigma >> clip for b=8
    _, sigma_raw = clipped_skew_from_vec(vec, AdapterOFTConfig(BLOCK_RIGHT, None, False))
    assert float(sigma_raw.min()) > CFG.clip  # test premise: genuinely over the clip

    r_right = make_orthogonal_blocks(R_BLOCKS, BLOCK_RIGHT, seed=8, angle=0.3)
    vec_v = conjugate_skew_vec(vec, r_right, CFG)

    _, sigma_v = clipped_skew_from_vec(vec_v, AdapterOFTConfig(BLOCK_RIGHT, None, False))
    torch.testing.assert_close(sigma_v, torch.full_like(sigma_v, CFG.clip), rtol=1e-5, atol=1e-6)

    identity = torch.eye(BLOCK_RIGHT).repeat(R_BLOCKS, 1, 1)
    vec_again = conjugate_skew_vec(vec_v, identity, CFG)
    torch.testing.assert_close(vec_again, vec_v, rtol=1e-5, atol=1e-6)


def test_identity_transport_is_noop():
    """W_v == W_b: recovered rotations are identity, adapter passes through."""
    w_base = _weights(9)
    fit = blockwise_procrustes_rotations(w_base, w_base.clone(), BLOCK_RIGHT, BLOCK_LEFT, iters=12)
    assert fit.right_identity_dev < 1e-5
    assert fit.left_identity_dev < 1e-5

    vec = _skew_vec(10)
    vec_v = conjugate_skew_vec(vec, fit.r_right, CFG)
    torch.testing.assert_close(vec_v, vec, rtol=1e-5, atol=1e-6)


def test_noise_degrades_gracefully(make_orthogonal_blocks):
    """Transport + noise: residual grows monotonically with noise, no blowup."""
    w_base = _weights(11)
    r_left = make_orthogonal_blocks(OUT // BLOCK_LEFT, BLOCK_LEFT, seed=12, angle=0.1)
    r_right = make_orthogonal_blocks(R_BLOCKS, BLOCK_RIGHT, seed=13, angle=0.1)
    transported = apply_block_rotations(w_base, r_left=r_left, r_right=r_right)
    delta_scale = float((transported - w_base).norm())

    gen = torch.Generator().manual_seed(14)
    noise = torch.randn(OUT, IN, generator=gen)
    residuals = []
    for eps in (0.0, 1e-3, 1e-1):
        w_variant = transported + eps * delta_scale * noise / float(noise.norm())
        fit = blockwise_procrustes_rotations(w_base, w_variant, BLOCK_RIGHT, BLOCK_LEFT, iters=200)
        residuals.append(fit.rel_residual)

    assert residuals[0] < 1e-4
    assert residuals[0] <= residuals[1] <= residuals[2]
    assert residuals[2] < 1.0  # noise is a fraction of the delta; most stays explained


def test_dense_transport_degrades_toward_transplant():
    """The regime the real lineup occupies: a DENSE (cross-block) transport.
    The block-diagonal fit explains little of it, and the conjugation collapses
    toward the transplant (a small structured nudge) rather than the >3x win of
    block-expressible transports -- asserted here so the synthetic-transport
    tests above cannot give false confidence about real checkpoints."""
    w_base = _weights(20)
    gen = torch.Generator().manual_seed(21)
    skew = torch.randn(IN, IN, generator=gen, dtype=torch.float64)
    skew = skew - skew.T
    skew = skew * (0.1 / torch.linalg.matrix_norm(skew, ord=2))
    r_dense = torch.matrix_exp(skew).to(torch.float32)
    w_variant = w_base @ r_dense  # right-side dense rotation, no block structure

    fit = blockwise_procrustes_rotations(w_base, w_variant, BLOCK_RIGHT, BLOCK_LEFT, iters=100)
    assert fit.rel_residual > 0.5  # mostly NOT block-expressible

    vec = _skew_vec(22)
    vec_v = conjugate_skew_vec(vec, fit.r_right, CFG)
    rel_change = float((vec_v - vec).norm() / vec.norm())
    assert rel_change < 0.2  # conj ~= transplant + small nudge

    # Against the TRUE frame-transported target, conjugation must not be worse
    # than the transplant (and is not expected to be dramatically better).
    target = merged_full(w_base, reconstruct_oft_rotation(vec, CFG)) @ r_dense
    conj_err = float((merged_full(w_variant, reconstruct_oft_rotation(vec_v, CFG)) - target).norm())
    transplant_err = float((merged_full(w_variant, reconstruct_oft_rotation(vec, CFG)) - target).norm())
    assert conj_err <= transplant_err * 1.05


def test_dora_transport_policies():
    mult = torch.tensor([0.7, 1.0, 1.3])
    torch.testing.assert_close(transport_dora_multiplier(mult, "copy"), mult)
    expected = 1.0 + 0.5 * (mult - 1.0)
    torch.testing.assert_close(transport_dora_multiplier(mult, "scale", s=0.5), expected)
    with pytest.raises(ValueError):
        transport_dora_multiplier(mult, "nonsense")
    with pytest.raises(ValueError):
        transport_dora_multiplier(mult, "scale")  # scale requires s


def test_pick_divisor_block_real_shapes():
    # The real lineup: adapter blocks are 480 (in=3840) and 512 (in=10240).
    assert pick_divisor_block(3840, 480) == 480
    assert pick_divisor_block(10240, 480) == 512  # 480 doesn't divide; 512 does
    assert pick_divisor_block(10240, 512) == 512
    with pytest.raises(ValueError):
        pick_divisor_block(7919, 480)  # prime: no candidate divides


def test_rotation_agreement(make_orthogonal_blocks):
    ra = make_orthogonal_blocks(R_BLOCKS, BLOCK_RIGHT, seed=15, angle=0.2)
    rb = make_orthogonal_blocks(R_BLOCKS, BLOCK_RIGHT, seed=16)
    assert rotation_agreement(ra, ra) < 1e-6
    assert rotation_agreement(ra, rb) > 0.3
