"""Tests for oft_dense_fit.encode_rotation_to_vec -- re-encoding a fitted
block-orthogonal rotation as a stored oft_R skew vector.

The encode is the exact Cayley inverse Q = (B - I)(B + I)^{-1} (antisymmetrized
against fp roundoff), while the module DECODES with the truncated Cayley-
Neumann-5 polynomial R(Q) = I + 2Q + 2Q^2 + 2Q^3 + Q^4. At warm-start angles
the truncation gap is ~||Q||^4, which the encoder must measure and report.
Blocks whose generator exceeds the spectral clip have the clip baked in on
reload -- the encoder must count and report them, not hide them.
"""

from modules.util.oft_align import AdapterOFTConfig, reconstruct_oft_rotation
from modules.util.oft_conjugate import clipped_skew_from_vec
from modules.util.oft_dense_fit import encode_rotation_to_vec, neumann5_rotation

import torch


def _cfg(clip=0.99, scaled=False):
    return AdapterOFTConfig(block_size=6, clip=clip, scaled=scaled)


def _small_vec(r, n_elements, theta, seed):
    gen = torch.Generator().manual_seed(seed)
    return theta * torch.randn(r, n_elements, generator=gen, dtype=torch.float64)


def test_roundtrip_through_module_reconstruction():
    """vec -> (module decode) B -> encode -> vec' -> (module decode) B' must
    close: B' ~= B to well below any training-relevant scale."""
    cfg = _cfg()
    vec = _small_vec(4, 15, 1e-2, seed=0)  # b=6 blocks, theta ~ trained scale
    q, _ = clipped_skew_from_vec(vec, cfg)
    b_target = neumann5_rotation(q)

    enc = encode_rotation_to_vec(b_target, cfg)
    assert enc.clipped_blocks == 0
    b_back = reconstruct_oft_rotation(enc.vec, cfg)
    assert float((b_back - b_target).norm()) < 1e-9

    # the reported truncation gap is the actual ||N5(Q') - B|| discrepancy
    assert 0.0 <= enc.neumann_dev < 1e-9
    # at these angles the vector itself round-trips to ~||Q||^4 relative error
    assert float((enc.vec - vec).norm() / vec.norm()) < 1e-4


def test_neumann5_matches_reference_reconstruction():
    """neumann5_rotation must be bit-faithful to oft_align's Neumann path
    (same polynomial, same coefficients) when fed the same clipped skew."""
    cfg = _cfg()
    vec = _small_vec(3, 15, 5e-3, seed=1)
    q, _ = clipped_skew_from_vec(vec, cfg)
    assert torch.allclose(neumann5_rotation(q), reconstruct_oft_rotation(vec, cfg), atol=1e-14)


def test_encode_output_is_strict_upper_triangle_encoding():
    cfg = _cfg()
    vec = _small_vec(2, 15, 1e-2, seed=2)
    q, _ = clipped_skew_from_vec(vec, cfg)
    enc = encode_rotation_to_vec(neumann5_rotation(q), cfg)
    assert enc.vec.shape == vec.shape
    # re-scatter and check skewness of the implied generator
    q2, _ = clipped_skew_from_vec(enc.vec, cfg)
    assert torch.allclose(q2, -q2.transpose(-2, -1), atol=1e-12)


def test_scaled_config_roundtrip():
    cfg = _cfg(scaled=True)
    vec = _small_vec(4, 15, 1e-2, seed=3)
    q, _ = clipped_skew_from_vec(vec, cfg)
    b_target = neumann5_rotation(q)
    enc = encode_rotation_to_vec(b_target, cfg)
    b_back = reconstruct_oft_rotation(enc.vec, cfg)
    assert float((b_back - b_target).norm()) < 1e-9


def test_overclip_block_is_reported_and_bakes_clip():
    """A block rotated past the clip angle (sigma(Q) > clip) cannot be stored
    faithfully: the module's reload clips it. The encoder must report it, and
    reloading the written vec must land on the CLIPPED rotation."""
    gen = torch.Generator().manual_seed(4)
    skew = torch.randn(1, 6, 6, generator=gen, dtype=torch.float64)
    skew = skew - skew.transpose(-2, -1)
    skew = skew * (2.0 / torch.linalg.matrix_norm(skew, ord=2).reshape(1, 1, 1))
    b_big = torch.matrix_exp(skew)  # rotation angle 2 rad >> 2*atan(0.99)

    cfg = _cfg()
    enc = encode_rotation_to_vec(b_big, cfg)
    assert enc.clipped_blocks == 1
    assert enc.sigma_max > 0.99

    b_back = reconstruct_oft_rotation(enc.vec, cfg)
    assert float((b_back - b_big).norm()) > 1e-2  # clip baked in: NOT faithful
    # and the reload matches what the module will actually produce: N5 of the
    # clipped generator
    q_clipped, _ = clipped_skew_from_vec(enc.vec, cfg)
    assert torch.allclose(b_back, neumann5_rotation(q_clipped), atol=1e-12)


def test_no_clip_config():
    cfg = _cfg(clip=None)
    vec = _small_vec(2, 15, 1e-2, seed=5)
    q, _ = clipped_skew_from_vec(vec, cfg)
    enc = encode_rotation_to_vec(neumann5_rotation(q), cfg)
    assert enc.clipped_blocks == 0
    b_back = reconstruct_oft_rotation(enc.vec, cfg)
    assert float((b_back - neumann5_rotation(q)).norm()) < 1e-9
