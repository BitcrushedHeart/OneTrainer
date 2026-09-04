"""Tests for oft_dense_fit.refine_deployed_fit -- LBFGS polish of the written
(oft_R vec, dora_multiplier) through the module's ACTUAL decode path
(scatter -> skew -> Neumann-5, spectral clip enforced by projection).

The ALS + exact-Cayley-encode pipeline optimizes an exactly-orthogonal block
rotation and then encodes it for a decoder that is NOT the exact Cayley map
(Neumann-5 truncation) -- so even a target that is exactly representable in
the DEPLOYED form retains a floor. The refinement must close that gap, must
never return worse than its input (keep-best on the exact deployed error), and
must only ever return reload-faithful vectors (no block past the clip).
"""

from modules.util.oft_align import AdapterOFTConfig, reconstruct_oft_rotation
from modules.util.oft_conjugate import clipped_skew_from_vec
from modules.util.oft_dense_fit import (
    dense_warmstart_for_tensor,
    encode_rotation_to_vec,
    fit_student_als,
    refine_deployed_fit,
    student_merged,
)

import torch


def _cfg(block=4, clip=0.99, scaled=False):
    return AdapterOFTConfig(block_size=block, clip=clip, scaled=scaled)


def _vec_at_sigma(r: int, b: int, sigma: float, seed: int) -> torch.Tensor:
    """Stored oft_R params whose per-block skew has spectral norm exactly sigma."""
    gen = torch.Generator().manual_seed(seed)
    skew = torch.randn(r, b, b, generator=gen, dtype=torch.float64)
    skew = skew - skew.transpose(-2, -1)
    skew = skew * (sigma / torch.linalg.matrix_norm(skew, ord=2)).reshape(r, 1, 1)
    rows, cols = torch.triu_indices(b, b, 1)
    return skew[:, rows, cols]


def _deployed(w_variant, vec, cfg, mult):
    """The honest deployed weight: module decode (clip + Neumann-5) + rowscale."""
    return student_merged(w_variant, reconstruct_oft_rotation(vec, cfg), mult)


def test_refine_never_worse_and_reports_honest_errors():
    """Unreachable dense target: refinement must (a) report err_init equal to
    the true deployed error of its input, (b) never return worse, and (c)
    report err_final equal to the true deployed error of its output."""
    cfg = _cfg(block=4)
    gen = torch.Generator().manual_seed(0)
    w_variant = torch.randn(10, 12, generator=gen, dtype=torch.float64)
    target = torch.randn(10, 12, generator=gen, dtype=torch.float64)

    student = fit_student_als(w_variant, target, 4, dora=True, iters=4)
    enc = encode_rotation_to_vec(student.r_blocks, cfg)
    err_manual = float((_deployed(w_variant, enc.vec, cfg, student.multiplier) - target).norm())

    res = refine_deployed_fit(enc.vec, student.multiplier, w_variant, target, cfg, iters=40)
    assert abs(res.err_init - err_manual) < 1e-9
    assert res.err_final <= res.err_init + 1e-12
    err_out = float((_deployed(w_variant, res.vec, cfg, res.multiplier) - target).norm())
    assert abs(res.err_final - err_out) < 1e-9
    assert all(torch.isfinite(torch.tensor(res.err_history)))


def test_refine_recovers_deployed_representable_target():
    """A target exactly representable in the DEPLOYED form at a moderate angle
    (sigma 0.6, well under the 0.99 clip): the ALS + exact-Cayley pipeline
    leaves a real floor (Neumann-5 is not the exact Cayley map there), which
    the refinement must close by orders of magnitude."""
    cfg = _cfg(block=6)
    gen = torch.Generator().manual_seed(1)
    out_f, in_f, b = 12, 18, 6
    w_variant = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    vec_true = _vec_at_sigma(in_f // b, b, 0.6, seed=2)
    m_true = 1.0 + 0.1 * torch.randn(out_f, generator=gen, dtype=torch.float64)
    target = _deployed(w_variant, vec_true, cfg, m_true)

    student = fit_student_als(w_variant, target, b, dora=True, iters=8)
    enc = encode_rotation_to_vec(student.r_blocks, cfg)
    pre_err = float((_deployed(w_variant, enc.vec, cfg, student.multiplier) - target).norm())
    assert pre_err > 1e-4  # the encode-mismatch floor is real at this angle

    res = refine_deployed_fit(enc.vec, student.multiplier, w_variant, target, cfg, iters=300)
    assert res.err_final < 1e-3 * pre_err
    assert res.err_final < 1e-4  # ~1e-6 relative to ||target||; LBFGS plateaus near line-search precision
    assert torch.allclose(res.multiplier, m_true, atol=1e-3)


def test_refine_projects_overclip_blocks_to_reload_faithful():
    """An init with a block past the clip: the module's reload would clip it,
    so refinement must project it (deployed-identical), report it, and never
    return a vector whose generator exceeds the clip."""
    cfg = _cfg(block=4)
    gen = torch.Generator().manual_seed(3)
    w_variant = torch.randn(8, 8, generator=gen, dtype=torch.float64)
    vec_init = _vec_at_sigma(2, 4, 1.5, seed=4)  # both blocks past clip 0.99
    target = _deployed(w_variant, vec_init, cfg, None)  # == decode of the CLIPPED init

    res = refine_deployed_fit(vec_init, None, w_variant, target, cfg, iters=20)
    assert res.err_init < 1e-9  # projection is exactly what the reload clip does
    assert res.err_final <= res.err_init
    assert res.projected_blocks == 2
    _, sigma = clipped_skew_from_vec(res.vec, cfg)
    assert float(sigma.max()) <= cfg.clip + 1e-9


def test_refine_plain_oft_keeps_multiplier_none():
    cfg = _cfg(block=4)
    gen = torch.Generator().manual_seed(5)
    w_variant = torch.randn(8, 8, generator=gen, dtype=torch.float64)
    target = torch.randn(8, 8, generator=gen, dtype=torch.float64)
    student = fit_student_als(w_variant, target, 4, dora=False, iters=3)
    enc = encode_rotation_to_vec(student.r_blocks, cfg)

    res = refine_deployed_fit(enc.vec, None, w_variant, target, cfg, iters=20)
    assert res.multiplier is None
    assert res.err_final <= res.err_init + 1e-12
    err_out = float((_deployed(w_variant, res.vec, cfg, None) - target).norm())
    assert abs(res.err_final - err_out) < 1e-9


def test_warmstart_pipeline_refine_integration():
    """dense_warmstart_for_tensor: refine_iters=0 must reproduce the pure
    ALS+encode result; refine_iters>0 must keep the same target, never
    regress, and report the written tensors' honest deployed error."""
    cfg = _cfg(block=4)
    gen = torch.Generator().manual_seed(6)
    out_f, in_f = 10, 8
    w_base = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    skew = torch.randn(in_f, in_f, generator=gen, dtype=torch.float64)
    skew = 0.1 * (skew - skew.T) / torch.linalg.matrix_norm(skew - skew.T, ord=2)
    w_variant = w_base @ torch.matrix_exp(skew) + 0.02 * torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    vec = _vec_at_sigma(in_f // 4, 4, 0.05, seed=7)
    mult = 1.0 + 0.05 * torch.randn(out_f, generator=gen, dtype=torch.float64)

    ws0 = dense_warmstart_for_tensor(vec, mult, w_base, w_variant, cfg, refine_iters=0)
    assert ws0.refine is None
    assert ws0.deployed_err == ws0.deployed_err_pre_refine

    ws1 = dense_warmstart_for_tensor(vec, mult, w_base, w_variant, cfg, refine_iters=40)
    assert ws1.refine is not None
    assert torch.allclose(ws0.target, ws1.target)
    assert ws1.deployed_err_pre_refine == ws0.deployed_err_pre_refine
    assert ws1.deployed_err <= ws1.deployed_err_pre_refine
    err_out = float((_deployed(w_variant, ws1.vec_new, cfg, ws1.mult_new) - ws1.target).norm())
    assert abs(ws1.deployed_err - err_out) < 1e-9
