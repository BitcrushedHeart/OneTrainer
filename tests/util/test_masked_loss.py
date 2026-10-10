"""Masked-loss helpers shared by the supervised loss and masked DPO."""

from types import SimpleNamespace

from modules.util.loss.masked_loss import (
    clamp_mask,
    masked_losses,
    masked_mse_per_sample,
    resolve_unmasked_weight,
)

import torch


def _config(unmasked_weight: float = 0.1):
    return SimpleNamespace(unmasked_weight=unmasked_weight)


def _half_mask(b: int = 2, h: int = 4, w: int = 4) -> torch.Tensor:
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, :, : w // 2] = 1.0
    return mask


def test_resolve_unmasked_weight_falls_back_to_the_global_scalar():
    assert resolve_unmasked_weight({}, _config(0.3), 4) == 0.3
    assert resolve_unmasked_weight({"unmasked_weight_override": 0.5}, _config(0.3), 4) == 0.3


def test_resolve_unmasked_weight_substitutes_the_global_where_negative():
    batch = {"unmasked_weight_override": torch.tensor([-1.0, 0.0, 0.7])}
    weight = resolve_unmasked_weight(batch, _config(0.3), 4)
    assert weight.shape == (3, 1, 1, 1)
    assert torch.allclose(weight.flatten(), torch.tensor([0.3, 0.0, 0.7]))


def test_clamp_mask_matches_scalar_clamp_for_a_uniform_per_sample_weight():
    mask = _half_mask(3)
    per_sample = torch.full((3, 1, 1, 1), 0.1)
    assert torch.equal(clamp_mask(mask, per_sample), clamp_mask(mask, 0.1))


def test_clamp_mask_applies_each_sample_its_own_floor():
    mask = _half_mask(2)
    clamped = clamp_mask(mask, torch.tensor([0.0, 0.25]).view(2, 1, 1, 1))
    assert clamped[0].min() == 0.0
    assert clamped[1].min() == 0.25
    assert clamped.max() == 1.0


def test_masked_losses_accepts_per_sample_weights():
    losses = torch.ones(2, 3, 4, 4)
    out = masked_losses(losses.clone(), _half_mask(2), torch.tensor([0.0, 0.5]).view(2, 1, 1, 1), False)
    assert out[0, :, :, 2:].sum() == 0.0
    assert torch.allclose(out[1, :, :, 2:], torch.full((3, 4, 2), 0.5))


def test_masked_mse_with_an_all_ones_mask_is_the_plain_mean():
    torch.manual_seed(0)
    pred, target = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    plain = (pred - target).pow(2).mean(dim=(1, 2, 3))
    assert torch.allclose(masked_mse_per_sample(pred, target, torch.ones(2, 1, 4, 4)), plain, atol=1e-6)


def test_masked_mse_is_independent_of_the_mask_area():
    # A uniform error map must give the same per-pixel average whatever the
    # mask covers: this is what keeps beta meaningful across pairs.
    pred = torch.zeros(1, 3, 8, 8)
    target = torch.full((1, 3, 8, 8), 2.0)
    full = masked_mse_per_sample(pred, target, torch.ones(1, 1, 8, 8))
    small = torch.zeros(1, 1, 8, 8)
    small[..., :2, :2] = 1.0
    assert torch.allclose(masked_mse_per_sample(pred, target, small), full)
    assert torch.allclose(full, torch.tensor([4.0]))


def test_masked_mse_only_sees_the_masked_region_at_weight_zero():
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 100.0], [1.0, 100.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    assert torch.allclose(masked_mse_per_sample(pred, target, mask), torch.tensor([1.0]))


def test_masked_mse_weights_the_background_by_the_unmasked_weight():
    pred = torch.zeros(1, 1, 1, 2)
    target = torch.tensor([[[[1.0, 3.0]]]])
    clamped = clamp_mask(torch.tensor([[[[1.0, 0.0]]]]), 0.5)
    # (1*1 + 0.5*9) / (1 + 0.5)
    assert torch.allclose(masked_mse_per_sample(pred, target, clamped), torch.tensor([5.5 / 1.5]))


def test_masked_mse_accumulates_in_fp32_from_low_precision_inputs():
    pred = torch.randn(2, 3, 4, 4).to(torch.bfloat16)
    target = torch.randn(2, 3, 4, 4).to(torch.bfloat16)
    out = masked_mse_per_sample(pred, target, _half_mask(2))
    assert out.dtype == torch.float32
    assert out.shape == (2,)
