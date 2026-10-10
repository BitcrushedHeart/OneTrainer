"""Low-sigma perceptual term: guards, weight, crop, feature distance, metrics.

CPU only. The fake VAE is a diffusers-shaped decoder (conv_in / mid_block /
up_blocks taking ``(sample, temb)``) small enough to run in milliseconds.
"""

import math
from types import SimpleNamespace

from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.modelSetup.mixin.ModelSetupPerceptualLossMixin import (
    cross_normalized_distance,
    decoder_feature_distance,
    perceptual_weight,
    random_latent_window,
)
from modules.util.config.TrainConfig import TrainConfig

import torch
from torch import nn

import pytest

LATENT_CHANNELS = 4


class _Block(nn.Module):
    def __init__(self, channels: int, upsample: bool):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample = upsample
        self.calls = 0
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, sample, temb=None):
        self.calls += 1
        self.seen_shapes.append(tuple(sample.shape))
        sample = torch.nn.functional.silu(self.conv(sample))
        if self.upsample:
            sample = torch.nn.functional.interpolate(sample, scale_factor=2, mode="nearest")
        return sample


class _FakeDecoder(nn.Module):
    def __init__(self, channels: int = 8):
        super().__init__()
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, channels, 3, padding=1)
        self.mid_block = _Block(channels, upsample=False)
        self.up_blocks = nn.ModuleList([_Block(channels, upsample=True), _Block(channels, upsample=False)])


class _FakeVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _FakeDecoder()
        self.post_quant_conv = None
        self.config = SimpleNamespace(latent_channels=LATENT_CHANNELS, scaling_factor=1.0, shift_factor=0.0)

    def decode(self, *args, **kwargs):
        raise AssertionError("the perceptual term must walk the decoder, not call decode()")


class _FakeModel:
    def __init__(self):
        self.vae = _FakeVae()
        self.noise_scheduler = SimpleNamespace(sigmas=torch.ones(1000))

    def unscale_latents(self, latents):
        return latents


class _StubZImageSetup(BaseZImageSetup):
    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError


def _config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.perceptual_strength = 1.0
    config.batch_size = 2
    config.gradient_accumulation_steps = 1
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _data(sigmas: list[float], height: int = 8, width: int = 8, seed: int = 0) -> tuple[dict, dict]:
    torch.manual_seed(seed)
    batch_size = len(sigmas)
    sigma = torch.tensor(sigmas, dtype=torch.float32).view(-1, 1, 1, 1)
    latent = torch.randn(batch_size, LATENT_CHANNELS, height, width)
    noise = torch.randn_like(latent)
    predicted = (noise - latent + 0.3 * torch.randn_like(latent)).requires_grad_(True)
    noisy = (1 - sigma) * latent + sigma * noise
    data = {
        "loss_type": "target",
        "timestep": (sigma.reshape(-1) * 1000 - 1).long().clamp_min(0),
        "predicted": predicted,
        "target": noise - latent,
        "latent_image": latent,
        "noisy_latent_image": noisy,
        "sigma": sigma,
        "predicted_latent": noisy - predicted * sigma,
    }
    batch = {"loss_weight": torch.ones(batch_size)}
    return batch, data


def _decoder_calls(model: _FakeModel) -> int:
    decoder = model.vae.decoder
    return decoder.mid_block.calls + sum(block.calls for block in decoder.up_blocks)


# -- weight ---------------------------------------------------------------------


def test_weight_formula_values():
    sigma = torch.tensor([0.05, 0.3, 0.5, 0.625, 0.75, 0.9])
    weight = perceptual_weight(sigma, strength=2.0, gamma=5.0, sigma_full=0.5, sigma_max=0.75)
    snr = ((1 - sigma) / sigma) ** 2
    # inside the full gate: strength * min(SNR, gamma)
    assert weight[0] == pytest.approx(2.0 * 5.0)  # SNR = 361, capped
    assert weight[1] == pytest.approx(2.0 * min(float(snr[1]), 5.0))
    assert weight[2] == pytest.approx(2.0 * 1.0)  # SNR = 1 at sigma 0.5
    # halfway through the cosine taper: gate 0.5
    assert weight[3] == pytest.approx(2.0 * float(snr[3]) * 0.5, rel=1e-5)
    assert weight[4] == 0.0
    assert weight[5] == 0.0


def test_weight_fades_out_at_low_sigma_relative_to_mse():
    # The pull on x0_hat is w * sigma^2: constant above 1/sqrt(gamma), vanishing below.
    sigma = torch.tensor([0.02, 0.1, 0.3, 0.45])
    weight = perceptual_weight(sigma, 1.0, 5.0, 0.5, 0.75)
    pull = weight * sigma**2
    assert pull[3] == pytest.approx(pull[2], rel=0.35)  # both capped-free: ~(1-s)^2
    assert pull[0] < 0.1 * pull[2]


def test_weight_step_gate_when_max_not_above_full():
    weight = perceptual_weight(torch.tensor([0.4, 0.6]), 1.0, 5.0, 0.5, 0.5)
    assert weight[0] > 0
    assert weight[1] == 0


# -- crop -----------------------------------------------------------------------


def test_random_latent_window_full_frame_and_bounds():
    import random

    rng = random.Random(0)
    assert random_latent_window(8, 8, 0, rng) == (0, 8, 0, 8)
    assert random_latent_window(8, 8, 16, rng) == (0, 8, 0, 8)
    for _ in range(50):
        y0, y1, x0, x1 = random_latent_window(8, 12, 4, rng)
        assert (y1 - y0, x1 - x0) == (4, 4)
        assert y0 >= 0 and y1 <= 8 and x0 >= 0 and x1 <= 12


# -- feature distance -------------------------------------------------------------


def test_cross_normalized_distance_zero_for_identical_and_uses_prediction_stats():
    torch.manual_seed(1)
    pred = torch.randn(2, 3, 4, 4) * 3 + 1
    assert torch.allclose(cross_normalized_distance(pred, pred.clone()), torch.zeros(2))

    target = pred + 0.5
    mean = pred.mean(dim=(2, 3), keepdim=True)
    std = pred.std(dim=(2, 3), keepdim=True) + 1e-6
    expected = (((pred - mean) / std - (target - mean) / std) ** 2).mean(dim=1).mean(dim=(1, 2))
    assert torch.allclose(cross_normalized_distance(pred, target), expected, atol=1e-5)


def test_cross_normalized_distance_masks_target_outliers():
    torch.manual_seed(2)
    pred = torch.randn(1, 3, 4, 4)
    target = pred.clone()
    target[0, :, 0, 0] += 1000.0  # one exploding position
    distance = cross_normalized_distance(pred, target)
    assert distance.item() < 1.0  # would be ~1e5 without the mask


def test_cross_normalized_distance_respects_latent_mask():
    torch.manual_seed(3)
    pred = torch.randn(1, 3, 8, 8)
    target = pred.clone()
    target[..., 4:, :] += 5.0  # differences only in the bottom half
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :] = 1.0  # keep only the top half (latent res, resized x2)
    assert cross_normalized_distance(pred, target, mask).item() == pytest.approx(0.0, abs=1e-6)


def test_decoder_feature_distance_backprops_to_prediction_only():
    decoder = _FakeDecoder()
    z_pred = torch.randn(2, LATENT_CHANNELS, 4, 4, requires_grad=True)
    z_target = torch.randn(2, LATENT_CHANNELS, 4, 4, requires_grad=True)
    distance = decoder_feature_distance(decoder, z_pred, z_target)
    assert distance.shape == (2,)
    assert decoder.up_blocks[0].calls == 2  # target walk + prediction walk (checkpoint recomputes on backward)
    distance.sum().backward()
    assert z_pred.grad is not None and torch.isfinite(z_pred.grad).all()
    assert z_target.grad is None


# -- mixin: guards ----------------------------------------------------------------


def test_off_leaves_losses_untouched():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.3, 0.9])
    losses = torch.tensor([1.0, 2.0])
    out = setup._add_perceptual_loss(model, batch, data, _config(perceptual_strength=0.0), losses)
    assert out is losses
    assert _decoder_calls(model) == 0
    assert setup.pop_perceptual_metrics() == {}


def test_no_grad_skips_decode():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.3, 0.3])
    with torch.no_grad():
        out = setup._add_perceptual_loss(model, batch, data, _config(), torch.tensor([1.0, 2.0]))
    assert torch.equal(out, torch.tensor([1.0, 2.0]))
    assert _decoder_calls(model) == 0


def test_all_samples_above_gate_skip_decode_but_report_fraction():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.8, 0.95])
    out = setup._add_perceptual_loss(model, batch, data, _config(), torch.tensor([1.0, 2.0]))
    assert torch.equal(out, torch.tensor([1.0, 2.0]))
    assert _decoder_calls(model) == 0
    assert setup.pop_perceptual_metrics() == {"fraction": 0.0}


@pytest.mark.parametrize("branch", ["dmd", "teacher"])
def test_distill_branches_skip(branch):
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.3, 0.3])
    config = _config()
    if branch == "dmd":
        batch["_distill_latent_input"] = data["noisy_latent_image"]
    else:
        config.distillation_teacher_lora_model_name = "teacher.safetensors"
    out = setup._add_perceptual_loss(model, batch, data, config, torch.tensor([1.0, 2.0]))
    assert torch.equal(out, torch.tensor([1.0, 2.0]))
    assert _decoder_calls(model) == 0


def test_offloaded_vae_warns_and_skips():
    setup = _StubZImageSetup(torch.device("cuda"), torch.device("cpu"), False)
    model = _FakeModel()  # VAE parameters on CPU
    batch, data = _data([0.3, 0.3])
    losses = torch.tensor([1.0, 2.0], device="meta")  # "training device" differs from the VAE's
    out = setup._add_perceptual_loss(model, batch, data, _config(), losses)
    assert out is losses
    assert _decoder_calls(model) == 0
    assert setup._perceptual_warned


# -- mixin: the term ----------------------------------------------------------------


def test_gated_sample_gets_term_and_grad_reaches_prediction():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.3, 0.9])
    losses = torch.tensor([1.0, 2.0])
    out = setup._add_perceptual_loss(model, batch, data, _config(perceptual_crop=0), losses)
    assert out[1].item() == pytest.approx(2.0)  # sigma 0.9 is outside the gate
    assert out[0].item() > 1.0
    # only the gated sample was decoded (target walk, prediction walk, then the
    # grad-norm diagnostic's backward recomputes the checkpointed prediction walk)
    assert model.vae.decoder.mid_block.seen_shapes == [(1, 8, 8, 8)] * 3
    out.mean().backward()
    assert data["predicted"].grad is not None
    assert torch.isfinite(data["predicted"].grad).all()
    assert data["predicted"].grad[1].abs().sum() == 0  # sample outside the gate untouched

    metrics = setup.pop_perceptual_metrics()
    assert metrics["fraction"] == pytest.approx(0.5)
    assert metrics["loss"] > 0
    assert metrics["raw"] > 0
    assert metrics["grad_norm_v"] > 0
    assert metrics["mse_grad_norm_v"] > 0
    assert setup.pop_perceptual_metrics() == {}


def test_crop_window_is_sliced_identically_for_prediction_and_target():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.3], height=8, width=8)
    # 32 px crop -> 4x4 latent window (third entry: checkpoint recompute in the diagnostic)
    setup._add_perceptual_loss(model, batch, data, _config(batch_size=1, perceptual_crop=32), torch.tensor([1.0]))
    assert model.vae.decoder.mid_block.seen_shapes == [(1, 8, 4, 4)] * 3


def test_weighted_term_matches_formula():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    setup._perceptual_rng.seed(0)
    model = _FakeModel()
    batch, data = _data([0.3])
    batch["loss_weight"] = torch.tensor([0.5])
    config = _config(batch_size=1, perceptual_strength=3.0, perceptual_crop=0)
    out = setup._add_perceptual_loss(model, batch, data, config, torch.zeros(1))

    with torch.no_grad():
        raw = decoder_feature_distance(model.vae.decoder, data["predicted_latent"], data["latent_image"])
    weight = perceptual_weight(data["sigma"], 3.0, 5.0, 0.5, 0.75)
    assert out.item() == pytest.approx(float(raw * weight * 0.5), rel=1e-4)


def test_grad_norm_diagnostic_only_on_first_micro_step():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    config = _config(batch_size=1, gradient_accumulation_steps=3, perceptual_crop=0)
    for _ in range(3):
        batch, data = _data([0.3])
        setup._add_perceptual_loss(model, batch, data, config, torch.zeros(1))
    assert setup._perceptual_counts["loss"] == 3
    assert setup._perceptual_counts["grad_norm_v"] == 1


def test_mse_grad_norm_is_analytic_gradient_of_mse_term():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.3])
    config = _config(batch_size=1, perceptual_crop=0)
    setup._add_perceptual_loss(model, batch, data, config, torch.zeros(1))
    reported = setup.pop_perceptual_metrics()["mse_grad_norm_v"]

    predicted = data["predicted"].detach().clone().requires_grad_(True)
    mse = torch.nn.functional.mse_loss(predicted, data["target"], reduction="none").mean(dim=(1, 2, 3))
    scale = config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
    (mse * config.mse_strength * scale * batch["loss_weight"]).sum().backward()
    assert reported == pytest.approx(float(predicted.grad.norm()), rel=1e-4)


def test_calculate_loss_wires_the_term():
    setup = _StubZImageSetup(torch.device("cpu"), torch.device("cpu"), False)
    model = _FakeModel()
    batch, data = _data([0.3, 0.9])
    config = _config(perceptual_crop=0)
    with_term = setup.calculate_loss(model, batch, data, config)
    config.perceptual_strength = 0.0
    without_term = setup.calculate_loss(model, batch, data, config)
    assert with_term.item() > without_term.item()
    assert math.isfinite(with_term.item())


def test_cross_normalized_distance_leaves_fp32_inputs_untouched():
    pred = torch.randn(1, 3, 8, 8, requires_grad=True)
    target = torch.randn(1, 3, 8, 8)
    before = target.clone()
    cross_normalized_distance(pred, target)
    assert torch.equal(target, before)
