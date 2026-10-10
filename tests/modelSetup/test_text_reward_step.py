"""End-to-end reward step on a fake flow model: the rollout, the one-step
x0, the no-grad detect, the latent window, the grad-bearing crop decode and
the CTC backward all run on CPU. The "VAE" is a differentiable 8x bilinear
upsample of a 3-channel latent, so a latent holding a downsampled render of
the target text decodes to something the frozen OCR can read."""

import importlib.util
import os
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupTextRewardMixin import ModelSetupTextRewardMixin, text_reward_sigmas
from modules.util.config.TrainConfig import TrainConfig
from modules.util.text_reward_util import TextRewardSample, TextSpec

import torch
import torch.nn.functional as F

import pytest

HAS_DOCTR = importlib.util.find_spec("doctr") is not None


class _FakeVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=3, scaling_factor=1.0, shift_factor=0.0)
        self.gain = torch.nn.Parameter(torch.ones(()))  # gives parameters() a dtype/device

    def decode(self, latent, return_dict=False):
        image = F.interpolate(latent * self.gain, scale_factor=8, mode="bilinear", align_corners=False)
        return (image * 2.0 - 1.0,)  # [0,1] latent -> [-1,1] pixels


class _FakeSetup(ModelSetupTextRewardMixin):
    """Velocity that lands exactly on ``target`` in one Euler step:
    x0 = x_k - sigma * v = target."""

    def __init__(self, target):
        super().__init__()
        self.train_device = torch.device("cpu")
        self.temp_device = torch.device("cpu")
        self.target = target
        self.grad_forwards = 0

    def predict_distill_velocity(
        self, model, batch, config, train_progress, latent, sigma, *, conditioning="conditional"
    ):
        if torch.is_grad_enabled():
            self.grad_forwards += 1
        s = sigma.view(-1, 1, 1, 1)
        return (latent - self.target) / s


def _render_latent(text: str, size: int):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (size, size), (215, 205, 190))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size // 8)
    except OSError:
        pytest.skip("no TrueType font available for the render")
    draw.text((size // 10, size // 2 - size // 16), text, fill=(25, 25, 35), font=font)
    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()[None] / 255.0
    return F.interpolate(pixels, scale_factor=1 / 8, mode="area")


@pytest.mark.skipif(not HAS_DOCTR, reason="python-doctr not installed")
@pytest.mark.parametrize("grad_steps", [1, 2])
def test_reward_step_backpropagates_ctc_into_the_policy(tmp_path, grad_steps):
    size = 768
    words = ("Open", "Daily", "£4.50")
    target = torch.nn.Parameter(_render_latent(" ".join(words), size))
    setup = _FakeSetup(target)
    setup.text_reward_model().augment = False

    model = SimpleNamespace(
        vae=_FakeVae(),
        train_dtype=SimpleNamespace(torch_dtype=lambda: torch.float32),
        unscale_latents=lambda x: x,
        vae_to=lambda device: None,
    )
    config = TrainConfig.default_values()
    config.text_reward_resolution = size
    config.text_reward_rollout_steps = 4
    config.text_reward_random_grad_step = False
    config.text_reward_grad_steps = grad_steps
    config.text_reward_max_decode_pixels = 512
    config.text_reward_save_interval = 1
    config.workspace_dir = str(tmp_path)
    progress = SimpleNamespace(global_step=11, epoch=0, epoch_step=11)
    sample = TextRewardSample(TextSpec("Sign", " ".join(words), words, "price"), torch.zeros(4, 8), None)

    backwards = []
    value = setup.calculate_text_reward_loss(model, sample, config, progress, backwards.append)

    # DRaFT-K: one grad forward per step of the chain, all backwarded at once
    assert setup.grad_forwards == grad_steps
    assert len(backwards) == 1
    assert torch.isfinite(value) and float(value) > 0
    loss = backwards[0]
    loss.backward()
    assert target.grad is not None and float(target.grad.abs().sum()) > 0
    # the gradient is confined to the decoded text window, not the whole latent
    nonzero_rows = (target.grad.abs().sum(dim=(0, 1, 3)) > 0).nonzero().flatten()
    assert 0 < len(nonzero_rows) < target.shape[-2]

    metrics = setup.get_last_text_reward_metrics()
    assert metrics["grad_steps"] == grad_steps
    if grad_steps == 1:
        assert metrics["x0_sigma"] == metrics["grad_sigma"]
    else:
        # the scored estimate sits one grid step below the first grad step
        assert 0.0 < metrics["x0_sigma"] < metrics["grad_sigma"]
    assert metrics["scored_words"] >= 2
    assert metrics["word_ratio"] > 0.5
    assert metrics["family_count/price"] == 1.0
    assert os.path.exists(os.path.join(tmp_path, "text_reward", "00000011.png"))
    assert os.path.exists(os.path.join(tmp_path, "text_reward", "00000011.txt"))


def test_sigma_grid_is_monotone_and_shifted():
    sigmas = text_reward_sigmas(8, 3.0, torch.device("cpu"))
    assert sigmas.shape == (1, 9)
    assert sigmas[0, 0] == 1.0 and sigmas[0, -1] == 0.0
    assert torch.all(sigmas[0, 1:] < sigmas[0, :-1])
    # shift > 1 pushes mass toward high sigma
    assert sigmas[0, 4] > 0.5
