"""GPU check for the perceptual term on a real diffusers AutoencoderKL. Skipped on CPU/CI.

Builds a tiny 16-channel Flux-shaped VAE from config (no weights download), walks
its decoder through the mixin on-device in bf16, and confirms the term is finite,
reaches the velocity output, and never touches the target's graph.
"""

from types import SimpleNamespace

from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.util.config.TrainConfig import TrainConfig

import torch

import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _StubZImageSetup(BaseZImageSetup):
    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError


def _tiny_flux_vae(device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=(32, 32, 32),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=8,
        use_quant_conv=False,
        use_post_quant_conv=False,
        scaling_factor=0.3611,
        shift_factor=0.1159,
    )
    return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)


def test_decoder_walk_on_real_autoencoder_kl():
    device = torch.device("cuda")
    vae = _tiny_flux_vae(device, torch.bfloat16)
    model = SimpleNamespace(
        vae=vae,
        unscale_latents=lambda z: z / vae.config.scaling_factor + vae.config.shift_factor,
        noise_scheduler=SimpleNamespace(sigmas=torch.ones(1000)),
    )
    setup = _StubZImageSetup(device, torch.device("cpu"), False)
    config = TrainConfig.default_values()
    config.perceptual_strength = 1.0
    config.perceptual_crop = 64  # 8x8 latent window from a 16x16 latent
    config.batch_size = 2
    config.gradient_accumulation_steps = 1

    torch.manual_seed(0)
    sigma = torch.tensor([0.3, 0.6], device=device).view(-1, 1, 1, 1)
    latent = torch.randn(2, 16, 16, 16, device=device)
    noise = torch.randn_like(latent)
    predicted = (noise - latent + 0.2 * torch.randn_like(latent)).requires_grad_(True)
    noisy = (1 - sigma) * latent + sigma * noise
    data = {
        "loss_type": "target",
        "timestep": torch.tensor([299, 599], device=device),
        "predicted": predicted,
        "target": noise - latent,
        "latent_image": latent,
        "noisy_latent_image": noisy,
        "sigma": sigma,
        "predicted_latent": noisy - predicted * sigma,
    }
    batch = {"loss_weight": torch.ones(2, device=device)}

    losses = setup._add_perceptual_loss(model, batch, data, config, torch.zeros(2, device=device))
    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    assert (losses > 0).all()  # both samples are inside the gate (0.6 is on the taper)
    losses.mean().backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert latent.grad is None

    metrics = setup.pop_perceptual_metrics()
    assert metrics["fraction"] == pytest.approx(1.0)
    assert metrics["grad_norm_v"] > 0 and metrics["mse_grad_norm_v"] > 0
