"""GPU end-to-end checks for teacher-match LPIPS. Skipped on CPU/CI.

These exercise the *real* BaseZImageSetup._distill_perceptual_loss (VAE decode +
LPIPS on-device) and the full run_distill_validation loop with a dummy VAE, to
confirm the device-colocation fix lets teacher_match_lpips actually compute.
"""

from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

import pytest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _ToggleAdapter:
    def __init__(self):
        self.hooked = False

    def hook_to_module(self):
        self.hooked = True

    def remove_hook_from_module(self):
        self.hooked = False


class _DummyVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 3, padding=1)  # latent (4ch) -> image (3ch)

    def decode(self, latent, return_dict=False):
        return (self.conv(latent),)


class _TrainDtype:
    @staticmethod
    def torch_dtype():
        return torch.float32


class _GpuModel(BaseModel):
    def __init__(self, device):
        super().__init__(ModelType.Z_IMAGE)
        self.student_adapter = _ToggleAdapter()
        self.fake_adapter = _ToggleAdapter()
        self.student_adapter.hook_to_module()
        self.base = nn.Conv2d(4, 4, 3, padding=1).to(device)
        self.student = nn.Conv2d(4, 4, 3, padding=1).to(device)
        self.fake = nn.Conv2d(4, 4, 3, padding=1).to(device)
        self.vae = _DummyVae().to(device)
        self.train_dtype = _TrainDtype()

    def unscale_latents(self, latent):
        return latent

    def to(self, device):
        return self

    def eval(self):
        return self

    def adapters(self):
        return [self.student_adapter]

    def fake_adapters(self):
        return [self.fake_adapter]


class _GpuSetup(BaseZImageSetup):
    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError

    # Route the "velocity" by active adapter, so no real transformer is needed.
    def predict_distill_velocity(
        self, model, batch, config, train_progress, latent, sigma, *, conditioning="conditional"
    ):
        if model.fake_adapter.hooked:
            return model.fake(latent)
        if model.student_adapter.hooked:
            return model.student(latent)
        return model.base(latent)


def _val_config():
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    config.distill_enabled = True
    config.distill_validation = True
    config.distill_validation_prompts = 1
    config.distill_target_steps = 2
    config.distill_teacher_steps = 4
    config.distill_teacher_steps_auto = False
    return config


def test_perceptual_loss_computes_on_gpu():
    device = torch.device("cuda")
    setup = _GpuSetup(device, torch.device("cpu"), False)
    model = _GpuModel(device)

    # Sanity: the VAE really is on cuda:0 and counts as co-located with train_device "cuda".
    vae_device = next(model.vae.parameters()).device
    assert vae_device.type == "cuda"
    assert setup._lpips_vae_colocated(vae_device, device) is True

    generated = torch.randn(1, 4, 64, 64, device=device, requires_grad=True)
    target = torch.randn(1, 4, 64, 64, device=device)

    value = setup._distill_perceptual_loss(model, generated, target, _val_config())

    assert value is not None, "LPIPS returned None on GPU (check the 'disabled after error' print above)"
    assert torch.isfinite(value)
    # It must be differentiable back into the generated latent (real training use).
    value.backward()
    assert generated.grad is not None and torch.isfinite(generated.grad).all()


def test_validation_loop_populates_teacher_match_lpips_on_gpu():
    device = torch.device("cuda")
    setup = _GpuSetup(device, torch.device("cpu"), False)
    model = _GpuModel(device)
    config = _val_config()

    # Stash a captured validation batch the way capture_distill_val_batch would.
    setup._distill_val_batches = [{"latent_image": torch.randn(1, 4, 64, 64, device=device)}]
    setup._distill_teacher_ref_cache = {}

    metrics = setup.run_distill_validation(model, config, TrainProgress())

    assert "teacher_match_mse" in metrics and torch.isfinite(torch.tensor(metrics["teacher_match_mse"]))
    assert "teacher_match_lpips" in metrics, f"teacher_match_lpips missing; got {sorted(metrics)}"
    assert torch.isfinite(torch.tensor(metrics["teacher_match_lpips"]))
    assert "diversity_ratio" in metrics
