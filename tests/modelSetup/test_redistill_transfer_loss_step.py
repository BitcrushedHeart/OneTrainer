"""CPU tests for the redistill Phase A training step.

Same toy-transformer approach as test_transfer_loss_step.py: real
calculate_redistill_transfer_loss orchestration (student-trajectory latent
source + prior_model teacher context + raw-weight swap + per-block loss), real
adapters and parameter partition; only transfer_fork_forward is replaced, and
the toy fork asserts it receives the injected (x_k, sigma_k)."""

from contextlib import nullcontext

# Plain-import form so isort keeps it FIRST: primes the factory import graph
# (create -> factory.import_dir) before any direct modelSetup import below.
import modules.util.create  # noqa: F401 -- import-side-effect required
from modules.model.BaseModel import BaseModel
from modules.modelSetup.ZImageLoRASetup import ZImageLoRASetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress
from modules.util.transfer_util import TransferForkState, run_prefix_and_capture

import torch
from torch import nn

import pytest

HIDDEN = 8
LAYERS = 4
LATE_START = 2


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False)])


class _Ffn(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(HIDDEN, HIDDEN, bias=False)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attn()
        self.feed_forward = _Ffn()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.attention.to_out[0](self.attention.to_q(x)) + self.feed_forward.w1(x)


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TinyRedistillModel(BaseModel):
    def __init__(self, config: TrainConfig, *, adapter_theta: float = 0.0):
        super().__init__(ModelType.Z_IMAGE)
        torch.manual_seed(0)
        self.transformer = _Stack()
        self.transformer.requires_grad_(False)
        self.autocast_context = nullcontext()
        self.transformer_lora = LoRAModuleWrapper(self.transformer, "transformer", config, [])
        if adapter_theta > 0.0:
            with torch.no_grad():
                for module in self.transformer_lora.lora_modules.values():
                    module.oft_R.weight.normal_(0, adapter_theta)
        self.transformer_lora.hook_to_module()
        self.transformer_fake_lora = None
        self.transfer_teacher_late_tensors = None

    def adapters(self):
        return [self.transformer_lora]

    def to(self, device):
        return self

    def eval(self):
        return self


class TinyRedistillSetup(ZImageLoRASetup):
    """Real ZImageLoRASetup with only the model-forward fork replaced."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_injected = None

    def transfer_fork_forward(
        self,
        model,
        batch,
        config,
        train_progress,
        *,
        deterministic=False,
        seed=None,
        latent_input=None,
        sigma=None,
    ):
        assert latent_input is not None and sigma is not None, "redistill must inject the trajectory sample"
        self.last_injected = (latent_input.detach().clone(), sigma.detach().clone())
        fork_block = model.transformer.layers[config.transfer_late_start]
        args, kwargs = run_prefix_and_capture(lambda: model.transformer(latent_input), fork_block)
        return TransferForkState(block_args=args, block_kwargs=kwargs, mask=None)


def _config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    config.peft_type = PeftType.OFT_2
    config.dora_oft = False
    config.oft_block_size = 4
    config.oft_scaled = False
    config.oft_clipped_norm = 0.99
    config.dropout_probability = 0.0
    config.redistill_enabled = True
    config.transfer_late_start = LATE_START
    config.transfer_families = "qkv,mlp"
    # target_steps=1 -> k is always 0: the rollout needs no predict() calls.
    config.distill_target_steps = 1
    config.distill_teacher_steps_auto = False
    return config


def _build(adapter_theta: float):
    config = _config()
    model = TinyRedistillModel(config, adapter_theta=adapter_theta)
    setup = TinyRedistillSetup(torch.device("cpu"), torch.device("cpu"), False)

    # Raw teacher = perturbed copies of the toy's own late weights (stand-in
    # for the old Turbo's late tensors).
    swap_modules = setup._transfer_swap_modules(model, config)
    model.transfer_teacher_late_tensors = {
        stem: module.weight.data.clone() + 0.01 for stem, module in swap_modules.items()
    }

    setup.create_parameters(model, config)
    model.transformer_lora.requires_grad_(True)
    for param in setup.frozen_parameters["transformer"]:
        param.requires_grad_(False)
    return model, setup, config


def _batch():
    return {"latent_image": torch.randn(2, HIDDEN)}


def test_loss_finite_and_grads_only_on_late_families():
    model, setup, config = _build(adapter_theta=0.05)
    loss = setup.calculate_redistill_transfer_loss(model, _batch(), config, TrainProgress())
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0

    loss.backward()
    for name, module in model.transformer_lora.lora_modules.items():
        layer = int(name.split(".")[1])
        is_trainable_family = layer >= LATE_START and (".to_q" in name or ".w1" in name)
        grad = module.oft_R.weight.grad
        if is_trainable_family:
            assert grad is not None and float(grad.abs().sum()) > 0.0, name
        else:
            assert grad is None, f"{name} must stay frozen"

    metrics = setup.get_last_transfer_metrics()
    assert "residual" in metrics and f"block_{LATE_START}" in metrics


def test_injected_trajectory_sample_used():
    model, setup, config = _build(adapter_theta=0.05)
    setup.calculate_redistill_transfer_loss(model, _batch(), config, TrainProgress())
    latent, sigma = setup.last_injected
    assert latent.shape == (2, HIDDEN)
    assert sigma.shape == (2,)
    # k=0 at a 1-step grid: the injected latent is the initial noise at sigma=1.
    assert float(sigma.max()) == pytest.approx(1.0)


def test_deterministic_seeded_loss_is_reproducible():
    model, setup, config = _build(adapter_theta=0.05)
    batch = _batch()
    first = float(
        setup.calculate_redistill_transfer_loss(
            model, batch, config, TrainProgress(), deterministic=True, seed=7
        ).detach()
    )
    torch.manual_seed(999)
    second = float(
        setup.calculate_redistill_transfer_loss(
            model, batch, config, TrainProgress(), deterministic=True, seed=7
        ).detach()
    )
    assert first == second


def test_teacher_weights_swapped_and_restored():
    model, setup, config = _build(adapter_theta=0.05)
    swap = setup._transfer_swap_modules(model, config)
    weight_snapshot = {stem: module.weight.data.clone() for stem, module in swap.items()}

    loss_a = float(setup.calculate_redistill_transfer_loss(model, _batch(), config, TrainProgress()))
    for stem, module in swap.items():
        assert torch.equal(module.weight.data, weight_snapshot[stem])

    model.transfer_teacher_late_tensors = {
        stem: torch.zeros_like(t) for stem, t in model.transfer_teacher_late_tensors.items()
    }
    loss_b = float(setup.calculate_redistill_transfer_loss(model, _batch(), config, TrainProgress()))
    assert loss_a != pytest.approx(loss_b)


def test_distill_update_mode_counts_from_phase_offset():
    _, setup, config = _build(adapter_theta=0.0)
    config.distill_fake_warmup_steps = 2
    config.distill_ttur_ratio = 1

    progress = TrainProgress()
    progress.global_step = 1000
    # Without the offset, step 1000 is far past warmup.
    assert setup._distill_update_mode(config, progress) in {"fake", "student"}

    setup.set_redistill_phase_b_start(1000)
    modes = []
    for step in range(1000, 1006):
        progress.global_step = step
        modes.append(setup._distill_update_mode(config, progress))
    # Warmup counts from the boundary: 2 fake warmup steps, then TTUR 1:1.
    assert modes == ["fake", "fake", "fake", "student", "fake", "student"]
