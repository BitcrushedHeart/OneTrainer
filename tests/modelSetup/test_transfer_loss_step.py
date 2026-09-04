"""CPU tests for the transfer-mode training step.

Drives the REAL pieces end to end on a tiny toy transformer: the generic
BaseModelSetup.calculate_transfer_loss orchestration (prior_model teacher
context + weight swap + per-block loss), the real BaseZImageSetup
transfer_late_forward and _transfer_swap_modules, the real
ZImageLoRASetup.create_parameters trainable/frozen partition, and real
OFTModule/DoRAOFTModule adapters in a real LoRAModuleWrapper. Only
transfer_fork_forward is replaced (the real one needs the full Z-Image
text/noise pipeline); the toy fork uses the same run_prefix_and_capture trap.
"""

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


class TinyTransferModel(BaseModel):
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
        self.transfer_teacher_late_tensors = None

    def adapters(self):
        return [self.transformer_lora]

    def to(self, device):
        return self

    def eval(self):
        return self


class TinySetup(ZImageLoRASetup):
    """Real ZImageLoRASetup with only the model-forward fork replaced."""

    def transfer_fork_forward(self, model, batch, config, train_progress, *, deterministic=False, seed=None):
        fork_block = model.transformer.layers[config.transfer_late_start]
        args, kwargs = run_prefix_and_capture(lambda: model.transformer(batch["latent_image"]), fork_block)
        return TransferForkState(block_args=args, block_kwargs=kwargs, mask=None)


def _config(*, dora: bool) -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    config.peft_type = PeftType.OFT_2
    config.dora_oft = dora
    config.oft_block_size = 4
    config.oft_scaled = False
    config.oft_clipped_norm = 0.99
    config.dropout_probability = 0.0
    config.transfer_enabled = True
    config.transfer_late_start = LATE_START
    config.transfer_families = "qkv,mlp"
    return config


def _build(dora: bool, adapter_theta: float):
    config = _config(dora=dora)
    model = TinyTransferModel(config, adapter_theta=adapter_theta)
    setup = TinySetup(torch.device("cpu"), torch.device("cpu"), False)

    # Teacher = the toy's own bare weights (a stand-in "Base + education" set);
    # covers ALL late adapter-covered stems incl. attention.to_out.0.
    swap_modules = setup._transfer_swap_modules(model, config)
    model.transfer_teacher_late_tensors = {stem: module.weight.data.clone() for stem, module in swap_modules.items()}

    # Real trainable/frozen partition + requires_grad emulation of setup_model.
    setup.create_parameters(model, config)
    model.transformer_lora.requires_grad_(True)
    for param in setup.frozen_parameters["transformer"]:
        param.requires_grad_(False)
    return model, setup, config


def test_swap_set_covers_all_late_adapted_modules():
    model, setup, config = _build(dora=False, adapter_theta=0.0)
    swap = setup._transfer_swap_modules(model, config)
    assert set(swap) == {
        f"layers.{layer}.{stem}"
        for layer in (2, 3)
        for stem in ("attention.to_q", "attention.to_out.0", "feed_forward.w1")
    }


def test_identity_adapter_matches_teacher_exactly():
    """Identity adapter + teacher == variant weights: residual is ~0. This also
    proves the teacher pass bypasses hooks AND uses the swapped weights -- any
    leakage of either would show up as a nonzero residual elsewhere."""
    model, setup, config = _build(dora=False, adapter_theta=0.0)
    batch = {"latent_image": torch.randn(2, HIDDEN)}
    loss = setup.calculate_transfer_loss(model, batch, config, TrainProgress())
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("dora", [False, True])
def test_prefix_once_late_twice_and_grads_only_late_families(dora):
    model, setup, config = _build(dora=dora, adapter_theta=0.05)
    batch = {"latent_image": torch.randn(2, HIDDEN)}

    loss = setup.calculate_transfer_loss(model, batch, config, TrainProgress())
    assert float(loss.detach()) > 0.0  # teacher (bare weights, hooks off) != adapted student

    for layer in (0, 1):
        assert model.transformer.layers[layer].calls == 1, "prefix must run exactly once"
    for layer in (2, 3):
        assert model.transformer.layers[layer].calls == 2, "late blocks run teacher + student"

    loss.backward()
    for name, module in model.transformer_lora.lora_modules.items():
        layer = int(name.split(".")[1])  # keys are bare relative names: layers.N.stem
        is_trainable_family = layer >= LATE_START and (".to_q" in name or ".w1" in name)
        grad = module.oft_R.weight.grad
        if is_trainable_family:
            assert grad is not None and float(grad.abs().sum()) > 0.0, name
            if dora:
                dgrad = module.dora_multiplier.grad
                assert dgrad is not None and float(dgrad.abs().sum()) > 0.0, name
        else:
            assert grad is None, f"{name} must stay frozen"
    for param in model.transformer.parameters():
        if not param.requires_grad:
            assert param.grad is None

    metrics = setup.get_last_transfer_metrics()
    assert "residual" in metrics and f"block_{LATE_START}" in metrics and "loss" in metrics


def test_model_state_restored_after_step():
    model, setup, config = _build(dora=True, adapter_theta=0.05)
    batch = {"latent_image": torch.randn(2, HIDDEN)}
    swap = setup._transfer_swap_modules(model, config)
    weight_snapshot = {stem: module.weight.data.clone() for stem, module in swap.items()}
    hooked_forwards = {name: module.orig_module.forward for name, module in model.transformer_lora.lora_modules.items()}

    setup.calculate_transfer_loss(model, batch, config, TrainProgress())

    for stem, module in swap.items():
        assert torch.equal(module.weight.data, weight_snapshot[stem])
    for name, module in model.transformer_lora.lora_modules.items():
        assert module.orig_module.forward == hooked_forwards[name], f"{name}: adapter hook not restored"


def test_deterministic_seeded_loss_is_reproducible():
    """The convergence tracker's entire design rests on validation being
    deterministic for a fixed (batch, seed): identical calls must produce
    identical residuals even after the global RNG is perturbed."""
    model, setup, config = _build(dora=True, adapter_theta=0.05)
    batch = {"latent_image": torch.randn(2, HIDDEN)}
    first = float(
        setup.calculate_transfer_loss(model, batch, config, TrainProgress(), deterministic=True, seed=5).detach()
    )
    torch.manual_seed(999)  # perturb global RNG between calls
    second = float(
        setup.calculate_transfer_loss(model, batch, config, TrainProgress(), deterministic=True, seed=5).detach()
    )
    assert first == second


def test_teacher_weights_actually_used():
    model, setup, config = _build(dora=False, adapter_theta=0.05)
    batch = {"latent_image": torch.randn(2, HIDDEN)}
    loss_a = float(setup.calculate_transfer_loss(model, batch, config, TrainProgress()))
    model.transfer_teacher_late_tensors = {
        stem: torch.zeros_like(t) for stem, t in model.transfer_teacher_late_tensors.items()
    }
    loss_b = float(setup.calculate_transfer_loss(model, batch, config, TrainProgress()))
    assert loss_a != pytest.approx(loss_b)


def test_gradient_checkpointing_path():
    model, setup, config = _build(dora=False, adapter_theta=0.05)
    transformer = model.transformer
    transformer.gradient_checkpointing = True
    transformer._gradient_checkpointing_func = lambda module, *args, **kwargs: torch.utils.checkpoint.checkpoint(
        module, *args, use_reentrant=False, **kwargs
    )
    batch = {"latent_image": torch.randn(2, HIDDEN)}
    loss = setup.calculate_transfer_loss(model, batch, config, TrainProgress())
    loss.backward()
    late_q = model.transformer_lora.lora_modules[f"layers.{LATE_START}.attention.to_q"]
    assert late_q.oft_R.weight.grad is not None
    assert float(late_q.oft_R.weight.grad.abs().sum()) > 0.0


def test_create_parameters_selects_only_families():
    model, setup, config = _build(dora=True, adapter_theta=0.0)
    collection = setup.create_parameters(model, config)
    groups = collection.parameter_groups if hasattr(collection, "parameter_groups") else None
    # Count trainable tensors: (oft_R + dora_multiplier) x (to_q + w1) x 2 late layers.
    selected = sum(
        1
        for module_name, module in model.transformer_lora.lora_modules.items()
        for _ in module.parameters()
        if int(module_name.split(".")[1]) >= LATE_START and (".to_q" in module_name or ".w1" in module_name)
    )
    frozen = len(setup.frozen_parameters["transformer"])
    total = sum(1 for m in model.transformer_lora.lora_modules.values() for _ in m.parameters())
    assert selected == 2 * 2 * 2  # 2 params/module x 2 family modules/layer x 2 layers
    assert selected + frozen == total
    assert groups is None or len(groups) >= 1
