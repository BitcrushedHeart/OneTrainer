"""Smoke test for the merge tool's wrapper-facing TrainConfig surface.

The merge orchestrator reconstructs a real TrainConfig from the adapter's
ot_config metadata (oft_merge._train_config_from_adapter) and hands it to
LoRAModuleWrapper. If the wrapper (or anything downstream) reads a field the
config no longer carries, we get an opaque AttributeError at runtime.

This wires a transfer-style TrainConfig into LoRAModuleWrapper against a
synthetic Z-Image-shaped module so any missing attribute on the OFT/DoRA-OFT
path surfaces in the test suite rather than in the user's backend log.

(Historical note: this file previously targeted a SimpleNamespace builder,
``_wrapper_config``/``_OFTAdapterMeta``, that the branch replaced with real
TrainConfig reconstruction.)
"""

from __future__ import annotations

from modules.module.LoRAModule import DoRAOFTModule, LoRAModuleWrapper, OFTModule
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
from torch import nn

import pytest


class _ToyTransformer(nn.Module):
    """A tiny stand-in for a real transformer: attention-style blocks with
    q/k/v Linear projections, mirroring the path names OneTrainer's OFT
    adapter uses (``layers.0.attention.to_v`` etc.)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([_AttnBlock(hidden) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _AttnBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attention = _Attn(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class _Attn(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.to_q = nn.Linear(hidden, hidden, bias=False)
        self.to_k = nn.Linear(hidden, hidden, bias=False)
        self.to_v = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_q(x) + self.to_k(x) + self.to_v(x)


def _make_config(*, dora_oft: bool, scaled_oft: bool = True, block_size: int = 16) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.OFT_2
    config.dora_oft = dora_oft
    config.oft_scaled = scaled_oft
    config.oft_block_size = block_size
    config.oft_clipped_norm = 0.99
    config.oft_block_share = False
    config.oft_cans = False
    config.dropout_probability = 0.0
    return config


def _build_wrapper(dora_oft: bool, layer_filter: list[str] | None = None) -> LoRAModuleWrapper:
    transformer = _ToyTransformer(hidden=64)
    return LoRAModuleWrapper(transformer, "transformer", _make_config(dora_oft=dora_oft), layer_filter or [])


def test_wrapper_config_satisfies_oft_path():
    """Building a LoRAModuleWrapper from a real TrainConfig must not raise
    AttributeError on the OFT (no DoRA) path."""
    wrapper = _build_wrapper(dora_oft=False)
    # Each of q/k/v on each of 2 layers -> 6 modules expected.
    assert len(wrapper.lora_modules) == 6
    for module in wrapper.lora_modules.values():
        assert isinstance(module, OFTModule)
        assert not isinstance(module, DoRAOFTModule)


def test_wrapper_config_satisfies_dora_oft_path():
    """Same as above but the DoRA-OFT path -- exercises dora_multiplier init."""
    wrapper = _build_wrapper(dora_oft=True)
    assert len(wrapper.lora_modules) == 6
    for module in wrapper.lora_modules.values():
        assert isinstance(module, DoRAOFTModule)
        assert module.dora_multiplier is not None


def test_wrapper_config_hook_apply_unhook_cycle():
    """End-to-end on the merge surface: hook, apply_to_module, unhook.
    Catches any missing config attr exercised only at apply time."""
    wrapper = _build_wrapper(dora_oft=True)
    wrapper.hook_to_module()
    wrapper.apply_to_module()
    wrapper.remove_hook_from_module()


def test_wrapper_config_layer_filter():
    """Layer filter must still work. Filter to just to_v -- 2 layers * 1
    projection = 2 modules."""
    transformer = _ToyTransformer(hidden=64)
    wrapper = LoRAModuleWrapper(transformer, "transformer", _make_config(dora_oft=True), ["to_v"])
    assert len(wrapper.lora_modules) == 2


@pytest.mark.parametrize(
    "attr",
    [
        # Every config.<name> reference grep'd from modules/module/LoRAModule.py.
        # If this list drifts from the real module surface, the next test fails
        # loud -- preferable to a cryptic AttributeError in the backend log.
        "debug_mode",
        "dora_oft",
        "dropout_probability",
        "layer_filter_regex",
        "lokr_decompose_both",
        "lokr_decompose_factor",
        "lokr_dim",
        "lokr_dora_on_output",
        "lokr_full_matrix",
        "lokr_use_tucker",
        "lokr_vec_trick",
        "lokr_weight_decompose",
        "lora_alpha",
        "lora_decompose",
        "lora_decompose_norm_epsilon",
        "lora_decompose_output_axis",
        "lora_rank",
        "oft_block_share",
        "oft_block_size",
        "oft_cans",
        "oft_clipped_norm",
        "oft_scaled",
        "peft_type",
        "train_device",
    ],
)
def test_train_config_has_wrapper_attr(attr):
    """Guard against drift between TrainConfig and LoRAModuleWrapper's reads."""
    assert hasattr(TrainConfig.default_values(), attr), f"TrainConfig missing wrapper-required attr: {attr}"
