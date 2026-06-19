"""Smoke test for the merge tool's SimpleNamespace config.

The merge orchestrator builds a SimpleNamespace stand-in for TrainConfig and
hands it to LoRAModuleWrapper. If the wrapper (or anything downstream)
reads a field that isn't on the namespace, we get an opaque
``'types.SimpleNamespace' object has no attribute X`` at runtime.

This test wires the orchestrator's actual _wrapper_config builder into
LoRAModuleWrapper against a synthetic Z-Image-shaped module so any missing
attribute on the OFT/DoRA-OFT path surfaces in CI rather than in the user's
backend log.
"""

from __future__ import annotations

from modules.module.LoRAModule import DoRAOFTModule, LoRAModuleWrapper, OFTModule
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.oft_merge import _OFTAdapterMeta, _wrapper_config

import torch
from torch import nn

import pytest


class _ToyTransformer(nn.Module):
    """A tiny stand-in for a real transformer: one attention-style block
    with q/k/v Linear projections, mirroring the path names OneTrainer's
    OFT adapter uses (``layers.0.attention.to_v`` etc.)."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        # Nested module so the wrapper exercises its named_modules() walk
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


def _make_meta(*, dora_oft: bool, scaled_oft: bool = True, block_size: int = 16) -> _OFTAdapterMeta:
    return _OFTAdapterMeta(
        model_type=ModelType.Z_IMAGE,
        peft_type=PeftType.OFT_2,
        lora_rank=0,
        lora_alpha=1.0,
        dora_oft=dora_oft,
        scaled_oft=scaled_oft,
        oft_block_size=block_size,
        oft_coft=False,
        oft_block_share=False,
        coft_eps=6e-5,
        layer_filter="",
        layer_filter_regex=False,
    )


def _build_wrapper(dora_oft: bool, layer_filter: list[str] | None = None) -> LoRAModuleWrapper:
    """Drive the same _wrapper_config builder the orchestrator uses."""
    meta = _make_meta(dora_oft=dora_oft)
    config = _wrapper_config(meta)
    transformer = _ToyTransformer(hidden=64)
    return LoRAModuleWrapper(transformer, "transformer", config, layer_filter or [])


def test_wrapper_config_satisfies_oft_path():
    """Building a LoRAModuleWrapper with the orchestrator's SimpleNamespace
    must not raise AttributeError on the OFT (no DoRA) path."""
    wrapper = _build_wrapper(dora_oft=False)
    # Each of q/k/v on each of 2 layers -> 6 modules expected.
    assert len(wrapper.lora_modules) == 6
    for module in wrapper.lora_modules.values():
        assert isinstance(module, OFTModule)
        assert not isinstance(module, DoRAOFTModule)


def test_wrapper_config_satisfies_dora_oft_path():
    """Same as above but the DoRA-OFT path -- exercises dora_scale init,
    which reads more config fields than plain OFT."""
    wrapper = _build_wrapper(dora_oft=True)
    assert len(wrapper.lora_modules) == 6
    for module in wrapper.lora_modules.values():
        assert isinstance(module, DoRAOFTModule)


def test_wrapper_config_hook_apply_unhook_cycle():
    """End-to-end on the merge surface: hook, apply_to_module, unhook.
    Catches any missing config attr exercised only at apply time."""
    wrapper = _build_wrapper(dora_oft=True)
    wrapper.hook_to_module()
    wrapper.apply_to_module()
    wrapper.remove_hook_from_module()


def test_wrapper_config_layer_filter_regex():
    """Layer filter must still work via the SimpleNamespace.
    Filter to just to_v -- 2 layers * 1 projection = 2 modules."""
    meta = _make_meta(dora_oft=True)
    config = _wrapper_config(meta)
    transformer = _ToyTransformer(hidden=64)
    wrapper = LoRAModuleWrapper(transformer, "transformer", config, ["to_v"])
    assert len(wrapper.lora_modules) == 2


@pytest.mark.parametrize(
    "attr",
    [
        # Every config.<name> reference grep'd from modules/module/LoRAModule.py.
        # If this list ever drifts from the real module surface, the next
        # test will fail loud -- preferable to a cryptic SimpleNamespace error
        # in the user's backend log.
        "coft_eps",
        "debug_mode",
        "dora_oft",
        "dropout_probability",
        "layer_filter_regex",
        "lora_alpha",
        "lora_decompose",
        "lora_decompose_norm_epsilon",
        "lora_decompose_output_axis",
        "lora_rank",
        "oft_block_share",
        "oft_block_size",
        "oft_coft",
        "peft_type",
        "scaled_oft",
        "train_device",
    ],
)
def test_wrapper_config_has_attr(attr):
    """Guard against future drift in LoRAModuleWrapper's config interface."""
    config = _wrapper_config(_make_meta(dora_oft=True))
    assert hasattr(config, attr), f"_wrapper_config missing required attr: {attr}"
