"""Bit-equivalence tests for OFT/DoRA-OFT apply_to_module.

The merge path bakes the OFT rotation (and DoRA magnitude) into the base
weight in place. To be safe to ship as a real "merge into checkpoint" tool,
``apply_to_module`` MUST produce a baked weight whose bare forward equals
the adapter-mode forward bit-for-bit (within float tolerance).

These tests anchor that invariant on randomized OFT params (so R is a
non-trivial rotation, not I) for both Linear and Conv2d, with and without
DoRA. Failure here means a merge would diverge from training behavior --
exactly the bug we hit when ComfyUI's Save Checkpoint ignored scaled_oft.
"""

from __future__ import annotations

from modules.module.LoRAModule import DoRAOFTModule, OFTModule

import torch
from torch import nn

import pytest


def _seeded_oft(
    cls,
    orig_module: nn.Module,
    block_size: int = 16,
    scaled_oft: bool = True,
    seed: int = 42,
):
    """Build an OFT/DoRA-OFT module with non-trivial random rotation params."""
    module = cls(
        prefix="test",
        orig_module=orig_module,
        oft_block_size=block_size,
        block_share=False,
        oft_scaled=scaled_oft,
    )
    torch.manual_seed(seed)
    # Small magnitude so the Cayley-Neumann series stays well within convergence
    # and the resulting R is genuinely orthogonal.
    with torch.no_grad():
        module.oft_R.weight.normal_(0, 0.05)
    return module


def _adapter_then_bare(module, orig_module: nn.Module, x: torch.Tensor):
    """Capture adapter-mode forward, then bake and run bare forward."""
    weight_snapshot = orig_module.weight.data.clone()

    module.hook_to_module()
    try:
        y_adapter = orig_module(x)
    finally:
        module.remove_hook_from_module()

    # Restore the snapshot so apply_to_module starts from the unmerged base.
    orig_module.weight.data.copy_(weight_snapshot)

    module.apply_to_module()
    y_baked = orig_module(x)

    return y_adapter, y_baked


@pytest.mark.parametrize("scaled_oft", [True, False])
def test_oft_linear_apply_matches_forward(scaled_oft):
    """OFT-only on a Linear layer: baked weight must reproduce adapter forward."""
    in_features, out_features, block_size = 64, 32, 16
    torch.manual_seed(0)
    orig_module = nn.Linear(in_features, out_features, bias=True)

    module = _seeded_oft(OFTModule, orig_module, block_size=block_size, scaled_oft=scaled_oft)

    x = torch.randn(4, in_features)
    y_adapter, y_baked = _adapter_then_bare(module, orig_module, x)

    torch.testing.assert_close(y_adapter, y_baked, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scaled_oft", [True, False])
def test_oft_conv2d_apply_matches_forward(scaled_oft):
    """OFT-only on a Conv2d layer: baked weight must reproduce adapter forward."""
    in_ch, out_ch, block_size = 16, 32, 8
    torch.manual_seed(0)
    orig_module = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

    module = _seeded_oft(OFTModule, orig_module, block_size=block_size, scaled_oft=scaled_oft)

    x = torch.randn(2, in_ch, 8, 8)
    y_adapter, y_baked = _adapter_then_bare(module, orig_module, x)

    torch.testing.assert_close(y_adapter, y_baked, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scaled_oft", [True, False])
def test_dora_oft_linear_apply_matches_forward(scaled_oft):
    """DoRA-OFT on Linear: dora_scale must be baked as a per-row multiplier."""
    in_features, out_features, block_size = 64, 32, 16
    torch.manual_seed(1)
    orig_module = nn.Linear(in_features, out_features, bias=True)

    module = _seeded_oft(DoRAOFTModule, orig_module, block_size=block_size, scaled_oft=scaled_oft)
    # Perturb dora_scale away from initial_norm so the test exercises the
    # actual scaling path rather than scale == 1.
    with torch.no_grad():
        module.dora_scale.mul_(0.7)

    x = torch.randn(4, in_features)
    y_adapter, y_baked = _adapter_then_bare(module, orig_module, x)

    torch.testing.assert_close(y_adapter, y_baked, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scaled_oft", [True, False])
def test_dora_oft_conv2d_apply_matches_forward(scaled_oft):
    """DoRA-OFT on Conv2d: dora_scale must bake into per-output-channel scaling."""
    in_ch, out_ch, block_size = 16, 32, 8
    torch.manual_seed(1)
    orig_module = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

    module = _seeded_oft(DoRAOFTModule, orig_module, block_size=block_size, scaled_oft=scaled_oft)
    with torch.no_grad():
        module.dora_scale.mul_(0.7)

    x = torch.randn(2, in_ch, 8, 8)
    y_adapter, y_baked = _adapter_then_bare(module, orig_module, x)

    torch.testing.assert_close(y_adapter, y_baked, rtol=1e-5, atol=1e-5)


def test_oft_linear_zero_oft_is_identity():
    """When oft_R.weight is zero, R = I and apply_to_module is a no-op."""
    in_features, out_features, block_size = 64, 32, 16
    torch.manual_seed(2)
    orig_module = nn.Linear(in_features, out_features, bias=True)
    weight_before = orig_module.weight.data.clone()

    module = OFTModule(
        prefix="test",
        orig_module=orig_module,
        oft_block_size=block_size,
        block_share=False,
        oft_scaled=True,
    )
    # oft_R is zero-initialized by OFTModule.initialize_weights -- R must be I.
    module.apply_to_module()

    torch.testing.assert_close(orig_module.weight.data, weight_before, rtol=0, atol=1e-7)


def test_dora_oft_zero_oft_with_unit_scale_is_identity():
    """DoRA-OFT with zero rotation params and dora_scale == initial_norm is a no-op."""
    in_features, out_features, block_size = 64, 32, 16
    torch.manual_seed(3)
    orig_module = nn.Linear(in_features, out_features, bias=True)
    weight_before = orig_module.weight.data.clone()

    module = DoRAOFTModule(
        prefix="test",
        orig_module=orig_module,
        oft_block_size=block_size,
        block_share=False,
        oft_scaled=True,
    )
    # dora_scale is initialized to initial_norm in DoRAOFTModule.initialize_weights,
    # so the per-row scale is 1.0 and apply_to_module should be a no-op.
    module.apply_to_module()

    torch.testing.assert_close(orig_module.weight.data, weight_before, rtol=1e-5, atol=1e-6)


def test_oft_linear_block_share_apply_matches_forward():
    """block_share=True path: a single R is broadcast across all blocks."""
    in_features, out_features, block_size = 64, 32, 16
    torch.manual_seed(4)
    orig_module = nn.Linear(in_features, out_features, bias=True)

    module = OFTModule(
        prefix="test",
        orig_module=orig_module,
        oft_block_size=block_size,
        block_share=True,
        oft_scaled=True,
    )
    with torch.no_grad():
        module.oft_R.weight.normal_(0, 0.05)

    x = torch.randn(4, in_features)
    y_adapter, y_baked = _adapter_then_bare(module, orig_module, x)

    torch.testing.assert_close(y_adapter, y_baked, rtol=1e-5, atol=1e-5)


def test_apply_to_module_preserves_orig_weight_dtype():
    """Apply must cast back to the original weight dtype (e.g., bf16)."""
    in_features, out_features, block_size = 64, 32, 16
    torch.manual_seed(5)
    orig_module = nn.Linear(in_features, out_features, bias=True)
    orig_module.to(torch.bfloat16)

    module = _seeded_oft(OFTModule, orig_module, block_size=block_size, scaled_oft=True)
    module.apply_to_module()

    assert orig_module.weight.dtype is torch.bfloat16


def test_dora_oft_linear_invariant_merged_row_norm_equals_dora_scale():
    """For DoRA-OFT on Linear, |merged_row[i]| must equal |dora_scale[i]|.

    This is the OFT-DoRA invariant: orthogonal R preserves row norms, so
    after scaling by dora_scale/initial_norm, each merged output row has
    norm == |dora_scale|. Verification gates depend on this property.
    """
    in_features, out_features, block_size = 64, 32, 16
    torch.manual_seed(6)
    orig_module = nn.Linear(in_features, out_features, bias=False)

    module = _seeded_oft(DoRAOFTModule, orig_module, block_size=block_size, scaled_oft=True)
    with torch.no_grad():
        module.dora_scale.mul_(0.7)

    expected_norms = module.dora_scale.detach().abs().view(-1)
    module.apply_to_module()
    actual_norms = orig_module.weight.data.float().norm(dim=1)

    torch.testing.assert_close(actual_norms, expected_norms, rtol=1e-4, atol=1e-5)
