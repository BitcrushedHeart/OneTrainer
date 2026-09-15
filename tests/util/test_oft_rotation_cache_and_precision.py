"""OFTRotationModule: shared triu index tables, two-matmul Neumann, fp32 Cayley.

CPU-only. Covers:
- the per-(block_size, device) index table is shared (not a buffer per module),
  survives .to(dtype)/.to(device) via the _apply re-bind, and leaves the
  state_dict surface of OFT and DoRA-OFT adapters unchanged;
- the Horner-factored 5-term Neumann series equals the three-matmul series and
  oft_align.reconstruct_oft_rotation;
- the rotation is computed in float32 with autocast off even inside a bf16
  autocast region, while output dtypes and the apply_to_module bake still agree;
- the forward compiles fullgraph (also under autocast) without recompiling.

CUDA cannot be exercised here; the device re-bind is tested with "meta" as the
non-CPU device (same _apply code path as .cuda()).
"""

from __future__ import annotations

import math

from modules.module.LoRAModule import DoRAOFTModule, LoRAModuleWrapper, OFTModule
from modules.module.oft_utils import OFTRotationModule, shared_triu_indices
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.oft_align import AdapterOFTConfig, reconstruct_oft_rotation

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.utils.checkpoint
from torch import nn

import pytest


def _rotation(block_size: int, r: int = 3, clip: float | None = 0.99, **kwargs) -> OFTRotationModule:
    return OFTRotationModule(
        r=r,
        n_elements=block_size * (block_size - 1) // 2,
        block_size=block_size,
        in_features=r * block_size,
        oft_clipped_norm=clip,
        **kwargs,
    )


def _ortho_error(R: torch.Tensor) -> float:
    R = R.double()
    eye = torch.eye(R.shape[-1], dtype=torch.float64)
    return (R.mT @ R - eye).abs().max().item()


def _legacy_neumann(module: OFTRotationModule, Q: torch.Tensor, num_neumann_terms: int = 5) -> torch.Tensor:
    """The pre-Horner _cayley_batch Neumann branch, verbatim (three bmm for 5 terms)."""
    b = Q.shape[0]
    block_size = module.block_size
    Q_skew = module._pytorch_skew_symmetric(Q, block_size)
    if module.oft_clipped_norm is not None:
        v_vec, u_vec = module._spectral_norm(Q_skew)
        u_raw = torch.bmm(Q_skew, v_vec)
        sigma = torch.sum(u_vec * u_raw, dim=1, keepdim=True)
        max_norm = module.oft_clipped_norm
        Q_skew = Q_skew * (max_norm / torch.clamp(sigma, min=max_norm))
    R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
    if num_neumann_terms > 1:
        R.add_(Q_skew, alpha=2.0)
        if num_neumann_terms > 2:
            Q_squared = torch.bmm(Q_skew, Q_skew)
            R.add_(Q_squared, alpha=2.0)

            Q_power = Q_squared
            for _ in range(3, num_neumann_terms - 1):
                Q_power = torch.bmm(Q_power, Q_skew)
                R.add_(Q_power, alpha=2.0)
            Q_power = torch.bmm(Q_power, Q_skew)
            R.add_(Q_power)
    return R


# --------------------------------------------------------------------------- #
# Shared index tables
# --------------------------------------------------------------------------- #


def test_same_block_size_shares_index_storage():
    a, b, c = _rotation(8), _rotation(8, r=5), _rotation(16)

    assert a.rows.data_ptr() == b.rows.data_ptr()
    assert a.cols.data_ptr() == b.cols.data_ptr()
    assert a.rows.data_ptr() != c.rows.data_ptr()
    assert a.cols.data_ptr() != c.cols.data_ptr()

    expected_rows, expected_cols = torch.triu_indices(8, 8, 1)
    assert torch.equal(a.rows, expected_rows) and torch.equal(a.cols, expected_cols)
    assert a.rows.dtype == torch.int64

    for module in (a, b, c):
        names = {name for name, _ in module.named_buffers()}
        assert not {"rows", "cols"} & names


def test_to_dtype_and_device_rebind_shared_indices():
    a, b = _rotation(8), _rotation(8)

    a.to(torch.float64)
    assert a.weight.dtype == torch.float64
    assert a.rows.dtype == torch.int64 and a.cols.dtype == torch.int64  # never dtype-cast
    assert a.rows.data_ptr() == b.rows.data_ptr()

    a.to("cpu")
    assert a.rows.device.type == "cpu"
    assert a.rows.data_ptr() == b.rows.data_ptr()

    # "meta" stands in for cuda: .to(device) goes through the same _apply override.
    a.to("meta")
    b.to("meta")
    assert a.weight.device.type == "meta"
    assert a.rows.device.type == "meta" and a.cols.device.type == "meta"
    assert a.rows is b.rows and a.cols is b.cols
    assert a.rows is shared_triu_indices(8, "meta")[0]

    # And back: a full round trip lands on the CPU table again, and still works.
    a.to_empty(device="cpu")
    assert a.rows is shared_triu_indices(8, "cpu")[0]
    with torch.no_grad():
        a.weight.zero_()
    torch.testing.assert_close(a._pytorch_skew_symmetric(a.weight, 8), torch.zeros(3, 8, 8, dtype=torch.float64))


def test_skew_symmetric_falls_back_to_indices_on_the_input_device():
    module = _rotation(8)  # bound to CPU
    vec = torch.randn(3, 28, device="meta")
    out = module._pytorch_skew_symmetric(vec, 8)
    assert out.device.type == "meta"
    assert module.rows.device.type == "cpu"  # the fallback does not re-bind


def test_skew_symmetric_roundtrip_uses_shared_indices():
    module = _rotation(8).double()
    vec = torch.randn(3, 28, dtype=torch.float64)
    skew = module._pytorch_skew_symmetric(vec, 8)
    torch.testing.assert_close(skew, -skew.mT)
    torch.testing.assert_close(module._pytorch_skew_symmetric_inv(skew, 8), vec)


def _toy_transformer(hidden: int = 64) -> nn.Module:
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)
            self.to_v = nn.Linear(hidden, hidden, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attn()

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(2)])

    return Toy()


def _wrapper(*, dora_oft: bool, scaled: bool) -> LoRAModuleWrapper:
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.OFT_2
    config.dora_oft = dora_oft
    config.oft_scaled = scaled
    config.oft_block_size = 16
    config.oft_clipped_norm = 0.99
    config.oft_block_share = False
    config.oft_cans = False
    config.dropout_probability = 0.0
    return LoRAModuleWrapper(_toy_transformer(), "transformer", config, [])


# Per-module state_dict suffixes the pre-change OFTRotationModule produced (rows/cols
# were non-persistent buffers, so they never appeared; u_state/v_state likewise).
_LEGACY_SUFFIXES = {
    (False, False): {"oft_R.weight", "oft_R.clipped_oft"},
    (False, True): {"oft_R.weight", "oft_R.clipped_oft", "oft_R.scaled_oft"},
    (True, False): {"oft_R.weight", "oft_R.clipped_oft", "dora_multiplier"},
    (True, True): {"oft_R.weight", "oft_R.clipped_oft", "oft_R.scaled_oft", "dora_multiplier"},
}


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("dora_oft", [False, True])
def test_wrapper_state_dict_keys_unchanged_and_legacy_file_loads_strict(dora_oft, scaled):
    wrapper = _wrapper(dora_oft=dora_oft, scaled=scaled)
    stems = [f"transformer.layers.{i}.attention.{p}" for i in range(2) for p in ("to_q", "to_k", "to_v")]
    expected = {f"{stem}.{suffix}" for stem in stems for suffix in _LEGACY_SUFFIXES[(dora_oft, scaled)]}

    assert set(wrapper.state_dict()) == expected

    # A state dict shaped exactly like a pre-change adapter file.
    gen = torch.Generator().manual_seed(0)
    legacy = {}
    for stem in stems:
        legacy[f"{stem}.oft_R.weight"] = torch.randn(4, 120, generator=gen) * 1e-2
        legacy[f"{stem}.oft_R.clipped_oft"] = torch.tensor(0.99)
        if scaled:
            legacy[f"{stem}.oft_R.scaled_oft"] = torch.tensor(True)
        if dora_oft:
            legacy[f"{stem}.dora_multiplier"] = 1.0 + 0.1 * torch.randn(64, generator=gen)

    fresh = _wrapper(dora_oft=dora_oft, scaled=scaled)
    fresh.load_state_dict(legacy, strict=True)
    for key, value in legacy.items():
        torch.testing.assert_close(fresh.state_dict()[key], value)


# --------------------------------------------------------------------------- #
# Two-matmul Neumann
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("clip", [None, 0.99])
@pytest.mark.parametrize("std", [2e-3, 0.05, 0.5])
def test_horner_neumann_matches_three_bmm_float64(clip, std):
    torch.manual_seed(1)
    module = _rotation(16, r=4, clip=clip).double().eval()  # eval: power-iteration state is not advanced
    Q = torch.randn(4, 120, dtype=torch.float64) * std

    if clip is not None and std == 0.5:
        # The clip must actually engage in this case, or it tests nothing.
        v, u = module._spectral_norm(module._pytorch_skew_symmetric(Q, 16))
        sigma = (u * torch.bmm(module._pytorch_skew_symmetric(Q, 16), v)).sum(dim=1)
        assert (sigma > clip).all()

    horner = module._cayley_batch(Q, 16, True, 5, False)
    reference = _legacy_neumann(module, Q)
    assert horner.dtype == torch.float64
    assert (horner - reference).abs().max().item() <= 1e-12


@pytest.mark.parametrize("clip", [None, 0.99])
def test_horner_neumann_gradients_match_three_bmm_float64(clip):
    torch.manual_seed(11)
    module = _rotation(16, r=4, clip=clip).double().eval()
    Q = (torch.randn(4, 120, dtype=torch.float64) * 0.5).requires_grad_(True)
    upstream = torch.randn(4, 16, 16, dtype=torch.float64)

    (grad_horner,) = torch.autograd.grad((module._cayley_batch(Q, 16, True, 5, False) * upstream).sum(), Q)
    (grad_legacy,) = torch.autograd.grad((_legacy_neumann(module, Q) * upstream).sum(), Q)
    assert (grad_horner - grad_legacy).abs().max().item() <= 1e-11


@pytest.mark.parametrize("terms", [1, 2, 3, 4, 6])
def test_generic_neumann_loop_kept_for_other_term_counts(terms):
    torch.manual_seed(2)
    module = _rotation(8, clip=None).double()
    Q = torch.randn(3, 28, dtype=torch.float64) * 0.05
    torch.testing.assert_close(
        module._cayley_batch(Q, 8, True, terms, False), _legacy_neumann(module, Q, terms), rtol=0, atol=0
    )


@pytest.mark.parametrize("scaled", [False, True])
def test_cayley_batch_matches_oft_align_reconstruct_without_clip(scaled):
    torch.manual_seed(3)
    block = 16
    module = _rotation(block, r=4, clip=None, oft_scaled=scaled).double()
    with torch.no_grad():
        module.weight.normal_(0, 0.05)
    scaling = 2 * math.sqrt(block - 1) if scaled else 1

    ours = module._cayley_batch(module.weight.detach() / scaling, block, True, 5, False)
    # block_size in the config is the pre-adjustment request; reconstruct derives the real one.
    theirs = reconstruct_oft_rotation(module.weight.detach(), AdapterOFTConfig(block_size=6, clip=None, scaled=scaled))
    assert (ours - theirs).abs().max().item() <= 1e-12


def test_cayley_batch_matches_oft_align_reconstruct_with_engaged_clip():
    # oft_align clips with the exact spectral norm (eigvalsh); the module uses a
    # power iteration. After converging the iteration the two agree closely, but
    # not to float64 round-off, hence the looser tolerance here.
    torch.manual_seed(4)
    block = 16
    module = _rotation(block, r=4, clip=0.99).double().eval()
    with torch.no_grad():
        module.weight.normal_(0, 0.5)
    module.warm_start_spectral_state(steps=2000)

    ours = module._cayley_batch(module.weight.detach(), block, True, 5, False)
    theirs = reconstruct_oft_rotation(
        module.weight.detach(), AdapterOFTConfig(block_size=block, clip=0.99, scaled=False)
    )
    assert (ours - theirs).abs().max().item() <= 1e-5


# --------------------------------------------------------------------------- #
# fp32 Cayley under autocast
# --------------------------------------------------------------------------- #


def _spy_cayley(monkeypatch, module: OFTRotationModule) -> list[dict]:
    calls = []
    original = module._cayley_batch

    def spy(Q, *args, **kwargs):
        R = original(Q, *args, **kwargs)
        calls.append(
            {
                "Q_dtype": Q.dtype,
                "R_dtype": R.dtype,
                "R": R.detach().clone(),
                "autocast": torch.is_autocast_enabled("cpu"),
            }
        )
        return R

    monkeypatch.setattr(module, "_cayley_batch", spy)
    return calls


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_rotation_is_fp32_and_orthogonal_under_bf16_autocast(monkeypatch, input_dtype):
    torch.manual_seed(5)
    block = 64
    module = _rotation(block, r=4, clip=0.99)
    with torch.no_grad():
        module.weight.normal_(0, 2e-3)
    calls = _spy_cayley(monkeypatch, module)

    x = torch.randn(2, 5, 4 * block, dtype=input_dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        y = module(x)

    assert len(calls) == 1
    call = calls[0]
    assert call["autocast"] is False
    assert call["Q_dtype"] == torch.float32
    assert call["R_dtype"] == torch.float32
    assert _ortho_error(call["R"]) <= 1e-5
    assert (call["R"].double().norm(dim=-1) - 1).abs().max().item() <= 1e-6

    assert y.dtype == input_dtype
    assert y.shape == x.shape


def test_rotation_output_dtype_without_autocast():
    module = _rotation(8)
    for dtype in (torch.float32, torch.bfloat16, torch.float64):
        x = torch.randn(3, 24, dtype=dtype)
        assert module(x).dtype == dtype


def test_spectral_state_stays_in_module_dtype_under_autocast():
    module = _rotation(16, clip=0.99)
    with torch.no_grad():
        module.weight.normal_(0, 0.3)
    before = module.u_state.clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        module(torch.randn(2, 48, dtype=torch.bfloat16))
    assert module.u_state.dtype == torch.float32 and module.v_state.dtype == torch.float32
    assert not torch.equal(module.u_state, before)  # training mode advanced the iteration


def _seeded(cls, orig_module, block_size, *, clip=0.99, scaled=True, seed=42):
    module = cls(
        prefix="test",
        orig_module=orig_module,
        oft_block_size=block_size,
        block_share=False,
        oft_scaled=scaled,
        oft_clipped_norm=clip,
    )
    torch.manual_seed(seed)
    with torch.no_grad():
        module.oft_R.weight.normal_(0, 0.05)
    return module


@pytest.mark.parametrize("scaled", [False, True])
def test_dora_oft_linear_forward_matches_bake(scaled):
    torch.manual_seed(6)
    orig = nn.Linear(64, 32, bias=True)
    module = _seeded(DoRAOFTModule, orig, 16, scaled=scaled)
    with torch.no_grad():
        module.dora_multiplier.mul_(0.7)
    x = torch.randn(4, 64)
    snapshot = orig.weight.data.clone()

    module.hook_to_module()
    try:
        y_adapter = orig(x)
    finally:
        module.remove_hook_from_module()
    orig.weight.data.copy_(snapshot)
    module.apply_to_module()
    y_baked = orig(x)

    torch.testing.assert_close(y_adapter, y_baked, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cls", [OFTModule, DoRAOFTModule])
def test_oft_conv2d_rotation_fp32_under_autocast(monkeypatch, cls):
    torch.manual_seed(7)
    # bias=False: DoRA-OFT strips/re-adds an fp32 bias, which promotes its output
    # to fp32 under autocast (unchanged behavior, unrelated to the rotation).
    orig = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
    module = _seeded(cls, orig, 8)
    calls = _spy_cayley(monkeypatch, module.oft_R)
    x = torch.randn(2, 16, 8, 8)

    module.hook_to_module()
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = orig(x)
        y_fp32 = orig(x)
    finally:
        module.remove_hook_from_module()

    assert [c["R_dtype"] for c in calls] == [torch.float32, torch.float32]
    assert calls[0]["autocast"] is False
    assert y.dtype == torch.bfloat16  # conv under autocast, as before
    assert y_fp32.dtype == torch.float32


def test_bf16_param_swap_matches_bf16_reference_semantics():
    """DPO EXISTING_ADAPTER swaps param.data to bf16: R is now built from its fp32 upcast."""
    torch.manual_seed(8)
    linear = nn.Linear(32, 32, bias=False)
    module = _seeded(OFTModule, linear, 8, clip=None)
    module.oft_R.weight.data = module.oft_R.weight.data.to(torch.bfloat16)
    module.hook_to_module()
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = linear(torch.randn(2, 32))
    finally:
        module.remove_hook_from_module()
    assert y.dtype == torch.bfloat16
    R = module.oft_R.cayley_rotation_full_precision()
    assert R.dtype == torch.float32
    assert _ortho_error(R) <= 1e-5


def test_cans_step_count_follows_original_weight_dtype(monkeypatch):
    steps_seen = []
    module = _rotation(8, clip=None, oft_cans=True)
    original = module._cans_newton_schulz_iteration

    def spy(G, steps=7, eps=1e-7):
        steps_seen.append((G.dtype, steps))
        return original(G=G, steps=steps, eps=eps)

    monkeypatch.setattr(module, "_cans_newton_schulz_iteration", spy)
    x = torch.randn(2, 24)

    module(x)  # fp32 weight, fp32 id_mat
    module.to(torch.bfloat16)  # fully bf16 module: previously G was bf16 -> 5 steps
    module(x.to(torch.bfloat16))
    module.to(torch.float32)
    module.weight.data = module.weight.data.to(torch.bfloat16)  # param-only swap: id_mat fp32 + Q bf16 -> fp32
    module(x)

    assert steps_seen == [(torch.float32, 7), (torch.float32, 5), (torch.float32, 7)]


# --------------------------------------------------------------------------- #
# torch.compile
# --------------------------------------------------------------------------- #


def _compile_modules():
    torch.manual_seed(9)
    m8a, m8b, m16 = _rotation(8, r=3), _rotation(8, r=3), _rotation(16, r=2)
    for m in (m8a, m8b, m16):
        with torch.no_grad():
            m.weight.normal_(0, 0.05)
    return m8a, m8b, m16


@pytest.mark.parametrize("use_autocast", [False, True])
def test_rotation_compiles_fullgraph_without_recompile(use_autocast):
    torch._dynamo.reset()
    m8a, m8b, m16 = _compile_modules()

    def fn(x24, x32):
        return m8a(x24).sum() + m16(x32).sum() + m8b(x24).sum()

    counter = torch._dynamo.testing.CompileCounter()
    compiled = torch.compile(fn, backend=counter, fullgraph=True)
    x24 = torch.randn(2, 5, 24, requires_grad=True)
    x32 = torch.randn(2, 5, 32)

    for _ in range(3):
        if use_autocast:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = compiled(x24, x32)
        else:
            out = compiled(x24, x32)
        out.float().backward()

    assert counter.frame_count == 1
    assert m8a.weight.grad is not None and m16.weight.grad is not None
    eager = fn(x24, x32)
    if not use_autocast:
        torch.testing.assert_close(out, eager)
    torch._dynamo.reset()


def test_hooked_oft_linears_compile_fullgraph_under_autocast():
    torch._dynamo.reset()
    torch.manual_seed(10)
    seq = nn.Sequential(nn.Linear(24, 32, bias=False), nn.Linear(32, 24, bias=False), nn.Linear(24, 8, bias=False))
    adapters = [_seeded(DoRAOFTModule, layer, 8) for layer in seq]
    for adapter in adapters:
        adapter.hook_to_module()

    counter = torch._dynamo.testing.CompileCounter()
    compiled = torch.compile(seq, backend=counter, fullgraph=True)
    try:
        for _ in range(2):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = compiled(torch.randn(3, 24))
            out.float().sum().backward()
    finally:
        for adapter in adapters:
            adapter.remove_hook_from_module()
    assert counter.frame_count == 1
    assert out.dtype == torch.bfloat16
    torch._dynamo.reset()


@pytest.mark.parametrize("backward_pass_autocast", ["same_as_forward", "off"])
def test_rotation_aot_eager_matches_eager_under_autocast(backward_pass_autocast):
    """aot_eager runs AOTAutograd functionalization over the in-place diagonal adds
    and the autocast-disabled region; the forward must match eager exactly.

    Gradients: AOTAutograd traces the backward under the ambient autocast
    ("same_as_forward", the default), which is eager's ``backward()`` called
    *inside* the autocast block, so the rotation's backward matmuls run in bf16
    there. The trainer calls backward outside autocast; only with
    backward_pass_autocast="off" does the compiled backward match that exactly.
    """
    torch._dynamo.reset()
    eager_mods = _compile_modules()
    compiled_mods = _compile_modules()
    for m in (*eager_mods, *compiled_mods):
        m.eval()  # keep the power-iteration state fixed so both runs see the same clip

    def make_fn(m8a, m8b, m16):
        def fn(x24, x32):
            return torch.cat([m8a(x24).flatten(), m16(x32).flatten(), m8b(x24).flatten()])

        return fn

    x24 = torch.randn(2, 5, 24)
    x32 = torch.randn(2, 5, 32)
    upstream = torch.randn(2 * 5 * 24 * 2 + 2 * 5 * 32)
    eager = make_fn(*eager_mods)

    with torch._functorch.config.patch(backward_pass_autocast=backward_pass_autocast):
        compiled = torch.compile(make_fn(*compiled_mods), backend="aot_eager", fullgraph=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out_c = compiled(x24, x32)
            out_e = eager(x24, x32)
            if backward_pass_autocast == "same_as_forward":
                (out_e.float() * upstream).sum().backward()
        if backward_pass_autocast == "off":
            (out_e.float() * upstream).sum().backward()
        (out_c.float() * upstream).sum().backward()

    assert out_c.dtype == out_e.dtype == torch.float32
    torch.testing.assert_close(out_c, out_e, rtol=0, atol=0)
    for mc, me in zip(compiled_mods, eager_mods, strict=True):
        torch.testing.assert_close(mc.weight.grad, me.weight.grad, rtol=0, atol=1e-6)
    torch._dynamo.reset()


def test_checkpointed_hooked_oft_block_compiles_fullgraph_like_training():
    """Mirror of the live path: a non-reentrant checkpoint around a block whose
    Linears are hooked by DoRA-OFT, compiled fullgraph under bf16 autocast
    (CheckpointLayer.compile(fullgraph=True) in checkpointing_util)."""
    torch._dynamo.reset()

    def build():
        torch.manual_seed(12)
        block = nn.Sequential(nn.Linear(24, 32, bias=False), nn.Linear(32, 24, bias=False))
        adapters = [_seeded(DoRAOFTModule, layer, 8) for layer in block]
        for adapter in adapters:
            adapter.oft_R.eval()  # fixed power-iteration state: eager and compiled see the same clip
            adapter.hook_to_module()
        return block, adapters

    def run(block, x):
        return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)

    eager_block, eager_adapters = build()
    compiled_block, compiled_adapters = build()
    counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(run, backend=counter, fullgraph=True)
    x = torch.randn(3, 24)
    upstream = torch.randn(3, 24)

    try:
        with torch._functorch.config.patch(backward_pass_autocast="off"):
            for _ in range(2):
                for adapter in (*eager_adapters, *compiled_adapters):
                    adapter.oft_R.weight.grad = None
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    out_c = compiled(compiled_block, x)
                    out_e = run(eager_block, x)
                (out_c.float() * upstream).sum().backward()
                (out_e.float() * upstream).sum().backward()
    finally:
        for adapter in (*eager_adapters, *compiled_adapters):
            adapter.remove_hook_from_module()

    assert counter.frame_count == 1
    assert out_c.dtype == torch.bfloat16
    torch.testing.assert_close(out_c, out_e, rtol=0, atol=0)
    for ac, ae in zip(compiled_adapters, eager_adapters, strict=True):
        torch.testing.assert_close(ac.oft_R.weight.grad, ae.oft_R.weight.grad, rtol=0, atol=1e-6)
    torch._dynamo.reset()
