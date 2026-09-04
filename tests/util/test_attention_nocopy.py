"""CPU tests for the EFFICIENT_NOCOPY attention backend.

The CUDA kernels are not exercised here (the op falls back to
F.scaled_dot_product_attention on CPU); what is checked is the op's contract (schema,
fake impl, autograd registration via torch.library.opcheck), that the fallback matches
SDPA for every mask form Z-Image produces, that the processor swap reaches every
attention module and reverses, that the swapped processor is numerically identical
to diffusers' own through the whole transformer (batch padding mask, caption refiner
mask, and no mask), and that it traces under dynamo with dynamic shapes.
"""

# Plain-import form so isort keeps the priming import before the direct
# modelSetup imports (see test_attention_backend.py).
import modules.util.create  # noqa: F401 -- import-side-effect required
from modules.modelSetup.ZImageLoRASetup import ZImageLoRASetup
from modules.util.attention_backend_util import apply_attention_backend
from modules.util.attention_nocopy import ZImageNoCopyEfficientAttnProcessor, efficient_attention_nocopy
from modules.util.enum.AttentionBackend import AttentionBackend

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

import pytest

# the tests directory is not a package: pytest puts tests/util on sys.path
from test_attention_backend import (  # noqa: I001
    EXPECTED_PROCESSORS,
    _build_tiny_transformer,
    _make_setup_config,
    _make_setup_model,
    _processors,
)

# ------------------------------------------------------------------ the op itself


def _qkv(batch=2, seq=11, heads=3, head_dim=8, seed=0, requires_grad=True):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn((batch, seq, heads, head_dim), generator=generator, dtype=torch.float32).requires_grad_(
            requires_grad
        )
        for _ in range(3)
    )


def _key_padding_mask(lengths, seq):
    mask = torch.zeros((len(lengths), seq), dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    return mask[:, None, None, :]


MASKS = {
    "none": lambda seq: None,
    "bool_padded": lambda seq: _key_padding_mask([seq, seq - 3], seq),
    "bool_all_true": lambda seq: _key_padding_mask([seq, seq], seq),
    "float_additive": lambda seq: _key_padding_mask([seq, seq - 5], seq).logical_not().float() * -1e9,
}


def _sdpa_reference(q, k, v, mask, scale=None):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=scale
    )
    return out.transpose(1, 2)


@pytest.mark.parametrize("mask_name", list(MASKS), ids=str)
def test_op_matches_sdpa_forward_and_backward(mask_name):
    seq = 11
    q, k, v = _qkv(seq=seq)
    mask = MASKS[mask_name](seq)
    grad_out = torch.randn((2, seq, 3, 8), generator=torch.Generator().manual_seed(1))

    out = efficient_attention_nocopy(q, k, v, mask)
    grads = torch.autograd.grad(out, (q, k, v), grad_out)

    ref = _sdpa_reference(q, k, v, mask)
    ref_grads = torch.autograd.grad(ref, (q, k, v), grad_out)

    assert out.shape == (2, seq, 3, 8)
    assert out.is_contiguous()
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    for g, r in zip(grads, ref_grads, strict=True):
        assert torch.allclose(g, r, atol=1e-6, rtol=1e-6)


def test_op_honours_scale():
    seq = 9
    q, k, v = _qkv(seq=seq, requires_grad=False)
    mask = MASKS["bool_padded"](seq)
    assert torch.allclose(
        efficient_attention_nocopy(q, k, v, mask, scale=0.3), _sdpa_reference(q, k, v, mask, scale=0.3), atol=1e-6
    )
    assert not torch.allclose(
        efficient_attention_nocopy(q, k, v, mask, scale=0.3), _sdpa_reference(q, k, v, mask), atol=1e-3
    )


def test_op_rejects_a_mask_that_is_not_4d():
    q, k, v = _qkv(requires_grad=False)
    with pytest.raises(ValueError, match=r"\(B, 1, 1, Lk\)"):
        efficient_attention_nocopy(q, k, v, torch.ones((2, 11), dtype=torch.bool))


@pytest.mark.parametrize("mask_name", ["none", "bool_padded"], ids=str)
def test_opcheck_forward_op(mask_name):
    """torch.library.opcheck validates the schema, the fake impl against the real
    outputs, and the autograd registration (the parts torch.compile relies on)."""
    seq = 11
    q, k, v = _qkv(seq=seq)
    mask = MASKS[mask_name](seq)
    torch.library.opcheck(torch.ops.onetrainer.sdpa_efficient_nocopy_fwd, (q, k, v, mask, None))


def test_opcheck_backward_op():
    seq = 11
    q, k, v = _qkv(seq=seq, requires_grad=False)
    mask = MASKS["bool_padded"](seq)
    out, lse = torch.ops.onetrainer.sdpa_efficient_nocopy_fwd(q, k, v, mask, None)
    grad_out = torch.randn_like(out)
    torch.library.opcheck(torch.ops.onetrainer.sdpa_efficient_nocopy_bwd, (grad_out, q, k, v, mask, out, lse, None))


def test_lse_shape_is_padded_to_32_like_the_cuda_kernel():
    q, k, v = _qkv(seq=11, requires_grad=False)
    _, lse = torch.ops.onetrainer.sdpa_efficient_nocopy_fwd(q, k, v, None, None)
    assert lse.shape == (2, 3, 32)
    assert lse.dtype == torch.float32


@pytest.mark.parametrize("mask_name", ["none", "bool_padded"], ids=str)
def test_op_traces_under_dynamo_with_dynamic_shapes(mask_name):
    """aot_eager runs dynamo + AOTAutograd (fake tensors, the registered autograd)
    without inductor codegen, so it works on any CPU. fullgraph=True means a graph
    break inside the op would fail the test, as it would fail OneTrainer's
    `compile(fullgraph=True)` on the transformer blocks."""
    torch._dynamo.reset()

    def fn(q, k, v, mask):
        return efficient_attention_nocopy(q, k, v, mask)

    compiled = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)
    for seq in (11, 16):
        q, k, v = _qkv(seq=seq)
        mask = MASKS[mask_name](seq)
        grad_out = torch.randn((2, seq, 3, 8), generator=torch.Generator().manual_seed(seq))
        out = compiled(q, k, v, mask)
        grads = torch.autograd.grad(out, (q, k, v), grad_out)
        ref = _sdpa_reference(q, k, v, mask)
        ref_grads = torch.autograd.grad(ref, (q, k, v), grad_out)
        assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
        for g, r in zip(grads, ref_grads, strict=True):
            assert torch.allclose(g, r, atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------- the processor


def _attention_module(dim=24, heads=3, seed=0) -> Attention:
    torch.manual_seed(seed)
    return Attention(
        query_dim=dim,
        cross_attention_dim=None,
        dim_head=dim // heads,
        heads=heads,
        qk_norm="rms_norm",
        eps=1e-5,
        bias=False,
        out_bias=False,
        processor=ZSingleStreamAttnProcessor(),
    )


def _processor_inputs(batch=2, seq=10, dim=24, head_dim=8, seed=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((batch, seq, dim), generator=generator)
    freqs = torch.polar(
        torch.ones((batch, seq, head_dim // 2)), torch.randn((batch, seq, head_dim // 2), generator=generator)
    )
    mask2d = _key_padding_mask([seq, seq - 4], seq)[:, 0, 0, :]
    return x, freqs, mask2d


@pytest.mark.parametrize("with_mask", [True, False], ids=["mask", "no_mask"])
def test_processor_matches_diffusers_processor(with_mask):
    attn = _attention_module()
    x, freqs, mask2d = _processor_inputs()
    mask = mask2d if with_mask else None

    def run(processor):
        attn.set_processor(processor)
        attn.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        out = attn(xi, attention_mask=mask, freqs_cis=freqs)
        out.square().mean().backward()
        return out.detach(), xi.grad.detach().clone(), {n: p.grad.detach().clone() for n, p in attn.named_parameters()}

    ref_out, ref_dx, ref_dw = run(ZSingleStreamAttnProcessor())
    out, dx, dw = run(ZImageNoCopyEfficientAttnProcessor())

    assert torch.allclose(out, ref_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(dx, ref_dx, atol=1e-6, rtol=1e-6)
    assert ref_dw.keys() == dw.keys() and len(dw) >= 4
    for name in ref_dw:
        assert torch.allclose(dw[name], ref_dw[name], atol=1e-6, rtol=1e-6), name


def test_processor_traces_under_dynamo():
    torch._dynamo.reset()
    attn = _attention_module()
    attn.set_processor(ZImageNoCopyEfficientAttnProcessor())
    x, freqs, mask2d = _processor_inputs()
    compiled = torch.compile(attn, backend="aot_eager", dynamic=True, fullgraph=True)
    xi = x.clone().requires_grad_(True)
    out = compiled(xi, attention_mask=mask2d, freqs_cis=freqs)
    out.square().mean().backward()
    attn.set_processor(ZSingleStreamAttnProcessor())
    xr = x.clone().requires_grad_(True)
    ref = attn(xr, attention_mask=mask2d, freqs_cis=freqs)
    ref.square().mean().backward()
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    assert torch.allclose(xi.grad, xr.grad, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------- apply / transformer


def test_apply_swaps_every_processor_and_any_other_backend_swaps_back():
    model = _build_tiny_transformer()
    assert apply_attention_backend(model, AttentionBackend.EFFICIENT_NOCOPY) == EXPECTED_PROCESSORS
    processors = _processors(model)
    assert len(processors) == EXPECTED_PROCESSORS
    assert all(type(p) is ZImageNoCopyEfficientAttnProcessor for p in processors)
    assert all(p._attention_backend is None for p in processors)

    assert apply_attention_backend(model, AttentionBackend.NATIVE_MATH) == EXPECTED_PROCESSORS
    processors = _processors(model)
    assert all(type(p) is ZSingleStreamAttnProcessor for p in processors)
    assert all(p._attention_backend == "_native_math" for p in processors)

    assert apply_attention_backend(model, AttentionBackend.EFFICIENT_NOCOPY) == EXPECTED_PROCESSORS
    assert apply_attention_backend(model, AttentionBackend.DEFAULT) == EXPECTED_PROCESSORS
    processors = _processors(model)
    assert all(type(p) is ZSingleStreamAttnProcessor for p in processors)
    assert all(p._attention_backend is None for p in processors)


def test_apply_is_idempotent():
    model = _build_tiny_transformer()
    apply_attention_backend(model, AttentionBackend.EFFICIENT_NOCOPY)
    first = [id(p) for p in _processors(model)]
    apply_attention_backend(model, AttentionBackend.EFFICIENT_NOCOPY)
    assert [id(p) for p in _processors(model)] == first


def test_apply_refuses_a_non_zimage_processor():
    from diffusers.models.attention_processor import AttnProcessor2_0

    attn = _attention_module()
    attn.set_processor(AttnProcessor2_0())
    attn.processor._attention_backend = None  # make it look like a dispatch-aware processor
    with pytest.raises(ValueError, match="only implemented for Z-Image"):
        apply_attention_backend(attn, AttentionBackend.EFFICIENT_NOCOPY)


def _batch_inputs(cap_lengths, seed=1):
    """Same-size latents, captions of the given lengths: with different lengths the
    transformer builds the (B, L) batch padding mask for the main layers and the
    caption refiner (transformer_z_image.py _build_unified_sequence / _prepare_sequence)."""
    generator = torch.Generator().manual_seed(seed)
    latents = [torch.randn(4, 1, 8, 8, generator=generator) for _ in cap_lengths]
    caps = [torch.randn(n, 16, generator=generator) for n in cap_lengths]
    timestep = torch.full((len(cap_lengths),), 0.3)
    return latents, timestep, caps


def _forward_backward(model, backend, cap_lengths):
    apply_attention_backend(model, backend)
    model.zero_grad(set_to_none=True)
    latents, timestep, caps = _batch_inputs(cap_lengths)
    latents = [latent.clone().requires_grad_(True) for latent in latents]
    output = model(latents, timestep, caps, return_dict=True).sample
    loss = torch.stack(output, dim=0).square().mean()
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    for i, latent in enumerate(latents):
        grads[f"__input_{i}__"] = latent.grad.detach().clone()
    return torch.stack(output, dim=0).detach(), grads


@pytest.mark.parametrize(
    "cap_lengths",
    [(7, 40), (7, 7), (7,)],
    ids=["batch_padded", "batch_equal_lengths", "batch_1"],
)
def test_transformer_matches_default_backend(cap_lengths, monkeypatch):
    """Caption lengths 7 and 40 pad to 32 and 64 tokens, so both the caption refiner
    and the main layers see a mask; 7 and 7 pad to the same length (no mask)."""
    from diffusers.models.transformers import transformer_z_image

    seen_masks = []
    original = transformer_z_image.dispatch_attention_fn

    def spy(query, key, value, attn_mask=None, **kwargs):
        seen_masks.append(attn_mask is not None)
        return original(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(transformer_z_image, "dispatch_attention_fn", spy)

    # caption positions run from 1, so a 64-token padded caption needs a longer RoPE axis
    model = _build_tiny_transformer(axes_lens=[128, 32, 32])
    ref_out, ref_grads = _forward_backward(model, AttentionBackend.DEFAULT, cap_lengths)
    assert len(seen_masks) == EXPECTED_PROCESSORS
    assert any(seen_masks) == (len(set(cap_lengths)) > 1), "mask presence did not match the batch shape"

    out, grads = _forward_backward(model, AttentionBackend.EFFICIENT_NOCOPY, cap_lengths)
    assert len(seen_masks) == EXPECTED_PROCESSORS, "EFFICIENT_NOCOPY must not go through dispatch_attention_fn"

    assert torch.allclose(out, ref_out, atol=1e-5, rtol=1e-5), f"max abs diff {(out - ref_out).abs().max().item():.3e}"
    assert grads.keys() == ref_grads.keys()
    assert len(grads) > len(cap_lengths)
    for name, ref_grad in ref_grads.items():
        assert torch.allclose(grads[name], ref_grad, atol=1e-5, rtol=1e-5), (
            f"grad {name} max abs diff {(grads[name] - ref_grad).abs().max().item():.3e}"
        )


def test_setup_optimizations_applies_nocopy():
    transformer = _build_tiny_transformer()
    model = _make_setup_model(transformer)
    setup = ZImageLoRASetup(torch.device("cpu"), torch.device("cpu"), False)

    setup.setup_optimizations(model, _make_setup_config(AttentionBackend.EFFICIENT_NOCOPY))

    processors = _processors(transformer)
    assert len(processors) == EXPECTED_PROCESSORS
    assert all(type(p) is ZImageNoCopyEfficientAttnProcessor for p in processors)
