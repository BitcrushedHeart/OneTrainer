"""Memory-efficient SDPA that torch.compile cannot turn into a dense mask.

Z-Image hands F.scaled_dot_product_attention a (B, 1, 1, L) boolean key-padding mask
whenever the samples in a batch have different joint sequence lengths (every real
batch: captions are stripped per prompt and re-padded only to a multiple of 32).
PyTorch picks the memory-efficient kernel for that call and, inside the composite
op, converts the mask to an additive bias and `expand`s it to (B, H, L, L) as a
stride-0 view. Under torch.compile that expand is part of the graph, and inductor's
stride constraint for `_scaled_dot_product_efficient_attention_backward` copies the
view into a dense (B, H, L, L) buffer (torch/_inductor/lowering.py `sdpa_constraint`
keeps expanded dims only for the *forward* op's bias): 16.6 GiB at batch 15, 30
heads, 4448 tokens, in every layer's compiled backward.

This module wraps the same two aten kernels in custom ops. The expand happens inside
the op, so inductor only ever sees the (B, 1, 1, L) mask and has nothing to
materialise. The ops work in (B, L, H, D), the kernels' native layout and the one the
diffusers processor already holds, so no extra permute copies are made either. On
CPU (tests) the ops fall back to F.scaled_dot_product_attention with a recomputing
backward, so the contract can be checked without a GPU.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor

# attention.cpp preprocess_mask: the bias' second-to-last stride must be a multiple of 8
_BIAS_ALIGNMENT = 8
# the efficient kernel's logsumexp is padded to a multiple of 32 along the sequence
_LSE_ALIGNMENT = 32


def _additive_bias(attn_mask: Tensor, dtype: torch.dtype, batch: int, heads: int, seq_q: int, seq_k: int) -> Tensor:
    """A (B, 1, 1, Lk) bool (True = attend) or float mask as an additive bias in `dtype`,
    aligned the way PyTorch's own preprocess_mask aligns it, expanded to (B, H, Lq, Lk)
    as a stride-0 view."""
    if attn_mask.dtype == torch.bool:
        bias = torch.zeros(attn_mask.shape, dtype=dtype, device=attn_mask.device)
        bias = bias.masked_fill(~attn_mask, float("-inf"))
    else:
        bias = attn_mask.to(dtype)
    last = bias.shape[-1]
    pad = (-last) % _BIAS_ALIGNMENT
    if pad:
        bias = F.pad(bias, (0, pad))[..., :last]
    return bias.expand(batch, heads, seq_q, seq_k)


def _lse_dim(seq_q):
    return (seq_q + _LSE_ALIGNMENT - 1) // _LSE_ALIGNMENT * _LSE_ALIGNMENT


def _reference(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None, scale: float | None) -> Tensor:
    """F.scaled_dot_product_attention on (B, L, H, D) tensors; the CPU path and the
    definition the CUDA path is held to."""
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale).transpose(1, 2).contiguous()


@torch.library.custom_op("onetrainer::sdpa_efficient_nocopy_fwd", mutates_args=())
def _nocopy_fwd(
    query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None, scale: float | None
) -> tuple[Tensor, Tensor]:
    batch, seq_q, heads, _ = query.shape
    seq_k = key.shape[1]
    if query.is_cuda:
        bias = None if attn_mask is None else _additive_bias(attn_mask, query.dtype, batch, heads, seq_q, seq_k)
        out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            bias,
            True,  # compute_log_sumexp: the backward needs it
            0.0,  # dropout_p
            False,  # is_causal
            scale=scale,
        )
        # the kernel writes (B, L, H, D) storage and returns its (B, H, L, D) transpose
        return out.transpose(1, 2).contiguous(), lse
    out = _reference(query, key, value, attn_mask, scale)
    lse = torch.zeros((batch, heads, _lse_dim(seq_q)), dtype=torch.float32, device=query.device)
    return out, lse


@_nocopy_fwd.register_fake
def _nocopy_fwd_fake(query, key, value, attn_mask, scale):
    batch, seq_q, heads, _ = query.shape
    out = query.new_empty((batch, seq_q, heads, value.shape[-1]))
    lse = query.new_empty((batch, heads, _lse_dim(seq_q)), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("onetrainer::sdpa_efficient_nocopy_bwd", mutates_args=())
def _nocopy_bwd(
    grad_out: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    out: Tensor,
    lse: Tensor,
    scale: float | None,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, seq_q, heads, _ = query.shape
    seq_k = key.shape[1]
    if query.is_cuda:
        bias = None if attn_mask is None else _additive_bias(attn_mask, query.dtype, batch, heads, seq_q, seq_k)
        # philox seed/offset are only read when dropout_p > 0
        philox = torch.empty((), dtype=torch.long, device=query.device)
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        # out and grad_out must be (B, H, L, D) views of (B, L, H, D)-contiguous storage
        # (pytorch/pytorch#138772); `out` is, by construction of the forward.
        o = out.contiguous().transpose(1, 2)
        go = grad_out.contiguous().transpose(1, 2)
        dq, dk, dv, _ = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            go, q, k, v, bias, o, lse, philox, philox, 0.0, [True, True, True, False], False, scale=scale
        )
        return dq.transpose(1, 2).contiguous(), dk.transpose(1, 2).contiguous(), dv.transpose(1, 2).contiguous()
    return _reference_backward(grad_out, query, key, value, attn_mask, scale)


def _reference_backward(grad_out, query, key, value, attn_mask, scale):
    """Closed-form SDPA backward (custom-op kernels run below autograd, so the CPU
    path cannot just differentiate `_reference`). fp32 math on (B, H, L, D) views."""
    dtype = query.dtype
    batch, seq_q, heads, head_dim = query.shape
    seq_k = key.shape[1]
    scale = head_dim**-0.5 if scale is None else scale
    q, k, v, go = (t.transpose(1, 2).float() for t in (query, key, value, grad_out))
    scores = (q @ k.transpose(-1, -2)) * scale
    if attn_mask is not None:
        scores = scores + _additive_bias(attn_mask, torch.float32, batch, heads, seq_q, seq_k)
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    d_probs = go @ v.transpose(-1, -2)
    d_scores = probs * (d_probs - (go * out).sum(dim=-1, keepdim=True))
    dq = (d_scores @ k) * scale
    dk = (d_scores.transpose(-1, -2) @ q) * scale
    dv = probs.transpose(-1, -2) @ go
    return tuple(g.transpose(1, 2).to(dtype).contiguous() for g in (dq, dk, dv))


@_nocopy_bwd.register_fake
def _nocopy_bwd_fake(grad_out, query, key, value, attn_mask, out, lse, scale):
    return (
        torch.empty_like(query, memory_format=torch.contiguous_format),
        torch.empty_like(key, memory_format=torch.contiguous_format),
        torch.empty_like(value, memory_format=torch.contiguous_format),
    )


def _setup_context(ctx, inputs, output):
    query, key, value, attn_mask, scale = inputs
    out, lse = output
    tensors = [query, key, value, out, lse]
    ctx.has_mask = attn_mask is not None
    if ctx.has_mask:
        tensors.append(attn_mask)
    ctx.save_for_backward(*tensors)
    ctx.scale = scale


def _backward(ctx, grad_out, grad_lse):
    query, key, value, out, lse, *rest = ctx.saved_tensors
    attn_mask = rest[0] if ctx.has_mask else None
    dq, dk, dv = torch.ops.onetrainer.sdpa_efficient_nocopy_bwd(
        grad_out, query, key, value, attn_mask, out, lse, ctx.scale
    )
    return dq, dk, dv, None, None


torch.library.register_autograd("onetrainer::sdpa_efficient_nocopy_fwd", _backward, setup_context=_setup_context)


def efficient_attention_nocopy(
    query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None, scale: float | None = None
) -> Tensor:
    """Scaled dot product attention on (B, L, H, D) tensors, returning (B, L, H, D).

    `attn_mask` is None or (B, 1, 1, Lk): bool (True = attend) or additive float. No
    dropout, not causal: the only form Z-Image uses."""
    if attn_mask is not None and attn_mask.ndim != 4:
        raise ValueError(f"attn_mask must be (B, 1, 1, Lk), got shape {tuple(attn_mask.shape)}")
    out, _ = torch.ops.onetrainer.sdpa_efficient_nocopy_fwd(query, key, value, attn_mask, scale)
    return out


def _apply_rotary_emb(x_in: Tensor, freqs_cis: Tensor) -> Tensor:
    # identical to the closure inside diffusers' ZSingleStreamAttnProcessor.__call__
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageNoCopyEfficientAttnProcessor(ZSingleStreamAttnProcessor):
    """diffusers' ZSingleStreamAttnProcessor with the attention call routed through
    `efficient_attention_nocopy` instead of `dispatch_attention_fn`. Everything else
    (projections, qk norm, RoPE, output projection) is kept line for line; keep it in
    step with transformer_z_image.py when diffusers is bumped."""

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
    ) -> Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            query = _apply_rotary_emb(query, freqs_cis)
            key = _apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # [batch, seq_len] -> [batch, 1, 1, seq_len]; the op broadcasts it internally
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = efficient_attention_nocopy(query, key, value, attention_mask)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output
