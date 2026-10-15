"""Attention-only micro-benchmark for the Z-Image training shapes: DEFAULT vs A vs B.

One file, no model load. Random q/k/v, the mask built the way the transformer builds
it (bool (B, 1, 1, L), right-padded keys, sample 0 unpadded). Three parts:

1. ``op``        -- fwd+bwd of the bare attention call per backend, eager and under
                    torch.compile(dynamic=True): time, peak memory, max abs diff of
                    output and grads vs an fp32 math reference (computed on a 2-sample
                    slice: attention is independent per sample, so the slice is exact),
                    and whether the compiled graph allocates a dense (B, H, L, L) buffer
                    (read from inductor's generated code, not inferred from memory).
                    Backends: default (what the trainer does today: PyTorch picks the
                    kernel), efficient / cudnn / math / flash forced, and ``nocopy``
                    (option B, OneTrainer's custom op). Batch 15 for all, then 18 and
                    20 for cudnn and nocopy.
2. ``processor`` -- diffusers' ZSingleStreamAttnProcessor, the code the trainer runs,
                    configured the way ``apply_attention_backend`` configures it:
                    DEFAULT, NATIVE_CUDNN (A), EFFICIENT_NOCOPY (B); eager and compiled;
                    vs eager DEFAULT.

Usage (VM; defaults are the live run's shapes: batch 15, L = 4096 image + 352 text):

    cd /content/OneTrainer && venv/bin/python scripts/perf/bench_sdpa_backends.py --json /tmp/sdpa.json

    # subsets / smaller
    venv/bin/python scripts/perf/bench_sdpa_backends.py --parts op --batches 15 --iters 3

Option B needs OneTrainer's modules importable (run from the repo root on the branch
that has modules/util/attention_nocopy.py); without it that arm is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from modules.util.attention_nocopy import ZImageNoCopyEfficientAttnProcessor, efficient_attention_nocopy
except Exception as _exc:  # noqa: BLE001
    ZImageNoCopyEfficientAttnProcessor = None
    efficient_attention_nocopy = None
    NOCOPY_IMPORT_ERROR = repr(_exc)
else:
    NOCOPY_IMPORT_ERROR = None

FORCED = {
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
}

# diffusers AttentionBackendName values, as OneTrainer's enum maps them
DIFFUSERS_NAME = {"DEFAULT": None, "NATIVE_CUDNN": "_native_cudnn", "NATIVE_EFFICIENT": "_native_efficient"}


# ----------------------------------------------------------------------------- inputs


def seq_lengths(batch: int, image_tokens: int, max_text: int, seq_multi: int = 32) -> list[int]:
    """Joint lengths per sample the way the transformer pads them: captions to a
    multiple of 32, the batch to its longest. Sample 0 carries the longest caption."""
    lengths = [image_tokens + max_text]
    for i in range(1, batch):
        text = max_text - ((i * 3) % max(1, max_text // seq_multi)) * seq_multi
        lengths.append(image_tokens + max(seq_multi, text))
    return lengths


def build_mask(lengths: list[int], device) -> torch.Tensor:
    """transformer_z_image.py:879-883 then ZSingleStreamAttnProcessor:129-130."""
    bsz, max_len = len(lengths), max(lengths)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
    for i, n in enumerate(lengths):
        attn_mask[i, :n] = 1
    return attn_mask[:, None, None, :]


def make_blhd(batch, seq, heads, head_dim, device, dtype, seed):
    """(B, L, H, D) tensors like the processor holds before its permute."""
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn((batch, seq, heads, head_dim), generator=g, device=device, dtype=dtype).requires_grad_(True)
        for _ in range(3)
    )


# ------------------------------------------------------------------------- the arms


def attention_fn(name: str):
    """(q, k, v, mask) -> out, all (B, L, H, D); mask (B, 1, 1, L) bool or None."""
    if name == "nocopy":
        if efficient_attention_nocopy is None:
            raise RuntimeError(f"nocopy op not importable: {NOCOPY_IMPORT_ERROR}")

        def fn(q, k, v, m):
            return efficient_attention_nocopy(q, k, v, m)

        return fn

    backend = FORCED.get(name)

    def fn(q, k, v, m):
        qh, kh, vh = (t.permute(0, 2, 1, 3) for t in (q, k, v))  # diffusers' _native_attention permute
        if backend is None:
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=m)
        else:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=m)
        return out.permute(0, 2, 1, 3)

    return fn


# -------------------------------------------------------------------- measurement


def sync():
    torch.cuda.synchronize()


def fwd_bwd(fn, q, k, v, m, grad_out):
    out = fn(q, k, v, m)
    out.backward(grad_out)
    return out


def measure(fn, q, k, v, m, grad_out, warmup, iters):
    """ms per fwd+bwd, peak MB above the resident tensors, detached out and grads."""
    for _ in range(warmup):
        fwd_bwd(fn, q, k, v, m, grad_out)
        for t in (q, k, v):
            t.grad = None
    sync()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        out = fwd_bwd(fn, q, k, v, m, grad_out)
        if i != iters - 1:
            for t in (q, k, v):
                t.grad = None
    end.record()
    sync()
    ms = start.elapsed_time(end) / iters
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    grads = tuple(t.grad.detach().clone() for t in (q, k, v))
    for t in (q, k, v):
        t.grad = None
    return ms, peak, out.detach().clone(), grads


def fp32_reference(q, k, v, m, grad_out, n):
    """Math-backend fp32 reference on the first n samples (sample 0 unpadded, the
    rest padded). Exact for those samples: attention never mixes batch elements."""
    qs, ks, vs = (t[:n].detach().float().requires_grad_(True) for t in (q, k, v))
    ms = None if m is None else m[:n]
    with sdpa_kernel(SDPBackend.MATH):
        out = attention_fn("math")(qs, ks, vs, ms)
    grads = torch.autograd.grad(out, (qs, ks, vs), grad_out[:n].float())
    return out.detach(), tuple(g.detach() for g in grads)


def diffs(out, grads, ref, n):
    d_out = (out[:n].float() - ref[0]).abs().max().item()
    d_grad = max((g[:n].float() - r).abs().max().item() for g, r in zip(grads, ref[1], strict=True))
    return d_out, d_grad


_ALLOC_RE = re.compile(r"empty_strided_cuda\(\(([^()]*)\)")


def dense_mask_buffers(codes: list[str], heads: int) -> list[str]:
    """Allocations of shape (b, H, s, s) in inductor's generated code: the dense
    (B, H, L, L) mask copy (or a math-path score matrix)."""
    found = []
    for code in codes:
        for match in _ALLOC_RE.finditer(code):
            dims = [d.strip() for d in match.group(1).split(",") if d.strip()]
            if len(dims) == 4 and dims[1] == str(heads) and dims[2] == dims[3]:
                found.append("(" + ", ".join(dims) + ")")
    return sorted(set(found))


def print_table(rows, header):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda cols: "  ".join(str(c).ljust(w) for c, w in zip(cols, widths, strict=True))  # noqa: E731
    print(line(header))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))


# ---------------------------------------------------------------------- part: op


def run_op(args, device, dtype, results):
    from torch._inductor.utils import run_and_get_code

    print(
        f"\n== op: H={args.heads} D={args.head_dim} L={args.seq} ({args.image_tokens} image + {args.text_tokens} text) {dtype}, "
        f"mask (B,1,1,L) bool, fp32 math reference on {args.ref_samples} samples"
    )
    rows = []
    eager_arms = ["default", "efficient", "cudnn", "math", "flash", "nocopy"]
    compiled_arms = ["default", "cudnn", "nocopy"]
    for batch in args.batches:
        lengths = seq_lengths(batch, args.image_tokens, args.text_tokens)
        mask = build_mask(lengths, device)
        q, k, v = make_blhd(batch, args.seq, args.heads, args.head_dim, device, dtype, args.seed)
        grad_out = torch.randn_like(q)
        ref = fp32_reference(q, k, v, mask, grad_out, args.ref_samples)
        torch.cuda.empty_cache()
        print(f"   batch {batch}: lengths {lengths[:4]}... ({len(set(lengths))} distinct)")

        arms = eager_arms if batch == args.batches[0] else ["cudnn", "nocopy"]
        for mode in ("eager", "compiled"):
            for name in arms:
                if mode == "compiled" and name not in compiled_arms:
                    continue
                record = {"part": "op", "mode": mode, "backend": name, "batch": batch}
                try:
                    fn = attention_fn(name)
                    dense = ""
                    if mode == "compiled":
                        torch._dynamo.reset()
                        compiled = torch.compile(fn, dynamic=True, fullgraph=True)

                        def once(q, k, v, m, go, compiled=compiled):
                            fwd_bwd(compiled, q, k, v, m, go)

                        _, codes = run_and_get_code(once, q, k, v, mask, grad_out)
                        for t in (q, k, v):
                            t.grad = None
                        buffers = dense_mask_buffers(codes, args.heads)
                        dense = "DENSE " + " ".join(buffers) if buffers else "no (B,H,L,L) buffer"
                        record["dense_buffers"] = buffers
                        fn = compiled
                    ms, peak, out, grads = measure(fn, q, k, v, mask, grad_out, args.warmup, args.iters)
                    d_out, d_grad = diffs(out, grads, ref, args.ref_samples)
                    record.update({"ms": ms, "peak_mb": peak, "max_diff_out": d_out, "max_diff_grad": d_grad})
                    rows.append(
                        [batch, mode, name, f"{ms:8.2f}", f"{peak:9.0f}", f"{d_out:.2e}", f"{d_grad:.2e}", dense]
                    )
                    del out, grads
                except Exception as exc:  # noqa: BLE001
                    reason = str(exc).splitlines()[0][:70] if str(exc) else type(exc).__name__
                    record["error"] = traceback.format_exc()[-3000:]
                    rows.append([batch, mode, name, "FAILED", "", "", "", reason])
                    for t in (q, k, v):
                        t.grad = None
                results.append(record)
                torch.cuda.empty_cache()
        del q, k, v, grad_out, mask, ref
        torch.cuda.empty_cache()
    print_table(
        rows, ["batch", "mode", "backend", "fwd+bwd ms", "peak MB", "max|dout|", "max|dgrad|", "compiled graph"]
    )
    b, h, n = args.batches[0], args.heads, args.seq
    print(
        f"   dense (B,H,L,L) bf16 at batch {b} = {b * h * n * n * 2 / 2**20:.0f} MB; 'flash' fails by design (mask present)"
    )
    print(
        "   diffs: bf16 kernel vs fp32 math on the reference samples; compare against the 'math' row (bf16 math) for the dtype floor"
    )


# --------------------------------------------------------------- part: processor


def run_processor(args, device, dtype, results):
    try:
        from diffusers.models.attention_processor import Attention
        from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor
    except Exception as exc:  # noqa: BLE001
        print(f"\n== processor: skipped, diffusers not importable ({exc})")
        return
    batch = args.batches[0]
    dim = args.heads * args.head_dim
    print(f"\n== processor: ZSingleStreamAttnProcessor dim={dim} batch={batch} L={args.seq}, vs eager DEFAULT")
    torch.manual_seed(args.seed)
    attn = Attention(
        query_dim=dim,
        cross_attention_dim=None,
        dim_head=args.head_dim,
        heads=args.heads,
        qk_norm="rms_norm",
        eps=1e-5,
        bias=False,
        out_bias=False,
        processor=ZSingleStreamAttnProcessor(),
    ).to(device=device, dtype=dtype)
    attn.requires_grad_(False)
    lengths = seq_lengths(batch, args.image_tokens, args.text_tokens)
    mask2d = build_mask(lengths, device)[:, 0, 0, :]
    x = torch.randn((batch, args.seq, dim), device=device, dtype=dtype)
    freqs = torch.polar(
        torch.ones((batch, args.seq, args.head_dim // 2), device=device),
        torch.randn((batch, args.seq, args.head_dim // 2), device=device),
    )
    grad_out = torch.randn_like(x)

    arms = ["DEFAULT", "NATIVE_CUDNN", "EFFICIENT_NOCOPY"]

    def install(name):
        if name == "EFFICIENT_NOCOPY":
            if ZImageNoCopyEfficientAttnProcessor is None:
                raise RuntimeError(f"not importable: {NOCOPY_IMPORT_ERROR}")
            attn.set_processor(ZImageNoCopyEfficientAttnProcessor())
        else:
            proc = ZSingleStreamAttnProcessor()
            proc._attention_backend = DIFFUSERS_NAME[name]  # what apply_attention_backend does
            attn.set_processor(proc)

    rows, ref = [], None
    for mode in ("eager", "compiled"):
        for name in arms:
            record = {"part": "processor", "mode": mode, "backend": name, "batch": batch}
            try:
                install(name)
                fn = attn
                if mode == "compiled":
                    torch._dynamo.reset()
                    fn = torch.compile(attn, dynamic=True, fullgraph=True)
                xi = x.detach().clone().requires_grad_(True)
                for _ in range(args.warmup):
                    fn(xi, attention_mask=mask2d, freqs_cis=freqs).backward(grad_out)
                    xi.grad = None
                sync()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                t0 = time.perf_counter()
                for i in range(args.iters):
                    out = fn(xi, attention_mask=mask2d, freqs_cis=freqs)
                    out.backward(grad_out)
                    if i != args.iters - 1:
                        xi.grad = None
                sync()
                ms = (time.perf_counter() - t0) * 1000 / args.iters
                peak = (torch.cuda.max_memory_allocated() - base) / 2**20
                out, grad = out.detach(), xi.grad.detach().clone()
                if ref is None:
                    ref = (out.clone(), grad.clone())
                d_out = (out.float() - ref[0].float()).abs().max().item()
                d_grad = (grad.float() - ref[1].float()).abs().max().item()
                record.update({"ms": ms, "peak_mb": peak, "max_diff_out": d_out, "max_diff_grad": d_grad})
                rows.append([mode, name, f"{ms:8.2f}", f"{peak:9.0f}", f"{d_out:.2e}", f"{d_grad:.2e}", ""])
                del out, grad, xi
            except Exception as exc:  # noqa: BLE001
                reason = str(exc).splitlines()[0][:70] if str(exc) else type(exc).__name__
                record["error"] = traceback.format_exc()[-3000:]
                rows.append([mode, name, "FAILED", "", "", "", reason])
            results.append(record)
            torch.cuda.empty_cache()
    print_table(rows, ["mode", "backend", "fwd+bwd ms", "peak MB", "max|dout|", "max|dgrad|", "note"])
    print(
        "   (diffs vs eager DEFAULT; includes the projections, qk norm and RoPE, i.e. the trainer's attention module)"
    )


# ------------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", default="15,18,20", help="first value runs every arm; the rest run cudnn and nocopy")
    p.add_argument("--heads", type=int, default=30)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--image-tokens", type=int, default=4096, help="64x64 latent patches at 1024x1024")
    p.add_argument("--text-tokens", type=int, default=352, help="longest caption, multiple of 32 (352 -> L=4448)")
    p.add_argument("--parts", default="op,processor")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--ref-samples", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    args.seq = args.image_tokens + args.text_tokens
    args.batches = [int(b) for b in args.batches.split(",")]

    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark needs a GPU", file=sys.stderr)
        return 2
    device, dtype = torch.device("cuda"), getattr(torch, args.dtype)
    props = torch.cuda.get_device_properties(device)
    print(
        f"torch {torch.__version__}  {props.name}  sm_{props.major}{props.minor}  {props.total_memory / 2**30:.0f} GiB"
    )
    print("default SDPA priority:", [SDPBackend(i).name for i in torch._C._get_sdp_priority_order() if i >= 0])
    print(
        f"free VRAM: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB; nocopy op: {'ok' if NOCOPY_IMPORT_ERROR is None else NOCOPY_IMPORT_ERROR}"
    )
    print(f"pid {os.getpid()}")

    results = []
    t0 = time.perf_counter()
    parts = set(args.parts.split(","))
    if "op" in parts:
        run_op(args, device, dtype, results)
    if "processor" in parts:
        run_processor(args, device, dtype, results)
    print(f"\ntotal {time.perf_counter() - t0:.0f} s")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {"args": vars(args), "device": props.name, "torch": torch.__version__, "results": results}, f, indent=2
            )
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
