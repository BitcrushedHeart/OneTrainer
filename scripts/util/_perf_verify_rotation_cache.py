"""GPU verification for the OFT rotation cache (config oft_rotation_cache; not run in CI).

At Z-Image sizes (per transformer block: 6 x Linear(in 3840, block 768) + 1 x
Linear(in 10240, block 1024), oft_clipped_norm 0.95, S = 4209 tokens) this measures:

(a) VRAM: torch.cuda.memory_allocated deltas of enabling the cache (bf16 leaves) on
    ``--blocks`` blocks, of the bf16 accumulators after one backward, and the fp32
    Q.grad the cache replaces during a window; plus a cuda -> cpu -> cuda round trip
    checking that the leaf keeps its identity and gradient hook (the .data = branch
    of OFTRotationModule._apply that CPU tests cannot reach).
(b) Speed: compiled rotation time per micro-step for one block, forward + non-reentrant
    checkpoint recompute + backward, cache on vs off (Phase-1 per-forward Cayley).
(c) Update step: materialize (dL/dR -> Q.grad through one fp32 R(Q)) + spectral advance
    + cache refresh for ``--blocks`` blocks.
(d) Accuracy: SR-accumulated (bf16) vs fp32-accumulated dL/dQ, relative error over
    ``--micro-steps`` synthetic micro-steps on the 1024-block module (heavy-tailed
    activation channels, lognormal per-micro-step loss scale).

Refuses to run while another process holds more than 4 GiB of VRAM (a live training
run) unless --force is given.

Usage:
    venv/Scripts/python.exe scripts/util/_perf_verify_rotation_cache.py
    venv/Scripts/python.exe scripts/util/_perf_verify_rotation_cache.py --blocks 30 --iters 20
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.module.oft_utils import OFTRotationModule, set_rotation_cache_seed

import torch
import torch._dynamo
import torch.utils.checkpoint

GIB = 1024**3
MIB = 1024**2
VRAM_LIMIT_BYTES = 4 * GIB
# (in_features, block_size) of the adapted Linears in one Z-Image transformer block.
BLOCK_LINEARS = [(3840, 768)] * 6 + [(10240, 1024)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=30, help="transformer blocks for (a) and (c) (default 30)")
    parser.add_argument("--tokens", type=int, default=4209, help="sequence length S (default 4209)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--std", type=float, default=2e-3, help="rotation weight std")
    parser.add_argument("--clip", type=float, default=0.95, help="oft_clipped_norm")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--micro-steps", type=int, default=70, help="micro-steps per window for (d)")
    parser.add_argument("--backend", default="inductor", help="torch.compile backend")
    parser.add_argument("--no-compile", action="store_true", help="time eager instead of torch.compile")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-update", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--force", action="store_true", help="run even if another process holds >4 GiB VRAM")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# VRAM guard (same policy as _perf_verify_oft_rotation.py)
# --------------------------------------------------------------------------- #


def _nvidia_smi(query: str, kind: str) -> list[list[str]] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return [[p.strip() for p in line.split(",")] for line in result.stdout.strip().splitlines() if line.strip()]


def find_vram_hogs(device: torch.device) -> list[str]:
    own_pid = os.getpid()
    hogs: list[str] = []
    known = False
    for row in _nvidia_smi("pid,process_name,used_memory", "compute-apps") or []:
        if len(row) < 3:
            continue
        pid, name, used = row[0], ",".join(row[1:-1]), row[-1]
        if pid.isdigit() and int(pid) == own_pid:
            continue
        if used.isdigit():
            known = True
            if int(used) * MIB > VRAM_LIMIT_BYTES:
                hogs.append(f"pid {pid} ({name}) uses {int(used)} MiB")
    if known:
        return hogs
    gpus = _nvidia_smi("index,memory.used", "gpu")
    if gpus:
        return [
            f"GPU {row[0]} has {int(row[1])} MiB in use (per-process usage unavailable)"
            for row in gpus
            if len(row) == 2 and row[1].isdigit() and int(row[1]) * MIB > VRAM_LIMIT_BYTES
        ]
    free, total = torch.cuda.mem_get_info(device)
    if total - free > VRAM_LIMIT_BYTES:
        hogs.append(f"{device} has {(total - free) / MIB:.0f} MiB in use (nvidia-smi unavailable)")
    return hogs


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _reset(device: torch.device) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def build_modules(blocks: int, args, device: torch.device) -> list[OFTRotationModule]:
    modules = []
    torch.manual_seed(0)
    for _ in range(blocks):
        for in_features, block_size in BLOCK_LINEARS:
            r = in_features // block_size
            module = OFTRotationModule(
                r=r,
                n_elements=block_size * (block_size - 1) // 2,
                block_size=block_size,
                in_features=in_features,
                oft_scaled=True,
                oft_clipped_norm=args.clip,
            ).to(device)
            with torch.no_grad():
                module.weight.normal_(0, args.std)
            module.warm_start_spectral_state()
            module.train()
            modules.append(module)
    return modules


def synthetic_rotation_grad(module: OFTRotationModule, gen: torch.Generator, tokens: int = 256) -> torch.Tensor:
    """sum over tokens of x^T g per block, heavy-tailed channels, lognormal loss scale; bf16."""
    device = module.weight.device
    b, r = module.block_size, module.r
    chan = torch.exp(torch.randn(r, 1, b, device=device, generator=gen))
    chan[..., :3] *= 30.0
    scale = math.exp(torch.randn((), device=device, generator=gen).item())
    x = torch.randn(r, tokens, b, device=device, generator=gen) * chan
    up = torch.randn(r, tokens, b, device=device, generator=gen) * 1e-4 * scale
    return torch.einsum("rtk,rtc->rkc", x, up).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# (a) memory
# --------------------------------------------------------------------------- #


def run_memory(args, device: torch.device) -> None:
    _reset(device)
    modules = build_modules(args.blocks, args, device)
    _reset(device)
    base = torch.cuda.memory_allocated(device)
    for module in modules:
        module.enable_rotation_cache(torch.bfloat16)
    _reset(device)
    leaves = torch.cuda.memory_allocated(device) - base

    gen = torch.Generator(device=device).manual_seed(0)
    set_rotation_cache_seed(0, device)
    base = torch.cuda.memory_allocated(device)
    for module in modules:
        module.rotation_leaf.backward(synthetic_rotation_grad(module, gen, tokens=8))
    _reset(device)
    accumulators = torch.cuda.memory_allocated(device) - base
    q_grad = sum(m.weight.numel() * 4 for m in modules)

    module = modules[-1]
    leaf, hook = module.rotation_leaf, module._rotation_leaf_hook
    module.to("cpu")
    module.to(device)
    same = module.rotation_leaf is leaf and module._rotation_leaf_hook is hook and leaf.device == device
    for m in modules:
        m.materialize_rotation_grad()
    print(f"(a) {len(modules)} rotation modules ({args.blocks} blocks)")
    print(f"    bf16 leaves        {leaves / GIB:7.3f} GiB")
    print(f"    bf16 accumulators  {accumulators / GIB:7.3f} GiB (during a window)")
    print(f"    fp32 Q.grad        {q_grad / GIB:7.3f} GiB (what Phase 1 holds during a window instead)")
    print(f"    net window delta   {(leaves + accumulators - q_grad) / GIB:+7.3f} GiB")
    print(f"    cuda->cpu->cuda keeps leaf identity and hook: {same}\n")
    del modules
    _reset(device)


# --------------------------------------------------------------------------- #
# (b) speed per micro-step
# --------------------------------------------------------------------------- #


def run_speed(args, device: torch.device) -> None:
    results = {}
    for cached in (False, True):
        torch._dynamo.reset()
        _reset(device)
        modules = build_modules(1, args, device)
        if cached:
            for module in modules:
                module.enable_rotation_cache(torch.bfloat16)
        xs = [
            torch.randn(args.batch, args.tokens, m.in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
            for m in modules
        ]
        upstream = [torch.randn_like(x) for x in xs]

        def block(*inputs, _modules=modules):
            return [m(x) for m, x in zip(_modules, inputs, strict=True)]

        def step(*inputs):
            return torch.utils.checkpoint.checkpoint(block, *inputs, use_reentrant=False)

        run = step if args.no_compile else torch.compile(step, backend=args.backend, fullgraph=True)
        fwd, bwd = [], []
        for i in range(args.warmup + args.iters):
            for x in xs:
                x.grad = None
            for m in modules:
                m.weight.grad = None
                m.rotation_grad_acc = None
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = run(*xs)
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            torch.autograd.backward(outputs, upstream)  # includes the checkpoint recompute
            torch.cuda.synchronize(device)
            t2 = time.perf_counter()
            if i >= args.warmup:
                fwd.append((t1 - t0) * 1e3)
                bwd.append((t2 - t1) * 1e3)
            del outputs
        results[cached] = (statistics.median(fwd), statistics.median(bwd))
    print(f"(b) one block, {args.tokens} tokens, {'eager' if args.no_compile else 'compile/' + args.backend}")
    for cached, (f, b) in results.items():
        label = "cache on " if cached else "cache off"
        print(f"    {label}: fwd {f:7.2f} ms  bwd+recompute {b:7.2f} ms  total {f + b:7.2f} ms")
    off, on = sum(results[False]), sum(results[True])
    print(f"    saved per micro-step x {args.blocks} blocks: {(off - on) * args.blocks:.0f} ms\n")


# --------------------------------------------------------------------------- #
# (c) update step
# --------------------------------------------------------------------------- #


def run_update(args, device: torch.device) -> None:
    _reset(device)
    modules = build_modules(args.blocks, args, device)
    gen = torch.Generator(device=device).manual_seed(1)
    set_rotation_cache_seed(1, device)
    for module in modules:
        module.enable_rotation_cache(torch.bfloat16)
        module.rotation_leaf.backward(synthetic_rotation_grad(module, gen, tokens=8))
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    t0 = time.perf_counter()
    scores = [m.materialize_rotation_grad() for m in modules]
    worst = torch.stack(scores).max().item()
    t1 = time.perf_counter()
    for module in modules:
        module.weight.grad = None
        module.advance_spectral_state()
        module.refresh_rotation_cache()
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(device) - base
    print(f"(c) update step over {len(modules)} modules")
    print(f"    materialize {(t1 - t0) * 1e3:8.1f} ms (stale score max {worst:.3f})")
    print(f"    advance + refresh {(t2 - t1) * 1e3:8.1f} ms   peak transient {peak / MIB:.0f} MiB\n")
    del modules
    _reset(device)


# --------------------------------------------------------------------------- #
# (d) SR vs fp32 accumulation
# --------------------------------------------------------------------------- #


def run_accuracy(args, device: torch.device) -> None:
    _reset(device)
    sr_module, fp32_module = build_modules(1, args, device)[-1], build_modules(1, args, device)[-1]
    sr_module.enable_rotation_cache(torch.bfloat16)
    fp32_module.enable_rotation_cache(torch.float32)
    gen = torch.Generator(device=device).manual_seed(2)
    for step in range(args.micro_steps):
        set_rotation_cache_seed(step, device)
        grad = synthetic_rotation_grad(sr_module, gen) / args.micro_steps
        sr_module.rotation_leaf.backward(grad)
        fp32_module.rotation_leaf.backward(grad.float())
    sr_module.materialize_rotation_grad()
    fp32_module.materialize_rotation_grad()
    rel = (
        sr_module.weight.grad.double() - fp32_module.weight.grad.double()
    ).norm() / fp32_module.weight.grad.double().norm()
    print(f"(d) {args.micro_steps} micro-steps on the {sr_module.block_size}-block module:")
    print(f"    dL/dQ relative error, SR bf16 vs fp32 accumulation: {rel.item():.3e}\n")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print(
            "CUDA is not available (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}); this script needs the GPU."
        )
        return 1
    device = torch.device(args.device)
    hogs = find_vram_hogs(device)
    if hogs and not args.force:
        print("Refusing to run: another process is using more than 4 GiB of VRAM:")
        for hog in hogs:
            print(f"  {hog}")
        print("Stop the training run first, or pass --force.")
        return 2

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(device)}")
    print(f"blocks {args.blocks}  tokens {args.tokens}  batch {args.batch}  std {args.std}  clip {args.clip}\n")
    if not args.skip_memory:
        run_memory(args, device)
    if not args.skip_speed:
        run_speed(args, device)
    if not args.skip_update:
        run_update(args, device)
    if not args.skip_accuracy:
        run_accuracy(args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
