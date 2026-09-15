"""GPU verification for the OFT rotation memory / precision change (not run in CI).

Measures, at Z-Image sizes (per transformer block: 6 x Linear(in 3840, block 768)
+ 1 x Linear(in 10240, block 1024), oft_clipped_norm 0.99, weights ~ N(0, 2e-4),
S = 4209 tokens):

(a) VRAM: torch.cuda.memory_allocated delta of constructing ``--blocks`` blocks of
    OFTRotationModules on CPU and moving them to CUDA (as LoRAModuleWrapper.to
    does), with the shared triu index table (NEW) vs the old per-module
    non-persistent ``rows``/``cols`` buffers (OLD, re-registered inline here).
(b) Speed: compiled Cayley forward / backward per block for OLD (bf16 autocast,
    three-bmm Neumann, re-implemented inline) vs NEW (fp32 with autocast off,
    two-matmul Horner; the module code), plus the full rotation forward+backward
    on activations, and an estimate per micro-step:
    blocks x (forward + checkpoint recompute + backward).

The compiled backward is traced under the ambient autocast by default
(torch._functorch.config.backward_pass_autocast = "same_as_forward"), so NEW's
backward matmuls run in bf16 unless that is set to "off"; both are timed.

Refuses to run while another process holds more than 4 GiB of VRAM (a live
training run) unless --force is given.

Usage:
    venv/Scripts/python.exe scripts/util/_perf_verify_oft_rotation.py
    venv/Scripts/python.exe scripts/util/_perf_verify_oft_rotation.py --iters 20 --tf32
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.module import oft_utils
from modules.module.oft_utils import OFTRotationModule

import torch
import torch._dynamo
import torch._functorch.config

GIB = 1024**3
MIB = 1024**2
VRAM_LIMIT_BYTES = 4 * GIB
# (in_features, block_size) of the adapted Linears in one Z-Image transformer block.
BLOCK_LINEARS = [(3840, 768)] * 6 + [(10240, 1024)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--blocks", type=int, default=30, help="transformer blocks (default 30)")
    parser.add_argument("--tokens", type=int, default=4209, help="sequence length S (default 4209)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--std", type=float, default=2e-4, help="rotation weight std")
    parser.add_argument("--clip", type=float, default=0.99, help="oft_clipped_norm")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--backend", default="inductor", help="torch.compile backend")
    parser.add_argument("--no-compile", action="store_true", help="time eager instead of torch.compile")
    parser.add_argument("--tf32", action="store_true", help="allow TF32 matmul (the live bf16 run keeps it off)")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--force", action="store_true", help="run even if another process holds >4 GiB VRAM")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# VRAM guard
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
    """Describe other VRAM users above the limit. Per-process via nvidia-smi when it
    reports usage (often [N/A] under Windows WDDM), else per-GPU totals, else torch."""
    own_pid = os.getpid()
    hogs: list[str] = []

    apps = _nvidia_smi("pid,process_name,used_memory", "compute-apps")
    known = False
    for row in apps or []:
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
# OLD implementation, re-created inline
# --------------------------------------------------------------------------- #


class OldBufferRotation(OFTRotationModule):
    """Pre-change registration: private non-persistent int64 rows/cols buffers per module."""

    def _bind_triu_indices(self, device: torch.device) -> None:
        if "rows" in self._buffers:
            return  # buffers follow .to() on their own, one private copy per module
        rows, cols = torch.triu_indices(self.block_size, self.block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)


def old_cayley(m: OFTRotationModule) -> torch.Tensor:
    """Pre-change forward Cayley: runs under the caller's autocast, three bmm."""
    Q = m.weight
    b = Q.shape[0]
    block = m.block_size
    Q_skew = m._pytorch_skew_symmetric(Q, block)
    if m.oft_clipped_norm is not None:
        v_vec, u_vec = m._spectral_norm(Q_skew)
        u_raw = torch.bmm(Q_skew, v_vec)
        sigma = torch.sum(u_vec * u_raw, dim=1, keepdim=True)
        Q_skew = Q_skew * (m.oft_clipped_norm / torch.clamp(sigma, min=m.oft_clipped_norm))
    R = torch.eye(block, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
    R.add_(Q_skew, alpha=2.0)
    Q_squared = torch.bmm(Q_skew, Q_skew)
    R.add_(Q_squared, alpha=2.0)
    Q_power = torch.bmm(Q_squared, Q_skew)
    R.add_(Q_power, alpha=2.0)
    Q_power = torch.bmm(Q_power, Q_skew)
    R.add_(Q_power)
    return R.to(Q.dtype)


def old_rotate(m: OFTRotationModule, x: torch.Tensor) -> torch.Tensor:
    """Pre-change OFTRotationModule.forward (x upcast to the weight dtype)."""
    required_dtype = x.dtype
    x = x.to(m.weight.dtype)
    R = old_cayley(m)
    x_reshaped = x.reshape(*x.shape[:-1], m.r, m.block_size)
    y = torch.einsum("...rk,rkc->...rc", x_reshaped, R)
    return y.reshape(x.shape).to(required_dtype)


def new_cayley(m: OFTRotationModule) -> torch.Tensor:
    return m.cayley_rotation_full_precision()


def new_rotate(m: OFTRotationModule, x: torch.Tensor) -> torch.Tensor:
    return m(x)


# --------------------------------------------------------------------------- #
# (a) memory
# --------------------------------------------------------------------------- #


def iter_block_modules(cls: type[OFTRotationModule], blocks: int, clip: float) -> Iterator[OFTRotationModule]:
    """Rotation modules for ``blocks`` transformer blocks, built lazily on CPU."""
    for _ in range(blocks):
        for in_features, block in BLOCK_LINEARS:
            yield cls(
                r=in_features // block,
                n_elements=block * (block - 1) // 2,
                block_size=block,
                in_features=in_features,
                oft_clipped_norm=clip,
            )


def _reset_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def measure_construction(cls: type[OFTRotationModule], args, device: torch.device) -> dict:
    for key in [k for k in oft_utils._TRIU_INDEX_CACHE if k[1].type == "cuda"]:
        del oft_utils._TRIU_INDEX_CACHE[key]  # count the shared table's own allocation too
    _reset_cuda(device)
    before = torch.cuda.memory_allocated(device)
    # built on CPU one at a time, then .to(device), like LoRAModuleWrapper.to
    modules = [m.to(device) for m in iter_block_modules(cls, args.blocks, args.clip)]
    torch.cuda.synchronize(device)
    delta = torch.cuda.memory_allocated(device) - before

    index_ptrs = {}
    state_bytes = 0
    for m in modules:
        for t in (m.rows, m.cols):
            index_ptrs[t.data_ptr()] = t.numel() * t.element_size()
        for name, t in [*m.named_parameters(), *m.named_buffers()]:
            if name not in ("rows", "cols"):
                state_bytes += t.numel() * t.element_size()
    del modules
    _reset_cuda(device)
    return {"delta": delta, "index": sum(index_ptrs.values()), "state": state_bytes}


# --------------------------------------------------------------------------- #
# (b) speed
# --------------------------------------------------------------------------- #


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def time_variant(
    label: str,
    fn: Callable[[], list[torch.Tensor]],
    modules: list[OFTRotationModule],
    upstream: list[torch.Tensor],
    leaves: list[torch.Tensor],
    args,
    device: torch.device,
    *,
    backward_pass_autocast: str = "same_as_forward",
) -> dict:
    torch._dynamo.reset()
    run = fn if args.no_compile else torch.compile(fn, backend=args.backend, fullgraph=True)
    fwd_ms, bwd_ms = [], []
    peak = 0
    with torch._functorch.config.patch(backward_pass_autocast=backward_pass_autocast):
        for i in range(args.warmup + args.iters):
            for m in modules:
                m.weight.grad = None
            for leaf in leaves:
                leaf.grad = None
            measured = i >= args.warmup
            if measured:
                _sync(device)
                torch.cuda.reset_peak_memory_stats(device)
                base_alloc = torch.cuda.memory_allocated(device)
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = run()
            _sync(device)
            t1 = time.perf_counter()
            torch.autograd.backward(outputs, upstream)
            _sync(device)
            t2 = time.perf_counter()
            if measured:
                fwd_ms.append((t1 - t0) * 1e3)
                bwd_ms.append((t2 - t1) * 1e3)
                peak = max(peak, torch.cuda.max_memory_allocated(device) - base_alloc)
            del outputs
    return {
        "label": label,
        "fwd": statistics.median(fwd_ms),
        "bwd": statistics.median(bwd_ms),
        "peak": peak,
    }


def run_speed(args, device: torch.device) -> list[dict]:
    torch.manual_seed(0)
    modules = [m.to(device) for m in iter_block_modules(OFTRotationModule, 1, args.clip)]
    for m in modules:
        with torch.no_grad():
            m.weight.normal_(0, args.std)
        m.train()

    r_upstream = [torch.randn(m.r, m.block_size, m.block_size, device=device) for m in modules]
    xs = [
        torch.randn(args.batch, args.tokens, m.in_features, device=device, dtype=torch.bfloat16, requires_grad=True)
        for m in modules
    ]
    y_upstream = [torch.randn_like(x) for x in xs]

    def cayley_old():
        return [old_cayley(m) for m in modules]

    def cayley_new():
        return [new_cayley(m) for m in modules]

    def rotate_old():
        return [old_rotate(m, x) for m, x in zip(modules, xs, strict=True)]

    def rotate_new():
        return [new_rotate(m, x) for m, x in zip(modules, xs, strict=True)]

    results = [
        time_variant("cayley OLD  bf16-autocast 3bmm", cayley_old, modules, r_upstream, [], args, device),
        time_variant("cayley NEW  fp32 Horner", cayley_new, modules, r_upstream, [], args, device),
        time_variant(
            "cayley NEW  fp32 Horner, fp32 bwd",
            cayley_new,
            modules,
            r_upstream,
            [],
            args,
            device,
            backward_pass_autocast="off",
        ),
        time_variant("rotate OLD  (x upcast)", rotate_old, modules, y_upstream, xs, args, device),
        time_variant("rotate NEW", rotate_new, modules, y_upstream, xs, args, device),
        time_variant(
            "rotate NEW  fp32 bwd",
            rotate_new,
            modules,
            y_upstream,
            xs,
            args,
            device,
            backward_pass_autocast="off",
        ),
    ]
    return results


def print_speed_table(results: list[dict], blocks: int) -> None:
    header = f"{'variant (per block, 7 Linears)':<36} {'fwd ms':>9} {'bwd ms':>9} {'peak MiB':>9} {'step est ms':>12}"
    print(header)
    print("-" * len(header))
    for row in results:
        # forward + checkpoint recompute + backward, for every block
        step = blocks * (2 * row["fwd"] + row["bwd"])
        print(f"{row['label']:<36} {row['fwd']:>9.2f} {row['bwd']:>9.2f} {row['peak'] / MIB:>9.0f} {step:>12.0f}")
    print(f"step est = {blocks} blocks x (fwd + recompute fwd + bwd); excludes everything but the OFT rotation.")


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

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(device)}  TF32 matmul {args.tf32}")
    print(
        f"blocks {args.blocks}  tokens {args.tokens}  batch {args.batch}  std {args.std}  clip {args.clip}  "
        f"{'eager' if args.no_compile else 'compile/' + args.backend}\n"
    )

    if not args.skip_memory:
        old = measure_construction(OldBufferRotation, args, device)
        new = measure_construction(OFTRotationModule, args, device)
        print("(a) construction + .to(cuda) of", args.blocks * len(BLOCK_LINEARS), "rotation modules")
        print(f"{'':<10} {'alloc delta MiB':>16} {'index MiB':>10} {'weight+state MiB':>17}")
        for name, row in (("OLD", old), ("NEW", new)):
            print(f"{name:<10} {row['delta'] / MIB:>16.1f} {row['index'] / MIB:>10.1f} {row['state'] / MIB:>17.1f}")
        print(f"saved: {(old['delta'] - new['delta']) / MIB:.1f} MiB\n")

    if not args.skip_speed:
        print("(b) per-block timing (median of", args.iters, "iters after", args.warmup, "warmup)")
        print_speed_table(run_speed(args, device), args.blocks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
