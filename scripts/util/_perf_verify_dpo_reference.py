"""GPU check for the host-RAM DPO reference snapshot.

Builds a synthetic adapter parameter set shaped like the live Z-Image DoRA-OFT
adapter (180 x (5, 294528) and 30 x (10, 523776) fp32 rotations plus one
(out,) magnitude per module: 420 tensors, 423.2M params) on the GPU, and runs
the EXISTING_ADAPTER reference pass two ways with train_dtype bf16:

* old: the snapshot cloned onto the GPU once and kept there (the previous
  BaseModelSetup.reference_model, reimplemented inline here);
* new: BaseModelSetup.reference_model as it is now, with the snapshot in
  pinned host RAM and uploaded into temporaries on every pass.

For each it prints the VRAM held between passes (torch.cuda.memory_allocated
above the policy params), the peak during a pass, and the median wall time to
enter and exit the reference context over --iters passes, synchronizing so the
upload is inside the timing. It also checks that the reference is bit-exact
during the pass and that the policy data and grads come back afterwards.

Expected: old holds ~807 MiB (the bf16 snapshot) between passes, new ~0; the
peak inside a pass is about the same for both; new pays one host-to-device
upload of ~807 MiB per pass (tens of ms on PCIe 4.0 x16), old only pointer swaps.

With --backup-dir DIR it also times a real dpo_reference.pt write against the
hard link used for an unchanged reference, on the drive DIR lives on, and
checks the link really shares one file. The files are deleted afterwards.

Refuses to start while the GPU already has more than --max-used-gib in use
(a live training run), unless --force.

Usage:
    python scripts/util/_perf_verify_dpo_reference.py [--iters 20] [--device 0] [--backup-dir F:/tmp/dpo_link_check] [--force]
"""

import argparse
import gc
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.modelSetup.BaseModelSetup import BaseModelSetup  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.DataType import DataType  # noqa: E402
from modules.util.enum.DPORefMode import DPORefMode  # noqa: E402
from modules.util.enum.TrainingMethod import TrainingMethod  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

# (rotation shape, magnitude shape) per adapter module, as in the live adapter.
LAYOUT = [((5, 294528), (3840,))] * 180 + [((10, 523776), (10240,))] * 30
MIB = 2**20
GIB = 2**30


def _gpu_used_gib(index: int) -> float | None:
    # Ask the driver before this process creates a CUDA context, so the figure
    # is what other processes (a training run, the desktop) already hold.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return float(out.stdout.strip().splitlines()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
    config.train_dtype = DataType.BFLOAT_16
    return config


def _build_adapters(device: torch.device) -> list[nn.Module]:
    adapters = []
    generator = torch.Generator(device=device).manual_seed(0)
    for rotation_shape, magnitude_shape in LAYOUT:
        module = nn.Module()
        module.oft_R = nn.Parameter(torch.randn(rotation_shape, device=device, generator=generator) * 1e-3)
        module.magnitude = nn.Parameter(torch.ones(magnitude_shape, device=device))
        adapters.append(module)
    return adapters


@contextmanager
def _old_reference_model(state: SimpleNamespace, adapters: list[nn.Module], config: TrainConfig):
    # The previous behaviour: clone onto the params' device once and keep it.
    dtype = config.train_dtype.torch_dtype()
    policy = [list(adapter.parameters()) for adapter in adapters]
    if state._dpo_ref_params is None:
        state._dpo_ref_params = [[p.data.to(dtype=dtype, copy=True) for p in params] for params in policy]
    policy_data = [[p.data for p in params] for params in policy]
    policy_grads = [[p.grad for p in params] for params in policy]
    try:
        for params in policy:
            for param in params:
                param.grad = None
        for params, ref_params in zip(policy, state._dpo_ref_params, strict=True):
            for param, ref_data in zip(params, ref_params, strict=True):
                param.data = ref_data
        yield
    finally:
        for params, data, grads in zip(policy, policy_data, policy_grads, strict=True):
            for param, d, g in zip(params, data, grads, strict=True):
                param.data = d
                param.grad = g


def _new_reference_model(state, adapters, config):
    model = SimpleNamespace(adapters=lambda: adapters)
    return BaseModelSetup.reference_model(state, model, config)


def _fresh_state() -> SimpleNamespace:
    return SimpleNamespace(
        _dpo_ref_params=None,
        _dpo_ref_version=0,
        _dpo_ref_saved=None,
        _dpo_ref_pinned_version=None,
        _DPO_REFERENCE_FILENAME=BaseModelSetup._DPO_REFERENCE_FILENAME,
    )


def _check_pass(state, adapters, enter, device) -> None:
    grads = [[p.grad for p in adapter.parameters()] for adapter in adapters]
    ptrs = [[p.data_ptr() for p in adapter.parameters()] for adapter in adapters]
    with enter():
        for adapter, ref_params in zip(adapters, state._dpo_ref_params, strict=True):
            for param, ref in zip(adapter.parameters(), ref_params, strict=True):
                assert param.dtype == torch.bfloat16, param.dtype
                assert param.grad is None
                assert torch.equal(param.data, ref.to(device)), "reference pass does not see the snapshot"
    for adapter, g_row, p_row in zip(adapters, grads, ptrs, strict=True):
        for param, g, ptr in zip(adapter.parameters(), g_row, p_row, strict=True):
            assert param.dtype == torch.float32 and param.data_ptr() == ptr and param.grad is g


def _measure(name, state, adapters, config, device, iters) -> dict:
    def enter():
        if name == "old":
            return _old_reference_model(state, adapters, config)
        return _new_reference_model(state, adapters, config)

    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)

    with enter():  # first pass captures the snapshot
        pass
    torch.cuda.synchronize(device)
    snapshot = state._dpo_ref_params
    _check_pass(state, adapters, enter, device)
    for _ in range(3):
        with enter():
            pass
    torch.cuda.synchronize(device)

    steady = torch.cuda.memory_allocated(device) - base
    torch.cuda.reset_peak_memory_stats(device)
    enter_times, exit_times = [], []
    for _ in range(iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        ctx = enter()
        ctx.__enter__()
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        ctx.__exit__(None, None, None)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        enter_times.append(t1 - t0)
        exit_times.append(t2 - t1)
    peak = torch.cuda.max_memory_allocated(device) - base
    assert state._dpo_ref_params is snapshot, "the snapshot list was replaced"

    host = sum(t.numel() * t.element_size() for row in snapshot for t in row if t.device.type == "cpu")
    pinned = all(t.is_pinned() for row in snapshot for t in row if t.device.type == "cpu")
    return {
        "steady_mib": steady / MIB,
        "peak_mib": peak / MIB,
        "enter_ms": statistics.median(enter_times) * 1e3,
        "exit_ms": statistics.median(exit_times) * 1e3,
        "host_mib": host / MIB,
        "host_pinned": pinned if host else None,
    }


def _host_pinned_stats() -> str:
    stats_fn = getattr(torch.cuda, "host_memory_stats", None)
    if stats_fn is None:
        return "n/a"
    try:
        stats = stats_fn()
    except RuntimeError:
        return "n/a"
    keys = [k for k in stats if "bytes" in k and k.endswith(".current")]
    return ", ".join(f"{k}={stats[k] / MIB:.0f} MiB" for k in sorted(keys)) or "n/a"


def _check_backup_link(state, backup_root: str) -> None:
    os.makedirs(backup_root, exist_ok=True)
    root = tempfile.mkdtemp(prefix="dpo_ref_link_", dir=backup_root)
    try:
        first, second = os.path.join(root, "b1"), os.path.join(root, "b2")
        os.makedirs(first)
        os.makedirs(second)
        state._dpo_ref_saved = None
        t0 = time.perf_counter()
        BaseModelSetup.save_dpo_reference(state, first)
        t1 = time.perf_counter()
        BaseModelSetup.save_dpo_reference(state, second)
        t2 = time.perf_counter()
        a = os.path.join(first, BaseModelSetup._DPO_REFERENCE_FILENAME)
        b = os.path.join(second, BaseModelSetup._DPO_REFERENCE_FILENAME)
        linked = os.path.samefile(a, b)
        print(
            f"backup file on {backup_root}: write {(t1 - t0) * 1e3:.0f} ms ({os.path.getsize(a) / MIB:.0f} MiB), "
            f"second save {(t2 - t1) * 1e3:.1f} ms, hard-linked={linked} (st_nlink={os.stat(b).st_nlink})"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--iters", type=int, default=20, help="timed reference passes per mode")
    parser.add_argument("--max-used-gib", type=float, default=4.0, help="refuse above this much VRAM already in use")
    parser.add_argument("--backup-dir", default=None, help="also time write vs hard link of the backup file here")
    parser.add_argument("--force", action="store_true", help="run even if the GPU is busy")
    args = parser.parse_args()

    used = _gpu_used_gib(args.device)
    if used is None:
        print("warning: nvidia-smi unavailable, cannot tell whether the GPU is busy")
        if not args.force:
            print("refusing to run without --force")
            return 2
    elif used > args.max_used_gib and not args.force:
        print(
            f"GPU {args.device} already has {used:.1f} GiB in use (> {args.max_used_gib} GiB); pass --force to run anyway"
        )
        return 2
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return 2

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    config = _config()
    adapters = _build_adapters(device)
    numel = sum(p.numel() for adapter in adapters for p in adapter.parameters())
    tensors = sum(1 for adapter in adapters for _ in adapter.parameters())
    print(f"synthetic adapter: {tensors} tensors, {numel / 1e6:.1f}M params, bf16 snapshot {numel * 2 / MIB:.0f} MiB")
    for adapter in adapters:
        for p in adapter.parameters():
            p.grad = torch.zeros_like(p)  # an accumulated fp32 grad, as mid-accumulation

    results = {}
    for name in ("old", "new"):
        state = _fresh_state()
        results[name] = _measure(name, state, adapters, config, device, args.iters)
        if name == "new":
            print(f"pinned host allocator: {_host_pinned_stats()}")
            if args.backup_dir:
                _check_backup_link(state, args.backup_dir)
        del state
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    print(
        f"{'mode':<5} {'VRAM between passes':>20} {'peak in pass':>13} {'enter (median)':>15} {'exit (median)':>14}  host snapshot"
    )
    for name, r in results.items():
        host = f"{r['host_mib']:.0f} MiB pinned={r['host_pinned']}" if r["host_mib"] else "-"
        print(
            f"{name:<5} {r['steady_mib']:>16.0f} MiB {r['peak_mib']:>9.0f} MiB "
            f"{r['enter_ms']:>12.1f} ms {r['exit_ms']:>11.1f} ms  {host}"
        )
    saved = results["old"]["steady_mib"] - results["new"]["steady_mib"]
    print(f"VRAM freed between reference passes: {saved:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
