import threading
import time
import traceback

import torch
import torch.utils._sympy.functions

from sympy import S
from tqdm.auto import tqdm


# code from https://github.com/pytorch/pytorch/blob/ed82d5fcfd80110565f69130f286c7bfec6db2dc/torch/utils/_sympy/functions.py#L481
# but accepts negative numbers, to avoid https://github.com/Nerogar/OneTrainer/issues/1126
# torch 2.12 merged the fix PR https://github.com/pytorch/pytorch/pull/169726, but according to Claude's analysis this bug
# still exists. TODO confirm manually by testing https://github.com/Nerogar/OneTrainer/issues/1126:
# Claude: "Mod.eval itself still raises on negative numbers, and inductor's tiling_utils still calls sympy is_constant() (which
# substitutes negative test values) on expressions that can contain Mod. can be removed once Mod.eval itself
# accepts negative numbers in a torch version we use"
@classmethod
def Mod_patched_eval(cls, p, q):
    # This was adapted from: sympy/core/mod.py

    # Triggered by
    # python test/test_dynamic_shapes.py -k TestDimConstraints.test_dim_constraints_solve_full
    # assert p.is_integer, p
    # assert q.is_integer, q

    if q.is_zero:
        raise ZeroDivisionError("Modulo by zero")

    # Three cases:
    #   1. p == 0
    #   2. p is either q or -q
    #   3. p is integer and q == 1
    if p is S.Zero or p in (q, -q) or q == 1:
        return S.Zero

    # Evaluate if they are both literals.
    if q.is_Number and p.is_Number:
        if p < 0:
            # raise AssertionError(p)
            pass
        if q < 1:
            raise AssertionError(q)
        return p % q

    # If q == 2, it's a matter of whether p is odd or even.
    if q.is_Number and q == 2:
        if p.is_even:
            return S.Zero
        if p.is_odd:
            return S.One

    # If p is a multiple of q.
    r = p / q
    if r.is_integer:
        return S.Zero

    # If p < q and its ratio is positive, then:
    #   - floor(p / q) = 0
    #   - p % q = p - floor(p / q) * q = p
    less = p < q
    if less.is_Boolean and bool(less) and r.is_positive:
        return p

    return None


_compile_progress_lock = threading.Lock()
_compile_progress_state = {
    "starts": 0,  # graphs that have started compiling
    "ends": 0,  # graphs that have finished compiling
    "active": {},  # compile_id -> (trigger_name, t_start)
    "trigger_counts": {},  # trigger_name -> int
    "total_seconds": 0.0,  # cumulative compile time
    "installed": False,
}


def _format_progress_line() -> str:
    state = _compile_progress_state
    parts = [f"started={state['starts']}", f"finished={state['ends']}"]
    if state["active"]:
        parts.append(f"in_flight={len(state['active'])}")
    if state["trigger_counts"]:
        triggers = ",".join(f"{name.lower()}={count}" for name, count in sorted(state["trigger_counts"].items()))
        parts.append(triggers)
    parts.append(f"compile_time={state['total_seconds']:.1f}s")
    return "[torch.compile] " + " ".join(parts)


def _on_compile_start(args):
    trigger_name = args.callback_trigger.name
    now = time.monotonic()
    with _compile_progress_lock:
        state = _compile_progress_state
        state["starts"] += 1
        state["active"][args.compile_id] = (trigger_name, now)
        state["trigger_counts"][trigger_name] = state["trigger_counts"].get(trigger_name, 0) + 1
        # tqdm.write goes through tqdm's lock so it interleaves cleanly with
        # the epoch/step bars that GenericTrainer keeps open during this call.
        tqdm.write(
            f"[torch.compile] compiling graph id={args.compile_id} trigger={trigger_name} "
            f"(started={state['starts']}, finished={state['ends']})"
        )


def _on_compile_end(args):
    now = time.monotonic()
    with _compile_progress_lock:
        state = _compile_progress_state
        entry = state["active"].pop(args.compile_id, None)
        elapsed = now - entry[1] if entry is not None else 0.0
        state["total_seconds"] += elapsed
        state["ends"] += 1
        line = _format_progress_line()
        trigger_name = entry[0] if entry is not None else args.callback_trigger.name
    tqdm.write(f"[torch.compile] finished graph id={args.compile_id} trigger={trigger_name} in {elapsed:.2f}s | {line}")


def _install_compile_progress_callbacks():
    if _compile_progress_state["installed"]:
        return
    from torch._dynamo import callback as dynamo_callback

    dynamo_callback.callback_handler.register_start_callback(_on_compile_start)
    dynamo_callback.callback_handler.register_end_callback(_on_compile_end)
    _compile_progress_state["installed"] = True


# Process-wide default for Dynamo's on-disk precompile cache. Seeded from the
# TORCH_CACHING_PRECOMPILE env var (which torch reads at import), overridden per
# run by set_precompile_cache() and applied to every compiling thread by
# init_compile().
_precompile_cache_enabled = bool(torch._dynamo.config.caching_precompile)


def set_precompile_cache(enabled: bool) -> None:
    """Turn Dynamo's cross-process precompile cache on or off for this run.

    With it on, every compiled graph's traced bytecode and guards are written to
    the inductor cache dir by save_precompile_cache(), and a later process that
    compiles the same code (the next queued run, a restarted backend) loads them
    instead of retracing: the first training step then costs seconds instead of
    minutes. Inductor's own FX-graph / Triton caches were already shared across
    processes; this adds the Dynamo tracing that sat in front of them.

    Must run before init_compile() in the training thread: since torch 2.12 the
    dynamo config is a ContextVar, so init_compile() applies the flag per thread.
    """
    global _precompile_cache_enabled
    _precompile_cache_enabled = bool(enabled)


def precompile_cache_enabled() -> bool:
    return _precompile_cache_enabled


def save_precompile_cache(label: str = "") -> None:
    """Write the graphs compiled so far to the on-disk Dynamo cache (no-op when off).

    Cheap (a few MB) and idempotent, so it runs after the first optimizer step,
    when the forward and checkpoint-recompute graphs exist, and again at the end
    of the run to pick up anything compiled later (new buckets, sampling).
    """
    if not _precompile_cache_enabled:
        return
    try:
        from torch._dynamo.utils import counters

        result = torch.compiler.save_cache_artifacts()
        saved = len(result[1].artifacts.get("precompile", [])) if result is not None else 0
        stats = dict(counters["dynamo_cache"])
        hits = stats.get("dynamo_cache_hit", 0)
        misses = stats.get("dynamo_cache_miss", 0)
        suffix = f" ({label})" if label else ""
        tqdm.write(
            f"[torch.compile] precompile cache{suffix}: wrote {saved} package(s); "
            f"loaded from cache this run: {hits} hit(s), {misses} miss(es)"
        )
    except Exception:
        traceback.print_exc()
        tqdm.write("[torch.compile] could not save the precompile cache; the next run will retrace.")


def init_compile():
    # cache_size_limit and recompile_limit are aliases for the same dynamo config value.
    # since torch 2.12, dynamo config overrides are stored in ContextVars (pytorch/pytorch#173568),
    # which threads don't inherit. init_compile() must therefore be called in every thread that can
    # trigger a compilation - including the autograd worker threads, which recompute checkpointed
    # modules during the backward pass.
    torch._dynamo.config.cache_size_limit = 8192
    torch._dynamo.config.caching_precompile = _precompile_cache_enabled
    _install_compile_progress_callbacks()


torch.utils._sympy.functions.Mod.eval = Mod_patched_eval
