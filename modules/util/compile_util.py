import threading
import time

import torch

from sympy import S
from tqdm.auto import tqdm


# code from https://github.com/pytorch/pytorch/blob/ed82d5fcfd80110565f69130f286c7bfec6db2dc/torch/utils/_sympy/functions.py#L481
# but accepts negative numbers, to avoid https://github.com/Nerogar/OneTrainer/issues/1126
# can be removed once https://github.com/pytorch/pytorch/pull/169726 is merged into a torch version we use
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


def init_compile():
    # cache_size_limit and recompile_limit are aliases for the same dynamo config value.
    # NOTE: dynamo config overrides are thread-local — this must run on the thread that
    # executes the forward passes (the training thread), not just any importing thread.
    torch._dynamo.config.cache_size_limit = 8192
    torch.utils._sympy.functions.Mod.eval = Mod_patched_eval
    _install_compile_progress_callbacks()
