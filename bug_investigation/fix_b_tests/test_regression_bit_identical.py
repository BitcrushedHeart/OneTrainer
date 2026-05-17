"""
Regression test: prove the Fix B contract.

For each accumulation_steps value:
  1. Run baseline (no stop) -- record per-update-step logged losses.
  2. Run restart-without-fix -- save mid-window with the *unfixed*
     payload (no accumulator, no grads), reload, continue. Asserts
     this diverges from baseline (the H5 bug reproduces).
  3. Run restart-with-fix -- save mid-window with the *Fix B*
     payload (accumulator + grads + scaler + RNG), reload, continue.
     Asserts this is BIT-IDENTICAL to baseline.

Bit-identicalness is the load-bearing claim of Fix B. Anything less
than exact equality at every logged step would mean the fix is
incomplete.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _shadow_trainer import (   # noqa: E402
    ShadowTrainer,
    fresh_optimizer,
    make_batches,
    make_seeded_model,
    set_global_seeds,
)


# Parametrise across the user's typical-usage range.  acc=150 is the
# expensive end and dominates wall-time; pick batches/steps so all three
# fit in well under a minute on CPU.
ACC_VALUES = [5, 50, 150]
SEED = 1234


def _build_trainer(seed: int, acc: int, enable_fix: bool) -> ShadowTrainer:
    set_global_seeds(seed)
    model = make_seeded_model(seed)
    opt = fresh_optimizer(model)
    return ShadowTrainer(
        model, opt,
        accumulation_steps=acc,
        enable_fix=enable_fix,
        concepts=[{"name": "c1", "path": "/tmp/x", "seed": 7,
                   "type": "image", "include_subdirectories": False}],
    )


def _run_baseline(acc: int, total_steps: int):
    """Single uninterrupted run."""
    t = _build_trainer(SEED, acc, enable_fix=False)
    batches = make_batches(total_steps, seed=99, batch_size=4)
    result = t.run_epoch(batches)
    return result.logs


def _run_restart(acc: int, total_steps: int, stop_after_step: int, enable_fix: bool):
    """Two-leg run: stop at stop_after_step, save, recreate, load, continue."""
    t1 = _build_trainer(SEED, acc, enable_fix=enable_fix)
    batches = make_batches(total_steps, seed=99, batch_size=4)

    # First leg: stop mid-window.
    result1 = t1.run_epoch(batches, stop_after_step=stop_after_step)
    assert result1.final_global_step == stop_after_step + 1, (
        f"stop didn't fire when expected: final_global_step="
        f"{result1.final_global_step}, stop_after_step={stop_after_step}"
    )

    with tempfile.TemporaryDirectory() as td:
        t1.save_to(td)

        # Second leg: fresh trainer, restore.
        set_global_seeds(SEED + 9999)   # irrelevant -- restored RNG should win
        model2 = make_seeded_model(SEED)   # same init weights
        opt2 = fresh_optimizer(model2)
        t2 = ShadowTrainer(
            model2, opt2,
            accumulation_steps=acc,
            enable_fix=enable_fix,
            concepts=t1.concepts,
        )
        warnings: list[str] = []
        t2.load_from(td, warn_callback=warnings.append)
        assert warnings == [], f"unexpected warnings on matched-load: {warnings}"

        remaining = batches[t2.global_step:]
        result2 = t2.run_epoch(remaining)

    return result1.logs + result2.logs


def _pick_stop_step(acc: int, total_steps: int) -> int:
    """Pick a stop step deep into a window (so k_lost is big and the
    bug bites hard).

    For acc=5 with steps 0..14, window 5..9 is in play; stop at 7 (3
    micro-batches already in window). For acc=150 with steps 0..299,
    stop at 250 (101 micro-batches already in second window).
    """
    # Find the start of the second window: 0..acc-1, acc..2acc-1, ...
    second_window_start = acc
    # Stop ~70% of the way through that window so k_lost is dominant.
    return second_window_start + int(acc * 0.7)


def _logs_to_dict(logs):
    return {step: loss for step, loss in logs}


@pytest.mark.parametrize("acc", ACC_VALUES)
def test_unfixed_save_diverges_from_baseline(acc: int):
    """Sanity: the bug exists in enable_fix=False mode."""
    # 3 full windows worth of steps + a buffer
    total_steps = acc * 3
    stop_step = _pick_stop_step(acc, total_steps)

    baseline = _logs_to_dict(_run_baseline(acc, total_steps))
    restart = _logs_to_dict(_run_restart(
        acc, total_steps, stop_step, enable_fix=False
    ))

    # The first update-step boundary >= stop_step must show a *different*
    # logged loss.  (Subsequent boundaries may also drift, but at least the
    # one straddling the stop must diverge -- that's the bug.)
    boundary_steps = sorted(baseline.keys() & restart.keys())
    affected = [s for s in boundary_steps if s >= stop_step]
    assert affected, "no update boundary after stop -- test misconfigured"

    first_affected = affected[0]
    delta = abs(baseline[first_affected] - restart[first_affected])
    assert delta > 1e-6, (
        f"acc={acc}: unfixed restart matched baseline at step {first_affected} "
        f"({baseline[first_affected]} vs {restart[first_affected]}); "
        f"the bug is NOT reproducing, so the fix can't be measured against it"
    )


@pytest.mark.parametrize("acc", ACC_VALUES)
def test_fixed_save_matches_baseline_bit_identical(acc: int):
    """The load-bearing claim of Fix B: baseline == restart, bit-identical."""
    total_steps = acc * 3
    stop_step = _pick_stop_step(acc, total_steps)

    baseline = _logs_to_dict(_run_baseline(acc, total_steps))
    restart = _logs_to_dict(_run_restart(
        acc, total_steps, stop_step, enable_fix=True
    ))

    assert sorted(baseline.keys()) == sorted(restart.keys()), (
        f"acc={acc}: different update-step sets -- "
        f"baseline={sorted(baseline)} restart={sorted(restart)}"
    )

    diffs = []
    for step in sorted(baseline.keys()):
        b, r = baseline[step], restart[step]
        if b != r:    # bit-identical required
            diffs.append((step, b, r, abs(b - r)))

    assert not diffs, (
        f"acc={acc}: Fix B is not bit-identical to baseline. "
        f"First divergences: {diffs[:3]}"
    )


if __name__ == "__main__":
    # Allow direct invocation for quick smoke runs.
    sys.exit(pytest.main([__file__, "-v"]))
