"""
RNG determinism test: two runs from the same seed with identical
stop/resume points produce bit-identical loss traces -- which exercises
the RNG-snapshot save/restore round-trip in the Fix B payload.
"""
from __future__ import annotations

import os
import sys
import tempfile

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

ACC = 6
SEED = 31415


def _build():
    set_global_seeds(SEED)
    return ShadowTrainer(
        make_seeded_model(SEED),
        fresh_optimizer(make_seeded_model(SEED)),
        accumulation_steps=ACC, enable_fix=True,
        concepts=[{"name": "c", "path": "/x", "seed": 1,
                   "type": "image", "include_subdirectories": False}],
    )


def _run_with_stop_resume(stop_step: int):
    t1 = _build()
    # Rebind optimizer to the right model (the dummy in _build is throwaway).
    t1.optimizer = fresh_optimizer(t1.model)
    batches = make_batches(24, seed=88, batch_size=2)
    r1 = t1.run_epoch(batches, stop_after_step=stop_step)
    with tempfile.TemporaryDirectory() as td:
        t1.save_to(td)
        set_global_seeds(0xDEAD)   # different post-save state to prove
                                   # RNG restore actually does work
        t2 = _build()
        t2.optimizer = fresh_optimizer(t2.model)
        t2.load_from(td)
        remaining = batches[t2.global_step:]
        r2 = t2.run_epoch(remaining)
    return r1.logs + r2.logs


def test_two_runs_identical_loss_traces():
    run_a = _run_with_stop_resume(stop_step=9)
    run_b = _run_with_stop_resume(stop_step=9)

    assert run_a == run_b, (
        f"runs diverged despite identical setup. "
        f"a={run_a}\nb={run_b}"
    )
