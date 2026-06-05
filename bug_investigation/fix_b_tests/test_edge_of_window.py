"""
Edge-of-window test. Verify Fix B is bit-identical at two boundary
conditions:
  - Stop one micro-batch BEFORE an accumulation boundary (k_lost = acc-1,
    the worst-case for the old bug).
  - Stop EXACTLY at the boundary (k_lost = 0; accumulated_loss is 0
    and .grad is None -- the trivial "save at clean boundary" case
    that the production code's automatic-backup path always satisfies).
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _shadow_trainer import (  # noqa: E402
    ShadowTrainer,
    fresh_optimizer,
    make_batches,
    make_seeded_model,
    set_global_seeds,
)

ACC = 5
SEED = 271


def _build():
    set_global_seeds(SEED)
    model = make_seeded_model(SEED)
    return ShadowTrainer(
        model,
        fresh_optimizer(model),
        accumulation_steps=ACC,
        enable_fix=True,
        concepts=[{"name": "c", "path": "/x", "seed": 1, "type": "image", "include_subdirectories": False}],
    )


def _baseline(total_steps):
    t = _build()
    batches = make_batches(total_steps, seed=42, batch_size=2)
    return dict(t.run_epoch(batches).logs)


def _restart(total_steps, stop_step):
    t1 = _build()
    batches = make_batches(total_steps, seed=42, batch_size=2)
    r1 = t1.run_epoch(batches, stop_after_step=stop_step)
    with tempfile.TemporaryDirectory() as td:
        t1.save_to(td)
        set_global_seeds(SEED + 7777)
        t2 = _build()
        t2.load_from(td)
        remaining = batches[t2.global_step :]
        r2 = t2.run_epoch(remaining)
    return dict(r1.logs + r2.logs)


@pytest.mark.parametrize(
    "stop_step,description",
    [
        (3, "one micro-batch BEFORE boundary at step 4 (k_lost=acc-1, worst case)"),
        (4, "EXACTLY AT boundary (k_lost=0, accumulator is 0, .grad is None)"),
    ],
)
def test_edge_of_window_bit_identical(stop_step, description):
    total_steps = ACC * 3
    base = _baseline(total_steps)
    restart = _restart(total_steps, stop_step)
    for step in sorted(base):
        assert base[step] == restart[step], (
            f"[{description}] step {step}: baseline={base[step]} restart={restart[step]}"
        )
