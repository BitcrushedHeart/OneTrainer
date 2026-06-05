"""
gradient_accumulation_steps mismatch test. Save with one acc value;
reload with a different one. The fix MUST warn-and-continue: restore
the partial state unchanged. The first optimizer step uses the saved
grads under the new acc config (one-time off-spec step); subsequent
steps proceed normally under the new acc value.

This matches the explicit user preference: losing hours of in-flight
gradient state is worse than one off-spec optimizer step.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

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

SEED = 9090


def _build(enable_fix: bool, acc: int):
    set_global_seeds(SEED)
    model = make_seeded_model(SEED)
    opt = fresh_optimizer(model)
    return ShadowTrainer(
        model,
        opt,
        accumulation_steps=acc,
        enable_fix=enable_fix,
        concepts=[{"name": "c", "path": "/x", "seed": 1, "type": "image", "include_subdirectories": False}],
    )


def test_acc_mismatch_warns_then_restores():
    SAVED_ACC = 10
    NEW_ACC = 5

    t1 = _build(enable_fix=True, acc=SAVED_ACC)
    batches = make_batches(40, seed=33, batch_size=2)
    t1.run_epoch(batches, stop_after_step=15)  # mid-window 10..19

    with tempfile.TemporaryDirectory() as td:
        t1.save_to(td)

        # Load with a DIFFERENT acc value.
        set_global_seeds(SEED + 1)
        t2 = _build(enable_fix=True, acc=NEW_ACC)
        warnings = []
        t2.load_from(td, warn_callback=warnings.append)

        acc_warnings = [w for w in warnings if "accumulation_steps mismatch" in w]
        assert acc_warnings, f"expected an acc-steps warning, got: {warnings}"

        # Partial state must have been restored.
        assert t2.state.accumulated_loss > 0.0, "partial accumulator was discarded under acc mismatch"
        assert t2.has_gradient, "saved grads were not applied under acc mismatch"

        remaining = batches[t2.global_step :]
        result = t2.run_epoch(remaining)
        for step, loss in result.logs:
            assert not math.isnan(loss) and math.isfinite(loss), f"step {step}: NaN/Inf after acc mismatch: {loss}"
