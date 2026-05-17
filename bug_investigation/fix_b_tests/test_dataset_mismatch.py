"""
Save mid-window with one dataset fingerprint; load with a different one.
The fix MUST warn-but-continue: restore the partial accumulator state
unchanged, no NaN, no crash. The first post-resume optimizer step uses
whatever effective batch the loaded grads imply.
"""
from __future__ import annotations

import math
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
SEED = 1010


def _build(enable_fix: bool, concepts: list):
    set_global_seeds(SEED)
    model = make_seeded_model(SEED)
    opt = fresh_optimizer(model)
    return ShadowTrainer(
        model, opt,
        accumulation_steps=ACC,
        enable_fix=enable_fix,
        concepts=concepts,
    )


def test_dataset_mismatch_warns_then_restores():
    save_concepts = [
        {"name": "a", "path": "/x/a", "seed": 1, "type": "image",
         "include_subdirectories": False},
        {"name": "b", "path": "/x/b", "seed": 2, "type": "image",
         "include_subdirectories": False},
    ]
    load_concepts = [
        {"name": "a", "path": "/x/a", "seed": 1, "type": "image",
         "include_subdirectories": False},
        # 'b' removed, 'c' added -> fingerprint mismatch
        {"name": "c", "path": "/x/c", "seed": 3, "type": "image",
         "include_subdirectories": False},
    ]

    t1 = _build(enable_fix=True, concepts=save_concepts)
    batches = make_batches(20, seed=55, batch_size=3)
    t1.run_epoch(batches, stop_after_step=10)  # mid-window 6..11

    with tempfile.TemporaryDirectory() as td:
        t1.save_to(td)

        # Load into a fresh trainer with DIFFERENT concepts.
        set_global_seeds(SEED + 1)
        t2 = ShadowTrainer(
            make_seeded_model(SEED),
            fresh_optimizer(make_seeded_model(SEED)),  # dummy; replaced
            accumulation_steps=ACC,
            enable_fix=True,
            concepts=load_concepts,
        )
        # Rebind optimizer to the right model.
        t2.optimizer = fresh_optimizer(t2.model)

        warnings = []
        t2.load_from(td, warn_callback=warnings.append)

        # Must have warned about the dataset mismatch.
        dataset_warnings = [w for w in warnings if "dataset fingerprint" in w]
        assert dataset_warnings, (
            f"expected a dataset-fingerprint warning, got: {warnings}"
        )

        # Partial state must have been restored anyway.
        assert t2.state.accumulated_loss > 0.0, (
            "partial accumulator was discarded (it shouldn't have been)"
        )
        assert t2.has_gradient, (
            "saved grads were not applied (they should have been despite "
            "the mismatch)"
        )

        # Continue training without crashing / NaN'ing.
        remaining = batches[t2.global_step:]
        result = t2.run_epoch(remaining)
        for step, loss in result.logs:
            assert not math.isnan(loss) and math.isfinite(loss), (
                f"step {step}: NaN/Inf in post-mismatch log: {loss}"
            )
