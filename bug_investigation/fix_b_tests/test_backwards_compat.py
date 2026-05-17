"""
Backwards-compat test. A legacy backup (saved by pre-Fix-B code) has
NO accumulator/accumulator.pt in it -- only model.pt, optimizer.pt,
meta.json. Loading such a backup with the new code must:
  - not crash
  - not emit any Fix-B-specific warning
  - leave the trainer with accumulator at defaults (0.0, no grads)
  - resume training cleanly (this is the pre-fix behaviour: if the
    legacy backup happened to be mid-window, the user gets the
    original spike on this one resume; that's the existing UX, not a
    regression)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

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


def test_legacy_backup_loads_without_accumulator_file():
    set_global_seeds(0)
    model = make_seeded_model(0)
    opt = fresh_optimizer(model)
    t1 = ShadowTrainer(
        model, opt, accumulation_steps=4,
        enable_fix=False,   # produces a legacy-style save
        concepts=[{"name": "c", "path": "/x", "seed": 0, "type": "image",
                   "include_subdirectories": False}],
    )
    batches = make_batches(12, seed=1, batch_size=2)
    t1.run_epoch(batches, stop_after_step=5)

    with tempfile.TemporaryDirectory() as td:
        t1.save_to(td)
        # Sanity: confirm there's no accumulator file.
        assert not os.path.exists(os.path.join(td, "accumulator")), (
            "shadow trainer wrote an accumulator file with enable_fix=False; "
            "test premise broken"
        )

        # Load with the NEW code (enable_fix=True).
        set_global_seeds(1)
        t2_model = make_seeded_model(0)
        t2 = ShadowTrainer(
            t2_model, fresh_optimizer(t2_model),
            accumulation_steps=4, enable_fix=True,
            concepts=t1.concepts,
        )
        warnings = []
        t2.load_from(td, warn_callback=warnings.append)
        # No Fix-B-specific warnings.
        assert not warnings, (
            f"unexpected warning loading legacy backup: {warnings}"
        )

        # Accumulator at defaults; no .grad on params.
        assert t2.state.accumulated_loss == 0.0
        assert not t2.has_gradient

        # Continue training without crashing.
        remaining = batches[t2.global_step:]
        result = t2.run_epoch(remaining)
        assert result.logs, "trainer should have logged at least one update step"
