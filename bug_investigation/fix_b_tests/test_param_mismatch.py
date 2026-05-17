"""
Parameter-set mismatch test. Saved param_grads contains keys that don't
exist in the current model. Behaviour:
  - Individually-missing keys are silently skipped.
  - When >10% of saved keys are missing, a warning is emitted.
  - Current-model params with no saved grad get .grad = None (clean).
  - No crash either way.
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch
import torch.nn as nn

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


def _wide_model(seed: int):
    """Bigger model so 1 spurious grad key stays under the 10% threshold.
    7 Linear layers -> 14 trainable params, 1/15 = 6.7% < 10%."""
    torch.manual_seed(seed)
    layers = []
    in_dim = 8
    for w in (16, 16, 16, 16, 16, 16, 8):
        layers.append(nn.Linear(in_dim, w))
        layers.append(nn.Tanh())
        in_dim = w
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def _capture_save(model, acc):
    set_global_seeds(7)
    opt = fresh_optimizer(model)
    t = ShadowTrainer(model, opt, accumulation_steps=acc, enable_fix=True,
                      concepts=[{"name": "c", "path": "/x", "seed": 0,
                                 "type": "image", "include_subdirectories": False}])
    batches = make_batches(16, seed=11, batch_size=2)
    t.run_epoch(batches, stop_after_step=acc + (acc // 2))   # mid-second-window
    return t, batches


def test_few_missing_keys_silent_skip():
    set_global_seeds(7)
    model = _wide_model(7)
    t, _ = _capture_save(model, acc=8)

    with tempfile.TemporaryDirectory() as td:
        t.save_to(td)

        # Inject ONE extra spurious key into the saved file (well under 10%).
        payload_path = os.path.join(td, "accumulator", "accumulator.pt")
        payload = torch.load(payload_path, weights_only=False)
        before = len(payload["param_grads"])
        # Add a fake key.
        any_t = next(iter(payload["param_grads"].values()))
        payload["param_grads"]["does_not_exist_in_model"] = any_t.clone()
        torch.save(payload, payload_path)
        assert (1 / (before + 1)) < 0.10, "test misconfigured (1 spurious > 10%)"

        set_global_seeds(7)
        wide2 = _wide_model(7)
        t2 = ShadowTrainer(
            wide2, fresh_optimizer(wide2),
            accumulation_steps=8, enable_fix=True,
            concepts=t.concepts,
        )
        warnings = []
        t2.load_from(td, warn_callback=warnings.append)
        param_warnings = [w for w in warnings if "saved grad keys" in w]
        assert not param_warnings, (
            f"unexpected warning for sub-10% missing: {param_warnings}"
        )


def test_many_missing_keys_warns():
    """If the model has fewer trainable params than what was saved (>10%
    missing), a warning fires but the load succeeds."""
    set_global_seeds(7)
    model = make_seeded_model(7)
    t, _ = _capture_save(model, acc=8)

    with tempfile.TemporaryDirectory() as td:
        t.save_to(td)

        # Stuff the payload with many bogus keys so 10%+ are missing.
        payload_path = os.path.join(td, "accumulator", "accumulator.pt")
        payload = torch.load(payload_path, weights_only=False)
        any_t = next(iter(payload["param_grads"].values()))
        for i in range(len(payload["param_grads"]) * 3):
            payload["param_grads"][f"bogus_{i}"] = any_t.clone()
        torch.save(payload, payload_path)

        set_global_seeds(7)
        t2 = ShadowTrainer(
            make_seeded_model(7),
            fresh_optimizer(make_seeded_model(7)),
            accumulation_steps=8, enable_fix=True,
            concepts=t.concepts,
        )
        t2.optimizer = fresh_optimizer(t2.model)
        warnings = []
        t2.load_from(td, warn_callback=warnings.append)
        param_warnings = [w for w in warnings if "saved grad keys" in w]
        assert param_warnings, f"expected param-mismatch warning, got: {warnings}"
        # Still loaded cleanly.
        assert t2.has_gradient


def test_current_param_with_no_saved_grad_is_silent():
    """A trainable parameter that has no entry in the saved param_grads
    dict gets .grad = None and no warning is raised."""
    set_global_seeds(7)
    model = make_seeded_model(7)
    t, _ = _capture_save(model, acc=8)

    with tempfile.TemporaryDirectory() as td:
        t.save_to(td)

        # Drop one key (a current model param will be missing on load).
        payload_path = os.path.join(td, "accumulator", "accumulator.pt")
        payload = torch.load(payload_path, weights_only=False)
        dropped_key = next(iter(payload["param_grads"].keys()))
        del payload["param_grads"][dropped_key]
        torch.save(payload, payload_path)

        set_global_seeds(7)
        t2 = ShadowTrainer(
            make_seeded_model(7),
            fresh_optimizer(make_seeded_model(7)),
            accumulation_steps=8, enable_fix=True,
            concepts=t.concepts,
        )
        t2.optimizer = fresh_optimizer(t2.model)
        warnings = []
        t2.load_from(td, warn_callback=warnings.append)
        assert not warnings, f"expected no warnings, got: {warnings}"
