"""
Adapter analogue: frozen base + small trainable head. Only the head's
parameters have ``requires_grad=True`` and only those should appear in
the saved ``param_grads`` dict. Bit-identical baseline-vs-restart must
still hold.
"""
from __future__ import annotations

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

ACC = 8
TOTAL_STEPS = 24
STOP_AFTER = 13   # mid-window 8..15
SEED = 4242


def _build(enable_fix: bool, seed: int = SEED):
    set_global_seeds(seed)
    # Freeze the first 2 Linear layers (the "base"); only the last layer
    # remains trainable -- the "adapter head".
    model = make_seeded_model(seed, freeze_first_n=2)
    opt = fresh_optimizer(model)
    return ShadowTrainer(
        model, opt,
        accumulation_steps=ACC,
        enable_fix=enable_fix,
        concepts=[{"name": "c1", "path": "/p/a", "seed": 1, "type": "image",
                   "include_subdirectories": False}],
    )


def test_only_trainable_params_in_saved_grads():
    t = _build(enable_fix=True)
    batches = make_batches(TOTAL_STEPS, seed=77, batch_size=2)
    t.run_epoch(batches, stop_after_step=STOP_AFTER)

    with tempfile.TemporaryDirectory() as td:
        t.save_to(td)
        payload = torch.load(os.path.join(td, "accumulator", "accumulator.pt"),
                             weights_only=False)
        saved_grad_keys = set(payload["param_grads"].keys())

    # Only the last Linear's weight + bias should appear.
    trainable_keys = {
        n for n, p in t.model.named_parameters() if p.requires_grad
    }
    assert saved_grad_keys == trainable_keys, (
        f"saved keys {saved_grad_keys} != trainable keys {trainable_keys}"
    )
    # The frozen base must NOT leak into the payload.
    frozen_keys = {
        n for n, p in t.model.named_parameters() if not p.requires_grad
    }
    assert not (saved_grad_keys & frozen_keys), (
        f"frozen params leaked into payload: {saved_grad_keys & frozen_keys}"
    )


def test_adapter_bit_identical_baseline_vs_restart():
    # Baseline.
    t = _build(enable_fix=True)
    batches = make_batches(TOTAL_STEPS, seed=77, batch_size=2)
    base = t.run_epoch(batches)

    # Restart.
    t1 = _build(enable_fix=True)
    result1 = t1.run_epoch(batches, stop_after_step=STOP_AFTER)
    with tempfile.TemporaryDirectory() as td:
        t1.save_to(td)
        set_global_seeds(SEED + 1)   # restored RNG should override
        t2 = _build(enable_fix=True, seed=SEED)
        t2.load_from(td)
        remaining = batches[t2.global_step:]
        result2 = t2.run_epoch(remaining)
    restart_logs = result1.logs + result2.logs

    base_d = dict(base.logs)
    rest_d = dict(restart_logs)
    for step in sorted(base_d):
        assert base_d[step] == rest_d[step], (
            f"step {step}: baseline={base_d[step]} restart={rest_d[step]}"
        )
