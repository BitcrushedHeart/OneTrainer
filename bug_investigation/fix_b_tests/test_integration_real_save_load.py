"""
Integration test: runs the REAL InternalModelSaverMixin._save_internal_data
and InternalModelLoaderMixin._load_internal_data against a minimal
duck-typed BaseModel.  Proves the production save/load wiring actually
round-trips the accumulator payload (param_grads, scaler, RNG,
fingerprint, accumulators).

This is the missing piece between the shadow-trainer regression test
(which proves the *algorithm*) and end-to-end manual training (which
proves the integration in a real workload).
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Make the OneTrainer source importable.
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin  # noqa: E402
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin  # noqa: E402


class _Saver(InternalModelSaverMixin):
    pass


class _Loader(InternalModelLoaderMixin):
    pass


def _build_mock_model(with_accumulator: bool):
    """Duck-typed BaseModel exposing only the attributes the mixins touch.

    The Mixins read:
      model.optimizer, model.param_group_mapping, model.train_config.optimizer.optimizer
      model.ema, model.train_progress.{epoch,epoch_step,epoch_sample,global_step,last_action_epoch}
      model.tensorboard_subdir
      model.accumulator_state                  (Fix B)
    """
    head = nn.Linear(4, 2)
    opt = torch.optim.SGD(head.parameters(), lr=0.01, momentum=0.9)
    # Run a step to populate optimizer state.
    head(torch.zeros(1, 4)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    train_progress = SimpleNamespace(
        epoch=2,
        epoch_step=15,
        epoch_sample=60,
        global_step=27,
        last_action_epoch={"sample": 2, "validate": 2},
    )
    inner_optimizer_cfg = SimpleNamespace(optimizer="SGD")
    train_config = SimpleNamespace(optimizer=inner_optimizer_cfg)
    model = SimpleNamespace(
        optimizer=opt,
        param_group_mapping=["head"],
        train_config=train_config,
        ema=None,
        train_progress=train_progress,
        tensorboard_subdir="run-1",
        accumulator_state=None,
        # Loader writes these on load:
        optimizer_state_dict=None,
        ema_state_dict=None,
        resumed_tensorboard_subdir=None,
    )

    if with_accumulator:
        # Synthesise a Fix B payload as the trainer would.
        param_grads = {
            "head.0": torch.randn(2, 4),
            "head.1": torch.randn(2),
        }
        rng = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
            "python": random.getstate(),
            "numpy": np.random.get_state(legacy=True),
        }
        model.accumulator_state = {
            "accumulated_loss": 0.1234,
            "accumulated_dpo_metrics": {"loss": 0.42, "_count": 3},
            "param_grads": param_grads,
            "scaler": None,
            "rng": rng,
            "fingerprint": {
                "gradient_accumulation_steps": 10,
                "dataset_hash": "abc123",
                "concept_count": 2,
            },
        }

    return model


def test_real_save_and_load_round_trip_full_payload():
    saver = _Saver()
    loader = _Loader()

    src = _build_mock_model(with_accumulator=True)
    with tempfile.TemporaryDirectory() as td:
        saver._save_internal_data(src, td)

        # Files written.
        assert os.path.exists(os.path.join(td, "optimizer", "optimizer.pt"))
        assert os.path.exists(os.path.join(td, "meta.json"))
        assert os.path.exists(os.path.join(td, "accumulator", "accumulator.pt")), (
            "accumulator.pt not written by the production save path"
        )

        # Load into a blank model.
        dst = _build_mock_model(with_accumulator=False)
        loader._load_internal_data(dst, td)

        # Train progress round-trips.
        assert dst.train_progress.global_step == 27
        assert dst.train_progress.last_action_epoch == {"sample": 2, "validate": 2}
        assert dst.resumed_tensorboard_subdir == "run-1"

        # Optimizer state round-trips.
        assert dst.optimizer_state_dict is not None
        assert "param_group_mapping" in dst.optimizer_state_dict

        # Accumulator round-trips.
        assert dst.accumulator_state is not None, "accumulator_state not populated by load"
        assert dst.accumulator_state["accumulated_loss"] == 0.1234
        assert dst.accumulator_state["accumulated_dpo_metrics"] == {"loss": 0.42, "_count": 3}
        assert set(dst.accumulator_state["param_grads"].keys()) == {"head.0", "head.1"}
        # Exact tensor equality.
        for k in ("head.0", "head.1"):
            torch.testing.assert_close(
                dst.accumulator_state["param_grads"][k],
                src.accumulator_state["param_grads"][k],
            )
        # Fingerprint preserved.
        assert dst.accumulator_state["fingerprint"]["gradient_accumulation_steps"] == 10
        assert dst.accumulator_state["fingerprint"]["dataset_hash"] == "abc123"


def test_real_save_without_accumulator_does_not_write_file():
    """When the saver receives a model with no staged accumulator state
    (e.g. embedding/finetune save paths that bypass training), it must
    NOT write a stale accumulator.pt."""
    saver = _Saver()
    model = _build_mock_model(with_accumulator=False)
    with tempfile.TemporaryDirectory() as td:
        saver._save_internal_data(model, td)
        assert not os.path.exists(os.path.join(td, "accumulator")), (
            "saver wrote an accumulator directory despite model.accumulator_state being None"
        )


def test_real_load_legacy_backup_no_crash_no_accumulator():
    """A backup written by old code (no accumulator dir) loads cleanly:
    no crash, accumulator_state stays None."""
    saver = _Saver()
    loader = _Loader()
    # Save WITHOUT the accumulator dir.
    src = _build_mock_model(with_accumulator=False)
    with tempfile.TemporaryDirectory() as td:
        saver._save_internal_data(src, td)
        assert not os.path.exists(os.path.join(td, "accumulator"))

        dst = _build_mock_model(with_accumulator=False)
        loader._load_internal_data(dst, td)
        assert dst.accumulator_state is None
        # Other fields still round-trip.
        assert dst.train_progress.global_step == 27
