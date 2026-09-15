"""Redistill phase-boundary mechanics on a GenericTrainer with faked model state."""

import json
import os

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch

import pytest
from safetensors.torch import save_file


class _FakeSetup:
    def __init__(self):
        self.phase_b_start = None
        self.restore_refreshes = []

    def set_redistill_phase_b_start(self, step):
        self.phase_b_start = step

    def refresh_adapters_after_param_restore(self, model, config, *, warm):
        self.restore_refreshes.append(warm)


class _FakeModel:
    def __init__(self):
        self.transfer_teacher_late_tensors = {"layers.2.attention.to_q": torch.zeros(2, 2)}


def _make_trainer(tmp_path, *, continue_last_backup=False, transfer_steps=100):
    config = TrainConfig.default_values()
    config.tensorboard = False
    config.workspace_dir = str(tmp_path / "workspace")
    config.cache_dir = str(tmp_path / "cache")
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.OFT_2
    config.redistill_enabled = True
    config.redistill_transfer_steps = transfer_steps
    config.continue_last_backup = continue_last_backup
    turbo = tmp_path / "turbo.safetensors"
    if not turbo.is_file():
        save_file({"x": torch.zeros(1)}, str(turbo))
    config.redistill_turbo_model_name = str(turbo)

    trainer = GenericTrainer(config, TrainCallbacks(), TrainCommands())
    trainer.model_setup = _FakeSetup()
    trainer.model = _FakeModel()
    trainer.parameters = [torch.nn.Parameter(torch.zeros(2))]
    return trainer


def _advance(trainer, step):
    progress = TrainProgress()
    progress.global_step = step
    trainer._GenericTrainer__maybe_advance_redistill_phase(progress)
    return progress


def test_stays_in_phase_a_before_budget(tmp_path):
    trainer = _make_trainer(tmp_path)
    _advance(trainer, 50)
    assert trainer._redistill_phase_b is False
    assert trainer.model.transfer_teacher_late_tensors is not None
    assert trainer.model_setup.phase_b_start is None


def test_advances_at_budget_frees_teacher_and_writes_marker(tmp_path):
    trainer = _make_trainer(tmp_path)
    _advance(trainer, 100)
    assert trainer._redistill_phase_b is True
    assert trainer.model.transfer_teacher_late_tensors is None
    assert trainer.model_setup.phase_b_start == 100

    marker_path = os.path.join(trainer.config.workspace_dir, "backup", "redistill-phase.json")
    with open(marker_path) as f:
        assert json.load(f)["phase_b_start_step"] == 100

    # Idempotent: a later call must not re-run the boundary work.
    trainer.model.transfer_teacher_late_tensors = {"sentinel": torch.zeros(1)}
    _advance(trainer, 150)
    assert trainer.model.transfer_teacher_late_tensors is not None


def test_advance_restores_transfer_best(tmp_path):
    trainer = _make_trainer(tmp_path)
    best_path = os.path.join(trainer.config.workspace_dir, "backup", "transfer-best.pt")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    torch.save([torch.full((2,), 7.0)], best_path)
    trainer._transfer_best_backup_path = best_path

    _advance(trainer, 100)
    assert float(trainer.parameters[0].data.sum()) == pytest.approx(14.0)
    # The restored adapter re-warms the OFT spectral state (and refreshes any rotation cache).
    assert trainer.model_setup.restore_refreshes == [True]


def test_resume_past_boundary_enters_phase_b_without_restore(tmp_path):
    # First run crosses the boundary and writes the marker.
    trainer = _make_trainer(tmp_path)
    _advance(trainer, 100)

    # Resumed run: marker read at construction, Phase B entered on the first
    # step with the ORIGINAL boundary step as the offset and no best-restore.
    resumed = _make_trainer(tmp_path, continue_last_backup=True)
    assert resumed._redistill_resume_start_step == 100
    best_path = os.path.join(resumed.config.workspace_dir, "backup", "transfer-best.pt")
    torch.save([torch.full((2,), 7.0)], best_path)
    resumed._transfer_best_backup_path = best_path

    _advance(resumed, 140)
    assert resumed._redistill_phase_b is True
    assert resumed.model_setup.phase_b_start == 100
    assert float(resumed.parameters[0].data.sum()) == pytest.approx(0.0)  # no restore on resume


def test_fresh_run_clears_stale_marker(tmp_path):
    trainer = _make_trainer(tmp_path)
    _advance(trainer, 100)
    fresh = _make_trainer(tmp_path, continue_last_backup=False)
    marker_path = os.path.join(fresh.config.workspace_dir, "backup", "redistill-phase.json")
    assert not os.path.isfile(marker_path)
    assert fresh._redistill_resume_start_step is None
