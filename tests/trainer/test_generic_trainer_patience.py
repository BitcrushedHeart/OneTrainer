from types import SimpleNamespace
from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.PatienceMode import PatienceMode

import torch

import pytest


def _make_trainer(tmp_path, mode, patience_checks, min_checks=0, second_best=False):
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = TrainConfig.default_values()
    trainer.config.patience = True
    trainer.config.patience_epochs = patience_checks
    trainer.config.patience_min_checks = min_checks
    trainer.config.patience_save_second_best = second_best
    trainer.config.patience_mode = mode
    trainer.config.workspace_dir = str(tmp_path)
    trainer.config.output_model_destination = str(tmp_path / "model.safetensors")
    trainer.config.backup_before_save = False
    trainer.config.tensorboard = False
    trainer.commands = TrainCommands()
    trainer.tensorboard = Mock()
    trainer.callbacks = Mock()
    trainer.parameters = [torch.nn.Parameter(torch.tensor([0.0]))]
    trainer._patience_counter = 0
    trainer._patience_checks = 0
    trainer._patience_best_loss = float("inf")
    trainer._patience_best_step = -1
    trainer._patience_best_backup_path = None
    trainer._patience_alt_backup_path = None
    trainer._patience_alt_step = -1
    trainer._patience_second_backup_path = None
    trainer._patience_second_loss = float("inf")
    trainer._patience_second_step = -1
    return trainer


def _checker(trainer):
    def check(step, loss):
        trainer.parameters[0].data.fill_(step)
        trainer._GenericTrainer__check_patience(loss, SimpleNamespace(global_step=step))

    return check


@pytest.mark.parametrize("mode", list(PatienceMode))
@pytest.mark.parametrize("patience_checks", [2, 6, 9])
def test_patience_stops_at_configured_threshold_and_saves_expected_weights(
    tmp_path,
    monkeypatch,
    mode,
    patience_checks,
):
    trainer = _make_trainer(tmp_path, mode, patience_checks)
    check = _checker(trainer)

    check(1, 1.0)
    check(2, 1.1)
    check(3, 0.9)  # A new best resets the consecutive failure count.
    stop_step = 3 + patience_checks
    for step in range(4, stop_step):
        check(step, 1.0)
        assert not trainer.commands.get_stop_command()
        assert trainer._patience_alt_backup_path is None
    check(stop_step, 1.0)
    assert trainer.commands.get_stop_command()
    assert trainer._patience_counter == patience_checks

    trainer.one_step_trained = True
    trainer.temp_device = torch.device("cpu")
    trainer.model = Mock(ema=None)
    trainer.grad_hook_handles = []
    saved = {}
    trainer.model_saver = Mock()
    trainer.model_saver.save.side_effect = lambda **kwargs: saved.update(
        {
            kwargs["output_model_destination"]: trainer.parameters[0].detach().clone(),
        }
    )
    monkeypatch.setattr("modules.trainer.GenericTrainer.multi.is_master", lambda: True)
    trainer.end()

    assert saved[str(tmp_path / "model.safetensors")].item() == 3
    if mode == PatienceMode.SAVE_ALTERNATIVE_AT_END:
        assert saved[str(tmp_path / "model-patience-alt.safetensors")].item() == stop_step
        assert len(saved) == 2
    else:
        assert len(saved) == 1


@pytest.mark.parametrize("mode", list(PatienceMode))
def test_patience_min_checks_ignores_warmup_validations(tmp_path, mode):
    trainer = _make_trainer(tmp_path, mode, patience_checks=2, min_checks=3)
    check = _checker(trainer)

    # Three warm-up checks: the untrained step-1 loss is the lowest of the whole
    # run, but it must be neither the best nor counted against patience.
    check(1, 0.1)
    check(2, 0.5)
    check(3, 0.5)
    assert trainer._patience_best_step == -1
    assert trainer._patience_best_loss == float("inf")
    assert trainer._patience_counter == 0
    assert trainer._patience_best_backup_path is None
    assert not trainer.commands.get_stop_command()

    # Tracking starts at check 4; the first tracked loss becomes the best.
    check(4, 0.9)
    assert trainer._patience_best_step == 4
    assert trainer._patience_counter == 0
    check(5, 0.8)
    assert trainer._patience_best_step == 5
    check(6, 1.0)
    assert not trainer.commands.get_stop_command()
    check(7, 1.0)
    assert trainer.commands.get_stop_command()
    assert trainer._patience_counter == 2
    assert trainer._patience_best_step == 5


def test_patience_min_checks_zero_keeps_old_behaviour(tmp_path):
    trainer = _make_trainer(tmp_path, PatienceMode.STOP_TRAINING, patience_checks=1, min_checks=0)
    check = _checker(trainer)
    check(1, 0.1)
    assert trainer._patience_best_step == 1
    check(2, 0.2)
    assert trainer.commands.get_stop_command()


def _finish(trainer, monkeypatch):
    trainer.one_step_trained = True
    trainer.temp_device = torch.device("cpu")
    trainer.model = Mock(ema=None)
    trainer.grad_hook_handles = []
    saved = {}
    trainer.model_saver = Mock()
    trainer.model_saver.save.side_effect = lambda **kwargs: saved.update(
        {
            kwargs["output_model_destination"]: trainer.parameters[0].detach().clone(),
        }
    )
    monkeypatch.setattr("modules.trainer.GenericTrainer.multi.is_master", lambda: True)
    trainer.end()
    return saved


@pytest.mark.parametrize("mode", list(PatienceMode))
def test_second_best_output_tracks_runner_up_validation(tmp_path, monkeypatch, mode):
    trainer = _make_trainer(tmp_path, mode, patience_checks=3, second_best=True)
    check = _checker(trainer)

    check(1, 1.0)  # best
    check(2, 0.8)  # new best; step 1 demoted to second
    assert trainer._patience_second_step == 1 and trainer._patience_second_loss == 1.0
    check(3, 0.9)  # not a best, but beats the runner-up -> becomes second
    assert trainer._patience_second_step == 3 and trainer._patience_second_loss == 0.9
    check(4, 0.95)  # worse than both: nothing changes
    assert trainer._patience_second_step == 3
    check(5, 0.7)  # new best; step 2 (0.8) demoted over the 0.9 runner-up
    assert trainer._patience_best_step == 5
    assert trainer._patience_second_step == 2 and trainer._patience_second_loss == 0.8
    for step in (6, 7):
        check(step, 1.1)
        assert not trainer.commands.get_stop_command()
    check(8, 1.1)
    assert trainer.commands.get_stop_command()

    saved = _finish(trainer, monkeypatch)
    assert saved[str(tmp_path / "model.safetensors")].item() == 5
    assert saved[str(tmp_path / "model-patience-second.safetensors")].item() == 2
    if mode == PatienceMode.SAVE_ALTERNATIVE_AT_END:
        assert saved[str(tmp_path / "model-patience-alt.safetensors")].item() == 8
        assert len(saved) == 3
    else:
        assert len(saved) == 2


def test_second_best_disabled_writes_no_extra_output(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, PatienceMode.STOP_TRAINING, patience_checks=1, second_best=False)
    check = _checker(trainer)
    check(1, 1.0)
    check(2, 0.8)
    check(3, 0.9)
    assert trainer._patience_second_backup_path is None
    saved = _finish(trainer, monkeypatch)
    assert list(saved) == [str(tmp_path / "model.safetensors")]
