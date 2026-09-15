from types import SimpleNamespace
from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.PatienceMode import PatienceMode

import pytest
import torch


@pytest.mark.parametrize("mode", list(PatienceMode))
@pytest.mark.parametrize("patience_checks", [2, 6, 9])
def test_patience_stops_at_configured_threshold_and_saves_expected_weights(
    tmp_path, monkeypatch, mode, patience_checks,
):
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = TrainConfig.default_values()
    trainer.config.patience = True
    trainer.config.patience_epochs = patience_checks
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
    trainer._patience_best_loss = float("inf")
    trainer._patience_best_step = -1
    trainer._patience_best_backup_path = None
    trainer._patience_alt_backup_path = None
    trainer._patience_alt_step = -1

    def check(step, loss):
        trainer.parameters[0].data.fill_(step)
        trainer._GenericTrainer__check_patience(loss, SimpleNamespace(global_step=step))

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
    trainer.model_saver.save.side_effect = lambda **kwargs: saved.update({
        kwargs["output_model_destination"]: trainer.parameters[0].detach().clone(),
    })
    monkeypatch.setattr("modules.trainer.GenericTrainer.multi.is_master", lambda: True)
    trainer.end()

    assert saved[str(tmp_path / "model.safetensors")].item() == 3
    if mode == PatienceMode.SAVE_ALTERNATIVE_AT_END:
        assert saved[str(tmp_path / "model-patience-alt.safetensors")].item() == stop_step
        assert len(saved) == 2
    else:
        assert len(saved) == 1
