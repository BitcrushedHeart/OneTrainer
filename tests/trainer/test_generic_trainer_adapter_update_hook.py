"""GenericTrainer calls BaseModelSetup.after_optimizer_step_adapters exactly once per
optimizer update (not per micro-step), after the step and before after_optimizer_step,
on every step path: plain, grad scaler, fused back pass (with and without scaler).

CPU-only. A real GenericTrainer.train() runs over mocked model/setup/loader with one
real parameter, so backward and the fused-back-pass grad hooks actually fire.
"""

from unittest.mock import Mock

import modules.trainer.GenericTrainer as generic_trainer_module
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch

import pytest


def _driven_trainer(tmp_path, monkeypatch, *, gradient_accumulation_steps, batches, use_scaler, fused):
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.workspace_dir = str(tmp_path / "workspace")
    config.concept_file_name = str(tmp_path / "missing-concepts.json")
    config.sample_definition_file_name = str(tmp_path / "missing-samples.json")
    config.tensorboard = False
    config.validation = False
    config.epochs = 1
    config.batch_size = 1
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.sample_after_unit = TimeUnit.NEVER
    config.backup_after_unit = TimeUnit.NEVER
    config.save_every_unit = TimeUnit.NEVER
    config.rolling_backup = False
    config.optimizer.optimizer = Optimizer.ADAMW  # supports the fused back pass
    config.optimizer.fused_back_pass = fused
    config.clip_grad_norm = None

    events: list[str] = []
    parameter = torch.nn.Parameter(torch.ones(3))

    trainer = GenericTrainer(config, Mock(), TrainCommands())
    trainer.tensorboard = Mock()
    trainer.parameters = [parameter]
    trainer.sample_queue = []
    trainer.sft_anchor_source = None
    trainer.dpo_addon_source = None

    optimizer = Mock(param_groups=[{"params": [parameter]}])
    optimizer.step.side_effect = lambda: events.append("optimizer.step")
    optimizer.step_parameter.side_effect = lambda *args: events.append("optimizer.step_parameter")
    optimizer.zero_grad.side_effect = lambda set_to_none=True: events.append("zero_grad")
    trainer.model = Mock(
        ema=None, accumulator_state=None, parameters=None, train_progress=TrainProgress(), optimizer=optimizer
    )

    setup = trainer.model_setup = Mock()
    setup.predict.side_effect = lambda *args, **kwargs: events.append("predict") or {}
    setup.calculate_loss.side_effect = lambda *args, **kwargs: (parameter * 2.0).sum()
    setup.after_optimizer_step_adapters.side_effect = lambda model, config: events.append("adapters")
    setup.after_optimizer_step.side_effect = lambda *args: events.append("after_optimizer_step")

    trainer.model_saver = Mock()
    trainer.data_loader = Mock()
    data_set = trainer.data_loader.get_data_set.return_value
    data_set.approximate_length.return_value = batches
    trainer.data_loader.get_data_loader.return_value = [{"concept_type": ["STANDARD"]} for _ in range(batches)]

    if use_scaler:
        scaler = Mock()
        scaler.scale.side_effect = lambda tensor: tensor
        scaler.step.side_effect = lambda opt: events.append("scaler.step")
        scaler.step_after_unscale_parameter_.side_effect = lambda opt: events.append("scaler.step_after_unscale")
        scaler.maybe_opt_step_parameter.side_effect = lambda *args: events.append("optimizer.step_parameter")
        monkeypatch.setattr(generic_trainer_module, "create_grad_scaler", lambda: scaler)
    monkeypatch.setattr(generic_trainer_module, "enable_grad_scaling", lambda *args: use_scaler)
    monkeypatch.setattr(generic_trainer_module, "gc", Mock())
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())
    return trainer, events, setup


@pytest.mark.parametrize(
    ("use_scaler", "fused", "step_event"),
    [
        (False, False, "optimizer.step"),
        (True, False, "scaler.step"),
        (False, True, "optimizer.step_parameter"),
        (True, True, "optimizer.step_parameter"),
    ],
    ids=["plain", "scaler", "fused", "fused_scaler"],
)
def test_adapter_hook_runs_once_per_optimizer_update(tmp_path, monkeypatch, use_scaler, fused, step_event):
    trainer, events, setup = _driven_trainer(
        tmp_path, monkeypatch, gradient_accumulation_steps=3, batches=6, use_scaler=use_scaler, fused=fused
    )

    trainer.train()

    assert events.count("predict") == 6
    assert setup.after_optimizer_step_adapters.call_count == 2  # 6 micro-steps / GA 3
    assert setup.after_optimizer_step.call_count == 2
    for call in setup.after_optimizer_step_adapters.call_args_list:
        assert call.args == (trainer.model, trainer.config)

    # Each update window: 3 predicts, the parameter step(s), zero_grad, then the
    # adapter hook, then the model-specific after_optimizer_step.
    windows = []
    current: list[str] = []
    for event in events:
        current.append(event)
        if event == "after_optimizer_step":
            windows.append(current)
            current = []
    assert current == []
    assert len(windows) == 2
    for window in windows:
        assert window.count("predict") == 3
        assert window.count("adapters") == 1
        assert step_event in window
        last_step = max(i for i, e in enumerate(window) if e == step_event)
        assert last_step < window.index("zero_grad") < window.index("adapters") < window.index("after_optimizer_step")
        assert window.index("adapters") > max(i for i, e in enumerate(window) if e == "predict")
