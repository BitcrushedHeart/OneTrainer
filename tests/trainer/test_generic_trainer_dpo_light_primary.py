"""GenericTrainer under ``rlhf_dpo_light_primary``: DPO-primary pair micro-steps
on the light path (calculate_dpo_loss_light, half-by-half forwards backwarded
inside) instead of the fused [chosen; rejected] forward and the common backward.

CPU-only. A real GenericTrainer.train() runs over mocked model/setup/loaders
with one real parameter, so the per-term 1 / GA scaling, the gradient
accumulation windows, the KTO batch and SFT-anchor terms folded into a pair
micro-step and the logged loss are compared between the two paths for real.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import modules.trainer.GenericTrainer as generic_trainer_module
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch

import pytest

BATCH_SIZE = 2
PAIRS = 6
# d(term)/d(parameter) per element before scaling, and each term's value.
DPO_GRAD_CHOSEN, DPO_GRAD_REJECTED, DPO_LOSS = 2.0, 1.0, 0.7
KTO_GRAD, KTO_LOSS = 3.0, 0.3
SFT_GRAD, SFT_LOSS, ANCHOR_WEIGHT = 5.0, 0.2, 0.5


class _FakeDataSet:
    def __init__(self, length: int):
        self.length = length
        self.loading_pipeline = SimpleNamespace(
            _LoadingPipeline__initial_index=0, _LoadingPipeline__mark_reset_serial_modules=False
        )

    def start_next_epoch(self):
        pass

    def approximate_length(self) -> int:
        return self.length


class _FakeDataLoader:
    def __init__(self, name: str, length: int):
        self.name = name
        self.data_set = _FakeDataSet(length)
        self.pulled: list[int] = []

    def get_data_set(self):
        return self.data_set

    def get_data_loader(self):
        for index in range(self.data_set.length):
            self.pulled.append(index)
            yield {"concept_type": ["STANDARD"] * BATCH_SIZE, "index": index, "source": self.name}


def _trainer(tmp_path, monkeypatch, *, light: bool, ga: int, kto: bool = False, anchor: bool = False):
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
    config.batch_size = BATCH_SIZE
    config.gradient_accumulation_steps = ga
    config.sample_after_unit = TimeUnit.NEVER
    config.backup_after_unit = TimeUnit.NEVER
    config.save_every_unit = TimeUnit.NEVER
    config.rolling_backup = False
    config.clip_grad_norm = None
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.DPO
    config.rlhf_dpo_light_primary = light
    config.rlhf_sft_anchor_weight = ANCHOR_WEIGHT if anchor else 0.0

    events: list[str] = []
    parameter = torch.nn.Parameter(torch.ones(3))
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    trainer.tensorboard = Mock()
    trainer.parameters = [parameter]
    trainer.sample_queue = []
    trainer.dpo_addon_source = None
    trainer._text_reward_scheduler = None
    trainer._text_reward_bank = None

    optimizer = Mock(param_groups=[{"params": [parameter]}])
    grads_at_step: list[torch.Tensor] = []
    optimizer.step.side_effect = lambda: grads_at_step.append(parameter.grad.detach().clone())
    optimizer.zero_grad.side_effect = lambda set_to_none=True: setattr(parameter, "grad", None)
    trainer.model = Mock(
        ema=None,
        accumulator_state=None,
        parameters=None,
        train_progress=TrainProgress(),
        optimizer=optimizer,
    )

    setup = trainer.model_setup = Mock()
    setup.pop_perceptual_metrics.return_value = {}

    def with_value(grad: float, value: float) -> torch.Tensor:
        # parameter is ones(3): the sum carries the gradient, the constant sets the value.
        return (parameter * grad).sum() - 3 * grad + value

    def dpo_fused(model, batch, config, train_progress):
        events.append("fused")
        return with_value(DPO_GRAD_CHOSEN + DPO_GRAD_REJECTED, DPO_LOSS)

    def dpo_light(model, batch, config, train_progress, backward_fn):
        events.append("light")
        backward_fn((parameter * DPO_GRAD_CHOSEN).sum())
        backward_fn((parameter * DPO_GRAD_REJECTED).sum())
        return torch.tensor(DPO_LOSS)

    setup.calculate_dpo_loss.side_effect = dpo_fused
    setup.calculate_dpo_loss_light.side_effect = dpo_light
    setup.get_last_dpo_metrics.return_value = {
        "loss": DPO_LOSS,
        "dpo_loss": DPO_LOSS,
        "chosen_reward": 0.2,
        "rejected_reward": 0.1,
        "reward_margin": 0.1,
        "accuracy": 1.0,
    }
    setup.calculate_kto_loss.side_effect = lambda *args, **kwargs: (
        events.append("kto") or with_value(KTO_GRAD, KTO_LOSS)
    )
    setup.get_last_kto_metrics.return_value = {
        "loss": KTO_LOSS,
        "liked_reward_sum": 0.4,
        "liked_count": 1.0,
        "disliked_reward_sum": -0.1,
        "disliked_count": 1.0,
        "accuracy": 1.0,
    }
    setup.predict.side_effect = lambda *args, **kwargs: events.append("anchor") or {}
    setup.calculate_loss.side_effect = lambda *args, **kwargs: with_value(SFT_GRAD, SFT_LOSS)

    trainer.model_saver = Mock()
    trainer.data_loader = _FakeDataLoader("pairs", PAIRS)
    trainer.kto_source = (
        generic_trainer_module.CyclingBatchSource(_FakeDataLoader("kto", PAIRS), "KTO") if kto else None
    )
    trainer.sft_anchor_source = (
        generic_trainer_module.CyclingBatchSource(_FakeDataLoader("anchor", PAIRS), "SFT-anchor") if anchor else None
    )

    monkeypatch.setattr(generic_trainer_module, "enable_grad_scaling", lambda *args: False)
    monkeypatch.setattr(generic_trainer_module, "gc", Mock())
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())
    return SimpleNamespace(
        trainer=trainer, config=config, events=events, grads=grads_at_step, optimizer=optimizer, setup=setup
    )


def _logged(run, prefix: str) -> list[tuple[str, float, int]]:
    rows = []
    for call in run.trainer.tensorboard.add_scalar.call_args_list:
        tag, value, step = call.args[:3]
        if tag.startswith(prefix):
            rows.append((tag, float(value), step))
    return rows


def _expected_grad(ga: int, kto: bool, anchor: bool) -> float:
    per_step = (
        DPO_GRAD_CHOSEN + DPO_GRAD_REJECTED + (KTO_GRAD if kto else 0.0) + (ANCHOR_WEIGHT * SFT_GRAD if anchor else 0.0)
    )
    return per_step * ga / ga  # ga micro-steps, each scaled by 1 / ga


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_flag_defaults_off_and_survives_round_trip_and_old_configs():
    config = TrainConfig.default_values()
    assert config.rlhf_dpo_light_primary is False
    assert not config.rlhf_dpo_light_primary_applies()

    config.rlhf_dpo_light_primary = True
    loaded = TrainConfig.default_values().from_dict(config.to_dict())
    assert loaded.rlhf_dpo_light_primary is True

    old = config.to_dict()
    del old["rlhf_dpo_light_primary"]
    assert TrainConfig.default_values().from_dict(old).rlhf_dpo_light_primary is False
    assert TrainConfig.default_values().from_dict({"__version": 26}).rlhf_dpo_light_primary is False


def test_flag_applies_to_dpo_primary_runs_only():
    config = TrainConfig.default_values()
    config.rlhf_dpo_light_primary = True
    assert not config.rlhf_dpo_light_primary_applies()  # RLHF off
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.DPO
    assert config.rlhf_dpo_light_primary_applies()
    config.rlhf_mode = RLHFMode.SFT_DPO
    assert not config.rlhf_dpo_light_primary_applies()  # add-on steps are light already


def test_fused_back_pass_is_rejected_with_the_light_path():
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.DPO
    config.rlhf_dpo_light_primary = True
    assert config.validate_rlhf_startup() is None
    config.optimizer.fused_back_pass = True
    assert "fused back pass" in config.validate_rlhf_startup()
    config.rlhf_dpo_light_primary = False
    assert config.validate_rlhf_startup() is None


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ga", [1, 2, 3])
@pytest.mark.parametrize("kto, anchor", [(False, False), (True, False), (False, True), (True, True)])
def test_light_path_matches_the_fused_path_gradient_loss_and_progress(tmp_path, monkeypatch, ga, kto, anchor):
    fused = _trainer(tmp_path / "fused", monkeypatch, light=False, ga=ga, kto=kto, anchor=anchor)
    fused.trainer.train()
    light = _trainer(tmp_path / "light", monkeypatch, light=True, ga=ga, kto=kto, anchor=anchor)
    light.trainer.train()

    # The fused run forwards once per pair micro-step; the light run never
    # touches calculate_dpo_loss and backwards inside calculate_dpo_loss_light.
    assert fused.events.count("fused") == PAIRS and "light" not in fused.events
    assert light.events.count("light") == PAIRS and "fused" not in light.events
    fused.setup.calculate_dpo_loss_light.assert_not_called()
    light.setup.calculate_dpo_loss.assert_not_called()
    for run in (fused, light):
        assert run.events.count("kto") == (PAIRS if kto else 0)
        assert run.events.count("anchor") == (PAIRS if anchor else 0)

    # Same optimizer steps, each with the same accumulated gradient.
    assert fused.optimizer.step.call_count == PAIRS // ga
    assert light.optimizer.step.call_count == PAIRS // ga
    expected = torch.full((3,), _expected_grad(ga, kto, anchor))
    for fused_grad, light_grad in zip(fused.grads, light.grads, strict=True):
        assert torch.allclose(fused_grad, expected), fused_grad
        assert torch.allclose(light_grad, expected), light_grad
        assert torch.allclose(fused_grad, light_grad, rtol=0.0, atol=1e-6)

    # Same progress counters and the same logged loss / DPO metrics.
    for run in (fused, light):
        progress = run.trainer.model.train_progress
        assert progress.global_step == PAIRS
        assert progress.epoch == 1 and progress.epoch_step == 0
    fused_loss, light_loss = _logged(fused, "loss/"), _logged(light, "loss/")
    assert fused_loss == pytest.approx(light_loss)
    assert [row for row in fused_loss if row[0] == "loss/train_step"]
    expected_loss = DPO_LOSS + (KTO_LOSS if kto else 0.0) + (ANCHOR_WEIGHT * SFT_LOSS if anchor else 0.0)
    for tag, value, _ in light_loss:
        if tag == "loss/train_step":
            assert value == pytest.approx(expected_loss, abs=1e-6)
    assert _logged(fused, "dpo/") == pytest.approx(_logged(light, "dpo/"))
    assert _logged(fused, "kto/") == pytest.approx(_logged(light, "kto/"))
    if anchor:
        assert any(tag == "dpo/sft_anchor_loss" for tag, _, _ in _logged(light, "dpo/"))


def test_light_path_backward_callback_scales_each_term_by_gradient_accumulation(tmp_path, monkeypatch):
    """Each callback backwards its term as handed over, scaled by 1 / GA
    (the fused path's loss / GA): the two halves land as separate backwards
    whose sum is the pair's gradient."""
    ga = 4
    run = _trainer(tmp_path, monkeypatch, light=True, ga=ga)
    seen: list[torch.Tensor] = []
    parameter = run.trainer.parameters[0]

    def dpo_light(model, batch, config, train_progress, backward_fn):
        before = parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros(3)
        backward_fn((parameter * DPO_GRAD_CHOSEN).sum())
        after_chosen = parameter.grad.detach().clone()
        backward_fn((parameter * DPO_GRAD_REJECTED).sum())
        seen.append(after_chosen - before)
        seen.append(parameter.grad.detach() - after_chosen)
        return torch.tensor(DPO_LOSS)

    run.setup.calculate_dpo_loss_light.side_effect = dpo_light
    run.trainer.train()

    assert len(seen) == 2 * PAIRS
    for i, delta in enumerate(seen):
        grad = DPO_GRAD_CHOSEN if i % 2 == 0 else DPO_GRAD_REJECTED
        assert torch.allclose(delta, torch.full((3,), grad / ga)), (i, delta)
