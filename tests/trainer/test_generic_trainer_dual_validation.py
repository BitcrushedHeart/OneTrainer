"""A run can validate twice per check: plain validation loss on the VALIDATION
concepts without DPO patterns (-> patience) and DPO accuracy/margin on the
VALIDATION concepts with patterns (-> DPO patience). The two kinds use different
data pipelines, so GenericTrainer keeps one loader per kind and __validate routes
each loader to its own pass."""

from types import SimpleNamespace
from unittest.mock import Mock

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.trainer import GenericTrainer as generic_trainer_module
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util import create
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType

import torch

import pytest

_DPO_METRICS = {
    "loss": 0.1,
    "dpo_loss": 0.1,
    "accuracy": 0.75,
    "chosen_reward": 0.4,
    "rejected_reward": 0.1,
    "reward_margin": 0.3,
}


def _standard_batch(name: str, seed: int) -> dict:
    return {
        "concept_name": [name],
        "concept_path": [f"/data/{name}"],
        "concept_seed": torch.tensor([seed]),
    }


def _loader(events: list[str], tag: str, batches: list) -> Mock:
    loader = Mock()
    loader.get_data_set.return_value.start_next_epoch.side_effect = lambda: events.append(f"{tag}_epoch")
    loader.get_data_set.return_value.approximate_length.return_value = len(batches)
    loader.get_data_loader.return_value = batches
    return loader


def _make_trainer(*, validation: bool, dpo_validation: bool, standard_losses=(0.5, 0.7)):
    trainer = GenericTrainer.__new__(GenericTrainer)
    config = TrainConfig.default_values()
    config.tensorboard = False
    config.validation = validation
    config.rlhf_enabled = dpo_validation
    config.rlhf_dpo_validation = dpo_validation
    config.patience = True
    trainer.config = config

    events: list[str] = []
    trainer.events = events
    trainer.callbacks = Mock()
    trainer.tensorboard = Mock()
    trainer.model = Mock()

    setup = trainer.model_setup = Mock()
    setup.setup_train_device.side_effect = lambda model, config: events.append("device")
    setup.calculate_dpo_loss.side_effect = lambda *args, **kwargs: events.append("dpo_forward")
    setup.get_last_dpo_metrics.return_value = dict(_DPO_METRICS)
    setup.predict.side_effect = lambda *args, **kwargs: events.append("standard_forward") or {}
    losses = iter(standard_losses)
    setup.calculate_loss.side_effect = lambda *args, **kwargs: torch.tensor(next(losses))

    # Validation is due on every call; the periodic gc never fires.
    trainer._GenericTrainer__needs_validate = lambda train_progress: True
    trainer._GenericTrainer__needs_gc = lambda train_progress: False
    trainer._GenericTrainer__check_patience = Mock()
    trainer._GenericTrainer__check_dpo_patience = Mock()

    trainer.validation_data_loader = None
    trainer.dpo_validation_data_loader = None
    return trainer


def _validate(trainer, step: int = 7):
    trainer._GenericTrainer__validate(SimpleNamespace(global_step=step))


def _scalar_tags(trainer) -> list[str]:
    return [call.args[0] for call in trainer.tensorboard.add_scalar.call_args_list]


# --------------------------------------------------------------------------- #
# __validate routing
# --------------------------------------------------------------------------- #


def test_both_loaders_run_both_passes_and_patience_sees_the_standard_loss():
    trainer = _make_trainer(validation=True, dpo_validation=True, standard_losses=(0.5, 0.7))
    events = trainer.events
    trainer.dpo_validation_data_loader = _loader(events, "dpo", [{"pair": 1}, {"pair": 2}, {"pair": 3}])
    trainer.validation_data_loader = _loader(
        events, "standard", [_standard_batch("cats", 1), _standard_batch("dogs", 2)]
    )

    _validate(trainer, step=7)

    # DPO pass: every pair scored under the DPO loss, the four dpo/val_* scalars
    # written and DPO patience fed the averaged accuracy / margin.
    assert trainer.model_setup.calculate_dpo_loss.call_count == 3
    tags = _scalar_tags(trainer)
    assert [t for t in tags if t.startswith("dpo/")] == [
        "dpo/val_accuracy",
        "dpo/val_chosen_reward",
        "dpo/val_rejected_reward",
        "dpo/val_reward_margin",
    ]
    trainer._GenericTrainer__check_dpo_patience.assert_called_once()
    acc, margin, progress = trainer._GenericTrainer__check_dpo_patience.call_args.args
    assert acc == pytest.approx(0.75) and margin == pytest.approx(0.3) and progress.global_step == 7

    # Standard pass: every plain batch predicted + scored, per-concept scalars
    # written, and patience fed the total average loss (not anything DPO).
    assert trainer.model_setup.predict.call_count == 2
    assert trainer.model_setup.calculate_loss.call_count == 2
    assert "loss/validation_step/cats" in tags and "loss/validation_step/dogs" in tags
    assert "loss/validation_step/total_average" in tags
    trainer._GenericTrainer__check_patience.assert_called_once()
    loss, progress = trainer._GenericTrainer__check_patience.call_args.args
    assert loss == pytest.approx(0.6) and progress.global_step == 7

    # Both loaders advance their epoch (caching can park the model on the temp
    # device) BEFORE the single setup_train_device, then DPO runs, then standard.
    assert events.count("device") == 1
    device_at = events.index("device")
    assert events.index("dpo_epoch") < device_at and events.index("standard_epoch") < device_at
    assert events[device_at + 1 :] == ["dpo_forward"] * 3 + ["standard_forward"] * 2
    trainer.callbacks.on_update_status.assert_called_once_with("Calculating validation loss")


def test_dpo_only_config_routes_the_single_loader_to_the_dpo_pass():
    trainer = _make_trainer(validation=False, dpo_validation=True)
    trainer.validation_data_loader = _loader(trainer.events, "dpo", [{"pair": 1}, {"pair": 2}])
    assert trainer.dpo_validation_data_loader is None

    _validate(trainer)

    assert trainer.model_setup.calculate_dpo_loss.call_count == 2
    trainer.model_setup.predict.assert_not_called()
    trainer.model_setup.calculate_loss.assert_not_called()
    trainer._GenericTrainer__check_dpo_patience.assert_called_once()
    trainer._GenericTrainer__check_patience.assert_not_called()
    assert all(tag.startswith("dpo/") for tag in _scalar_tags(trainer))
    assert trainer.events == ["dpo_epoch", "device", "dpo_forward", "dpo_forward"]


def test_standard_only_config_is_unchanged():
    trainer = _make_trainer(validation=True, dpo_validation=False, standard_losses=(1.0, 3.0))
    trainer.validation_data_loader = _loader(
        trainer.events, "standard", [_standard_batch("cats", 1), _standard_batch("cats", 1)]
    )

    _validate(trainer, step=3)

    trainer.model_setup.calculate_dpo_loss.assert_not_called()
    trainer._GenericTrainer__check_dpo_patience.assert_not_called()
    assert trainer.model_setup.calculate_loss.call_count == 2
    trainer._GenericTrainer__check_patience.assert_called_once()
    loss, progress = trainer._GenericTrainer__check_patience.call_args.args
    assert loss == pytest.approx(2.0) and progress.global_step == 3
    # A single concept gets its own scalar and no total_average line.
    assert _scalar_tags(trainer) == ["loss/validation_step/cats"]
    assert trainer.events == ["standard_epoch", "device", "standard_forward", "standard_forward"]


def test_both_loaders_empty_skips_device_setup_and_status():
    trainer = _make_trainer(validation=True, dpo_validation=True)
    trainer.dpo_validation_data_loader = _loader(trainer.events, "dpo", [])
    trainer.validation_data_loader = _loader(trainer.events, "standard", [])

    _validate(trainer)

    assert trainer.events == ["dpo_epoch", "standard_epoch"]
    trainer.callbacks.on_update_status.assert_not_called()
    trainer._GenericTrainer__check_patience.assert_not_called()
    trainer._GenericTrainer__check_dpo_patience.assert_not_called()


def test_empty_dpo_loader_still_runs_the_standard_pass():
    trainer = _make_trainer(validation=True, dpo_validation=True, standard_losses=(0.25,))
    trainer.dpo_validation_data_loader = _loader(trainer.events, "dpo", [])
    trainer.validation_data_loader = _loader(trainer.events, "standard", [_standard_batch("cats", 1)])

    _validate(trainer)

    trainer.model_setup.calculate_dpo_loss.assert_not_called()
    trainer._GenericTrainer__check_dpo_patience.assert_not_called()
    trainer._GenericTrainer__check_patience.assert_called_once()
    assert trainer._GenericTrainer__check_patience.call_args.args[0] == pytest.approx(0.25)
    assert trainer.events == ["dpo_epoch", "standard_epoch", "device", "standard_forward"]


def test_validation_not_due_touches_nothing():
    trainer = _make_trainer(validation=True, dpo_validation=True)
    trainer._GenericTrainer__needs_validate = lambda train_progress: False
    trainer.dpo_validation_data_loader = _loader(trainer.events, "dpo", [{"pair": 1}])
    trainer.validation_data_loader = _loader(trainer.events, "standard", [_standard_batch("cats", 1)])

    _validate(trainer)

    assert trainer.events == []


def test_end_clears_the_dpo_validation_loader(monkeypatch):
    trainer = _make_trainer(validation=True, dpo_validation=True)
    trainer.one_step_trained = False
    trainer.temp_device = torch.device("cpu")
    trainer.model = Mock(ema=None)
    trainer.grad_hook_handles = []
    trainer.model_loader = trainer.model_saver = trainer.model_sampler = trainer.data_loader = Mock()
    trainer.sft_anchor_source = trainer.dpo_addon_source = None
    trainer.parameters = []
    trainer.sample_queue = []
    trainer.dpo_validation_data_loader = Mock()
    trainer.validation_data_loader = Mock()
    monkeypatch.setattr(generic_trainer_module.multi, "is_master", lambda: True)

    trainer.end()

    assert trainer.validation_data_loader is None
    assert trainer.dpo_validation_data_loader is None


# --------------------------------------------------------------------------- #
# loader construction: the dpo_validation kwarg
# --------------------------------------------------------------------------- #


class _RecordingLoader(BaseDataLoader):
    def _create_dataset(self, config, model, model_setup, train_progress, is_validation):
        self.seen_config = config
        self.seen_is_validation = is_validation
        return Mock()


def _dpo_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_validation = True
    config.batch_size = 4
    return config


@pytest.mark.parametrize(
    ("dpo_validation", "expected_rlhf"),
    [(None, True), (True, True), (False, False)],
)
def test_base_data_loader_dpo_validation_kwarg_picks_the_pipeline(dpo_validation, expected_rlhf):
    config = _dpo_config()
    loader = _RecordingLoader(
        torch.device("cpu"),
        torch.device("cpu"),
        config,
        Mock(),
        Mock(),
        Mock(),
        is_validation=True,
        dpo_validation=dpo_validation,
    )
    assert loader.seen_is_validation is True
    assert loader.seen_config.rlhf_enabled is expected_rlhf
    assert loader.is_dpo_validation is expected_rlhf
    # Validation loaders still get the usual single-sample, single-GPU copy,
    # and the caller's config is never touched.
    assert loader.seen_config.batch_size == 1 and loader.seen_config.multi_gpu is False
    assert config.rlhf_enabled is True and config.batch_size == 4


def test_base_data_loader_default_without_dpo_validation_follows_the_config():
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_validation = False
    loader = _RecordingLoader(
        torch.device("cpu"), torch.device("cpu"), config, Mock(), Mock(), Mock(), is_validation=True
    )
    assert loader.seen_config.rlhf_enabled is False
    assert loader.is_dpo_validation is False

    training_loader = _RecordingLoader(torch.device("cpu"), torch.device("cpu"), config, Mock(), Mock(), Mock())
    assert training_loader.seen_config.rlhf_enabled is True
    assert training_loader.is_dpo_validation is False


def test_create_data_loader_forwards_dpo_validation(monkeypatch):
    """The kwarg travels BaseTrainer -> create.create_data_loader -> the loader class."""
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = _dpo_config()
    trainer.train_device = torch.device("cpu")
    trainer.temp_device = torch.device("cpu")
    model, setup, progress = Mock(), Mock(), Mock()

    constructed = []

    class _FakeLoader:
        def __init__(self, *args, **kwargs):
            constructed.append((args, kwargs))

    monkeypatch.setattr(create.factory, "get", lambda *args: _FakeLoader)

    loader = trainer.create_data_loader(model, setup, progress, is_validation=True, dpo_validation=False)
    assert isinstance(loader, _FakeLoader)
    args, kwargs = constructed[-1]
    assert args == (trainer.train_device, trainer.temp_device, trainer.config, model, setup, progress, True, False)
    assert kwargs == {"dpo_validation": False}

    # No kind given: the loader receives None and keeps its config-driven default.
    trainer.create_data_loader(model, setup, progress, is_validation=True)
    assert constructed[-1][1] == {"dpo_validation": None}


# --------------------------------------------------------------------------- #
# concept selection for a plain validation loader
# --------------------------------------------------------------------------- #


def _concept(name: str, concept_type: ConceptType, dpo: bool) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.name = name
    concept.path = f"/data/{name}"
    concept.enabled = True
    concept.type = concept_type
    if dpo:
        concept.dpo_chosen_pattern = "chosen/{}"
        concept.dpo_rejected_pattern = "rejected/{}"
    return concept


def test_select_concepts_plain_validation_loader_drops_dpo_pattern_concepts():
    config = TrainConfig.default_values()
    config.rlhf_enabled = False
    config.concepts = [
        _concept("plain_val", ConceptType.VALIDATION, dpo=False),
        _concept("paired_val", ConceptType.VALIDATION, dpo=True),
        _concept("train", ConceptType.STANDARD, dpo=False),
        _concept("paired_train", ConceptType.STANDARD, dpo=True),
    ]
    mixin = DataLoaderMgdsMixin()

    assert [c.name for c in mixin._select_concepts(config, is_validation=True)] == ["plain_val"]
    # The training-side selection is untouched: a plain (non-anchor) training
    # loader keeps every STANDARD concept.
    assert [c.name for c in mixin._select_concepts(config, is_validation=False)] == ["train", "paired_train"]


def test_select_concepts_dpo_validation_loader_keeps_only_dpo_pattern_concepts():
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.concepts = [
        _concept("plain_val", ConceptType.VALIDATION, dpo=False),
        _concept("paired_val", ConceptType.VALIDATION, dpo=True),
    ]
    mixin = DataLoaderMgdsMixin()

    assert [c.name for c in mixin._select_concepts(config, is_validation=True)] == ["paired_val"]
