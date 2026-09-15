"""Backup stall fixes: no base-model offload for LoRA backups, no hash/git work
for INTERNAL headers, and the epoch-start gc freeze. CPU-only, mocked model."""

import gc
import hashlib
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import modules.trainer.GenericTrainer as generic_trainer_module
from modules.modelSaver.mixin import DtypeModelSaverMixin as dtype_saver_module
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.modelSaver.mixin.LoRASaverMixin import LoRASaverMixin
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch

import pytest
from safetensors import safe_open


class _StopBeforeTrainingStep(Exception):
    """Raised from the "Training ..." status update, right after the backup/save block."""


def _driven_trainer(tmp_path, monkeypatch, training_method: TrainingMethod):
    """A real GenericTrainer whose model/setup/saver/loader are mocks, instrumented so
    train() runs one epoch start and the first batch's backup/save block, then stops."""
    config = TrainConfig.default_values()
    config.training_method = training_method
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.workspace_dir = str(tmp_path / "workspace")
    config.concept_file_name = str(tmp_path / "missing-concepts.json")
    config.sample_definition_file_name = str(tmp_path / "missing-samples.json")
    config.tensorboard = False
    config.epochs = 1
    config.sample_after_unit = TimeUnit.NEVER
    config.backup_after_unit = TimeUnit.NEVER
    config.save_every_unit = TimeUnit.NEVER
    config.rolling_backup = False

    events: list[tuple] = []

    def status(message):
        if message == "Training ...":
            raise _StopBeforeTrainingStep

    trainer = GenericTrainer(config, Mock(), TrainCommands())
    trainer.callbacks.on_update_status.side_effect = status
    trainer.model = Mock(ema=None, accumulator_state=None, parameters=None, train_progress=TrainProgress())
    trainer.model.to.side_effect = lambda device: events.append(("model.to", device))
    trainer.model_setup = Mock()
    trainer.model_setup.setup_train_device.side_effect = lambda *args: events.append(("setup_train_device",))
    trainer.model_setup.save_dpo_reference.side_effect = lambda path: events.append(("save_dpo_reference", path))
    trainer.model_saver = Mock()
    trainer.model_saver.save.side_effect = lambda *args, **kwargs: events.append(("model_saver.save", args, kwargs))
    trainer.data_loader = Mock()
    data_set = trainer.data_loader.get_data_set.return_value
    data_set.approximate_length.return_value = 1
    data_set.start_next_epoch.side_effect = lambda: events.append(("start_next_epoch",))
    trainer.data_loader.get_data_loader.return_value = [{"batch": 0}]
    trainer.parameters = [torch.nn.Parameter(torch.zeros(1))]
    trainer.sample_queue = []
    trainer.sft_anchor_source = None
    trainer.dpo_addon_source = None

    fake_gc = Mock()
    fake_gc.unfreeze.side_effect = lambda: events.append(("gc.unfreeze",))
    fake_gc.freeze.side_effect = lambda: events.append(("gc.freeze",))
    monkeypatch.setattr(generic_trainer_module, "gc", fake_gc)
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: events.append(("torch_gc",)))
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())
    return trainer, events


def _run_until_training_step(trainer):
    with pytest.raises(_StopBeforeTrainingStep):
        trainer.train()


def _names(events):
    return [event[0] for event in events]


@pytest.mark.parametrize(
    ("training_method", "backup", "save", "expect_offload"),
    [
        (TrainingMethod.LORA, True, False, False),
        (TrainingMethod.LORA, True, True, False),
        (TrainingMethod.LORA, False, True, True),  # a plain save keeps the old offload
        (TrainingMethod.FINE_TUNE, True, False, True),
        (TrainingMethod.EMBEDDING, True, False, True),
    ],
)
def test_backup_block_offloads_base_model_only_when_needed(
    tmp_path, monkeypatch, training_method, backup, save, expect_offload
):
    trainer, events = _driven_trainer(tmp_path, monkeypatch, training_method)
    if backup:
        trainer.commands.backup()
    if save:
        trainer.commands.save()

    _run_until_training_step(trainer)

    names = _names(events)
    saves = [event for event in events if event[0] == "model_saver.save"]
    assert len(saves) == int(backup) + int(save)
    first_save = names.index("model_saver.save")
    offloads = [i for i, event in enumerate(events) if event[0] == "model.to"]
    if expect_offload:
        assert len(offloads) == 1
        assert events[offloads[0]][1] == trainer.temp_device
        assert offloads[0] < first_save
    else:
        assert offloads == []

    if backup:
        _, args, _ = saves[0]
        assert args[2] == ModelFormat.INTERNAL
        backup_path = args[3]
        assert backup_path.startswith(str(tmp_path / "workspace" / "backup"))
        assert ("save_dpo_reference", backup_path) in events
        # __backup restores the train device before a same-step __save runs.
        assert "setup_train_device" in names[first_save:]
    if save:
        _, _, kwargs = saves[-1]
        assert kwargs["output_model_format"] == trainer.config.output_model_format
    # The block always ends by putting the model back on the train device.
    assert names[-1] == "setup_train_device"


def test_epoch_start_unfreezes_collects_then_freezes(tmp_path, monkeypatch):
    trainer, events = _driven_trainer(tmp_path, monkeypatch, TrainingMethod.LORA)

    _run_until_training_step(trainer)

    names = _names(events)
    assert names.count("gc.unfreeze") == 1
    assert names.count("gc.freeze") == 1
    unfreeze = names.index("gc.unfreeze")
    assert names[unfreeze : unfreeze + 3] == ["gc.unfreeze", "torch_gc", "gc.freeze"]
    # Runs after the epoch's loaders advanced and the model was placed for training.
    assert names.index("start_next_epoch") < unfreeze
    assert names.index("setup_train_device") < unfreeze


def test_collect_garbage_and_freeze_uses_real_gc_and_end_thaws(tmp_path, monkeypatch):
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = TrainConfig.default_values()
    trainer.config.tensorboard = False
    trainer.one_step_trained = False
    trainer.model = None
    trainer.tensorboard = None
    trainer.grad_hook_handles = []
    monkeypatch.setattr(generic_trainer_module, "torch_gc", Mock())

    gc.unfreeze()
    try:
        trainer._collect_garbage_and_freeze()
        generic_trainer_module.torch_gc.assert_called_once_with()
        assert gc.get_freeze_count() > 0
        trainer.end()
        assert gc.get_freeze_count() == 0
    finally:
        gc.unfreeze()


def test_end_unfreezes_before_final_save(tmp_path, monkeypatch):
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = TrainConfig.default_values()
    trainer.config.workspace_dir = str(tmp_path)
    trainer.config.output_model_destination = str(tmp_path / "model.safetensors")
    trainer.config.backup_before_save = False
    trainer.config.tensorboard = False
    trainer.callbacks = Mock()
    trainer.tensorboard = None
    trainer.parameters = []
    trainer._patience_alt_backup_path = None
    trainer.one_step_trained = True
    trainer.temp_device = torch.device("cpu")
    trainer.model = Mock(ema=None)
    trainer.grad_hook_handles = []
    events = []
    trainer.model_saver = Mock()
    trainer.model_saver.save.side_effect = lambda **kwargs: events.append("save")
    fake_gc = Mock()
    fake_gc.unfreeze.side_effect = lambda: events.append("gc.unfreeze")
    monkeypatch.setattr(generic_trainer_module, "gc", fake_gc)
    monkeypatch.setattr("modules.trainer.GenericTrainer.multi.is_master", lambda: True)

    trainer.end()

    fake_gc.unfreeze.assert_called_once_with()
    fake_gc.freeze.assert_not_called()
    assert events == ["gc.unfreeze", "save"]


# ---------------------------------------------------------------------------
# safetensors header: INTERNAL skips hash + git, exported formats keep them
# ---------------------------------------------------------------------------


def _header_model():
    return SimpleNamespace(model_spec=None, train_config=None, model_type=ModelType.Z_IMAGE)


def _state_dict():
    return {"b.weight": torch.arange(6, dtype=torch.float32).view(2, 3), "a.alpha": torch.tensor(1.0)}


def _expected_hash(state_dict):
    sha = hashlib.sha256()
    for tensor in OrderedDict(sorted(state_dict.items())).values():
        sha.update(tensor.contiguous().flatten().view(torch.uint8).numpy().tobytes())
    return f"0x{sha.hexdigest()}"


@pytest.fixture
def hash_spy(monkeypatch):
    calls = []
    original = DtypeModelSaverMixin._DtypeModelSaverMixin__calculate_safetensors_hash

    def spy(self, state_dict=None):
        calls.append(state_dict)
        return original(self, state_dict)

    monkeypatch.setattr(DtypeModelSaverMixin, "_DtypeModelSaverMixin__calculate_safetensors_hash", spy)
    return calls


@pytest.fixture
def forbid_git(monkeypatch):
    def fail():
        raise AssertionError("git subprocess must not run for INTERNAL headers")

    monkeypatch.setattr(dtype_saver_module.git_util, "get_git_branch", fail)
    monkeypatch.setattr(dtype_saver_module.git_util, "get_git_revision", fail)


@pytest.fixture
def fake_git(monkeypatch):
    calls = []

    def branch():
        calls.append("branch")
        return "test-branch"

    def revision():
        calls.append("revision")
        return "abc1234"

    monkeypatch.setattr(dtype_saver_module.git_util, "get_git_branch", branch)
    monkeypatch.setattr(dtype_saver_module.git_util, "get_git_revision", revision)
    return calls


def test_internal_header_skips_hash_and_git(hash_spy, forbid_git):
    header = DtypeModelSaverMixin()._create_safetensors_header(_header_model(), _state_dict(), internal_format=True)

    assert hash_spy == []
    assert "modelspec.hash_sha256" not in header
    assert "ot_branch" not in header
    assert "ot_revision" not in header
    assert header["modelspec.sai_model_spec"] == "1.0.0"
    assert "modelspec.date" in header


def test_internal_header_drops_hash_inherited_from_loaded_model_spec(hash_spy, forbid_git):
    model = _header_model()
    model.model_spec = dtype_saver_module.ModelSpec(hash_sha256="0xstale-base-model-hash")

    header = DtypeModelSaverMixin()._create_safetensors_header(model, _state_dict(), internal_format=True)

    assert hash_spy == []
    assert "modelspec.hash_sha256" not in header


def test_default_header_keeps_hash_and_git(hash_spy, fake_git):
    state_dict = _state_dict()
    header = DtypeModelSaverMixin()._create_safetensors_header(_header_model(), state_dict)

    assert len(hash_spy) == 1
    assert header["modelspec.hash_sha256"] == _expected_hash(state_dict)
    assert header["ot_branch"] == "test-branch"
    assert header["ot_revision"] == "abc1234"
    assert fake_git == ["branch", "revision"]


class _TinyLoRASaver(LoRASaverMixin):
    def _get_convert_key_sets(self, model):
        return None

    def _get_state_dict(self, model):
        return _state_dict()


def _read_metadata(path):
    with safe_open(str(path), framework="pt") as f:
        return f.metadata()


def test_lora_internal_save_writes_header_without_hash_or_git(tmp_path, hash_spy, forbid_git):
    _TinyLoRASaver()._save(_header_model(), ModelFormat.INTERNAL, str(tmp_path / "backup"), None)

    metadata = _read_metadata(tmp_path / "backup" / "lora" / "lora.safetensors")
    assert hash_spy == []
    assert "modelspec.hash_sha256" not in metadata
    assert "ot_branch" not in metadata
    assert "ot_revision" not in metadata


@pytest.mark.parametrize("output_format", [ModelFormat.SAFETENSORS, ModelFormat.LEGACY_SAFETENSORS])
@pytest.mark.parametrize("enable_omi_format", [False, True])
def test_lora_exported_save_keeps_hash_and_git(tmp_path, hash_spy, fake_git, output_format, enable_omi_format):
    destination = tmp_path / "out.safetensors"
    _TinyLoRASaver()._save(_header_model(), output_format, str(destination), None, enable_omi_format)

    metadata = _read_metadata(destination)
    assert len(hash_spy) == 1
    assert metadata["modelspec.hash_sha256"] == _expected_hash(_state_dict())
    assert metadata["ot_branch"] == "test-branch"
    assert metadata["ot_revision"] == "abc1234"
