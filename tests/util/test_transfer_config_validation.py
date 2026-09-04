"""TrainConfig transfer_* fields: defaults, migration, startup validation."""

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch

import pytest
from safetensors.torch import save_file


def _touch_safetensors(path) -> str:
    save_file({"x": torch.zeros(1)}, str(path))
    return str(path)


@pytest.fixture
def transfer_config(tmp_path):
    """A config that passes validate_transfer_startup."""
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.OFT_2
    config.transfer_enabled = True
    config.lora_model_name = _touch_safetensors(tmp_path / "warmstart.safetensors")
    config.transfer_base_model_name = _touch_safetensors(tmp_path / "base.safetensors")
    config.transfer_teacher_adapter_name = _touch_safetensors(tmp_path / "teacher_adapter.safetensors")
    return config


def test_defaults_off_and_validation_silent():
    config = TrainConfig.default_values()
    assert config.transfer_enabled is False
    assert config.transfer_late_start == 22
    assert config.transfer_families == "qkv,mlp"
    assert config.validate_transfer_startup() is None


def test_old_config_dict_migrates():
    """A pre-transfer config dict (version 22) round-trips and gains defaults."""
    config = TrainConfig.default_values().from_dict({"__version": 22})
    assert config.transfer_enabled is False
    assert config.transfer_patience_checks == 5
    assert config.transfer_min_relative_improvement == pytest.approx(0.005)


def test_valid_config_passes(transfer_config):
    assert transfer_config.validate_transfer_startup() is None


def test_missing_paths_fail(transfer_config):
    transfer_config.transfer_teacher_adapter_name = ""
    assert "teacher adapter" in transfer_config.validate_transfer_startup()

    transfer_config.transfer_teacher_adapter_name = r"Z:\does\not\exist.safetensors"
    assert "teacher adapter" in transfer_config.validate_transfer_startup()


def test_missing_base_fails(transfer_config):
    transfer_config.transfer_base_model_name = ""
    assert "Base checkpoint" in transfer_config.validate_transfer_startup()


def test_missing_warmstart_fails(transfer_config):
    transfer_config.lora_model_name = ""
    assert "lora_model_name" in transfer_config.validate_transfer_startup()


def test_wrong_peft_type_fails(transfer_config):
    transfer_config.peft_type = PeftType.LORA
    assert "OFT" in transfer_config.validate_transfer_startup()


def test_distill_conflict_fails(transfer_config):
    transfer_config.distill_enabled = True
    assert "istill" in transfer_config.validate_transfer_startup()


def test_rlhf_conflict_fails(transfer_config):
    transfer_config.rlhf_enabled = True
    assert "RLHF" in transfer_config.validate_transfer_startup()


def test_legacy_teacher_conflict_fails(transfer_config):
    transfer_config.distillation_teacher_lora_model_name = "something.safetensors"
    assert "teacher" in transfer_config.validate_transfer_startup()


def test_bad_families_fail(transfer_config):
    transfer_config.transfer_families = "qkv,bogus"
    assert "bogus" in transfer_config.validate_transfer_startup()


def test_dropout_conflict_fails(transfer_config):
    """Adapter dropout draws from the global RNG, which would make the
    deterministic convergence validation stochastic (patience would count
    noise; best-snapshot would select the minimum of a noise process)."""
    transfer_config.dropout_probability = 0.1
    assert "dropout" in transfer_config.validate_transfer_startup()


def test_patience_without_validation_fails(transfer_config):
    transfer_config.transfer_validation = False
    assert "validation" in transfer_config.validate_transfer_startup()


def test_output_dtype_warning(transfer_config, capsys):
    transfer_config.output_dtype = DataType.BFLOAT_16
    assert transfer_config.validate_transfer_startup() is None
    assert "FLOAT_32" in capsys.readouterr().out


def test_teacher_equals_warmstart_warns_but_passes(transfer_config, capsys):
    """Pointing the teacher adapter at the warm-start file is almost certainly a
    mistake (the teacher must be the ORIGINAL Base-frame adapter) -- warn loud,
    do not block: the files could legitimately coincide for a sentinel variant."""
    transfer_config.transfer_teacher_adapter_name = transfer_config.lora_model_name
    assert transfer_config.validate_transfer_startup() is None
    assert "warm-start" in capsys.readouterr().out.lower()
