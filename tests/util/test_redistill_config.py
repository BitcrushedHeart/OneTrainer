"""TrainConfig redistill_* fields: defaults, migration, startup validation."""

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch

import pytest
from safetensors.torch import save_file


def _touch_safetensors(path) -> str:
    save_file({"x": torch.zeros(1)}, str(path))
    return str(path)


@pytest.fixture
def redistill_config(tmp_path):
    """A config that passes validate_redistill_startup."""
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.OFT_2
    config.redistill_enabled = True
    config.redistill_turbo_model_name = _touch_safetensors(tmp_path / "turbo.safetensors")
    return config


def test_defaults_off_and_validation_silent():
    config = TrainConfig.default_values()
    assert config.redistill_enabled is False
    assert config.prompt_only_data is False
    assert config.redistill_transfer_steps == 2000
    assert config.validate_redistill_startup() is None
    assert config.validate_prompt_only_startup() is None
    assert config.distill_phase_active() is False


def test_old_config_dict_migrates():
    config = TrainConfig.default_values().from_dict({"__version": 24})
    assert config.redistill_enabled is False
    assert config.prompt_only_data is False
    assert config.redistill_turbo_model_name == ""
    assert config.redistill_transfer_steps == 2000


def test_valid_config_passes(redistill_config):
    assert redistill_config.validate_redistill_startup() is None
    assert redistill_config.distill_phase_active() is True


def test_mode_exclusions(redistill_config):
    redistill_config.distill_enabled = True
    assert "Distill" in redistill_config.validate_redistill_startup()
    redistill_config.distill_enabled = False
    redistill_config.transfer_enabled = True
    assert "Transfer" in redistill_config.validate_redistill_startup()
    redistill_config.transfer_enabled = False
    redistill_config.rlhf_enabled = True
    assert "RLHF" in redistill_config.validate_redistill_startup()


def test_missing_turbo_checkpoint_fails(redistill_config):
    redistill_config.redistill_turbo_model_name = ""
    assert "redistill_turbo_model_name" in redistill_config.validate_redistill_startup()
    redistill_config.redistill_turbo_model_name = r"Z:\does\not\exist.safetensors"
    assert "redistill_turbo_model_name" in redistill_config.validate_redistill_startup()


def test_shared_mode_constraints(redistill_config):
    redistill_config.peft_type = PeftType.LORA
    assert "OFT" in redistill_config.validate_redistill_startup()
    redistill_config.peft_type = PeftType.OFT_2

    redistill_config.gradient_accumulation_steps = 2
    assert "gradient_accumulation_steps" in redistill_config.validate_redistill_startup()
    redistill_config.gradient_accumulation_steps = 1

    redistill_config.dropout_probability = 0.1
    assert "dropout" in redistill_config.validate_redistill_startup()
    redistill_config.dropout_probability = 0.0

    redistill_config.ema = EMAMode.GPU
    assert "EMA" in redistill_config.validate_redistill_startup()
    redistill_config.ema = EMAMode.OFF

    redistill_config.transfer_patience_enabled = True
    redistill_config.transfer_validation = False
    assert "transfer_validation" in redistill_config.validate_redistill_startup()


def test_prompt_only_constraints():
    config = TrainConfig.default_values()
    config.prompt_only_data = True
    assert "distill" in config.validate_prompt_only_startup()
    config.distill_enabled = True
    assert config.validate_prompt_only_startup() is None
    config.rlhf_enabled = True
    assert "RLHF" in config.validate_prompt_only_startup()
    config.rlhf_enabled = False
    config.validation = True
    assert "validation" in config.validate_prompt_only_startup()
