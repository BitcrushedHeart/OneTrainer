from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DistillCfgMode import DistillCfgMode
from modules.util.enum.DistillFakeAdapterType import DistillFakeAdapterType
from modules.util.enum.DistillVramMode import DistillVramMode
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


def test_distill_config_defaults_round_trip():
    config = TrainConfig.default_values()

    assert config.distill_enabled is False
    assert config.distill_target_steps == 4
    assert config.distill_teacher_steps == 12
    assert config.distill_teacher_steps_auto is True
    assert config.distill_endpoint_loss_weight == 0.25
    assert config.distill_kl_loss_weight == 1.0
    assert config.distill_ttur_ratio == 5
    assert config.distill_fake_warmup_steps == 500
    assert config.distill_vram_mode == DistillVramMode.SPLIT
    assert config.distill_cfg_mode == DistillCfgMode.AUTO
    assert config.distill_fake_adapter_type == DistillFakeAdapterType.LORA
    assert config.distill_fake_adapter_rank == 16

    loaded = TrainConfig.default_values().from_dict(config.to_dict())

    assert loaded.distill_enabled is False
    assert loaded.distill_target_steps == 4
    assert loaded.distill_fake_adapter_type == DistillFakeAdapterType.LORA


def test_distill_validation_rejects_conflicting_or_unsupported_modes():
    config = TrainConfig.default_values()
    config.distill_enabled = True
    config.rlhf_enabled = True
    assert config.validate_distill_startup() == "Distill and RLHF can not be enabled in the same training run."

    config.rlhf_enabled = False
    config.training_method = TrainingMethod.FINE_TUNE
    assert config.validate_distill_startup() == "Distill is currently implemented for LoRA training only."

    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.FLUX_DEV_1
    assert config.validate_distill_startup() == "Distill is currently implemented for Z-Image only."

    config.model_type = ModelType.Z_IMAGE
    assert config.validate_distill_startup() is None
