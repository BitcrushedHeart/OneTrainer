from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TrainingMethod import TrainingMethod

import pytest


def _sft_dpo_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    return config


def test_rlhf_mode_has_sft_dpo_member():
    assert RLHFMode("SFT_DPO") == RLHFMode.SFT_DPO
    assert str(RLHFMode.SFT_DPO) == "SFT_DPO"


def test_sft_dpo_defaults_round_trip():
    config = TrainConfig.default_values()
    assert config.rlhf_mode == RLHFMode.DPO
    assert config.rlhf_dpo_addon_weight == 1.0
    assert config.rlhf_dpo_addon_interval == 0

    loaded = TrainConfig.default_values().from_dict(config.to_dict())
    assert loaded.rlhf_dpo_addon_weight == 1.0
    assert loaded.rlhf_dpo_addon_interval == 0


def test_old_config_dict_migrates_addon_fields():
    config = TrainConfig.default_values().from_dict({"__version": 26})
    assert config.rlhf_mode == RLHFMode.DPO
    assert config.rlhf_dpo_addon_weight == 1.0
    assert config.rlhf_dpo_addon_interval == 0


def test_ref_rebase_defaults_round_trip():
    config = TrainConfig.default_values()
    assert config.rlhf_dpo_ref_rebase is False
    assert config.rlhf_dpo_ref_rebase_interval == 0

    loaded = TrainConfig.default_values().from_dict(config.to_dict())
    assert loaded.rlhf_dpo_ref_rebase is False
    assert loaded.rlhf_dpo_ref_rebase_interval == 0


def test_old_config_dict_migrates_ref_rebase_fields():
    config = TrainConfig.default_values().from_dict({"__version": 27})
    assert config.rlhf_dpo_ref_rebase is False
    assert config.rlhf_dpo_ref_rebase_interval == 0


def test_ref_rebase_needs_existing_adapter_reference():
    # NEW_ADAPTER measures against the bare base model: there is no snapshot
    # to move, so asking for a rebase is a misconfiguration, not a no-op.
    config = _sft_dpo_config()
    config.rlhf_dpo_ref_mode = DPORefMode.NEW_ADAPTER
    config.rlhf_dpo_ref_rebase = True
    assert "Existing Adapter" in config.validate_rlhf_startup()

    config.rlhf_dpo_ref_rebase = False
    config.rlhf_dpo_ref_rebase_interval = 500
    assert "Existing Adapter" in config.validate_rlhf_startup()

    config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
    config.rlhf_dpo_ref_rebase = True
    assert config.validate_rlhf_startup() is None


def test_ref_rebase_interval_rejects_negative():
    config = _sft_dpo_config()
    config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
    config.rlhf_dpo_ref_rebase_interval = -1
    assert "rebase interval" in config.validate_rlhf_startup()


def test_primary_mode_helpers():
    config = TrainConfig.default_values()
    assert not config.rlhf_dpo_primary()
    assert not config.rlhf_sft_primary()

    config.rlhf_enabled = True
    assert config.rlhf_dpo_primary()
    assert not config.rlhf_sft_primary()

    config.rlhf_mode = RLHFMode.SFT_DPO
    assert not config.rlhf_dpo_primary()
    assert config.rlhf_sft_primary()


def test_effective_ref_mode_keeps_legacy_derivation():
    config = TrainConfig.default_values()
    assert config.effective_dpo_ref_mode() == DPORefMode.NEW_ADAPTER

    config.lora_model_name = "adapter.safetensors"
    assert config.effective_dpo_ref_mode() == DPORefMode.EXISTING_ADAPTER


def test_effective_ref_mode_honors_explicit_existing_adapter_without_loaded_adapter():
    # Resuming a finetune from a backup: no lora_model_name, yet the reference
    # must be the adapter as it stands at the first DPO step, not the bare base.
    config = TrainConfig.default_values()
    config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
    assert config.effective_dpo_ref_mode() == DPORefMode.EXISTING_ADAPTER


def test_validate_rlhf_startup_accepts_valid_sft_dpo_config():
    assert _sft_dpo_config().validate_rlhf_startup() is None


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_validate_rlhf_startup_rejects_non_positive_addon_weight(weight):
    config = _sft_dpo_config()
    config.rlhf_dpo_addon_weight = weight
    assert config.validate_rlhf_startup() is not None


def test_validate_rlhf_startup_rejects_negative_interval():
    config = _sft_dpo_config()
    config.rlhf_dpo_addon_interval = -1
    assert config.validate_rlhf_startup() is not None


def test_validate_rlhf_startup_rejects_fused_back_pass():
    config = _sft_dpo_config()
    config.optimizer.fused_back_pass = True
    assert "fused back pass" in config.validate_rlhf_startup()


def test_validate_rlhf_startup_rejects_non_lora():
    config = _sft_dpo_config()
    config.training_method = TrainingMethod.FINE_TUNE
    assert config.validate_rlhf_startup() is not None


def test_validate_rlhf_startup_ignores_disabled_rlhf():
    config = TrainConfig.default_values()
    config.rlhf_enabled = False
    config.rlhf_dpo_addon_weight = 0.0
    config.training_method = TrainingMethod.FINE_TUNE
    assert config.validate_rlhf_startup() is None
