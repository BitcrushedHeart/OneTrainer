"""Startup validation and epoch wiring of the on-image text reward add-on."""

from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TextRewardSource import TextRewardSource
from modules.util.text_reward_util import TextRewardBank, TextSpec

import pytest


def _config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.text_reward_enabled = True
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_validation_accepts_plain_and_sft_dpo_runs():
    assert _config().validate_text_reward_startup() is None
    assert _config(rlhf_enabled=True, rlhf_mode=RLHFMode.SFT_DPO).validate_text_reward_startup() is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_type": ModelType.FLUX_DEV_1},
        {"text_reward_weight": 0.0},
        {"text_reward_samples_per_epoch": 0},
        {"text_reward_rollout_steps": 0},
        {"text_reward_min_words": 3, "text_reward_max_words": 2},
        {"distill_enabled": True},
        {"transfer_enabled": True},
        {"rlhf_enabled": True, "rlhf_mode": RLHFMode.DPO},
        {"text_reward_grad_sigma_min": 0.7, "text_reward_grad_sigma_max": 0.5},
        {"text_reward_grad_sigma_max": 1.5},
        {"text_reward_init_sigma": 1.0},
        {"text_reward_init_sigma": -0.1},
        {"text_reward_init_sigma": 0.3, "text_reward_grad_sigma_min": 0.4, "text_reward_grad_sigma_max": 0.7},
        {"text_reward_init_sigma": 0.8, "text_reward_rollout_steps": 1},
    ],
)
def test_validation_rejects_bad_configs(overrides):
    assert _config(**overrides).validate_text_reward_startup() is not None


def test_disabled_reward_never_validates():
    config = _config(text_reward_enabled=False, text_reward_weight=0.0)
    assert config.validate_text_reward_startup() is None


def test_config_round_trips_through_migration():
    config = TrainConfig.default_values()
    data = config.to_dict()
    for key in ("text_reward_enabled", "text_reward_samples_per_epoch", "text_reward_weight"):
        assert key in data
    stale = {k: v for k, v in data.items() if not k.startswith("text_reward_")}
    stale["__version"] = 27
    restored = TrainConfig.default_values().from_dict(stale)
    assert restored.text_reward_enabled is False
    assert restored.text_reward_samples_per_epoch == 1000

    # a version-28 file predates the hybrid fields: migration 29 fills them in
    v28 = {k: v for k, v in data.items() if k not in ("text_reward_init_sigma", "text_reward_grad_sigma_min")}
    v28["__version"] = 28
    v28["text_reward_source"] = "OFFLINE_IMAGES"
    restored = TrainConfig.default_values().from_dict(v28)
    # the hybrid settings are the defaults, so an old preset picks them up on load
    assert restored.text_reward_init_sigma == 0.8
    assert restored.text_reward_grad_sigma_min == 0.45 and restored.text_reward_grad_sigma_max == 0.7
    assert restored.text_reward_source == TextRewardSource.OFFLINE_IMAGES


def test_hybrid_window_validates():
    config = _config(text_reward_init_sigma=0.8, text_reward_grad_sigma_min=0.4, text_reward_grad_sigma_max=0.7)
    assert config.validate_text_reward_startup() is None


def test_epoch_start_builds_bank_and_sizes_interval(capsys):
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = _config(text_reward_samples_per_epoch=1000)
    trainer.callbacks = Mock()
    trainer.model = Mock()
    trainer.model.train_progress.epoch = 3
    specs = [TextSpec("Sign", "Open", ("Open",), "words")] * 1000
    trainer.model_setup = Mock()
    trainer.model_setup.build_text_reward_bank.return_value = TextRewardBank(specs, [None] * 1000, None)
    trainer.data_loader = Mock()
    trainer.data_loader.get_data_set.return_value.approximate_length.return_value = 96000
    trainer.sft_anchor_source = None
    trainer.dpo_addon_source = None
    trainer.kto_source = None
    from modules.util.text_reward_util import TextRewardScheduler

    trainer._text_reward_scheduler = TextRewardScheduler(0)
    trainer._text_reward_bank = None

    trainer._GenericTrainer__start_secondary_epochs()

    trainer.model_setup.build_text_reward_bank.assert_called_once_with(trainer.model, trainer.config, 3)
    assert trainer._text_reward_scheduler.interval == 96
    assert len(trainer._text_reward_bank) == 1000
    assert "one rollout every 96" in capsys.readouterr().out


def test_offline_interval_uses_dataset_size_times_repeats(capsys):
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = _config(text_reward_source=TextRewardSource.OFFLINE_IMAGES, text_reward_dataset_repeats=7)
    trainer.callbacks = Mock()
    trainer.model = Mock()
    trainer.model.train_progress.epoch = 0
    specs = [TextSpec("Sign", "Open", ("Open",), "offline")] * 127
    trainer.model_setup = Mock()
    trainer.model_setup.build_text_reward_bank.return_value = TextRewardBank(
        specs, [None] * 127, None, [f"{i}.png" for i in range(127)]
    )
    trainer.data_loader = Mock()
    trainer.data_loader.get_data_set.return_value.approximate_length.return_value = 92148
    trainer.sft_anchor_source = None
    trainer.dpo_addon_source = None
    trainer.kto_source = None
    from modules.util.text_reward_util import TextRewardScheduler

    trainer._text_reward_scheduler = TextRewardScheduler(0)
    trainer._text_reward_bank = None
    trainer._GenericTrainer__start_secondary_epochs()
    assert trainer._text_reward_scheduler.interval == 104
