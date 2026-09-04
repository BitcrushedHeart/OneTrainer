from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TrainingMethod import TrainingMethod

import pytest


def test_generic_trainer_rejects_invalid_sft_dpo_startup_without_loading_model():
    config = TrainConfig.default_values()
    config.tensorboard = False
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_addon_weight = 0.0

    with pytest.raises(RuntimeError, match="add-on weight"):
        GenericTrainer(config, TrainCallbacks(), TrainCommands())
