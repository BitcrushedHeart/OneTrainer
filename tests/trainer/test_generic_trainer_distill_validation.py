import pytest

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


def test_generic_trainer_rejects_invalid_distill_startup_without_loading_model():
    config = TrainConfig.default_values()
    config.tensorboard = False
    config.distill_enabled = True
    config.training_method = TrainingMethod.FINE_TUNE
    config.model_type = ModelType.Z_IMAGE

    with pytest.raises(RuntimeError, match="Distill is currently implemented for LoRA training only."):
        GenericTrainer(config, TrainCallbacks(), TrainCommands())
