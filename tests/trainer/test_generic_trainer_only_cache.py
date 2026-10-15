"""A cache-only run (only_cache) drives every loader a training run reads:
the main loader, the secondary sources and the validation loaders, once per
epoch, so the cache holds exactly what training and validation will load."""

from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

import pytest


def _loader(events: list, name: str) -> Mock:
    loader = Mock()
    loader.get_data_set.return_value.start_next_epoch.side_effect = lambda: events.append(name)
    return loader


def _trainer(tmp_path, epochs: int, *, validation: bool, dpo_validation: bool):
    config = TrainConfig.default_values()
    config.only_cache = True
    config.epochs = epochs
    config.workspace_dir = str(tmp_path / "workspace")
    config.tensorboard = False
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    events: list[str] = []
    trainer.model = Mock(train_progress=TrainProgress())
    trainer.data_loader = _loader(events, "train")
    trainer.dpo_addon_source = None
    trainer.sft_anchor_source = None
    trainer.kto_source = None
    trainer.validation_data_loader = _loader(events, "validation") if validation else None
    trainer.dpo_validation_data_loader = _loader(events, "dpo_validation") if dpo_validation else None
    return trainer, events


@pytest.mark.parametrize(
    ("validation", "dpo_validation", "expected"),
    [
        (True, True, ["train", "validation", "dpo_validation"]),
        (True, False, ["train", "validation"]),
        (False, False, ["train"]),
    ],
)
def test_only_cache_starts_every_validation_loader_each_epoch(tmp_path, validation, dpo_validation, expected):
    trainer, events = _trainer(tmp_path, epochs=3, validation=validation, dpo_validation=dpo_validation)
    trainer.train()
    assert events == expected * 3


def test_only_cache_resumes_from_the_progress_epoch(tmp_path):
    trainer, events = _trainer(tmp_path, epochs=4, validation=True, dpo_validation=False)
    trainer.model.train_progress.epoch = 2
    trainer.train()
    assert events == ["train", "validation"] * 2
