import contextlib
import json
import os
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.util.TrainProgress import TrainProgress

import torch


class InternalModelLoaderMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def _load_internal_data(
            self,
            model: BaseModel,
            model_name: str,
    ):
        if os.path.exists(os.path.join(model_name, "meta.json")):
            # train progress
            with open(os.path.join(model_name, "meta.json"), "r") as meta_file:
                meta = json.load(meta_file)
                train_progress = TrainProgress(
                    epoch=meta['train_progress']['epoch'],
                    epoch_step=meta['train_progress']['epoch_step'],
                    epoch_sample=meta['train_progress']['epoch_sample'],
                    global_step=meta['train_progress']['global_step'],
                )
                if 'last_action_epoch' in meta:
                    train_progress.last_action_epoch = dict(meta['last_action_epoch'])
                elif train_progress.epoch_step > 0:
                    # Legacy backup taken mid-epoch: start-of-epoch actions
                    # already fired in the pre-fix session. Pre-fill markers
                    # to prevent a duplicate fire on the first resumed batch.
                    train_progress.last_action_epoch = {
                        'validate': train_progress.epoch,
                        'sample': train_progress.epoch,
                    }
                else:
                    train_progress.last_action_epoch = {}
                resumed_tb_subdir = meta.get('tensorboard_subdir', None)

            # optimizer
            with contextlib.suppress(FileNotFoundError):
                model.optimizer_state_dict = torch.load(os.path.join(model_name, "optimizer", "optimizer.pt"),
                                                        weights_only=True)

            # ema
            with contextlib.suppress(FileNotFoundError):
                model.ema_state_dict = torch.load(os.path.join(model_name, "ema", "ema.pt"), weights_only=True)

            # accumulator state (Fix B): contains the in-flight gradient-
            # accumulation snapshot from the save side. Optional -- legacy
            # backups won't have it, and the trainer falls back to today's
            # behavior. weights_only=False because the payload mixes
            # tensors with python dicts/tuples (RNG state); the file lives
            # in the same trust boundary as optimizer.pt.
            with contextlib.suppress(FileNotFoundError):
                model.accumulator_state = torch.load(
                    os.path.join(model_name, "accumulator", "accumulator.pt"),
                    weights_only=False,
                )

            # meta
            model.train_progress = train_progress
            model.resumed_tensorboard_subdir = resumed_tb_subdir
