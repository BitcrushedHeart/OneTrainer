import contextlib
import json
import os
from abc import ABCMeta, abstractmethod

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes
from modules.util.TrainProgress import TrainProgress

import torch


class BaseModelLoader(metaclass=ABCMeta):

    def _load_internal_state(
            self,
            model: BaseModel,
            base_model_name: str,
    ):
        with open(os.path.join(base_model_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )
            train_progress.last_action_epoch = dict(meta.get('last_action_epoch', {}))
            resumed_tb_subdir = meta.get('tensorboard_subdir', None)

        # optimizer
        with contextlib.suppress(FileNotFoundError):
            model.optimizer_state_dict = torch.load(os.path.join(base_model_name, "optimizer", "optimizer.pt"),
                                                    weights_only=True)

        # ema
        with contextlib.suppress(FileNotFoundError):
            model.ema_state_dict = torch.load(os.path.join(base_model_name, "ema", "ema.pt"), weights_only=True)

        # meta
        model.train_progress = train_progress
        model.resumed_tensorboard_subdir = resumed_tb_subdir


    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ) -> BaseModel | None:
        pass
