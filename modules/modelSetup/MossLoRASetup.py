from modules.model.MossModel import MossModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseMossSetup import BaseMossSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


class MossLoRASetup(
    BaseMossSetup,
):
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        debug_mode: bool,
    ):
        super().__init__(
            train_device=train_device,
            temp_device=temp_device,
            debug_mode=debug_mode,
        )

    def create_parameters(
        self,
        model: MossModel,
        config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(
            parameter_group_collection, "transformer", model.transformer_lora, config.transformer
        )
        return parameter_group_collection

    def __setup_requires_grad(
        self,
        model: MossModel,
        config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad(
            "transformer", model.transformer_lora, config.transformer, model.train_progress
        )

    def setup_model(
        self,
        model: MossModel,
        config: TrainConfig,
    ):
        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)
        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
        self,
        model: MossModel,
        config: TrainConfig,
    ):
        # The perceptual term decodes x0_hat crops every gated step; keep the
        # VAE resident instead of paging it per step.
        vae_on_train_device = not config.latent_caching or config.perceptual_strength > 0
        text_encoder_on_train_device = not config.latent_caching

        model.text_encoder_to(self.train_device if text_encoder_on_train_device else self.temp_device)
        model.vae_to(self.train_device if vae_on_train_device else self.temp_device)
        model.transformer_to(self.train_device)

        model.text_encoder.eval()
        model.vae.eval()

        if config.transformer.train:
            model.transformer.train()
        else:
            model.transformer.eval()

    def after_optimizer_step(self, model: MossModel, config: TrainConfig, train_progress: TrainProgress):
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, MossLoRASetup, ModelType.MOSS, TrainingMethod.LORA)
