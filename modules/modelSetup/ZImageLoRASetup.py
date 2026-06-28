import copy

from modules.model.ZImageModel import ZImageModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.BaseZImageSetup import BaseZImageSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroupCollection
from modules.util.optimizer_util import init_model_parameters
from modules.util.TrainProgress import TrainProgress

import torch


def _infer_dora_shape_from_state_dict(state_dict: dict) -> tuple[int, float]:
    """Read (rank, alpha) from a DoRA/LoRA state dict so the teacher wrapper
    can be constructed with the same shape the file expects."""
    rank = None
    for k, v in state_dict.items():
        if k.endswith(".lora_down.weight"):
            rank = v.shape[0]
            break
    if rank is None:
        raise ValueError(
            "Distillation teacher state dict has no '*.lora_down.weight' key — "
            "the file does not look like a DoRA/LoRA checkpoint."
        )

    alpha = float(rank)  # sensible default if no alpha is present
    for k, v in state_dict.items():
        if k.endswith(".alpha"):
            alpha = float(v.item()) if hasattr(v, "item") else float(v)
            break

    return rank, alpha


def _build_teacher_config_view(config: TrainConfig, teacher_state_dict: dict) -> TrainConfig:
    """Shallow-copy the live config and override only the fields LoRAModuleWrapper
    reads at construction, so a DoRA teacher wrapper can be built without
    mutating the student's config."""
    teacher_rank, teacher_alpha = _infer_dora_shape_from_state_dict(teacher_state_dict)

    teacher_config = copy.copy(config)
    teacher_config.peft_type = PeftType.LORA
    teacher_config.lora_decompose = True
    teacher_config.dora_oft = False
    teacher_config.lora_rank = teacher_rank
    teacher_config.lora_alpha = teacher_alpha
    return teacher_config


def _build_fake_config_view(config: TrainConfig) -> TrainConfig:
    fake_config = copy.copy(config)
    fake_type = str(config.distill_fake_adapter_type).upper()

    fake_config.lora_rank = config.distill_fake_adapter_rank
    fake_config.lora_alpha = float(config.distill_fake_adapter_rank)

    if fake_type == "LORA":
        fake_config.peft_type = PeftType.LORA
        fake_config.lora_decompose = False
        fake_config.dora_oft = False
    elif fake_type == "DORA":
        fake_config.peft_type = PeftType.LORA
        fake_config.lora_decompose = True
        fake_config.dora_oft = False
    elif fake_type == "OFT":
        fake_config.peft_type = PeftType.OFT_2
        fake_config.dora_oft = False
        fake_config.oft_block_size = config.distill_fake_adapter_rank
    else:
        raise ValueError(f"Unsupported distill fake adapter type: {config.distill_fake_adapter_type}")

    return fake_config


class ZImageLoRASetup(
    BaseZImageSetup,
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
        model: ZImageModel,
        config: TrainConfig,
    ) -> NamedParameterGroupCollection:
        parameter_group_collection = NamedParameterGroupCollection()

        self._create_model_part_parameters(
            parameter_group_collection, "transformer", model.transformer_lora, config.transformer
        )
        if config.distill_enabled and model.transformer_fake_lora is not None:
            self._create_model_part_parameters(
                parameter_group_collection,
                "transformer_fake",
                model.transformer_fake_lora,
                config.transformer,
            )
        return parameter_group_collection

    def __setup_requires_grad(
        self,
        model: ZImageModel,
        config: TrainConfig,
    ):
        model.text_encoder.requires_grad_(False)
        model.transformer.requires_grad_(False)
        model.vae.requires_grad_(False)

        self._setup_model_part_requires_grad(
            "transformer", model.transformer_lora, config.transformer, model.train_progress
        )
        if config.distill_enabled and model.transformer_fake_lora is not None:
            self._setup_model_part_requires_grad(
                "transformer_fake", model.transformer_fake_lora, config.transformer, model.train_progress
            )

    def setup_model(
        self,
        model: ZImageModel,
        config: TrainConfig,
    ):
        teacher_path = config.distillation_teacher_lora_model_name
        if teacher_path:
            if config.peft_type != PeftType.OFT_2 or not config.dora_oft:
                raise ValueError(
                    "Distillation requires the student to be DoRA-OFT: set peft_type=OFT_2 and dora_oft=True."
                )
            if model.teacher_lora_state_dict is None:
                raise ValueError(
                    f"Distillation teacher LoRA configured ({teacher_path}) but the "
                    "loader did not populate teacher_lora_state_dict. Verify the file path."
                )

        model.transformer_lora = LoRAModuleWrapper(
            model.transformer, "transformer", config, config.layer_filter.split(",")
        )

        if model.lora_state_dict:
            model.transformer_lora.load_state_dict(model.lora_state_dict)
            model.lora_state_dict = None

        model.transformer_lora.set_dropout(config.dropout_probability)
        model.transformer_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
        model.transformer_lora.hook_to_module()

        if model.teacher_lora_state_dict is not None:
            teacher_config = _build_teacher_config_view(config, model.teacher_lora_state_dict)
            # Use the student's layer_filter so coverage matches.
            model.transformer_teacher_lora = LoRAModuleWrapper(
                model.transformer, "transformer", teacher_config, config.layer_filter.split(",")
            )
            model.transformer_teacher_lora.load_state_dict(model.teacher_lora_state_dict)
            model.teacher_lora_state_dict = None
            model.transformer_teacher_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            model.transformer_teacher_lora.requires_grad_(False)
            # Intentionally do NOT call hook_to_module(); the distillation context
            # manager toggles this per training step.

        if config.distill_enabled:
            fake_config = _build_fake_config_view(config)
            model.transformer_fake_lora = LoRAModuleWrapper(
                model.transformer, "transformer_fake", fake_config, config.layer_filter.split(",")
            )
            model.transformer_fake_lora.set_dropout(config.dropout_probability)
            model.transformer_fake_lora.to(dtype=config.lora_weight_dtype.torch_dtype())
            # Intentionally do NOT call hook_to_module(); DMD2 toggles the fake
            # score adapter only during fake-score updates/forwards.

        params = self.create_parameters(model, config)
        self.__setup_requires_grad(model, config)

        init_model_parameters(model, params, self.train_device)

    def setup_train_device(
        self,
        model: ZImageModel,
        config: TrainConfig,
    ):
        vae_on_train_device = not config.latent_caching
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

    def after_optimizer_step(self, model: ZImageModel, config: TrainConfig, train_progress: TrainProgress):
        self.__setup_requires_grad(model, config)


factory.register(BaseModelSetup, ZImageLoRASetup, ModelType.Z_IMAGE, TrainingMethod.LORA)
