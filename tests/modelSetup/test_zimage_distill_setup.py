import torch
from torch import nn

from modules.util import create  # noqa: F401 - primes model setup factory imports before direct setup import
import modules.modelSetup.ZImageLoRASetup as zimage_lora_setup_module
from modules.model.ZImageModel import ZImageModel
from modules.modelSetup.ZImageLoRASetup import ZImageLoRASetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


class FakeLoRAWrapper(nn.Module):
    instances = []

    def __init__(self, module, name, config, layer_filter):
        super().__init__()
        self.module = module
        self.name = name
        self.rank = config.lora_rank
        self.hooked = False
        self.weight = nn.Parameter(torch.zeros(1))
        FakeLoRAWrapper.instances.append(self)

    def hook_to_module(self):
        self.hooked = True

    def remove_hook_from_module(self):
        self.hooked = False

    def set_dropout(self, probability):
        pass

    def load_state_dict(self, state_dict, strict=True):
        return None


def test_zimage_lora_setup_creates_unhooked_fake_adapter_when_distill_enabled(monkeypatch):
    FakeLoRAWrapper.instances = []
    monkeypatch.setattr(zimage_lora_setup_module, "LoRAModuleWrapper", FakeLoRAWrapper)
    monkeypatch.setattr(
        zimage_lora_setup_module,
        "init_model_parameters",
        lambda model, params, train_device: setattr(model, "parameters", params),
    )

    model = ZImageModel(ModelType.Z_IMAGE)
    model.train_config = TrainConfig.default_values()
    model.text_encoder = nn.Linear(1, 1)
    model.transformer = nn.Linear(1, 1)
    model.vae = nn.Linear(1, 1)

    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    config.distill_enabled = True
    config.distill_fake_adapter_rank = 7

    setup = ZImageLoRASetup(torch.device("cpu"), torch.device("cpu"), False)
    setup.setup_model(model, config)

    assert model.transformer_lora is not None
    assert model.transformer_lora.hooked is True
    assert model.transformer_fake_lora is not None
    assert model.transformer_fake_lora.hooked is False
    assert model.transformer_fake_lora.rank == 7
    assert "transformer_fake" in model.parameters.unique_name_mapping
