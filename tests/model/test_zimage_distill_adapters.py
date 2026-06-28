import torch

from modules.model.ZImageModel import ZImageModel
from modules.util.enum.ModelType import ModelType


class FakeAdapter:
    def __init__(self):
        self.devices = []

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        self.devices.append(device)
        return self


class FakeModule:
    def to(self, device=None, **kwargs):
        return self


def test_zimage_model_exposes_and_moves_fake_distill_adapter():
    model = ZImageModel(ModelType.Z_IMAGE)
    model.transformer = FakeModule()
    fake = FakeAdapter()
    model.transformer_fake_lora = fake

    assert model.fake_adapters() == [fake]

    model.transformer_to(torch.device("cpu"))

    assert fake.devices[-1] == torch.device("cpu")
