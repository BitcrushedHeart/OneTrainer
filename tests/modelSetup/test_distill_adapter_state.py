import pytest
import torch

from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod


class FakeAdapter:
    def __init__(self):
        self.hooked = False
        self.hook_count = 0
        self.remove_count = 0

    def hook_to_module(self):
        self.hooked = True
        self.hook_count += 1

    def remove_hook_from_module(self):
        self.hooked = False
        self.remove_count += 1


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__(ModelType.Z_IMAGE)
        self.student_adapter = FakeAdapter()
        self.teacher_adapter = FakeAdapter()
        self.fake_adapter = FakeAdapter()
        self.student_adapter.hook_to_module()

    def to(self, device: torch.device):
        return self

    def eval(self):
        return self

    def adapters(self):
        return [self.student_adapter]

    def teacher_adapters(self):
        return [self.teacher_adapter]

    def fake_adapters(self):
        return [self.fake_adapter]


class TinySetup(BaseModelSetup):
    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_optimizations(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        raise NotImplementedError

    def calculate_loss(self, model, batch, data, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError


def _config():
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    return config


def test_distill_adapter_states_are_mutually_exclusive_and_restore_student():
    model = TinyModel()
    setup = TinySetup(torch.device("cpu"), torch.device("cpu"), False)
    config = _config()

    with setup.adapter_state(model, config, "teacher"):
        assert model.student_adapter.hooked is False
        assert model.teacher_adapter.hooked is False
        assert model.fake_adapter.hooked is False
    assert model.student_adapter.hooked is True

    with setup.adapter_state(model, config, "student"):
        assert model.student_adapter.hooked is True
        assert model.teacher_adapter.hooked is False
        assert model.fake_adapter.hooked is False

    with setup.adapter_state(model, config, "fake"):
        assert model.student_adapter.hooked is False
        assert model.teacher_adapter.hooked is False
        assert model.fake_adapter.hooked is True
    assert model.student_adapter.hooked is True
    assert model.fake_adapter.hooked is False


def test_distill_adapter_state_restores_after_exception():
    model = TinyModel()
    setup = TinySetup(torch.device("cpu"), torch.device("cpu"), False)
    config = _config()

    with pytest.raises(ValueError):
        with setup.adapter_state(model, config, "fake"):
            raise ValueError("boom")

    assert model.student_adapter.hooked is True
    assert model.teacher_adapter.hooked is False
    assert model.fake_adapter.hooked is False
