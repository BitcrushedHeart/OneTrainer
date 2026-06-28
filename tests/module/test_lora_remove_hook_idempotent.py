"""Regression for the DMD2 adapter-state crash.

The fake-score adapter is constructed but intentionally left un-hooked until its
first activation. ``adapter_state`` removes hooks from *all* adapters on entry,
so ``LoRAModuleWrapper.remove_hook_from_module()`` must be a safe no-op on a
never-hooked wrapper instead of raising ``AttributeError: 'LoRAModule' object
has no attribute 'orig_forward'``.

Uses a real ``LoRAModuleWrapper`` built from a real ``TrainConfig`` (the toy
adapter in test_distill_loss_step.py stubs hook/remove and cannot catch this).
"""

from types import SimpleNamespace

# Resolve OneTrainer's modelSetup <-> optimizer_util circular import once.
import modules.util.create  # noqa: F401 -- import-side-effect required
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
from torch import nn


class _TinyNet(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _StubSetup(BaseModelSetup):
    def __init__(self):
        super().__init__(torch.device("cpu"), torch.device("cpu"), False)

    def create_parameters(self, model, config):
        return None

    def setup_optimizations(self, model, config):
        pass

    def setup_model(self, model, config):
        pass

    def setup_train_device(self, model, config):
        pass

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        return {}

    def calculate_loss(self, model, batch, data, config):
        return None

    def after_optimizer_step(self, model, config, train_progress):
        pass


def _lora_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.train_device = "cpu"
    config.peft_type = PeftType.LORA
    config.lora_rank = 4
    config.lora_alpha = 4.0
    config.lora_decompose = False
    config.dropout_probability = 0.0
    config.training_method = TrainingMethod.LORA
    return config


def _make_wrapper(net):
    return LoRAModuleWrapper(net, "transformer", _lora_config(), [])


def test_remove_hook_on_never_hooked_wrapper_is_noop():
    wrapper = _make_wrapper(_TinyNet())
    # never hooked -> orig_forward was never assigned; remove must not raise
    wrapper.remove_hook_from_module()
    assert all(not m.is_applied for m in wrapper.lora_modules.values())


def test_hook_remove_remove_is_idempotent():
    wrapper = _make_wrapper(_TinyNet())
    wrapper.hook_to_module()
    wrapper.remove_hook_from_module()
    wrapper.remove_hook_from_module()  # second remove is a no-op
    assert all(not m.is_applied for m in wrapper.lora_modules.values())


def test_adapter_state_cycles_with_unhooked_fake():
    net = _TinyNet()
    student = _make_wrapper(net)
    fake = _make_wrapper(net)
    student.hook_to_module()  # baseline: only student hooked, fake never hooked

    model = SimpleNamespace(
        adapters=lambda: [student],
        teacher_adapters=list,
        fake_adapters=lambda: [fake],
    )
    setup = _StubSetup()
    config = SimpleNamespace(training_method=TrainingMethod.LORA)

    # The first adapter_state entry used to crash removing the unhooked fake.
    with setup.adapter_state(model, config, "student"):
        assert all(m.is_applied for m in student.lora_modules.values())
        assert all(not m.is_applied for m in fake.lora_modules.values())
    with setup.adapter_state(model, config, "fake"):
        assert all(m.is_applied for m in fake.lora_modules.values())
        assert all(not m.is_applied for m in student.lora_modules.values())

    # baseline restored: student hooked, fake unhooked
    assert all(m.is_applied for m in student.lora_modules.values())
    assert all(not m.is_applied for m in fake.lora_modules.values())
