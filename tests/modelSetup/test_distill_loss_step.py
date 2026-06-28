from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn


class ToggleAdapter:
    def __init__(self):
        self.hooked = False

    def hook_to_module(self):
        self.hooked = True

    def remove_hook_from_module(self):
        self.hooked = False


class TinyDistillModel(BaseModel):
    def __init__(self):
        super().__init__(ModelType.Z_IMAGE)
        self.student_adapter = ToggleAdapter()
        self.fake_adapter = ToggleAdapter()
        self.student_adapter.hook_to_module()
        self.base = nn.Linear(2, 2, bias=False)
        self.student = nn.Linear(2, 2, bias=False)
        self.fake = nn.Linear(2, 2, bias=False)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def to(self, device):
        return self

    def eval(self):
        return self

    def adapters(self):
        return [self.student_adapter]

    def fake_adapters(self):
        return [self.fake_adapter]


class TinyDistillSetup(BaseModelSetup):
    def __init__(self, train_device, temp_device, debug_mode):
        super().__init__(train_device, temp_device, debug_mode)
        self.velocity_calls = []

    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_optimizations(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        x = batch["latent_image"]
        if model.fake_adapter.hooked:
            predicted = model.fake(x)
        elif model.student_adapter.hooked:
            predicted = model.student(x)
        else:
            predicted = model.base(x)
        return {
            "predicted": predicted,
            "predicted_latent": predicted,
            "latent_image": batch["target_latent"],
        }

    def predict_distill_velocity(
        self,
        model,
        batch,
        config,
        train_progress,
        latent,
        sigma,
        *,
        conditioning="conditional",
    ):
        if model.fake_adapter.hooked:
            state = "fake"
            predicted = model.fake(latent)
        elif model.student_adapter.hooked:
            state = "student"
            predicted = model.student(latent)
        else:
            state = "teacher"
            predicted = model.base(latent)

        self.velocity_calls.append((state, conditioning, float(sigma.flatten()[0].detach().cpu())))
        if conditioning == "unconditional":
            return predicted * 0.5
        return predicted

    def calculate_loss(self, model, batch, data, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError


def _config():
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    config.distill_enabled = True
    config.distill_fake_warmup_steps = 1
    config.distill_ttur_ratio = 2
    config.distill_target_steps = 4
    config.distill_teacher_steps = 12
    return config


def test_distill_loss_step_warmup_updates_fake_adapter_only():
    model = TinyDistillModel()
    setup = TinyDistillSetup(torch.device("cpu"), torch.device("cpu"), False)
    progress = TrainProgress()
    progress.global_step = 0
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    loss = setup.calculate_distill_loss(model, batch, _config(), progress)
    loss.backward()

    assert model.fake.weight.grad is not None
    assert model.student.weight.grad is None
    assert model.base.weight.grad is None
    assert setup.get_last_distill_metrics()["mode"] == "fake"
    student_calls = [call for call in setup.velocity_calls if call[0] == "student"]
    assert len(student_calls) == _config().distill_target_steps


def test_distill_loss_step_ttur_student_update_excludes_fake_gradients():
    model = TinyDistillModel()
    setup = TinyDistillSetup(torch.device("cpu"), torch.device("cpu"), False)
    progress = TrainProgress()
    progress.global_step = 3
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    loss = setup.calculate_distill_loss(model, batch, _config(), progress)
    loss.backward()

    assert model.student.weight.grad is not None
    assert model.fake.weight.grad is None
    assert model.base.weight.grad is None
    assert setup.get_last_distill_metrics()["mode"] == "student"


def test_distill_loss_step_runs_metadata_cfg_baked_scheduler_trajectory():
    torch.manual_seed(123)
    model = TinyDistillModel()
    setup = TinyDistillSetup(torch.device("cpu"), torch.device("cpu"), False)
    progress = TrainProgress()
    progress.global_step = 3
    config = _config()
    batch = {
        "latent_image": torch.zeros(2, 2),
        "target_latent": torch.zeros(2, 2),
        "distill_initial_latent": torch.ones(2, 2),
        "distill_cfg_scale": torch.tensor([1.0, 2.0]),
        "distill_teacher_steps": torch.tensor([12, 12]),
    }

    loss = setup.calculate_distill_loss(model, batch, config, progress)
    loss.backward()

    student_calls = [call for call in setup.velocity_calls if call[0] == "student"]
    assert len(student_calls) == config.distill_target_steps
    # CFG>1 on at least one sample must trigger an unconditional teacher pass for
    # the mu_real (cfg-baked) velocity. The fake adapter is conditional-only.
    # The KL is evaluated at a random re-noised sigma, never the sigma=0 endpoint.
    assert any(call[0] == "teacher" and call[1] == "unconditional" for call in setup.velocity_calls)
    assert any(call[0] == "fake" and call[1] == "conditional" for call in setup.velocity_calls)
    assert all(call[2] > 0.0 for call in setup.velocity_calls if call[0] in ("teacher", "fake"))
    metrics = setup.get_last_distill_metrics()
    assert metrics["target_steps"] == 4
    assert metrics["teacher_steps"] == 12
    assert metrics["cfg_scale"] == 1.5


def test_distill_student_kl_gradient_is_nonzero_when_fake_differs_from_teacher():
    # Regression guard for the original defect: when mu_fake != mu_real the
    # distribution-matching surrogate MUST move the student. The old code
    # (fake trained to copy the teacher, KL evaluated at the clean sigma=0
    # endpoint) produced a zero KL gradient and would fail this test.
    torch.manual_seed(7)
    model = TinyDistillModel()
    with torch.no_grad():
        # Force fake != teacher so (v_fake - v_real) is non-zero.
        model.fake.weight.copy_(model.base.weight + 1.0)
        model.student.weight.copy_(torch.eye(2))
    setup = TinyDistillSetup(torch.device("cpu"), torch.device("cpu"), False)
    progress = TrainProgress()
    progress.global_step = 3  # student update step under the _config() TTUR
    config = _config()
    config.distill_endpoint_loss_weight = 0.0  # isolate the KL term
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    loss = setup.calculate_distill_loss(model, batch, config, progress)
    loss.backward()

    assert setup.get_last_distill_metrics()["mode"] == "student"
    assert model.student.weight.grad is not None
    assert model.student.weight.grad.abs().sum().item() > 0.0
