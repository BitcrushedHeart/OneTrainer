from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.distill_dmd2_util import build_distill_sigma_matrix
from modules.util.enum.DistillVramMode import DistillVramMode
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

import pytest


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

        self.velocity_calls.append(
            (state, conditioning, float(sigma.flatten()[0].detach().cpu()), torch.is_grad_enabled(), latent)
        )
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
    # The critic sample is a random-k one-step estimate: k prefix steps + one
    # one-step forward, all under no_grad.
    student_calls = [call for call in setup.velocity_calls if call[0] == "student"]
    assert 1 <= len(student_calls) <= _config().distill_target_steps
    assert all(call[3] is False for call in student_calls)


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
    config.distill_backprop_random_step = False  # pin k to the final step for a deterministic count
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


# --- Paired teacher-match + stochastic backprop-step (per-step one-step) tests ---


def _setup():
    return TinyDistillSetup(torch.device("cpu"), torch.device("cpu"), False)


def _student_progress(global_step: int) -> TrainProgress:
    progress = TrainProgress()
    progress.global_step = global_step
    return progress


def test_x0_has_grad_for_an_early_backprop_step():
    # Guards the autograd defect of the naive design: if steps after k run under
    # no_grad and x0 is the final latent, x0 is detached for any non-final k.
    # The one-step x0 = x_k - sigma_k * v_k must carry grad for ANY chosen k.
    config = _config()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}
    for global_step in range(3, 400, 3):  # student-mode steps under _config TTUR
        model = TinyDistillModel()
        setup = _setup()
        loss = setup.calculate_distill_loss(model, batch, config, _student_progress(global_step))
        if setup._last_grad_step_index == 0:  # an early, non-final step was chosen
            assert loss.requires_grad
            loss.backward()
            assert model.student.weight.grad is not None
            assert model.student.weight.grad.abs().sum().item() > 0.0
            return
    raise AssertionError("random backprop step never selected an early step")


def test_teacher_match_target_depends_on_initial_noise():
    # Well-posedness / diversity guard: different z (and k) -> different teacher
    # target and different student gradient. The old unpaired-image objective
    # produced a noise-independent target and collapsed.
    config = _config()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}
    model = TinyDistillModel()
    setup = _setup()
    out = []
    for global_step in (3, 6):
        model.student.weight.grad = None
        loss = setup.calculate_distill_loss(model, batch, config, _student_progress(global_step))
        assert setup.get_last_distill_metrics()["mode"] == "student"
        loss.backward()
        out.append((setup._last_distill_x0_teacher.clone(), model.student.weight.grad.clone()))
    assert (out[0][0] - out[1][0]).abs().mean().item() > 1e-4
    assert (out[0][1] - out[1][1]).abs().sum().item() > 1e-6


def test_teacher_match_ignores_dataset_image_latent():
    # Regression guard for the collapse bug: the target no longer depends on the
    # dataset image, so changing the image leaves the student gradient unchanged.
    config = _config()
    model = TinyDistillModel()
    setup = _setup()

    def grad_for(image):
        model.student.weight.grad = None
        batch = {"latent_image": image, "target_latent": image}
        loss = setup.calculate_distill_loss(model, batch, config, _student_progress(3))
        loss.backward()
        return model.student.weight.grad.clone()

    g_zeros = grad_for(torch.zeros(4, 2))
    g_fives = grad_for(torch.full((4, 2), 5.0))
    assert torch.allclose(g_zeros, g_fives)


def test_teacher_match_produces_student_gradient_in_isolation():
    config = _config()
    config.distill_kl_loss_weight = 0.0  # isolate the teacher-match term
    model = TinyDistillModel()
    with torch.no_grad():
        model.base.weight.copy_(torch.eye(2) * 2.0)  # teacher != student -> nonzero MSE
        model.student.weight.copy_(torch.eye(2))
    setup = _setup()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    loss = setup.calculate_distill_loss(model, batch, config, _student_progress(3))
    loss.backward()

    assert model.student.weight.grad is not None
    assert model.student.weight.grad.abs().sum().item() > 0.0
    assert model.fake.weight.grad is None


def test_random_backprop_step_covers_all_steps_and_is_reproducible():
    config = _config()  # distill_target_steps == 4
    setup = _setup()
    model = TinyDistillModel()
    sigmas = build_distill_sigma_matrix(
        config.distill_target_steps, torch.tensor([12, 12, 12, 12]), "AUTO", torch.device("cpu")
    )
    z = torch.randn(4, 2)
    batch = {"latent_image": torch.randn(4, 2)}
    seen = set()
    with setup.adapter_state(model, config, "student"):
        for global_step in range(200):
            _, _, _, k = setup._student_onestep_x0(model, batch, config, _student_progress(global_step), z, sigmas)
            seen.add(k)
        # reproducible for a fixed global step
        first = setup._student_onestep_x0(model, batch, config, _student_progress(42), z, sigmas)[3]
        second = setup._student_onestep_x0(model, batch, config, _student_progress(42), z, sigmas)[3]
    assert seen == {0, 1, 2, 3}
    assert first == second


def test_target_steps_one_selects_step_zero():
    config = _config()
    config.distill_target_steps = 1
    model = TinyDistillModel()
    setup = _setup()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}
    loss = setup.calculate_distill_loss(model, batch, config, _student_progress(3))
    loss.backward()
    assert setup._last_grad_step_index == 0
    assert model.student.weight.grad is not None
    assert model.student.weight.grad.abs().sum().item() > 0.0


def test_exactly_one_grad_bearing_student_forward():
    config = _config()
    model = TinyDistillModel()
    setup = _setup()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}
    setup.calculate_distill_loss(model, batch, config, _student_progress(3))
    grad_student_calls = [c for c in setup.velocity_calls if c[0] == "student" and c[3] is True]
    assert len(grad_student_calls) == 1


def test_distill_metric_key_sets_match_across_modes():
    config = _config()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    setup_fake = _setup()
    setup_fake.calculate_distill_loss(TinyDistillModel(), batch, config, _student_progress(0))  # warmup -> fake
    fake_keys = set(setup_fake.get_last_distill_metrics().keys())

    setup_student = _setup()
    setup_student.calculate_distill_loss(TinyDistillModel(), batch, config, _student_progress(3))  # student
    student_keys = set(setup_student.get_last_distill_metrics().keys())

    assert setup_fake.get_last_distill_metrics()["mode"] == "fake"
    assert setup_student.get_last_distill_metrics()["mode"] == "student"
    assert fake_keys == student_keys
    assert {"loss", "fake_loss", "kl_loss", "endpoint_loss"} <= student_keys


def test_cfg_scale_controls_unconditional_teacher_pass():
    config = _config()

    setup_one = _setup()
    batch_one = {
        "latent_image": torch.zeros(2, 2),
        "target_latent": torch.zeros(2, 2),
        "distill_cfg_scale": torch.tensor([1.0, 1.0]),
        "distill_teacher_steps": torch.tensor([12, 12]),
    }
    setup_one.calculate_distill_loss(TinyDistillModel(), batch_one, config, _student_progress(3))
    assert not any(c[1] == "unconditional" for c in setup_one.velocity_calls)

    setup_gt1 = _setup()
    batch_gt1 = dict(batch_one, distill_cfg_scale=torch.tensor([2.0, 2.0]))
    setup_gt1.calculate_distill_loss(TinyDistillModel(), batch_gt1, config, _student_progress(3))
    assert any(c[0] == "teacher" and c[1] == "unconditional" for c in setup_gt1.velocity_calls)


def test_lpips_unavailable_path_is_finite():
    config = _config()
    config.distill_endpoint_lpips_weight = 1.0  # base _distill_perceptual_loss returns None -> skipped
    model = TinyDistillModel()
    setup = _setup()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}
    loss = setup.calculate_distill_loss(model, batch, config, _student_progress(3))
    assert torch.isfinite(loss)
    assert setup.get_last_distill_metrics()["lpips_loss"] == 0.0


def _clone_model(source: TinyDistillModel) -> TinyDistillModel:
    clone = TinyDistillModel()
    with torch.no_grad():
        clone.base.weight.copy_(source.base.weight)
        clone.student.weight.copy_(source.student.weight)
        clone.fake.weight.copy_(source.fake.weight)
    return clone


def test_concurrent_mode_matches_split_loss_and_gradient():
    # CONCURRENT fuses the two frozen-teacher passes into one double-batch
    # forward; the objective must be unchanged.
    torch.manual_seed(11)
    config_split = _config()
    config_concurrent = _config()
    config_concurrent.distill_vram_mode = DistillVramMode.CONCURRENT
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    model_split = TinyDistillModel()
    model_concurrent = _clone_model(model_split)

    setup_split = _setup()
    loss_split = setup_split.calculate_distill_loss(model_split, batch, config_split, _student_progress(3))
    loss_split.backward()

    setup_concurrent = _setup()
    loss_concurrent = setup_concurrent.calculate_distill_loss(
        model_concurrent, batch, config_concurrent, _student_progress(3)
    )
    loss_concurrent.backward()

    assert setup_split.get_last_distill_metrics()["mode"] == "student"
    assert setup_concurrent.get_last_distill_metrics()["mode"] == "student"
    assert torch.allclose(loss_split, loss_concurrent, rtol=1e-6, atol=1e-7)
    assert torch.allclose(model_split.student.weight.grad, model_concurrent.student.weight.grad, rtol=1e-6, atol=1e-7)
    assert torch.allclose(
        setup_split._last_distill_x0_teacher, setup_concurrent._last_distill_x0_teacher, rtol=1e-6, atol=1e-7
    )


def test_concurrent_mode_fuses_teacher_passes_into_one_call():
    config = _config()
    config.distill_vram_mode = DistillVramMode.CONCURRENT
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    setup = _setup()
    setup.calculate_distill_loss(TinyDistillModel(), batch, config, _student_progress(3))

    teacher_calls = [c for c in setup.velocity_calls if c[0] == "teacher"]
    assert len(teacher_calls) == 1
    # Fused call carries [x_k; x_sigma]: double the batch of the student forwards.
    student_latent = next(c[4] for c in setup.velocity_calls if c[0] == "student")
    assert teacher_calls[0][4].shape[0] == 2 * student_latent.shape[0]


def test_concurrent_mode_fuses_cfg_teacher_passes_per_conditioning():
    # With CFG > 1 the teacher needs conditional + unconditional velocities;
    # CONCURRENT still fuses x_k/x_sigma so exactly one call runs per conditioning.
    config = _config()
    config.distill_vram_mode = DistillVramMode.CONCURRENT
    batch = {
        "latent_image": torch.zeros(2, 2),
        "target_latent": torch.zeros(2, 2),
        "distill_cfg_scale": torch.tensor([2.0, 2.0]),
        "distill_teacher_steps": torch.tensor([12, 12]),
    }

    setup = _setup()
    setup.calculate_distill_loss(TinyDistillModel(), batch, config, _student_progress(3))

    teacher_cond = [c for c in setup.velocity_calls if c[0] == "teacher" and c[1] == "conditional"]
    teacher_uncond = [c for c in setup.velocity_calls if c[0] == "teacher" and c[1] == "unconditional"]
    assert len(teacher_cond) == 1
    assert len(teacher_uncond) == 1


def test_sigma_shift_changes_training_grid_but_keeps_gradient_flow():
    config = _config()
    config.distill_sigma_shift = 1.5
    model = TinyDistillModel()
    setup = _setup()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    loss = setup.calculate_distill_loss(model, batch, config, _student_progress(3))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.student.weight.grad is not None
    assert model.student.weight.grad.abs().sum().item() > 0.0


def test_fake_mode_final_step_estimate_equals_full_rollout_endpoint():
    # With random backprop step off, the critic's sample (one-step x0 from the
    # final grid step) is mathematically the full rollout endpoint:
    # x_final = x_k + (0 - sigma_k) * v_k = x_k - sigma_k * v_k. This makes the
    # pinned-step configuration exactly reproduce the old full-rollout critic.
    config = _config()
    config.distill_backprop_random_step = False
    model = TinyDistillModel()
    setup = _setup()
    sigmas = build_distill_sigma_matrix(
        config.distill_target_steps, torch.tensor([12, 12, 12, 12]), "AUTO", torch.device("cpu")
    )
    z = torch.randn(4, 2)
    batch = {"latent_image": torch.randn(4, 2)}

    with setup.adapter_state(model, config, "student"), torch.no_grad():
        x0, _, _, k = setup._student_onestep_x0(model, batch, config, _student_progress(0), z, sigmas)
        latent = z
        for step_index in range(config.distill_target_steps):
            velocity = model.student(latent)
            delta = (sigmas[:, step_index + 1] - sigmas[:, step_index]).unsqueeze(-1)
            latent = latent + delta * velocity

    assert k == config.distill_target_steps - 1
    assert torch.allclose(x0, latent, rtol=1e-5, atol=1e-6)


def test_fake_mode_pinned_step_runs_the_full_rollout():
    config = _config()
    config.distill_backprop_random_step = False
    setup = _setup()
    batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}

    setup.calculate_distill_loss(TinyDistillModel(), batch, config, _student_progress(0))

    student_calls = [c for c in setup.velocity_calls if c[0] == "student"]
    assert len(student_calls) == config.distill_target_steps


def test_fake_mode_critic_samples_cover_all_backprop_steps_over_time():
    # The critic must be trained on one-step estimates from every trajectory
    # step, matching the distribution the student update's KL queries.
    config = _config()
    counts = set()
    for global_step in range(40):
        progress = _student_progress(global_step)
        if BaseModelSetup._distill_update_mode(config, progress) != "fake":
            continue
        setup = _setup()
        batch = {"latent_image": torch.randn(4, 2), "target_latent": torch.zeros(4, 2)}
        setup.calculate_distill_loss(TinyDistillModel(), batch, config, progress)
        counts.add(len([c for c in setup.velocity_calls if c[0] == "student"]))
    assert counts == {1, 2, 3, 4}


def test_one_grad_forward_with_inplace_buffer_is_autograd_safe():
    # Proves the invariant behind "exactly one grad-bearing student forward":
    # OFT spectral-norm mutates its u/v buffers in place each forward, so a second
    # grad-bearing forward corrupts the first forward's saved graph.
    class PowerIterModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))
            self.register_buffer("scale", torch.ones(2))

        def forward(self, x):
            with torch.no_grad():
                self.scale.mul_(0.9).add_(0.1)  # power-iteration-style in-place update
            return (x @ self.weight) * self.scale  # uses (and saves) the mutated buffer

    module = PowerIterModule()
    x0 = torch.randn(3, 2)
    with torch.no_grad():  # no-grad rollout: mutates the buffer but builds no graph
        h = module(x0)
        h = module(h)
    x_k = h.detach()

    out = module(x_k)  # single grad-bearing forward
    out.pow(2).mean().backward()  # must not raise
    assert module.weight.grad is not None

    module.weight.grad = None
    a = module(x_k)
    b = module(a)  # second grad forward re-mutates the buffer a's graph saved
    with pytest.raises(RuntimeError):
        b.pow(2).mean().backward()
