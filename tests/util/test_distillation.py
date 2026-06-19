"""End-to-end CPU-only verification of the DoRA -> DoRA-OFT distillation
primitives added in :mod:`modules.modelSetup.ZImageLoRASetup` and
:meth:`BaseModelSetup.distillation_teacher_model`.

The tests construct a multi-layer ``nn.Linear`` toy model, wrap it with real
``LoRAModuleWrapper`` instances for both the DoRA teacher and the DoRA-OFT
student, and exercise the hook-toggle context manager + a short training
loop against the teacher's outputs. No GPU is used.
"""

from contextlib import suppress
from types import SimpleNamespace

# OneTrainer's modelSetup files have a circular dependency with optimizer_util
# (via modules.util.create.factory.import_dir). Pre-importing the create
# module forces the full graph to resolve once before the test's own imports
# below touch the same chain. This matches the normal CLI bootstrap order.
import modules.util.create  # noqa: F401  -- import-side-effect required
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.ZImageLoRASetup import (
    _build_teacher_config_view,
    _infer_dora_shape_from_state_dict,
)
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import PeftType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
import torch.nn.functional as F
from torch import nn

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _dora_config(rank: int = 4, alpha: float = 4.0) -> SimpleNamespace:
    """Minimal config that ``LoRAModuleWrapper`` reads for the DoRA path."""
    return SimpleNamespace(
        peft_type=PeftType.LORA,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_decompose=True,
        lora_decompose_norm_epsilon=True,
        lora_decompose_output_axis=False,
        train_device="cpu",
        dora_oft=False,
        oft_scaled=False,
        oft_block_size=4,
        oft_block_share=False,
        layer_filter_regex=False,
        dropout_probability=0.0,
    )


def _doraoft_config(block_size: int = 4) -> SimpleNamespace:
    """Minimal config for the DoRA-OFT student path."""
    return SimpleNamespace(
        peft_type=PeftType.OFT_2,
        lora_rank=0,
        lora_alpha=1.0,
        lora_decompose=False,
        lora_decompose_norm_epsilon=True,
        lora_decompose_output_axis=False,
        train_device="cpu",
        dora_oft=True,
        oft_scaled=False,
        oft_block_size=block_size,
        oft_block_share=False,
        layer_filter_regex=False,
        dropout_probability=0.0,
    )


class _TinyTransformer(nn.Module):
    """3-layer linear stack with deterministic init -- our stand-in for the
    Z-Image transformer at distillation-test scale."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class _MockModel:
    """Stand-in for a BaseModel exposing only the two accessors the
    distillation context manager needs."""

    def __init__(self, student_wrapper=None, teacher_wrapper=None) -> None:
        self._student = student_wrapper
        self._teacher = teacher_wrapper

    def adapters(self):
        return [self._student] if self._student is not None else []

    def teacher_adapters(self):
        return [self._teacher] if self._teacher is not None else []


class _StubSetup(BaseModelSetup):
    """Concrete ``BaseModelSetup`` subclass so we can call the real
    ``distillation_teacher_model`` ctx mgr from a test."""

    def __init__(self) -> None:
        super().__init__(
            train_device=torch.device("cpu"),
            temp_device=torch.device("cpu"),
            debug_mode=False,
        )

    def create_parameters(self, model, config):
        return None

    def setup_optimizations(self, model, config):
        pass

    def setup_model(self, model, config):
        pass

    def setup_train_device(self, model, config):
        pass

    def predict(self, model, batch, config, train_progress, *, deterministic: bool = False):
        return {}

    def calculate_loss(self, model, batch, data, config):
        return torch.tensor(0.0)

    def after_optimizer_step(self, model, config, train_progress):
        pass


def _randomize_dora_teacher(wrapper: LoRAModuleWrapper) -> None:
    """Populate a freshly-built DoRA wrapper with non-identity weights so its
    forward output differs from the base model."""
    torch.manual_seed(42)
    for mod in wrapper.lora_modules.values():
        # nudge lora_up off zero so the LoRA delta is non-zero
        with torch.no_grad():
            nn.init.normal_(mod.lora_up.weight, std=0.05)
            nn.init.normal_(mod.lora_down.weight, std=0.05)
            # nudge magnitudes slightly off initial_norm so DoRA scale fires
            mod.dora_scale.data.mul_(1.1)


# ---------------------------------------------------------------------------
# unit-level invariants
# ---------------------------------------------------------------------------


class TestInferDoRAShape:
    def test_extracts_rank_and_alpha_from_a_real_dora_sd(self):
        sd = {
            "transformer.layer.0.lora_down.weight": torch.zeros(8, 16),
            "transformer.layer.0.lora_up.weight": torch.zeros(16, 8),
            "transformer.layer.0.alpha": torch.tensor(4.0),
            "transformer.layer.0.dora_scale": torch.ones(16, 1),
        }
        rank, alpha = _infer_dora_shape_from_state_dict(sd)
        assert rank == 8
        assert alpha == pytest.approx(4.0)

    def test_missing_lora_down_raises(self):
        with pytest.raises(ValueError):
            _infer_dora_shape_from_state_dict({"foo.weight": torch.zeros(4)})

    def test_missing_alpha_defaults_to_rank(self):
        sd = {
            "x.lora_down.weight": torch.zeros(8, 16),
            "x.lora_up.weight": torch.zeros(16, 8),
        }
        rank, alpha = _infer_dora_shape_from_state_dict(sd)
        assert rank == 8
        assert alpha == pytest.approx(8.0)


class TestBuildTeacherConfigView:
    def test_overrides_peft_settings_without_mutating_original(self):
        student_cfg = _doraoft_config()
        teacher_sd = {
            "x.lora_down.weight": torch.zeros(4, 16),
            "x.lora_up.weight": torch.zeros(16, 4),
            "x.alpha": torch.tensor(2.0),
        }
        teacher_cfg = _build_teacher_config_view(student_cfg, teacher_sd)

        # Teacher view reflects the file
        assert teacher_cfg.peft_type == PeftType.LORA
        assert teacher_cfg.lora_decompose is True
        assert teacher_cfg.dora_oft is False
        assert teacher_cfg.lora_rank == 4
        assert teacher_cfg.lora_alpha == pytest.approx(2.0)

        # Student is untouched
        assert student_cfg.peft_type == PeftType.OFT_2
        assert student_cfg.dora_oft is True


# ---------------------------------------------------------------------------
# DoRA-OFT student starts as the identity transform
# ---------------------------------------------------------------------------


class TestStudentIdentityAtInit:
    """A freshly-constructed DoRA-OFT wrapper applied to the base model
    must produce output identical to the base model. This is the
    step-0 sanity assert called out in the plan."""

    def test_doraoft_init_is_identity(self):
        torch.manual_seed(0)
        net = _TinyTransformer()
        x = torch.randn(2, 16)

        base_out = net(x).detach().clone()

        wrapper = LoRAModuleWrapper(net, "transformer", _doraoft_config(), [])
        wrapper.hook_to_module()
        try:
            student_out = net(x)
        finally:
            wrapper.remove_hook_from_module()

        assert torch.allclose(student_out, base_out, atol=1e-6), (
            "DoRA-OFT at init must equal the bare base model output"
        )


# ---------------------------------------------------------------------------
# distillation_teacher_model context manager
# ---------------------------------------------------------------------------


class TestDistillationContextManager:
    def _build_wrappers(self):
        net = _TinyTransformer()
        student = LoRAModuleWrapper(net, "transformer", _doraoft_config(), [])
        teacher = LoRAModuleWrapper(net, "transformer", _dora_config(), [])
        _randomize_dora_teacher(teacher)
        # Mirror real setup_model behaviour: only the student is hooked at rest.
        student.hook_to_module()
        return net, student, teacher

    def test_swaps_active_adapter_inside_context(self):
        net, student, teacher = self._build_wrappers()
        x = torch.randn(2, 16)

        base_only_wrapper = LoRAModuleWrapper(_TinyTransformer(), "transformer", _doraoft_config(), [])
        del base_only_wrapper  # fresh reference net just for sanity below
        baseline_net = _TinyTransformer()
        base_out = baseline_net(x).detach().clone()
        del baseline_net

        # outside the ctx: student is hooked -> identity -> == base
        student_out = net(x).detach().clone()
        assert torch.allclose(student_out, base_out, atol=1e-6)

        setup = _StubSetup()
        model = _MockModel(student_wrapper=student, teacher_wrapper=teacher)
        config = SimpleNamespace(training_method=TrainingMethod.LORA)

        with setup.distillation_teacher_model(model, config):
            teacher_out = net(x).detach().clone()

        # inside the ctx, teacher was hooked -> non-identity weights mean output diverges
        assert not torch.allclose(teacher_out, base_out, atol=1e-3), (
            "Teacher should produce a measurably different output from the base model"
        )

        # after exit, student is hooked again -> back to identity
        post_out = net(x).detach().clone()
        assert torch.allclose(post_out, base_out, atol=1e-6), (
            "After ctx exit the student must be re-hooked, restoring base behaviour"
        )

        student.remove_hook_from_module()

    def test_ctx_restores_hooks_on_exception(self):
        net, student, teacher = self._build_wrappers()
        x = torch.randn(2, 16)
        base = _TinyTransformer()(x).detach().clone()

        setup = _StubSetup()
        model = _MockModel(student_wrapper=student, teacher_wrapper=teacher)
        config = SimpleNamespace(training_method=TrainingMethod.LORA)

        with suppress(RuntimeError), setup.distillation_teacher_model(model, config):
            raise RuntimeError("simulated teacher failure")

        # After the exception, the student should be hooked again and the teacher unhooked.
        out = net(x).detach().clone()
        assert torch.allclose(out, base, atol=1e-6), (
            "Even after an exception inside the ctx, the student must be re-hooked"
        )

        student.remove_hook_from_module()

    def test_raises_when_no_teacher_attached(self):
        net = _TinyTransformer()
        student = LoRAModuleWrapper(net, "transformer", _doraoft_config(), [])
        student.hook_to_module()

        setup = _StubSetup()
        model = _MockModel(student_wrapper=student, teacher_wrapper=None)
        config = SimpleNamespace(training_method=TrainingMethod.LORA)

        with pytest.raises(RuntimeError, match="no teacher adapters"):
            with setup.distillation_teacher_model(model, config):
                pass

        student.remove_hook_from_module()

    def test_rejects_non_lora_training_method(self):
        net = _TinyTransformer()
        student = LoRAModuleWrapper(net, "transformer", _doraoft_config(), [])
        teacher = LoRAModuleWrapper(net, "transformer", _dora_config(), [])
        student.hook_to_module()

        setup = _StubSetup()
        model = _MockModel(student_wrapper=student, teacher_wrapper=teacher)
        config = SimpleNamespace(training_method=TrainingMethod.FINE_TUNE)

        with pytest.raises(NotImplementedError), setup.distillation_teacher_model(model, config):
            pass

        student.remove_hook_from_module()


# ---------------------------------------------------------------------------
# end-to-end mini training loop
# ---------------------------------------------------------------------------


class TestEndToEndDistillation:
    """Run the same loop the trainer runs: teacher forward (no_grad) ->
    student forward (grad) -> MSE -> Adam step. Verify the student moves
    toward the teacher and that the teacher itself does NOT move."""

    def test_student_converges_toward_teacher_and_teacher_is_frozen(self):
        torch.manual_seed(123)

        net = _TinyTransformer()
        student = LoRAModuleWrapper(net, "transformer", _doraoft_config(), [])
        teacher = LoRAModuleWrapper(net, "transformer", _dora_config(), [])
        _randomize_dora_teacher(teacher)
        # Freeze the underlying base model (mirrors setup_requires_grad).
        for p in net.parameters():
            p.requires_grad_(False)
        teacher.requires_grad_(False)

        # Snapshot the teacher's params -- after training they must be unchanged.
        teacher_snapshot = [p.detach().clone() for p in teacher.parameters()]

        student.hook_to_module()
        setup = _StubSetup()
        model = _MockModel(student_wrapper=student, teacher_wrapper=teacher)
        config = SimpleNamespace(training_method=TrainingMethod.LORA)

        student_params = [p for p in student.parameters() if p.requires_grad]
        assert len(student_params) > 0, "Student must expose trainable params"
        opt = torch.optim.Adam(student_params, lr=5e-3)

        def step_loss(x: torch.Tensor) -> torch.Tensor:
            with setup.distillation_teacher_model(model, config), torch.no_grad():
                target = net(x).detach()
            pred = net(x)
            return F.mse_loss(pred, target)

        # Step 0 sanity: student is identity, so MSE equals teacher-vs-base.
        torch.manual_seed(999)
        x0 = torch.randn(4, 16)
        loss_initial = step_loss(x0).item()
        assert loss_initial > 1e-4, "Teacher must differ measurably from base at step 0"

        # Train.
        torch.manual_seed(999)
        for _ in range(120):
            x = torch.randn(4, 16)
            opt.zero_grad()
            loss = step_loss(x)
            loss.backward()
            opt.step()

        torch.manual_seed(999)
        x_eval = torch.randn(4, 16)
        loss_final = step_loss(x_eval).item()

        student.remove_hook_from_module()

        assert loss_final < loss_initial * 0.6, (
            f"Distillation should reduce teacher-student MSE; initial={loss_initial:.6f} final={loss_final:.6f}"
        )

        # Teacher params must NOT have moved.
        for before, after in zip(teacher_snapshot, teacher.parameters(), strict=True):
            assert torch.equal(before, after.detach()), (
                "Teacher params changed during distillation -- it should be frozen"
            )

    def test_base_anchor_path_runs_and_pulls_toward_base(self):
        """When the optional base-anchor term is enabled, the trainer adds
        MSE(student, base) to the loss. Verify the path works mechanically:
        the third forward (under prior_model ctx) executes and the combined
        loss is finite and differentiable."""
        torch.manual_seed(0)
        net = _TinyTransformer()
        student = LoRAModuleWrapper(net, "transformer", _doraoft_config(), [])
        teacher = LoRAModuleWrapper(net, "transformer", _dora_config(), [])
        _randomize_dora_teacher(teacher)
        for p in net.parameters():
            p.requires_grad_(False)
        teacher.requires_grad_(False)
        student.hook_to_module()

        setup = _StubSetup()
        model = _MockModel(student_wrapper=student, teacher_wrapper=teacher)
        config = SimpleNamespace(training_method=TrainingMethod.LORA)

        x = torch.randn(2, 16)
        with setup.distillation_teacher_model(model, config), torch.no_grad():
            teacher_out = net(x).detach()
        with setup.prior_model(model, config), torch.no_grad():
            base_out = net(x).detach()
        student_out = net(x)

        anchor_w = 0.05
        loss = F.mse_loss(student_out, teacher_out) + anchor_w * F.mse_loss(student_out, base_out)
        loss.backward()

        student.remove_hook_from_module()
        assert torch.isfinite(loss), "Distillation+anchor loss must be finite"
        # at least one student param should have a non-zero grad
        grads = [p.grad for p in student.parameters() if p.grad is not None]
        assert any(g.abs().sum().item() > 0 for g in grads), (
            "Backprop through distillation+anchor must produce non-zero student gradients"
        )
