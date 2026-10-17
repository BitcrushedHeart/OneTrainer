"""GenericTrainer wiring of the OFT rotation cache (config oft_rotation_cache).

CPU-only. Covers startup validation (hard errors only when the cache would be active,
a warning otherwise), the update-step ordering (materialize after gradient reduction
and before unscale/clip/step; the per-update adapter call after the model-specific
after_optimizer_step with no forward in between, before EMA and validation), a
mid-window stop whose in-flight rotation gradients land in the backup's accumulator
state exactly as without the cache, the redistill best-restore, and the sampling swaps.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import modules.trainer.GenericTrainer as generic_trainer_module
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.module.oft_utils import OFTRotationModule
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.checkpointing_util import CheckpointLayer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

import pytest
from safetensors.torch import save_file


def _cache_config(tmp_path=None, **overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.tensorboard = False
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.OFT_2
    config.oft_rotation_cache = True
    config.train_dtype = DataType.BFLOAT_16
    if tmp_path is not None:
        config.workspace_dir = str(tmp_path / "workspace")
        config.cache_dir = str(tmp_path / "cache")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# --------------------------------------------------------------------------- #
# startup validation (amendment 2)
# --------------------------------------------------------------------------- #


def test_active_cache_accepts_a_valid_run(capsys):
    config = _cache_config()
    assert config.oft_rotation_cache_applies()
    GenericTrainer(config, TrainCallbacks(), TrainCommands())
    assert "rotation cache" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c: setattr(c.optimizer, "fused_back_pass", True), "fused back pass"),
        (lambda c: setattr(c, "multi_gpu", True), "single-GPU"),
        (lambda c: setattr(c, "train_dtype", DataType.FLOAT_16), "bfloat16"),
        (lambda c: setattr(c, "train_dtype", DataType.FLOAT_32), "bfloat16"),
    ],
    ids=["fused_back_pass", "multi_gpu", "float16", "float32"],
)
def test_active_cache_rejects_unsupported_runs(mutate, match):
    config = _cache_config()
    mutate(config)
    with pytest.raises(RuntimeError, match=match):
        GenericTrainer(config, TrainCallbacks(), TrainCommands())


@pytest.mark.parametrize(
    ("training_method", "peft_type"),
    [
        (TrainingMethod.LORA, PeftType.LORA),
        (TrainingMethod.LORA, PeftType.LOKR),
        (TrainingMethod.FINE_TUNE, PeftType.OFT_2),
    ],
)
def test_flag_on_a_non_oft_run_warns_and_is_ignored(capsys, training_method, peft_type):
    # A shared preset: the cache flag is on, and so are options it would reject.
    config = _cache_config(training_method=training_method, peft_type=peft_type, train_dtype=DataType.FLOAT_16)
    config.optimizer.fused_back_pass = True
    assert not config.oft_rotation_cache_applies()
    assert config.validate_oft_rotation_cache_startup() is None
    GenericTrainer(config, TrainCallbacks(), TrainCommands())
    out = capsys.readouterr().out
    assert out.count("OFT rotation cache is enabled") == 1 and "ignoring it" in out


def test_flag_off_neither_warns_nor_validates(capsys):
    config = _cache_config(oft_rotation_cache=False, train_dtype=DataType.FLOAT_16)
    config.optimizer.fused_back_pass = True
    GenericTrainer(config, TrainCallbacks(), TrainCommands())
    assert "rotation cache" not in capsys.readouterr().out


def test_config_defaults_off_and_old_configs_load_with_it_off():
    assert TrainConfig.default_values().oft_rotation_cache is False
    data = TrainConfig.default_values().to_dict()
    data.pop("oft_rotation_cache")
    assert TrainConfig.default_values().from_dict(data).oft_rotation_cache is False
    data["oft_rotation_cache"] = True
    assert TrainConfig.default_values().from_dict(data).oft_rotation_cache is True


# --------------------------------------------------------------------------- #
# update-step ordering and the call-site move
# --------------------------------------------------------------------------- #


def _ordering_trainer(
    tmp_path, monkeypatch, *, use_scaler: bool, gradient_accumulation_steps: int = 3, batches: int = 6
):
    config = _cache_config(
        tmp_path,
        train_device="cpu",
        temp_device="cpu",
        concept_file_name=str(tmp_path / "missing-concepts.json"),
        sample_definition_file_name=str(tmp_path / "missing-samples.json"),
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        sample_after_unit=TimeUnit.NEVER,
        backup_after_unit=TimeUnit.NEVER,
        save_every_unit=TimeUnit.NEVER,
        rolling_backup=False,
        clip_grad_norm=1.0,
        validation=True,
        ema=EMAMode.GPU,
        # DPO primary run with a per-update reference rebase and the adaptive beta:
        # both run between the optimizer step and the adapter call.
        rlhf_enabled=True,
        rlhf_mode=RLHFMode.DPO,
        rlhf_dpo_ref_mode=DPORefMode.EXISTING_ADAPTER,
        rlhf_dpo_ref_rebase_interval=1,
        rlhf_dpo_adaptive_beta=True,
    )
    config.optimizer.optimizer = Optimizer.ADAMW

    events: list[str] = []
    parameter = nn.Parameter(torch.ones(3))
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    trainer.tensorboard = Mock()
    trainer.parameters = [parameter]
    trainer.sample_queue = []
    trainer.sft_anchor_source = None
    trainer.dpo_addon_source = None
    trainer._GenericTrainer__validate = lambda train_progress: events.append("validate")
    trainer._dpo_beta_controller = None  # set up by start(), which this test skips
    trainer._dpo_ref_rebase_bucket = 0

    optimizer = Mock(param_groups=[{"params": [parameter]}])
    optimizer.step.side_effect = lambda: events.append("step")
    optimizer.zero_grad.side_effect = lambda set_to_none=True: events.append("zero_grad")
    ema = Mock()
    ema.step.side_effect = lambda *args: events.append("ema")
    ema.get_current_decay.return_value = 0.9
    trainer.model = Mock(
        ema=ema, accumulator_state=None, parameters=None, train_progress=TrainProgress(), optimizer=optimizer
    )

    setup = trainer.model_setup = Mock()
    # No perceptual term: the trainer skips the metric pop when the hook is None.
    setup.pop_perceptual_metrics = None
    metrics = {
        "loss": 0.1,
        "dpo_loss": 0.1,
        "chosen_reward": 0.2,
        "rejected_reward": 0.1,
        "reward_margin": 0.1,
        "accuracy": 1.0,
    }
    setup.calculate_dpo_loss.side_effect = lambda *args, **kwargs: events.append("forward") or (parameter * 2.0).sum()
    setup.predict.side_effect = lambda *args, **kwargs: events.append("forward") or {}
    setup.get_last_dpo_metrics.return_value = metrics
    setup.before_clip_adapters.side_effect = lambda model, config: events.append("before_clip")
    setup.rebase_dpo_reference.side_effect = lambda *args: events.append("rebase")
    setup.set_dpo_runtime_beta.side_effect = lambda *args: events.append("beta")
    setup.report_to_tensorboard.side_effect = lambda *args: events.append("report")
    setup.after_optimizer_step.side_effect = lambda *args: events.append("after_optimizer_step")
    setup.after_optimizer_step_adapters.side_effect = lambda model, config: events.append("adapters")

    trainer.model_saver = Mock()
    trainer.data_loader = Mock()
    trainer.data_loader.get_data_set.return_value.approximate_length.return_value = batches
    trainer.data_loader.get_data_loader.return_value = [{"concept_type": ["STANDARD"]} for _ in range(batches)]

    monkeypatch.setattr(
        generic_trainer_module.multi, "reduce_grads_mean", lambda *args, **kwargs: events.append("reduce")
    )
    monkeypatch.setattr(generic_trainer_module.multi, "finish_async", lambda *args, **kwargs: events.append("reduce"))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args, **kwargs: events.append("clip"))
    if use_scaler:
        scaler = Mock()
        scaler.scale.side_effect = lambda tensor: tensor
        scaler.unscale_.side_effect = lambda opt: events.append("unscale")
        scaler.step.side_effect = lambda opt: events.append("step")
        monkeypatch.setattr(generic_trainer_module, "create_grad_scaler", lambda: scaler)
    monkeypatch.setattr(generic_trainer_module, "enable_grad_scaling", lambda *args: use_scaler)
    monkeypatch.setattr(generic_trainer_module, "gc", Mock())
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())
    return trainer, events, setup


@pytest.mark.parametrize("use_scaler", [False, True], ids=["plain", "scaler"])
def test_update_step_order_and_no_forward_between_step_and_adapter_call(tmp_path, monkeypatch, use_scaler):
    trainer, events, setup = _ordering_trainer(tmp_path, monkeypatch, use_scaler=use_scaler)
    trainer.train()

    assert events.count("forward") == 6
    assert setup.before_clip_adapters.call_count == 2  # once per update, not per micro-step
    assert setup.after_optimizer_step_adapters.call_count == 2

    windows, current = [], []
    for event in events:
        current.append(event)
        if event == "ema":
            windows.append(current)
            current = []
    assert len(windows) == 2
    for window in windows:
        first = {name: window.index(name) for name in reversed(window)}
        before_clip = window.index("before_clip")
        assert first["reduce"] < before_clip
        if use_scaler:
            assert before_clip < first["unscale"] < first["clip"] < first["step"]
        else:
            assert before_clip < first["clip"] < first["step"]
        order = ["step", "zero_grad", "rebase", "beta", "report", "after_optimizer_step", "adapters", "ema"]
        positions = [window.index(name) for name in order]
        assert positions == sorted(positions), window
        between = window[window.index("step") : window.index("adapters")]
        assert "forward" not in between and "validate" not in between
    # validation of the update micro-step runs after the adapter call (and EMA)
    assert events.index("validate", events.index("adapters")) > events.index("ema")


# --------------------------------------------------------------------------- #
# mid-window stop: backup staging and end() with a real cache
# --------------------------------------------------------------------------- #

HIDDEN = 16


class _RealSetup(BaseModelSetup):
    def __init__(self, events):
        super().__init__(torch.device("cpu"), torch.device("cpu"), True)
        self.events = events

    def create_parameters(self, model, config):
        return None

    def setup_optimizations(self, model, config):
        pass

    def setup_model(self, model, config):
        pass

    def setup_train_device(self, model, config):
        pass

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        return {"y": model.toy(batch["x"]), "upstream": batch["upstream"]}

    def calculate_loss(self, model, batch, data, config):
        return (data["y"] * data["upstream"]).sum()

    def after_optimizer_step(self, model, config, train_progress):
        pass

    def report_to_tensorboard(self, *args, **kwargs):
        pass

    def before_clip_adapters(self, model, config):
        self.events.append("before_clip")
        super().before_clip_adapters(model, config)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(HIDDEN, HIDDEN)
        self.to_k = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        return self.to_k(torch.tanh(self.to_q(x)))


def _tiny_model(config: TrainConfig, events: list[str], *, cached: bool):
    torch.manual_seed(0)
    toy = nn.Sequential(_Block(), _Block())
    wrapper = LoRAModuleWrapper(toy, "transformer", config, [])
    with torch.no_grad():
        for module in wrapper.lora_modules.values():
            module.oft_R.weight.normal_(0, 0.3)
            module.oft_R.warm_start_spectral_state()
    wrapper.hook_to_module()
    for i, block in enumerate(toy):
        toy[i] = CheckpointLayer(block, None, torch.device("cpu"))
    if cached:
        for module in wrapper.lora_modules.values():
            module.oft_R.enable_rotation_cache(torch.float32)  # exact accumulation for the comparison
    params = wrapper.parameters()
    model = SimpleNamespace(
        toy=toy,
        wrapper=wrapper,
        ema=None,
        accumulator_state=None,
        train_progress=TrainProgress(),
        optimizer=torch.optim.SGD(params, lr=1e-2),
        parameters=SimpleNamespace(iter_named_parameters=lambda: [(f"p{i}", p) for i, p in enumerate(params)]),
        adapters=lambda: [wrapper],
        fake_adapters=list,
        to=lambda device: events.append("model.to"),
    )
    return model, params


def _stopped_run(tmp_path, monkeypatch, *, cached: bool):
    config = _cache_config(
        tmp_path,
        oft_rotation_cache=cached,
        train_device="cpu",
        temp_device="cpu",
        concept_file_name=str(tmp_path / "missing-concepts.json"),
        sample_definition_file_name=str(tmp_path / "missing-samples.json"),
        output_model_destination=str(tmp_path / "out" / "model.safetensors"),
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=3,
        sample_after_unit=TimeUnit.NEVER,
        backup_after_unit=TimeUnit.NEVER,
        save_every_unit=TimeUnit.NEVER,
        rolling_backup=False,
        backup_before_save=True,
        oft_block_size=8,
        dora_oft=True,
        oft_scaled=True,
        oft_clipped_norm=0.95,
        dropout_probability=0.0,
    )
    events: list[str] = []
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    trainer.tensorboard = Mock()
    trainer.model, params = _tiny_model(config, events, cached=cached)
    trainer.parameters = params
    trainer.sample_queue = []
    trainer.sft_anchor_source = None
    trainer.dpo_addon_source = None
    setup = trainer.model_setup = _RealSetup(events)

    gen = torch.Generator().manual_seed(1)
    batches = [
        {
            "concept_type": ["STANDARD"],
            "x": torch.randn(2, HIDDEN, generator=gen),
            "upstream": torch.randn(2, HIDDEN, generator=gen),
        }
        for _ in range(6)
    ]
    original_predict = setup.predict

    def predict(model, batch, config, train_progress, **kwargs):
        if batch is batches[4]:
            trainer.commands.stop()  # stop two micro-steps into the second window
        return original_predict(model, batch, config, train_progress, **kwargs)

    setup.predict = predict
    trainer.data_loader = Mock()
    trainer.data_loader.get_data_set.return_value.approximate_length.return_value = 6
    trainer.data_loader.get_data_loader.return_value = batches

    staged = {}

    def save(*args, **kwargs):
        events.append("save")
        if trainer.model.accumulator_state is not None and not staged:
            staged.update({k: v.clone() for k, v in trainer.model.accumulator_state["param_grads"].items()})

    trainer.model_saver = Mock()
    trainer.model_saver.save.side_effect = save
    monkeypatch.setattr(generic_trainer_module, "gc", Mock())
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())

    trainer.train()
    rotations = [m.oft_R for m in trainer.model.wrapper.lora_modules.values()]
    pending = [r.rotation_grad_acc is not None for r in rotations]
    trainer.end()
    return staged, events, pending


def test_mid_window_stop_backup_holds_the_same_grads_with_and_without_cache(tmp_path, monkeypatch):
    staged_on, events_on, pending_on = _stopped_run(tmp_path / "on", monkeypatch, cached=True)
    staged_off, events_off, pending_off = _stopped_run(tmp_path / "off", monkeypatch, cached=False)

    assert all(pending_on) and not any(pending_off)  # the stop left gradients in the cache
    assert staged_on.keys() == staged_off.keys() and staged_on
    for key in staged_on:
        assert (staged_on[key] - staged_off[key]).abs().max().item() <= 1e-6, key
    # end(): materialize before the offload, then the backup (whose staging is a no-op).
    first_to = events_on.index("model.to")
    assert "before_clip" in events_on[:first_to]
    assert events_on.index("save") > first_to
    assert "before_clip" not in events_off  # without the cache none of the new calls run


# --------------------------------------------------------------------------- #
# redistill Phase A -> B best restore
# --------------------------------------------------------------------------- #


def test_redistill_restore_materializes_before_and_warms_and_refreshes_after(tmp_path):
    turbo = tmp_path / "turbo.safetensors"
    save_file({"x": torch.zeros(1)}, str(turbo))
    config = _cache_config(
        tmp_path, redistill_enabled=True, redistill_transfer_steps=100, redistill_turbo_model_name=str(turbo)
    )
    trainer = GenericTrainer(config, TrainCallbacks(), TrainCommands())
    trainer.model = SimpleNamespace(transfer_teacher_late_tensors=None)
    trainer.parameters = [nn.Parameter(torch.zeros(2))]
    seen = []
    trainer.model_setup = SimpleNamespace(
        set_redistill_phase_b_start=lambda step: None,
        before_clip_adapters=lambda model, config: seen.append(
            ("materialize", trainer.parameters[0].data.sum().item())
        ),
        refresh_adapters_after_param_restore=lambda model, config, *, warm: seen.append(
            ("refresh", warm, trainer.parameters[0].data.sum().item())
        ),
    )
    best_path = os.path.join(config.workspace_dir, "backup", "transfer-best.pt")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    torch.save([torch.full((2,), 7.0)], best_path)
    trainer._transfer_best_backup_path = best_path

    progress = TrainProgress()
    progress.global_step = 100
    trainer._GenericTrainer__maybe_advance_redistill_phase(progress)

    assert seen == [("materialize", 0.0), ("refresh", True, 14.0)]


# --------------------------------------------------------------------------- #
# sampling swaps
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("ema", "schedule_free"), [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("cached", [True, False])
def test_sampling_refreshes_the_cache_around_every_parameter_swap(monkeypatch, ema, schedule_free, cached):
    config = _cache_config(
        oft_rotation_cache=cached,
        ema=EMAMode.GPU if ema else EMAMode.OFF,
        non_ema_sampling=True,
        samples=[],
    )
    config.optimizer.optimizer = Optimizer.SCHEDULE_FREE_ADAMW if schedule_free else Optimizer.ADAMW
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    events = []
    model_ema = None
    if ema:
        model_ema = Mock()
        model_ema.copy_ema_to.side_effect = lambda *a, **k: events.append("copy_ema_to")
        model_ema.copy_temp_to.side_effect = lambda *a, **k: events.append("copy_temp_to")
    optimizer = Mock()
    optimizer.eval.side_effect = lambda: events.append("optimizer.eval")
    optimizer.train.side_effect = lambda: events.append("optimizer.train")
    trainer.model = Mock(ema=model_ema, optimizer=optimizer)
    trainer.parameters = []
    setup = trainer.model_setup = Mock()
    setup.refresh_adapters_after_param_restore.side_effect = lambda model, config, *, warm: events.append(
        f"refresh(warm={warm})"
    )
    setup.setup_train_device.side_effect = lambda *a: events.append("setup_train_device")
    trainer._GenericTrainer__sample_loop = lambda **kwargs: events.append(f"sample{kwargs.get('folder_postfix', '')}")
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)

    trainer._GenericTrainer__sample_during_training(TrainProgress(), torch.device("cpu"))

    refresh = ["refresh(warm=False)"] if cached else []
    expected = []
    if schedule_free:
        expected.append("optimizer.eval")
    if ema:
        expected.append("copy_ema_to")
    if ema or schedule_free:
        expected += refresh
    expected.append("sample")
    if ema:
        expected += ["copy_temp_to", *refresh, "sample - no-ema"]
    expected.append("setup_train_device")
    if schedule_free:
        expected += ["optimizer.train", *refresh]
    assert events == expected


def test_rotation_modules_module_import_is_shared():
    # The trainer seeds the same generator the module hook draws from.
    from modules.module import oft_utils

    assert generic_trainer_module.set_rotation_cache_seed is oft_utils.set_rotation_cache_seed
    assert issubclass(OFTRotationModule, nn.Module)
