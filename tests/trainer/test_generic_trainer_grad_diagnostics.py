"""GenericTrainer wiring of the gradient attribution diagnostics
(config log_grad_components): per-objective norms and cosines under both
add-on schedules, the per-concept probes, the split-by-concept backward, the
OFT rotation-cache materialisation per micro-step, and that the diagnostics
off leave the gradients untouched.

CPU-only. A toy "model": one parameter p in R^3, a sample's prediction is
p * x_i, so every gradient is known in closed form and the per-sample,
per-concept and per-objective pieces can be checked exactly.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import Mock

import modules.trainer.GenericTrainer as generic_trainer_module
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.RLHFAddonSchedule import RLHFAddonSchedule
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.grad_attribution_util import cosine
from modules.util.TrainProgress import TrainProgress

import torch

import pytest

BATCH = 2
X = {  # per-sample inputs, keyed by image path stem
    "A1": torch.tensor([1.0, 0.0, 0.0]),
    "A2": torch.tensor([0.0, 1.0, 0.0]),
    "B1": torch.tensor([0.0, 0.0, 1.0]),
    "B2": torch.tensor([1.0, 1.0, 0.0]),
    "P1": torch.tensor([0.5, -0.5, 0.0]),
    "P2": torch.tensor([0.0, 0.5, 0.5]),
}
T = torch.tensor([0.5, 0.5, 0.5])  # target for every sample
DPO_WEIGHT = 0.5
TEXT_DIR = torch.tensor([5.0, 5.0, -5.0])


def _concept(path: str, *, dpo: bool = False) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.path = path
    concept.name = os.path.basename(path)
    if dpo:
        concept.dpo_chosen_pattern = "chosen/{}"
        concept.dpo_rejected_pattern = "rejected/{}"
    return concept


class _Loader:
    def __init__(self, batches):
        self.batches = batches
        self.data_set = SimpleNamespace(start_next_epoch=lambda: None, approximate_length=lambda: len(batches))

    def get_data_set(self):
        return self.data_set

    def get_data_loader(self):
        return iter(self.batches)


def _batch(root, concept: str, stems: list[str]) -> dict:
    return {
        "concept_type": ["STANDARD"] * len(stems),
        "image_path": [os.path.join(root, concept, f"{s}.png") for s in stems],
        "x": torch.stack([X[s] for s in stems]),
    }


def _pair_batch(root, stems: list[str]) -> dict:
    return {
        "image_path": [os.path.join(root, "P", "chosen", f"{s}.png") for s in stems],
        "x": torch.stack([X[s] for s in stems]),
    }


def _trainer(
    tmp_path, monkeypatch, *, diagnostics: bool, interval: int, schedule, ga: int, text: bool = False, progress=None
):
    root = str(tmp_path / "data")
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.workspace_dir = str(tmp_path / "workspace")
    config.sample_definition_file_name = str(tmp_path / "missing-samples.json")
    config.concepts = [
        _concept(os.path.join(root, "A")),
        _concept(os.path.join(root, "B")),
        _concept(os.path.join(root, "P"), dpo=True),
    ]
    config.tensorboard = False
    config.validation = False
    config.epochs = 1
    config.batch_size = BATCH
    config.gradient_accumulation_steps = ga
    config.sample_after_unit = TimeUnit.NEVER
    config.backup_after_unit = TimeUnit.NEVER
    config.save_every_unit = TimeUnit.NEVER
    config.rolling_backup = False
    config.clip_grad_norm = 1e9  # keeps grad/norm logged without ever clipping
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_addon_weight = DPO_WEIGHT
    config.rlhf_addon_schedule = schedule
    config.log_grad_components = diagnostics
    config.log_grad_components_interval = interval
    config.text_reward_weight = 0.25
    config.text_reward_samples_per_epoch = 2

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    config.text_reward_enabled = text
    trainer.tensorboard = Mock()
    trainer.parameters = [parameter]
    trainer.sample_queue = []
    trainer.sft_anchor_source = None
    trainer.kto_source = None
    trainer._text_reward_bank = None

    grads: list[torch.Tensor] = []
    optimizer = Mock(param_groups=[{"params": [parameter]}])
    optimizer.step.side_effect = lambda: grads.append(parameter.grad.detach().clone())
    optimizer.zero_grad.side_effect = lambda set_to_none=True: setattr(parameter, "grad", None)
    trainer.model = Mock(
        ema=None,
        accumulator_state=None,
        parameters=None,
        train_progress=progress if progress is not None else TrainProgress(),
        optimizer=optimizer,
    )

    setup = trainer.model_setup = Mock()
    setup.pop_perceptual_metrics.return_value = {}
    state = {"losses": None}
    # fixed probes: the toy loss of the sample at the pinned timestep fraction
    setup.probe_loss.side_effect = lambda model, batch, config, tp, t: float(
        ((parameter.detach().unsqueeze(0) * batch["x"] - T).pow(2).sum(dim=1)).mean() * (1.0 + t)
    )
    setup.probe_dpo_margin.side_effect = lambda model, batch, config, tp, t: float(batch["x"].sum() * t)

    def predict(model, batch, config, train_progress):
        pred = parameter.unsqueeze(0) * batch["x"]
        return {"predicted": pred, "target": T.expand_as(pred), "timestep": torch.full((pred.shape[0],), 0.3)}

    def calculate_loss(model, batch, data, config):
        losses = (data["predicted"] - data["target"]).pow(2).sum(dim=1)
        state["losses"] = losses.detach()
        return losses.mean()

    setup.predict.side_effect = predict
    setup.calculate_loss.side_effect = calculate_loss
    setup.get_last_per_sample_losses.side_effect = lambda: state["losses"]
    setup.timestep_fraction.side_effect = lambda model, t: t.float()

    def dpo_light(model, batch, config, train_progress, backward_fn):
        # chosen half: logp = -||p x - T||^2 per pair, weight g = -1 (a sigmoid
        # objective at margin 0 has dL/dlogp_c = -beta/2; any constant works);
        # rejected half: the same inputs scaled by 2 with g = +1.
        pred_c = parameter.unsqueeze(0) * batch["x"]
        logp_c = -(pred_c - T).pow(2).sum(dim=1)
        pair_losses = torch.full((pred_c.shape[0],), 0.7)
        backward_fn(
            (-1.0 * logp_c).sum(),
            predicted=pred_c,
            half="chosen",
            image_paths=batch["image_path"],
            timestep=torch.full((pred_c.shape[0],), 0.6),
            losses=pair_losses,
        )
        pred_r = parameter.unsqueeze(0) * (2.0 * batch["x"])
        logp_r = -(pred_r - T).pow(2).sum(dim=1)
        backward_fn(
            (1.0 * logp_r).sum(),
            predicted=pred_r,
            half="rejected",
            image_paths=batch["image_path"],
            timestep=torch.full((pred_r.shape[0],), 0.6),
        )
        return torch.tensor(0.7)

    setup.calculate_dpo_loss_light.side_effect = dpo_light
    setup.get_last_dpo_metrics.return_value = {
        "loss": 0.7,
        "dpo_loss": 0.7,
        "chosen_reward": 0.2,
        "rejected_reward": 0.1,
        "reward_margin": 0.1,
        "accuracy": 1.0,
    }

    def text_reward(model, sample, config, train_progress, backward_fn):
        backward_fn((parameter * TEXT_DIR).sum())
        return torch.tensor(1.0)

    setup.calculate_text_reward_loss.side_effect = text_reward
    setup.get_last_text_reward_metrics.return_value = {"ctc": 1.0}
    from modules.util.text_reward_util import TextRewardBank, TextRewardScheduler, TextSpec

    trainer._text_reward_scheduler = TextRewardScheduler(0)
    specs = [TextSpec("Sign", f"t{i}", (f"t{i}",), "words") for i in range(2)]
    setup.build_text_reward_bank.side_effect = lambda model, config, epoch: TextRewardBank(specs, [None, None], None)

    sft_batches = [
        _batch(root, "A", ["A1", "A2"]),
        _batch(root, "A", ["A1", "B1"]),
        _batch(root, "B", ["B1", "B2"]),
        _batch(root, "B", ["B2", "A2"]),
    ]
    sft_batches[1]["image_path"][1] = os.path.join(root, "B", "B1.png")  # a mixed A/B batch
    sft_batches[3]["image_path"][1] = os.path.join(root, "A", "A2.png")
    trainer.data_loader = _Loader(sft_batches)
    dpo_loader = _Loader([_pair_batch(root, ["P1", "P2"]), _pair_batch(root, ["P2", "P1"])])
    trainer.dpo_addon_source = generic_trainer_module.CyclingBatchSource(dpo_loader, "DPO add-on")

    monkeypatch.setattr(generic_trainer_module, "enable_grad_scaling", lambda *args: False)
    monkeypatch.setattr(generic_trainer_module, "gc", Mock())
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())
    return SimpleNamespace(trainer=trainer, config=config, grads=grads, parameter=parameter, setup=setup, root=root)


def _scalars(run) -> dict[int, dict[str, float]]:
    """{global_step: {tag: value}} of everything the trainer logged."""
    out: dict[int, dict[str, float]] = {}
    for call in run.trainer.tensorboard.add_scalar.call_args_list:
        tag, value, step = call.args[:3]
        if tag.startswith(("grad/", "loss/concept/", "dpo/concept/")):
            out.setdefault(step, {})[tag] = value
    return out


def _grad_of(parameter, batches, scale: float, *, half_scale: float = 1.0) -> torch.Tensor:
    """Closed-form gradient of the toy standard loss over ``batches``."""
    total = torch.zeros(3)
    for batch in batches:
        pred = parameter.detach().unsqueeze(0) * (half_scale * batch["x"])
        total += (2 * (pred - T) * (half_scale * batch["x"])).sum(dim=0) / batch["x"].shape[0] * scale
    return total


# --------------------------------------------------------------------------- #
# off: nothing changes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("schedule", list(RLHFAddonSchedule))
def test_diagnostics_off_leave_gradients_and_logging_untouched(tmp_path, monkeypatch, schedule):
    off = _trainer(tmp_path / "off", monkeypatch, diagnostics=False, interval=50, schedule=schedule, ga=2)
    off.trainer.train()
    on = _trainer(tmp_path / "on", monkeypatch, diagnostics=True, interval=1, schedule=schedule, ga=2)
    on.trainer.train()

    assert off.trainer._grad_attribution is None
    off.setup.set_grad_probe.assert_not_called()
    assert not any(
        tag.startswith(("grad/norm_", "grad/cos", "grad/concept", "loss/concept"))
        for s in _scalars(off).values()
        for tag in s
    )
    assert len(off.grads) == len(on.grads) > 0
    for a, b in zip(off.grads, on.grads, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------- #
# per-objective norms and cosines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("schedule", list(RLHFAddonSchedule))
def test_per_kind_norms_and_cosines_match_the_known_terms(tmp_path, monkeypatch, schedule):
    run = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=0, schedule=schedule, ga=2, text=True)
    run.trainer.train()
    run.setup.set_grad_probe.assert_called_once_with(True)
    logged = _scalars(run)
    assert logged, "no update step logged"
    for step, scalars in logged.items():
        assert "grad/norm_total" in scalars and "grad/norm" in scalars, step
        # the sum of the attributed pieces is the gradient clip_grad_norm_ saw
        assert scalars["grad/norm_total"] == pytest.approx(scalars["grad/norm"], rel=1e-5)
        kinds = [k.removeprefix("grad/norm_") for k in scalars if k.startswith("grad/norm_") and k != "grad/norm_total"]
        for a_i, a in enumerate(kinds):
            for b in kinds[a_i + 1 :]:
                assert -1.0 - 1e-6 <= scalars[f"grad/cos_{a}_{b}"] <= 1.0 + 1e-6
    all_tags = {tag for scalars in logged.values() for tag in scalars}
    assert {"grad/norm_sft", "grad/norm_dpo", "grad/norm_text", "grad/cos_sft_dpo"} <= all_tags
    if schedule == RLHFAddonSchedule.PIGGYBACK:
        # interval 1 (4 standard vs 2 pairs -> every 2nd) and the text reward
        # blocked on pair steps: a window holds both anyway
        assert "grad/cos_sft_text" in all_tags or "grad/cos_dpo_text" in all_tags
    # the text term is a fixed direction: its norm is exactly ||TEXT_DIR|| * weight / GA per rollout
    text_norms = [s["grad/norm_text"] for s in logged.values() if "grad/norm_text" in s]
    assert text_norms and all(n == pytest.approx(TEXT_DIR.norm().item() * 0.25 / 2, rel=1e-5) for n in text_norms)


def test_independent_window_attribution_is_exact(tmp_path, monkeypatch):
    # INDEPENDENT, GA 2, schedule [S D S S D S]: window 0 = (S0, D0), window 1 = (S1, S2), window 2 = (D1, S3)
    run = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=0, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2)
    run.trainer.train()
    logged = _scalars(run)
    steps = sorted(logged)
    assert len(steps) == 3
    p = run.parameter  # the optimizer is a mock: the parameter never changes
    sft = run.trainer.data_loader.batches
    pairs = run.trainer.dpo_addon_source.data_loader.batches
    # chosen half: g = -1 on logp = -||..||^2 -> +dL/dp of the squared error; rejected: g = +1, inputs x2

    def dpo_vec(pair):
        count = pair["x"].shape[0]
        return (_grad_of(p, [pair], 1.0) - _grad_of(p, [pair], 1.0, half_scale=2.0)) * count * DPO_WEIGHT / 2

    expected = [
        {"sft": _grad_of(p, [sft[0]], 0.5), "dpo": dpo_vec(pairs[0])},
        {"sft": _grad_of(p, [sft[1], sft[2]], 0.5)},
        {"sft": _grad_of(p, [sft[3]], 0.5), "dpo": dpo_vec(pairs[1])},
    ]
    for step, want in zip(steps, expected, strict=True):
        scalars = logged[step]
        for kind, vector in want.items():
            assert scalars[f"grad/norm_{kind}"] == pytest.approx(vector.norm().item(), rel=1e-5), (step, kind)
        if "dpo" in want:
            assert scalars["grad/cos_sft_dpo"] == pytest.approx(cosine([want["sft"]], [want["dpo"]]), abs=1e-5)
            assert scalars["grad/norm_total"] == pytest.approx((want["sft"] + want["dpo"]).norm().item(), rel=1e-5)
        else:
            assert "grad/cos_sft_dpo" not in scalars
            assert scalars["grad/norm_total"] == pytest.approx(want["sft"].norm().item(), rel=1e-5)


def test_piggyback_and_independent_attribute_the_same_terms(tmp_path, monkeypatch):
    piggy = _trainer(
        tmp_path / "p", monkeypatch, diagnostics=True, interval=0, schedule=RLHFAddonSchedule.PIGGYBACK, ga=1
    )
    piggy.trainer.train()
    indep = _trainer(
        tmp_path / "i", monkeypatch, diagnostics=True, interval=0, schedule=RLHFAddonSchedule.INDEPENDENT, ga=1
    )
    indep.trainer.train()
    # GA 1: every PIGGYBACK pair step is (standard + pair) in one update; under
    # INDEPENDENT the same pair is its own update. The attributed DPO norms
    # are the same numbers either way, and so are the SFT ones.
    p_dpo = sorted(s["grad/norm_dpo"] for s in _scalars(piggy).values() if "grad/norm_dpo" in s)
    i_dpo = sorted(s["grad/norm_dpo"] for s in _scalars(indep).values() if "grad/norm_dpo" in s)
    assert p_dpo == pytest.approx(i_dpo)
    p_sft = sorted(s["grad/norm_sft"] for s in _scalars(piggy).values() if "grad/norm_sft" in s)
    i_sft = sorted(s["grad/norm_sft"] for s in _scalars(indep).values() if "grad/norm_sft" in s)
    assert p_sft == pytest.approx(i_sft)
    # under PIGGYBACK the pair rides on a standard step: its window has both pieces
    assert any("grad/cos_sft_dpo" in s for s in _scalars(piggy).values())
    assert not any("grad/cos_sft_dpo" in s for s in _scalars(indep).values())


# --------------------------------------------------------------------------- #
# per-concept: cheap stats every update, conflict probe every N
# --------------------------------------------------------------------------- #


def test_per_concept_stats_are_logged_every_update(tmp_path, monkeypatch):
    run = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=0, schedule=RLHFAddonSchedule.INDEPENDENT, ga=1)
    run.trainer.train()
    logged = _scalars(run)
    all_tags = {tag for s in logged.values() for tag in s}
    for tag in ("A", "B"):
        assert {
            f"loss/concept/{tag}/tnorm",
            f"grad/concept/{tag}/out_norm",
            f"grad/concept/{tag}/out_norm_tnorm",
        } <= all_tags
    assert {
        "grad/concept/P/out_norm_chosen",
        "grad/concept/P/out_norm_rejected",
        "grad/concept/P/out_norm",
        "loss/concept/P/tnorm",
    } <= all_tags
    assert not any(tag.endswith(("/norm", "/cos_rest")) for tag in all_tags if tag.startswith("grad/concept/"))
    # the first standard update (batch A1, A2): raw out-norm is ||2 (p x - T)|| / B per sample, mean over the two
    first = logged[min(logged)]
    p = run.parameter.detach()
    norms = [(2 * (p * X[s] - T) / BATCH).norm().item() for s in ("A1", "A2")]
    assert first["grad/concept/A/out_norm"] == pytest.approx(sum(norms) / 2, rel=1e-5)
    # the first time a timestep bin is seen the normalised value is relative to that batch's own mean
    assert first["grad/concept/A/out_norm_tnorm"] == pytest.approx(1.0, rel=1e-5)
    assert first["loss/concept/A/tnorm"] == pytest.approx(1.0, rel=1e-5)


def test_split_by_concept_backward_is_equivalent_and_reports_norms_and_rest_cosines(tmp_path, monkeypatch):
    plain = _trainer(
        tmp_path / "plain", monkeypatch, diagnostics=True, interval=0, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2
    )
    plain.trainer.train()
    split = _trainer(
        tmp_path / "split", monkeypatch, diagnostics=True, interval=1, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2
    )
    split.trainer.train()
    assert len(plain.grads) == len(split.grads) == 3
    for a, b in zip(plain.grads, split.grads, strict=True):
        assert torch.allclose(a, b, atol=1e-6)

    logged = _scalars(split)
    steps = sorted(logged)
    # window 1 = standard batches [A1, B1] and [B1, B2]: concept A gets A1's
    # share, concept B gets B1 + B1 + B2's; together they are the window's gradient
    p = split.parameter
    scalars = logged[steps[1]]
    a_vec = (2 * (p * X["A1"] - T) * X["A1"]) / BATCH / 2
    b_vec = sum(((2 * (p * X[s] - T) * X[s]) / BATCH / 2 for s in ("B1", "B1", "B2")), torch.zeros(3))
    assert scalars["grad/concept/A/norm"] == pytest.approx(a_vec.norm().item(), rel=1e-5)
    assert scalars["grad/concept/B/norm"] == pytest.approx(b_vec.norm().item(), rel=1e-5)
    assert scalars["grad/concept/A/cos_rest"] == pytest.approx(cosine([a_vec], [b_vec]), abs=1e-5)
    assert scalars["grad/concept/B/cos_rest"] == pytest.approx(cosine([b_vec], [a_vec]), abs=1e-5)
    assert (a_vec + b_vec).norm().item() == pytest.approx(scalars["grad/norm_sft"], rel=1e-5)
    # window 0 = (standard [A1, A2], pair batch): one standard concept and the DPO concept
    scalars = logged[steps[0]]
    assert {"grad/concept/A/norm", "grad/concept/P/norm", "grad/concept/A/cos_rest", "grad/concept/P/cos_rest"} <= set(
        scalars
    )
    assert scalars["grad/concept/A/norm"] == pytest.approx(scalars["grad/norm_sft"], rel=1e-5)
    assert scalars["grad/concept/P/norm"] == pytest.approx(scalars["grad/norm_dpo"], rel=1e-5)
    assert scalars["grad/concept/A/cos_rest"] == pytest.approx(scalars["grad/cos_sft_dpo"], abs=1e-5)


def test_conflict_probe_runs_only_every_n_updates(tmp_path, monkeypatch):
    run = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=2, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2)
    run.trainer.train()
    logged = _scalars(run)
    probed = [step for step, s in logged.items() if any(t.endswith("/cos_rest") for t in s)]
    unprobed = [step for step, s in logged.items() if not any(t.endswith("/cos_rest") for t in s)]
    assert probed and unprobed
    # updates 0 and 2 are probe windows, update 1 is not; an update is logged
    # at the global step of its last micro-step (1, 3, 5 at GA 2)
    assert sorted(probed) == [1, 5] and unprobed == [3]


# --------------------------------------------------------------------------- #
# OFT rotation cache: materialised per micro-step, same gradient
# --------------------------------------------------------------------------- #


def _cached_run(tmp_path, monkeypatch, *, diagnostics: bool):
    """A real OFT v2 adapter with the rotation cache on (the rotation-cache
    test's tiny model and real BaseModelSetup), GA 3 over 6 batches."""
    from test_generic_trainer_rotation_cache import HIDDEN, _cache_config, _RealSetup, _tiny_model

    config = _cache_config(
        tmp_path,
        oft_rotation_cache=True,
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
        oft_block_size=8,
        dora_oft=True,
        oft_scaled=True,
        oft_clipped_norm=0.95,
        dropout_probability=0.0,
        clip_grad_norm=1e9,
        log_grad_components=diagnostics,
        log_grad_components_interval=1,
    )
    events: list[str] = []
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    trainer.tensorboard = Mock()
    trainer.model, params = _tiny_model(config, events, cached=True)
    trainer.parameters = params
    trainer.sample_queue = []
    trainer.sft_anchor_source = None
    trainer.dpo_addon_source = None
    trainer.kto_source = None
    trainer.model_setup = _RealSetup(events)

    gen = torch.Generator().manual_seed(1)
    batches = [
        {
            "concept_type": ["STANDARD"],
            "x": torch.randn(2, HIDDEN, generator=gen),
            "upstream": torch.randn(2, HIDDEN, generator=gen),
        }
        for _ in range(6)
    ]
    trainer.data_loader = Mock()
    trainer.data_loader.get_data_set.return_value.approximate_length.return_value = 6
    trainer.data_loader.get_data_loader.return_value = batches

    grads: list[list[torch.Tensor]] = []
    pending: list[bool] = []
    rotations = [m.oft_R for m in trainer.model.wrapper.lora_modules.values()]
    original_step = trainer.model.optimizer.step

    def step(*args, **kwargs):
        grads.append([p.grad.detach().clone() for p in params])
        pending.append(any(r.rotation_grad_acc is not None for r in rotations))
        return original_step(*args, **kwargs)

    trainer.model.optimizer.step = step
    trainer.model_saver = Mock()
    monkeypatch.setattr(generic_trainer_module, "gc", Mock())
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())
    trainer.train()
    return SimpleNamespace(trainer=trainer, grads=grads, pending=pending, events=events)


def test_rotation_cache_materialised_per_micro_step_gives_the_same_gradient(tmp_path, monkeypatch):
    off = _cached_run(tmp_path / "off", monkeypatch, diagnostics=False)
    on = _cached_run(tmp_path / "on", monkeypatch, diagnostics=True)
    assert len(off.grads) == len(on.grads) == 2
    for a, b in zip(off.grads, on.grads, strict=True):
        for x, y in zip(a, b, strict=True):
            assert (x - y).abs().max().item() <= 1e-6
    # nothing left in the cache at either update, and the per-micro-step
    # materialisation ran (before_clip once per micro-step plus the update's own)
    assert not any(on.pending) and not any(off.pending)
    assert on.events.count("before_clip") > off.events.count("before_clip") == 2
    logged = _scalars(on)
    for step, scalars in logged.items():
        assert scalars["grad/norm_total"] == pytest.approx(scalars["grad/norm"], rel=1e-5), step
        assert scalars["grad/norm_sft"] == pytest.approx(scalars["grad/norm"], rel=1e-5), step


def test_validation_rejects_fused_back_pass_and_negative_interval():
    config = TrainConfig.default_values()
    config.log_grad_components = True
    assert config.validate_grad_diagnostics_startup() is None
    config.log_grad_components_interval = -1
    assert "interval" in config.validate_grad_diagnostics_startup()
    config.log_grad_components_interval = 0
    config.optimizer.fused_back_pass = True
    assert "fused back pass" in config.validate_grad_diagnostics_startup()
    config.log_grad_components = False
    assert config.validate_grad_diagnostics_startup() is None
    loaded = TrainConfig.default_values().from_dict(
        {k: v for k, v in config.to_dict().items() if not k.startswith("log_grad")}
    )
    assert loaded.log_grad_components is False and loaded.log_grad_components_interval == 50


# --------------------------------------------------------------------------- #
# per-image visit ledger and fixed probes
# --------------------------------------------------------------------------- #


def test_visit_ledger_logs_paired_deltas_for_revisited_images(tmp_path, monkeypatch):
    # standard batches [A1, A2], [A1, B1], [B1, B2], [B2, A2]: A1, A2, B1 and B2 are each seen twice
    run = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=0, schedule=RLHFAddonSchedule.INDEPENDENT, ga=1)
    run.trainer.train()
    ledger = run.trainer._visit_ledger
    assert ledger is not None
    assert len(ledger.visits(os.path.join(run.root, "A", "A1.png"))) == 2
    assert len(ledger.visits(os.path.join(run.root, "P", "chosen", "P1.png"))) == 2  # pairs keyed by chosen path
    logged = _scalars(run)
    last = logged[max(logged)]
    for tag, revisited in (("A", 2.0), ("B", 2.0), ("P", 2.0)):
        assert last[f"loss/concept/{tag}/revisited"] == revisited
        for quantity in ("visit_delta", "since_first", "frac_improving"):
            assert f"loss/concept/{tag}/{quantity}" in last
    # the parameter never changes (mock optimizer) and the normaliser's bin
    # means move, so deltas are finite numbers in a sane range
    assert -1.0 <= last["loss/concept/A/visit_delta"] <= 1.0
    assert 0.0 <= last["loss/concept/A/frac_improving"] <= 1.0


def test_fixed_probes_are_captured_from_the_stream_and_run_on_the_probe_cadence(tmp_path, monkeypatch):
    run = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=2, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2)
    run.trainer.train()
    probes = run.trainer._probe_set
    assert probes.count("A") == 2 and probes.count("B") == 2 and probes.count("P") == 2  # every distinct image
    assert probes.kind("P") == "dpo" and probes.kind("A") == "sft"
    sample = dict(probes.samples("A"))[os.path.join(run.root, "A", "A1.png")]
    assert sample["x"].shape == (1, 3) and sample["image_path"] == [os.path.join(run.root, "A", "A1.png")]
    logged = _scalars(run)
    probed = sorted(step for step, s in logged.items() if "loss/concept/A/probe" in s)
    assert probed == [1, 5]  # updates 0 and 2 at GA 2 (the conflict-probe cadence)
    first, second = logged[1], logged[5]
    assert first["loss/concept/A/probe_n"] == 2.0 and "loss/concept/A/probe_delta" not in first
    assert second["loss/concept/A/probe_delta"] == pytest.approx(
        second["loss/concept/A/probe"] - first["loss/concept/A/probe"]
    )
    assert "dpo/concept/P/probe_margin" in first and "dpo/concept/P/probe_margin_delta" in second
    # the probe value is the mean over the set's images and the three fixed timesteps
    p = run.parameter.detach()
    expected = sum(((p * X[s] - T).pow(2).sum() * (1.0 + t)).item() for s in ("A1", "A2") for t in (0.2, 0.5, 0.8)) / 6
    assert first["loss/concept/A/probe"] == pytest.approx(expected, rel=1e-5)
    assert os.path.isfile(os.path.join(run.trainer._diagnostics_dir, "probes.pt"))
    assert os.path.isfile(os.path.join(run.trainer._diagnostics_dir, "visit_ledger.json"))


def test_diagnostics_state_is_restored_on_resume_and_dropped_on_a_fresh_run(tmp_path, monkeypatch):
    first = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=1, schedule=RLHFAddonSchedule.INDEPENDENT, ga=1)
    first.trainer.train()
    images = len(first.trainer._visit_ledger)
    assert images > 0 and first.trainer._probe_set.total() == 6
    first.trainer._GenericTrainer__save_diagnostics_state()  # what a backup or end() does

    resumed = _trainer(
        tmp_path,
        monkeypatch,
        diagnostics=True,
        interval=1,
        schedule=RLHFAddonSchedule.INDEPENDENT,
        ga=1,
        progress=TrainProgress(epoch=0, epoch_step=0, epoch_sample=0, global_step=7),
    )
    resumed.trainer._GenericTrainer__setup_grad_diagnostics()
    assert len(resumed.trainer._visit_ledger) == images
    assert resumed.trainer._probe_set.total() == 6

    fresh = _trainer(tmp_path, monkeypatch, diagnostics=True, interval=1, schedule=RLHFAddonSchedule.INDEPENDENT, ga=1)
    fresh.trainer._GenericTrainer__setup_grad_diagnostics()
    assert len(fresh.trainer._visit_ledger) == 0 and fresh.trainer._probe_set.total() == 0
    assert not os.path.isfile(os.path.join(fresh.trainer._diagnostics_dir, "probes.pt"))
