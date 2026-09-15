"""calculate_dpo_loss_light with the rejected half's graph held across the
chosen half (rlhf_dpo_light_reuse_rejected_graph): 4 forwards instead of 5 and
the same gradient, loss and metrics as the one-graph-at-a-time sequence.

Drives the real method on CPU with a tiny deterministic two-layer net standing
in for the transformer. The fake predict wraps every grad-enabled output in an
identity whose graph node saves a marker tensor, so a weakref to the marker
tells whether that forward's graph is still alive."""

import weakref
from contextlib import contextmanager

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_light_util import dpo_objective
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TrainingMethod import TrainingMethod

import torch

import pytest

BATCH, DIM = 3, 5
ADDON_SCALE = 0.5  # stands in for rlhf_dpo_addon_weight / gradient_accumulation_steps
TE_DROPOUT = (0.25, 0.5, 0.125, 0.75)

HELD_SEQUENCE = [
    ("chosen", False, True),
    ("rejected", False, True),
    ("rejected", True, False),
    ("chosen", True, False),
]
FALLBACK_SEQUENCE = [
    ("chosen", False, True),
    ("rejected", False, True),
    ("rejected", False, False),
    ("chosen", True, False),
    ("rejected", True, False),
]


class _SaveMarker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, marker):
        ctx.save_for_backward(marker)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(DIM, 7)
        self.fc2 = torch.nn.Linear(7, DIM)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


class _Model:
    def __init__(self):
        torch.manual_seed(0)
        self.net = _Net()
        torch.manual_seed(1)
        self.reference_net = _Net().requires_grad_(False)


def _te_configs(config: TrainConfig):
    return (config.text_encoder, config.text_encoder_2, config.text_encoder_3, config.text_encoder_4)


class _FakeSetup(BaseModelSetup):
    def __init__(self, fail_on=None):
        super().__init__(torch.device("cpu"), torch.device("cpu"), False)
        self.fail_on = fail_on
        self.calls = []  # (half, grad enabled, inside reference_model)
        self.markers = []  # (half, weakref to a tensor saved by that forward's graph)
        self.dropout_seen = []
        self.reference_depth = 0

    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_optimizations(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError

    @contextmanager
    def reference_model(self, model, config):
        self.reference_depth += 1
        try:
            yield
        finally:
            self.reference_depth -= 1

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        half = batch["half"]
        grad = torch.is_grad_enabled()
        self.calls.append((half, grad, self.reference_depth > 0))
        self.dropout_seen.append(tuple(te.dropout_probability for te in _te_configs(config)))
        if self.fail_on == (half, grad):
            raise RuntimeError(f"forward failed on the {half} half")
        net = model.reference_net if self.reference_depth > 0 else model.net
        predicted = net(batch["latent_image"])
        if grad:
            marker = torch.zeros(1)
            predicted = _SaveMarker.apply(predicted, marker)
            self.markers.append((half, weakref.ref(marker)))
            del marker
        return {
            "predicted": predicted,
            "target": batch["target"],
            "timestep": batch["timestep"],
            "predicted_latent": predicted * 2.0,
        }

    def calculate_loss(self, model, batch, data, config):
        return (data["predicted"] - data["target"]).pow(2).mean() + 0.1 * data["predicted"].abs().mean()

    def rejected_graphs_alive(self) -> list[bool]:
        return [ref() is not None for half, ref in self.markers if half == "rejected"]


def _batch(beta_override=None) -> dict:
    generator = torch.Generator().manual_seed(7)
    batch = {
        "half": "chosen",
        "half_rejected": "rejected",
        "latent_image": torch.randn(BATCH, DIM, generator=generator),
        "latent_image_rejected": torch.randn(BATCH, DIM, generator=generator),
        "target": torch.randn(BATCH, DIM, generator=generator),
        "target_rejected": torch.randn(BATCH, DIM, generator=generator),
        "timestep": torch.tensor([0.1, 0.6, 0.9]),
    }
    if beta_override is not None:
        batch["dpo_beta_override"] = beta_override
    return batch


def _config(*, reuse=True, objective=DPOObjective.SIGMOID, label_smoothing=0.0, mix=0.0) -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_beta = 2.0
    config.rlhf_dpo_objective = objective
    config.rlhf_dpo_ipo_tau = 3.0
    config.rlhf_dpo_label_smoothing = label_smoothing
    config.rlhf_supervised_mix = mix
    config.rlhf_dpo_timestep_margin_logging = True
    config.rlhf_dpo_light_reuse_rejected_graph = reuse
    config.gradient_checkpointing = GradientCheckpointingMethod.ON
    config.enable_activation_offloading = True
    config.layer_offload_fraction = 0.0
    for te, dropout in zip(_te_configs(config), TE_DROPOUT, strict=True):
        te.dropout_probability = dropout
    return config


def _run(config: TrainConfig, batch: dict, fail_on=None):
    setup = _FakeSetup(fail_on=fail_on)
    model = _Model()
    liveness_after_backward = []

    def backward_fn(term):
        (term * ADDON_SCALE).backward()
        liveness_after_backward.append(setup.rejected_graphs_alive())

    loss = setup.calculate_dpo_loss_light(model, batch, config, None, backward_fn)
    return setup, model, loss, liveness_after_backward


def _grads(model: _Model) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.net.parameters()]


def _fused_reference(config: TrainConfig, batch: dict):
    """The same objective differentiated in one piece: both halves' log-probs
    live, the SFT mix on the chosen half, one backward."""
    model = _Model()

    def logp(net, x, target):
        return -(net(x) - target).pow(2).mean(dim=1, dtype=torch.float32)

    with torch.no_grad():
        ref_c = logp(model.reference_net, batch["latent_image"], batch["target"])
        ref_r = logp(model.reference_net, batch["latent_image_rejected"], batch["target_rejected"])
    setup = _FakeSetup()
    beta = setup._resolve_dpo_beta(config, batch)
    pred_c = model.net(batch["latent_image"])
    pol_c = -(pred_c - batch["target"]).pow(2).mean(dim=1, dtype=torch.float32)
    pol_r = logp(model.net, batch["latent_image_rejected"], batch["target_rejected"])
    loss, *_ = dpo_objective(pol_c, pol_r, ref_c, ref_r, **setup._dpo_objective_kwargs(config, beta))
    if config.rlhf_supervised_mix > 0:
        loss = loss + config.rlhf_supervised_mix * setup.calculate_loss(
            model, batch, {"predicted": pred_c, "target": batch["target"]}, config
        )
    (loss * ADDON_SCALE).backward()
    return loss.detach(), _grads(model)


OBJECTIVE_CASES = {
    "sigmoid": ({}, None),
    "sigmoid_label_smoothing": ({"label_smoothing": 0.2}, None),
    "ipo": ({"objective": DPOObjective.IPO}, None),
    "per_sample_beta": ({}, torch.tensor([0.0, 1.5, 4.0])),
}


@pytest.mark.parametrize("mix", [0.0, 0.3], ids=["no_mix", "supervised_mix"])
@pytest.mark.parametrize("case", list(OBJECTIVE_CASES))
def test_held_rejected_graph_matches_the_five_forward_sequence(case, mix):
    config_kwargs, beta_override = OBJECTIVE_CASES[case]

    held_setup, held_model, held_loss, _ = _run(_config(reuse=True, mix=mix, **config_kwargs), _batch(beta_override))
    base_setup, base_model, base_loss, _ = _run(_config(reuse=False, mix=mix, **config_kwargs), _batch(beta_override))

    assert held_setup.calls == HELD_SEQUENCE
    assert base_setup.calls == FALLBACK_SEQUENCE

    for held_grad, base_grad in zip(_grads(held_model), _grads(base_model), strict=True):
        assert torch.count_nonzero(held_grad) > 0
        assert torch.allclose(held_grad, base_grad, rtol=0.0, atol=1e-6)
    assert torch.allclose(held_loss, base_loss, rtol=0.0, atol=1e-7)
    assert not held_loss.requires_grad

    assert held_setup.get_last_dpo_metrics().keys() == base_setup.get_last_dpo_metrics().keys()
    for key, value in held_setup.get_last_dpo_metrics().items():
        assert value == pytest.approx(base_setup.get_last_dpo_metrics()[key], abs=1e-6), key
    if beta_override is not None:
        assert "beta_mean" in held_setup.get_last_dpo_metrics()

    # Both sequences are the exact gradient of the objective, not merely equal
    # to each other.
    fused_loss, fused_grads = _fused_reference(_config(mix=mix, **config_kwargs), _batch(beta_override))
    assert torch.allclose(held_loss, fused_loss, rtol=0.0, atol=1e-6)
    for held_grad, fused_grad in zip(_grads(held_model), fused_grads, strict=True):
        assert torch.allclose(held_grad, fused_grad, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("reuse", [True, False])
def test_text_encoder_dropout_is_zero_for_every_forward_and_restored(reuse):
    config = _config(reuse=reuse)
    setup, *_ = _run(config, _batch())
    assert setup.dropout_seen == [(0.0, 0.0, 0.0, 0.0)] * len(setup.calls)
    assert tuple(te.dropout_probability for te in _te_configs(config)) == TE_DROPOUT


@pytest.mark.parametrize(
    "checkpointing, activation_offloading, layer_offload_fraction",
    [
        (GradientCheckpointingMethod.OFF, False, 0.0),
        (GradientCheckpointingMethod.CPU_OFFLOADED, True, 0.0),
        (GradientCheckpointingMethod.CPU_OFFLOADED, False, 0.5),
    ],
    ids=["checkpointing_off", "activation_offloading", "layer_offloading"],
)
def test_unsafe_memory_setups_fall_back_to_five_forwards(checkpointing, activation_offloading, layer_offload_fraction):
    config = _config(reuse=True)
    config.gradient_checkpointing = checkpointing
    config.enable_activation_offloading = activation_offloading
    config.layer_offload_fraction = layer_offload_fraction

    setup, *_ = _run(config, _batch())

    assert setup.calls == FALLBACK_SEQUENCE


def test_cpu_offloaded_checkpointing_without_offloading_holds_the_graph():
    # CPU_OFFLOADED with neither activation nor layer offloading builds plain
    # non-reentrant checkpoints (conductor.offload_activated() is False).
    config = _config(reuse=True)
    config.gradient_checkpointing = GradientCheckpointingMethod.CPU_OFFLOADED
    config.enable_activation_offloading = False
    config.layer_offload_fraction = 0.0

    setup, *_ = _run(config, _batch())

    assert setup.calls == HELD_SEQUENCE


def test_held_rejected_graph_lives_through_the_chosen_backward_and_is_released_by_its_own():
    setup, _, _, liveness = _run(_config(reuse=True, mix=0.3), _batch())

    # After the chosen backward the rejected graph (built first) is still held;
    # the rejected backward frees it.
    assert liveness == [[True], [False]]
    assert all(ref() is None for _, ref in setup.markers)


def test_fallback_never_holds_a_rejected_graph_during_the_chosen_backward():
    setup, _, _, liveness = _run(_config(reuse=False), _batch())

    assert liveness == [[], [False]]
    assert all(ref() is None for _, ref in setup.markers)


@pytest.mark.parametrize("reuse", [True, False])
def test_exception_in_the_chosen_forward_restores_state_and_drops_the_held_graph(reuse):
    config = _config(reuse=reuse)
    setup = _FakeSetup(fail_on=("chosen", True))
    model = _Model()
    backward_calls = []

    with pytest.raises(RuntimeError, match="chosen half") as excinfo:
        setup.calculate_dpo_loss_light(model, _batch(), config, None, backward_calls.append)

    # excinfo's traceback still references the method's frame here.
    assert excinfo.traceback is not None
    assert setup.rejected_graphs_alive() == ([False] if reuse else [])
    assert tuple(te.dropout_probability for te in _te_configs(config)) == TE_DROPOUT
    assert setup.reference_depth == 0
    assert backward_calls == []
    assert all(p.grad is None for p in model.net.parameters())


def test_flag_defaults_on_and_survives_round_trip_and_old_configs():
    config = TrainConfig.default_values()
    assert config.rlhf_dpo_light_reuse_rejected_graph is True

    config.rlhf_dpo_light_reuse_rejected_graph = False
    loaded = TrainConfig.default_values().from_dict(config.to_dict())
    assert loaded.rlhf_dpo_light_reuse_rejected_graph is False

    # A config saved before the field existed loads the default.
    old = config.to_dict()
    del old["rlhf_dpo_light_reuse_rejected_graph"]
    assert TrainConfig.default_values().from_dict(old).rlhf_dpo_light_reuse_rejected_graph is True
    assert TrainConfig.default_values().from_dict({"__version": 26}).rlhf_dpo_light_reuse_rejected_graph is True
