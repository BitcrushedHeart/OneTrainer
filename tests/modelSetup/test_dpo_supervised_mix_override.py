"""Per-concept DPO supervised mix (ConceptConfig.supervised_mix_override).

The resolver turns the per-sample column into either the global scalar (no
column, or every sample resolving to one value) or a per-sample tensor. A
per-sample mix is folded into the batch's loss_weight, so the supervised term
is mean_i(mix_i * loss_i) through the real (masked) loss code. Drives the real
methods on CPU with a per-pixel net and the real flow-matching losses."""

from contextlib import contextmanager

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TrainingMethod import TrainingMethod

import torch

import pytest

BATCH, CHANNELS, H, W = 3, 3, 4, 6


def _config(mix: float = 0.25, *, masked: bool = False, unmasked_weight: float = 0.1) -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_beta = 2.0
    config.rlhf_dpo_objective = DPOObjective.SIGMOID
    config.rlhf_supervised_mix = mix
    config.rlhf_dpo_light_reuse_rejected_graph = True
    config.gradient_checkpointing = GradientCheckpointingMethod.ON
    config.masked_training = masked
    config.unmasked_weight = unmasked_weight
    for te in (config.text_encoder, config.text_encoder_2, config.text_encoder_3, config.text_encoder_4):
        te.dropout_probability = 0.0
    return config


def _resolve(config: TrainConfig, batch: dict):
    return BaseModelSetup._resolve_supervised_mix(config, batch)


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


def test_batch_without_the_column_returns_the_global_value_untouched():
    for mix in (0.25, 0.0, -1.0):
        resolved = _resolve(_config(mix), {"latent_image": torch.zeros(2, 4)})
        assert isinstance(resolved, float)
        assert resolved == mix


def test_all_samples_inheriting_collapse_to_the_global_scalar():
    resolved = _resolve(_config(0.25), {"supervised_mix_override": torch.tensor([-1.0, -1.0, -5.0])})
    assert isinstance(resolved, float)
    assert resolved == 0.25


def test_negative_inherits_zero_switches_off_and_positive_wins():
    resolved = _resolve(_config(0.25), {"supervised_mix_override": torch.tensor([-1.0, 0.0, 3.0])})
    assert isinstance(resolved, torch.Tensor)
    assert resolved.dtype == torch.float32
    assert torch.equal(resolved, torch.tensor([0.25, 0.0, 3.0]))


def test_one_override_shared_by_the_whole_batch_is_a_scalar():
    resolved = _resolve(_config(0.25), {"supervised_mix_override": torch.tensor([3.0, 3.0])})
    assert isinstance(resolved, float)
    assert resolved == 3.0
    assert not BaseModelSetup._supervised_mix_active(
        _resolve(_config(0.25), {"supervised_mix_override": torch.zeros(2)})
    )


def test_global_off_still_lets_a_concept_switch_the_term_on():
    config = _config(0.0)
    assert not BaseModelSetup._supervised_mix_active(_resolve(config, {"supervised_mix_override": -torch.ones(2)}))
    resolved = _resolve(config, {"supervised_mix_override": torch.tensor([-1.0, 2.0])})
    assert BaseModelSetup._supervised_mix_active(resolved)
    assert torch.equal(resolved, torch.tensor([0.0, 2.0]))


def test_a_negative_global_mix_inherits_as_off():
    resolved = _resolve(_config(-0.5), {"supervised_mix_override": torch.tensor([-1.0, 1.0])})
    assert torch.equal(resolved, torch.tensor([0.0, 1.0]))


def test_integer_override_column_is_promoted_to_float():
    resolved = _resolve(_config(0.25), {"supervised_mix_override": torch.tensor([-1, 2])})
    assert resolved.dtype == torch.float32
    assert torch.equal(resolved, torch.tensor([0.25, 2.0]))


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(CHANNELS, 5, 1)
        self.c2 = torch.nn.Conv2d(5, CHANNELS, 1)

    def forward(self, x):
        return self.c2(torch.tanh(self.c1(x)))


class _Model:
    def __init__(self):
        torch.manual_seed(0)
        self.net = _Net()
        torch.manual_seed(1)
        self.reference_net = _Net().requires_grad_(False)


class _FakeSetup(BaseModelSetup, ModelSetupDiffusionLossMixin):
    """calculate_loss is the real flow-matching loss (per-sample loss_weight,
    masked or not, then the batch mean), like every concrete setup."""

    def __init__(self):
        super().__init__(torch.device("cpu"), torch.device("cpu"), False)
        self.reference_depth = 0
        self.loss_calls = 0

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
        net = model.reference_net if self.reference_depth > 0 else model.net
        return {
            "loss_type": "target",
            "predicted": net(batch["latent_image"]),
            "target": batch["target"],
        }

    def calculate_loss(self, model, batch, data, config):
        self.loss_calls += 1
        return self._flow_matching_losses(batch, data, config, torch.device("cpu")).mean()


def _mask() -> torch.Tensor:
    mask = torch.zeros(BATCH, 1, H, W)
    mask[..., : W // 2] = 1.0
    return mask


def _batch(mix_override=None, *, masked: bool = False, seed: int = 7) -> dict:
    generator = torch.Generator().manual_seed(seed)
    batch = {
        "latent_image": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "latent_image_rejected": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "target": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "target_rejected": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "loss_weight": torch.tensor([1.0, 0.5, 2.0]),
    }
    if masked:
        batch["latent_mask"] = _mask()
    if mix_override is not None:
        batch["supervised_mix_override"] = mix_override
    return batch


def _per_sample_losses(setup: _FakeSetup, batch: dict, data: dict, config: TrainConfig) -> torch.Tensor:
    """The unweighted per-sample supervised loss (loss_weight 1)."""
    plain = dict(batch)
    plain["loss_weight"] = torch.ones(BATCH)
    return setup._flow_matching_losses(plain, data, config, torch.device("cpu"))


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
def test_per_sample_mix_weights_each_samples_supervised_loss(masked):
    config = _config(0.25, masked=masked)
    setup, model = _FakeSetup(), _Model()
    batch = _batch(masked=masked)
    original_loss_weight = batch["loss_weight"]
    data = setup.predict(model, batch, config, None)
    mix = torch.tensor([0.0, 2.0, 0.5])

    mixed = setup._mixed_supervised_loss(model, batch, data, config, mix)

    expected = (mix * batch["loss_weight"] * _per_sample_losses(setup, batch, data, config)).mean()
    assert torch.allclose(mixed, expected, rtol=0.0, atol=1e-7)
    # The weighting rides on a copy: the caller's batch keeps its loss_weight.
    assert batch["loss_weight"] is original_loss_weight
    assert torch.equal(batch["loss_weight"], torch.tensor([1.0, 0.5, 2.0]))


def test_masked_term_goes_through_the_masked_loss():
    # Minor-edit concepts: with a low unmasked weight, changing the target
    # outside the mask barely moves the supervised term.
    config = _config(0.25, masked=True, unmasked_weight=0.0)
    setup, model = _FakeSetup(), _Model()
    batch = _batch(masked=True)
    data = setup.predict(model, batch, config, None)
    mix = torch.tensor([1.0, 0.0, 3.0])
    before = setup._mixed_supervised_loss(model, batch, data, config, mix)

    outside = data["target"].clone()
    outside[..., W // 2 :] += 5.0
    after = setup._mixed_supervised_loss(model, batch, {**data, "target": outside}, config, mix)
    assert torch.allclose(before, after, rtol=0.0, atol=1e-7)


def test_scalar_mix_is_exactly_the_old_arithmetic():
    config = _config(0.25)
    setup, model = _FakeSetup(), _Model()
    batch = _batch()
    data = setup.predict(model, batch, config, None)
    assert torch.equal(
        setup._mixed_supervised_loss(model, batch, data, config, 0.25),
        0.25 * setup.calculate_loss(model, batch, data, config),
    )


def test_uniform_per_sample_mix_matches_the_scalar():
    config = _config(0.25)
    setup, model = _FakeSetup(), _Model()
    batch = _batch()
    data = setup.predict(model, batch, config, None)
    uniform = setup._mixed_supervised_loss(model, batch, data, config, torch.full((BATCH,), 3.0))
    scalar = setup._mixed_supervised_loss(model, batch, data, config, 3.0)
    assert torch.allclose(uniform, scalar, rtol=1e-6, atol=0.0)


def _fused(config: TrainConfig, batch: dict):
    setup, model = _FakeSetup(), _Model()
    loss = setup.calculate_dpo_loss(model, batch, config, None)
    loss.backward()
    return setup, loss.detach(), [p.grad.clone() for p in model.net.parameters()]


def _light(config: TrainConfig, batch: dict):
    setup, model = _FakeSetup(), _Model()
    loss = setup.calculate_dpo_loss_light(model, batch, config, None, lambda term: term.backward())
    return setup, loss.detach(), [p.grad.clone() for p in model.net.parameters()]


@pytest.mark.parametrize("path", [_fused, _light], ids=["fused", "light"])
def test_an_all_inherit_column_is_bit_for_bit_the_old_run(path):
    config = _config(0.25)
    _, old_loss, old_grads = path(config, _batch())
    _, new_loss, new_grads = path(config, _batch(-torch.ones(BATCH)))
    assert torch.equal(old_loss, new_loss)
    for old, new in zip(old_grads, new_grads, strict=True):
        assert torch.equal(old, new)


@pytest.mark.parametrize("path", [_fused, _light], ids=["fused", "light"])
def test_no_supervised_forward_when_every_sample_resolves_to_zero(path):
    setup, *_ = path(_config(0.25), _batch(torch.zeros(BATCH)))
    assert setup.loss_calls == 0
    setup, *_ = path(_config(0.0), _batch(-torch.ones(BATCH)))
    assert setup.loss_calls == 0
    setup, *_ = path(_config(0.0), _batch(torch.tensor([-1.0, 0.0, 2.0])))
    assert setup.loss_calls == 1


@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
def test_per_sample_mix_in_the_dpo_loss(masked):
    config = _config(0.25, masked=masked)
    override = torch.tensor([0.0, 2.0, -1.0])  # off, 2, inherit 0.25
    _, dpo_only, _ = _fused(config, _batch(torch.zeros(BATCH), masked=masked))
    _, fused_loss, fused_grads = _fused(config, _batch(override, masked=masked))

    setup, model = _FakeSetup(), _Model()
    batch = _batch(masked=masked)
    with torch.no_grad():
        data = setup.predict(model, batch, config, None)
    mix = torch.tensor([0.0, 2.0, 0.25])
    expected = (mix * batch["loss_weight"] * _per_sample_losses(setup, batch, data, config)).mean()
    assert torch.allclose(fused_loss - dpo_only, expected, rtol=0.0, atol=1e-6)

    # The light path backwards the same objective half by half.
    _, light_loss, light_grads = _light(config, _batch(override, masked=masked))
    assert torch.allclose(light_loss, fused_loss, rtol=0.0, atol=1e-6)
    for light, fused in zip(light_grads, fused_grads, strict=True):
        assert torch.allclose(light, fused, rtol=0.0, atol=1e-6)
