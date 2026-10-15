"""BaseModelSetup.probe_loss / probe_dpo_margin: a validation-style no-grad
forward at a pinned timestep with fixed noise that leaves the training step
untouched (gradients, the RNG training draws from, the per-sample loss stash,
the model's mode), and is deterministic on an unchanged model."""

from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin

import torch

import pytest
from test_dpo_light_reorder import _batch, _config, _FakeSetup, _Model


def _sft_batch() -> dict:
    batch = _batch()
    return {
        "half": "chosen",
        "latent_image": batch["latent_image"],
        "target": batch["target"],
        "timestep": batch["timestep"],
    }


def test_probe_loss_is_deterministic_and_matches_the_training_loss_function():
    setup, model, config = _FakeSetup(), _Model(), _config()
    batch = _sft_batch()
    first = setup.probe_loss(model, batch, config, None, 0.2)
    second = setup.probe_loss(model, batch, config, None, 0.2)
    assert first == second
    with torch.no_grad():
        output = setup.predict(model, batch, config, None, deterministic=True)
        expected = float(setup.calculate_loss(model, batch, output, config))
    assert first == expected
    assert setup.calls[-1] == ("chosen", False, False)  # no grad, not the reference


def test_probe_does_not_touch_grads_rng_stash_or_mode():
    setup, model, config = _FakeSetup(), _Model(), _config()
    batch = _sft_batch()
    for p in model.net.parameters():
        p.grad = torch.full_like(p, 0.25)
    model.net.train()
    setup._last_per_sample_losses = torch.tensor([9.0])
    torch.manual_seed(123)
    draw_before = torch.rand(3)
    torch.manual_seed(123)
    _ = torch.rand(3)
    setup.probe_loss(model, batch, config, None, 0.8)
    setup.probe_dpo_margin(model, _batch(), config, None, 0.5)
    draw_after = torch.rand(3)  # the next draw training would make
    torch.manual_seed(123)
    _ = torch.rand(3)
    assert torch.equal(draw_after, torch.rand(3)), "the probe must not advance the RNG training draws from"
    assert all(torch.equal(p.grad, torch.full_like(p, 0.25)) for p in model.net.parameters())
    assert all(p.grad is not None for p in model.net.parameters())
    assert model.net.training
    assert torch.equal(setup._last_per_sample_losses, torch.tensor([9.0]))
    assert setup._probe_timestep_fraction is None
    assert setup._dpo_paired_half is None
    assert not torch.equal(draw_before, draw_after)  # it is the next draw, not a repeat of the first


def test_probe_dpo_margin_is_the_pairs_reward_margin_and_deterministic():
    setup, model, config = _FakeSetup(), _Model(), _config()
    batch = _batch()
    margin = setup.probe_dpo_margin(model, batch, config, None, 0.5)
    assert margin == setup.probe_dpo_margin(model, batch, config, None, 0.5)
    # the fake forward ignores the timestep, so the batched DPO path's own
    # margin metric on the same batch is the same number
    setup.calculate_dpo_loss(model, batch, config, None)
    assert margin == setup.get_last_dpo_metrics()["reward_margin"]
    assert "chosen_reward" in setup.get_last_dpo_metrics()


class _Noise(ModelSetupNoiseMixin):
    pass


def test_deterministic_timesteps_follow_the_pinned_fraction():
    mixin = _Noise()
    generator = torch.Generator(device="cpu")
    config = SimpleNamespace(timestep_shift=1.0)
    assert mixin._get_timestep_discrete(1000, True, generator, 4, config).tolist() == [499]
    assert mixin._get_timestep_continuous(True, generator, 2, config).tolist() == [0.5, 0.5]
    mixin._probe_timestep_fraction = 0.2
    assert mixin._get_timestep_discrete(1000, True, generator, 4, config).tolist() == [199]
    assert mixin._get_timestep_continuous(True, generator, 2, config).tolist() == pytest.approx([0.2, 0.2])
    mixin._probe_timestep_fraction = 0.8
    assert mixin._get_timestep_discrete(1000, True, generator, 4, config).tolist() == [799]
    mixin._probe_timestep_fraction = None
    assert mixin._get_timestep_discrete(1000, True, generator, 4, config).tolist() == [499]
