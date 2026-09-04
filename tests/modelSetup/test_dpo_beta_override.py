from types import SimpleNamespace

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig

import torch

import pytest


def _config(beta: float = 200.0) -> TrainConfig:
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_beta = beta
    return config


def _resolve(runtime_beta, batch, config=None):
    setup = SimpleNamespace(_dpo_runtime_beta=runtime_beta)
    return BaseModelSetup._resolve_dpo_beta(setup, config or _config(), batch)


def test_batch_without_the_override_column_keeps_the_global_scalar():
    beta = _resolve(None, {"latent_image": torch.zeros(2, 4)})
    assert isinstance(beta, float)
    assert beta == 200.0


def test_zero_override_inherits_the_global_beta_and_positive_override_wins():
    beta = _resolve(None, {"dpo_beta_override": torch.tensor([0.0, 50.0, 0.0, 800.0])})
    assert torch.equal(beta, torch.tensor([200.0, 50.0, 200.0, 800.0]))


def test_negative_override_inherits_rather_than_inverting_the_preference():
    # A negative beta would flip the sign of the objective's logits, training
    # the model to prefer the rejected image. It must never be taken literally.
    beta = _resolve(None, {"dpo_beta_override": torch.tensor([-50.0, 30.0])})
    assert torch.equal(beta, torch.tensor([200.0, 30.0]))


def test_adaptive_beta_without_overrides_reproduces_the_absolute_value():
    # Backward compatibility: the ratio form must return exactly what the old
    # "replace the config beta" code returned.
    assert _resolve(137.5, {"latent_image": torch.zeros(2, 4)}) == pytest.approx(137.5)


def test_adaptive_beta_scales_every_concept_by_the_same_factor():
    # The controller ran to 2x the configured beta. Each concept's beta scales
    # by 2x; the overrides keep their configured proportions instead of being
    # flattened onto the controller's scalar.
    beta = _resolve(400.0, {"dpo_beta_override": torch.tensor([0.0, 50.0, 800.0])})
    assert torch.allclose(beta, torch.tensor([400.0, 100.0, 1600.0]))
    assert torch.allclose(beta / beta[0], torch.tensor([1.0, 0.25, 4.0]))


def test_adaptive_beta_is_a_noop_when_it_equals_the_configured_beta():
    beta = _resolve(200.0, {"dpo_beta_override": torch.tensor([0.0, 50.0])})
    assert torch.allclose(beta, torch.tensor([200.0, 50.0]))


def test_zero_global_beta_does_not_divide_by_zero():
    beta = _resolve(10.0, {"dpo_beta_override": torch.tensor([0.0, 50.0])}, config=_config(beta=0.0))
    assert torch.equal(beta, torch.tensor([0.0, 50.0]))


def test_integer_override_column_is_promoted_to_float():
    beta = _resolve(None, {"dpo_beta_override": torch.tensor([0, 50])})
    assert beta.dtype == torch.float32
    assert torch.equal(beta, torch.tensor([200.0, 50.0]))
