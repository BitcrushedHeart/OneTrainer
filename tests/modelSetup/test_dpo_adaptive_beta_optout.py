"""The adaptive beta controller's input with per-concept opt-outs.

A concept with ``dpo_adaptive_beta`` off keeps its configured beta (see
test_dpo_beta_override) and its margins stay out of the controller: the DPO
metrics carry the margin summed over the samples that take part plus their
count, the trainer pools those across micro-batches and ranks, and a step with
no such sample leaves the controller alone.
"""

from types import SimpleNamespace

from modules.dataLoader.BaseDataLoader import _fill_inherit_columns
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_addon_util import accumulate_dpo_metrics
from modules.util.dpo_beta_controller import controller_margin_terms

import torch

import pytest


def _config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_beta = 200.0
    config.rlhf_dpo_chosen_anchor = 0.0
    config.rlhf_dpo_concept_logging = False
    config.rlhf_dpo_live_saturation = False
    config.rlhf_dpo_timestep_margin_logging = False
    return config


def _record(margin: list[float], adaptive_mask=None) -> dict:
    """The per-micro-batch DPO metrics for a batch with the given raw margins."""
    setup = SimpleNamespace()
    margin_t = torch.tensor(margin)
    chosen = margin_t + 1.0
    rejected = torch.ones_like(margin_t)
    BaseModelSetup._record_dpo_metrics(
        setup,
        None,
        _config(),
        torch.tensor(0.5),
        torch.tensor(0.5),
        chosen,
        rejected,
        margin_t,
        None,
        len(margin),
        None,
        adaptive_mask=adaptive_mask,
    )
    return setup._last_dpo_metrics


def test_without_a_mask_the_terms_cover_every_sample():
    metrics = _record([0.1, 0.3, -0.2])
    assert metrics["adaptive_margin_sum"] == pytest.approx(0.2)
    assert metrics["adaptive_count"] == 3.0
    # The logged batch-mean margin is untouched.
    assert metrics["reward_margin"] == pytest.approx(0.2 / 3)


def test_the_mask_keeps_opted_out_samples_out_of_the_terms_but_not_the_log():
    metrics = _record([0.1, 12.0, -0.2], adaptive_mask=torch.tensor([True, False, True]))
    assert metrics["adaptive_margin_sum"] == pytest.approx(-0.1)
    assert metrics["adaptive_count"] == 2.0
    assert metrics["reward_margin"] == pytest.approx(11.9 / 3)


def test_a_fully_opted_out_batch_contributes_nothing():
    metrics = _record([5.0, 7.0], adaptive_mask=torch.tensor([False, False]))
    assert metrics["adaptive_margin_sum"] == 0.0
    assert metrics["adaptive_count"] == 0.0


def test_controller_input_is_the_mean_over_included_samples_of_the_window():
    # Two micro-batches: the off-policy pairs (huge margins) are masked out.
    accumulated = accumulate_dpo_metrics(None, _record([0.1, 12.0], adaptive_mask=torch.tensor([True, False])))
    accumulated = accumulate_dpo_metrics(
        accumulated, _record([0.3, 0.2, 15.0], adaptive_mask=torch.tensor([True, True, False]))
    )
    margin_sum, included = controller_margin_terms(accumulated)
    assert included == 3.0
    assert margin_sum / included == pytest.approx(0.2)
    # ... while the logged margin still averages the batch means over the window.
    assert accumulated["reward_margin"] / accumulated["_count"] == pytest.approx((12.1 / 2 + 15.5 / 3) / 2)


def test_a_window_without_included_samples_reports_a_zero_count():
    accumulated = accumulate_dpo_metrics(None, _record([5.0], adaptive_mask=torch.tensor([False])))
    margin_sum, included = controller_margin_terms(accumulated)
    assert included == 0.0
    assert margin_sum == 0.0


def test_without_opt_outs_the_controller_input_equals_the_batch_mean_margin():
    # Equal-sized micro-batches: the per-sample mean is the mean of the batch
    # means, i.e. the value the controller received before the opt-out existed.
    accumulated = None
    for margins in ([0.1, 0.3], [-0.2, 0.6], [0.05, 0.15]):
        accumulated = accumulate_dpo_metrics(accumulated, _record(margins))
    margin_sum, included = controller_margin_terms(accumulated)
    assert included == 6.0
    assert margin_sum / included == pytest.approx(accumulated["reward_margin"] / accumulated["_count"])


def test_metrics_without_the_terms_fall_back_to_the_batch_mean_margin():
    # Older model setups (and the trainer's mocked tests) know only reward_margin.
    accumulated = {"reward_margin": 0.9, "_count": 3}
    assert controller_margin_terms(accumulated) == (pytest.approx(0.3), 1.0)


def test_rows_cached_before_the_field_existed_take_part_in_adaptive_beta():
    # A sourceless batch mixing an old snapshot row (no key) with a new one:
    # the missing entry becomes True so the batch collates, like the override.
    batch = [
        {"dpo_beta_override": torch.tensor(0.0), "dpo_adaptive_beta": torch.tensor(False)},
        {"dpo_beta_override": torch.tensor(0.0)},
    ]
    filled = _fill_inherit_columns(batch)
    assert filled[0]["dpo_adaptive_beta"].item() is False
    assert filled[1]["dpo_adaptive_beta"].item() is True
    assert filled[1]["dpo_adaptive_beta"].dtype == torch.bool
