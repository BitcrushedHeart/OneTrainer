"""Per-concept adaptive beta in the model setup.

Each adaptive DPO concept has its own controller: the DPO metrics carry the
margin sum and count per concept key (attributed by chosen image path), and
``_resolve_dpo_beta`` scales each sample's configured beta by its own concept's
ratio. Samples under no adaptive concept keep following the pooled controller,
opted-out concepts stay unscaled and untracked, and without ratios (adaptive
beta off, or no controller updated yet) nothing changes.
"""

from types import MethodType, SimpleNamespace

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_addon_util import accumulate_dpo_metrics
from modules.util.dpo_beta_controller import DPOBetaController, controller_concept_terms, controller_margin_terms

import torch

import pytest

KEYS = ["off_policy", "on_policy"]


def _key_of_path(path):
    # "<concept>/chosen/<name>": the concept is the first folder.
    return path.split("/")[0] if path else None


def _config(beta: float = 200.0) -> TrainConfig:
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_beta = beta
    config.rlhf_dpo_chosen_anchor = 0.0
    config.rlhf_dpo_concept_logging = False
    config.rlhf_dpo_live_saturation = False
    config.rlhf_dpo_timestep_margin_logging = False
    return config


def _setup(keys=KEYS, key_of_path=_key_of_path, ratios=None, runtime_beta=None):
    setup = SimpleNamespace(
        _dpo_runtime_beta=runtime_beta,
        _dpo_runtime_beta_ratios=dict(ratios or {}),
        _dpo_concept_keys=list(keys),
        _dpo_concept_key_of_path=key_of_path,
        _dpo_concept_index_cache={},
        _dpo_adaptive_mask=BaseModelSetup._dpo_adaptive_mask,
    )
    setup._dpo_concept_indices = MethodType(BaseModelSetup._dpo_concept_indices, setup)
    return setup


def _record(setup, margin: list[float], paths, adaptive_mask=None) -> dict:
    margin_t = torch.tensor(margin)
    BaseModelSetup._record_dpo_metrics(
        setup,
        None,
        _config(),
        torch.tensor(0.5),
        torch.tensor(0.5),
        margin_t + 1.0,
        torch.ones_like(margin_t),
        margin_t,
        None,
        len(margin),
        None,
        image_paths=paths,
        adaptive_mask=adaptive_mask,
    )
    return setup._last_dpo_metrics


def _resolve(setup, batch, config=None):
    return BaseModelSetup._resolve_dpo_beta(setup, config or _config(), batch)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_metrics_carry_each_concepts_own_margin_terms():
    paths = ["on_policy/chosen/a", "off_policy/chosen/b", "on_policy/chosen/c", "elsewhere/chosen/d"]
    metrics = _record(_setup(), [0.0006, 0.0001, 0.0004, 0.5], paths)
    assert metrics["adaptive_margin_sum/on_policy"] == pytest.approx(0.001)
    assert metrics["adaptive_count/on_policy"] == 2.0
    assert metrics["adaptive_margin_sum/off_policy"] == pytest.approx(0.0001)
    assert metrics["adaptive_count/off_policy"] == 1.0
    # The stray sample only reaches the pooled terms, which still cover everything.
    assert metrics["adaptive_margin_sum"] == pytest.approx(0.5011)
    assert metrics["adaptive_count"] == 4.0
    assert metrics["reward_margin"] == pytest.approx(0.5011 / 4)


def test_opted_out_samples_stay_out_of_every_term():
    paths = ["on_policy/chosen/a", "on_policy/chosen/b"]
    metrics = _record(_setup(), [0.1, 12.0], paths, adaptive_mask=torch.tensor([True, False]))
    assert metrics["adaptive_margin_sum/on_policy"] == pytest.approx(0.1)
    assert metrics["adaptive_count/on_policy"] == 1.0
    assert metrics["adaptive_count/off_policy"] == 0.0
    assert metrics["adaptive_margin_sum"] == pytest.approx(0.1)
    assert metrics["adaptive_count"] == 1.0


def test_without_keys_or_paths_the_metrics_are_the_pooled_ones_only():
    metrics = _record(_setup(keys=[]), [0.1, 0.3], ["on_policy/chosen/a", "off_policy/chosen/b"])
    assert "adaptive_margin_sum/on_policy" not in metrics
    assert metrics["adaptive_count"] == 2.0
    metrics = _record(_setup(), [0.1, 0.3], None)
    assert "adaptive_margin_sum/on_policy" not in metrics
    assert metrics["adaptive_count"] == 2.0


def test_window_terms_sum_per_concept_and_lay_out_absent_concepts_as_zeros():
    setup = _setup()
    accumulated = accumulate_dpo_metrics(None, _record(setup, [0.2, 0.4], ["on_policy/chosen/a", "on_policy/chosen/b"]))
    accumulated = accumulate_dpo_metrics(accumulated, _record(setup, [0.6], ["on_policy/chosen/c"]))
    terms = controller_concept_terms(accumulated, KEYS)
    assert terms[0] == 0.0 and terms[1] == 0.0  # off_policy: no sample this window
    assert terms[2] == pytest.approx(1.2) and terms[3] == 3.0
    # Older metrics without the per-concept block reduce to zeros of the same length.
    assert controller_concept_terms({"reward_margin": 0.3, "_count": 1}, KEYS) == [0.0, 0.0, 0.0, 0.0]
    assert controller_margin_terms(accumulated) == (pytest.approx(1.2), 3.0)


def test_concept_indices_are_cached_and_unknown_keys_map_to_minus_one():
    seen = []

    def key_of_path(path):
        seen.append(path)
        return _key_of_path(path)

    setup = _setup(key_of_path=key_of_path)
    paths = ["on_policy/chosen/a", "elsewhere/x", "off_policy/chosen/b", "on_policy/chosen/a"]
    assert setup._dpo_concept_indices(paths).tolist() == [1, -1, 0, 1]
    assert seen == paths[:3]  # the repeated path came from the cache


# --------------------------------------------------------------------------- #
# beta resolution
# --------------------------------------------------------------------------- #


def test_each_sample_takes_its_own_concepts_ratio():
    setup = _setup(ratios={"on_policy": 2.0, "off_policy": 0.5}, runtime_beta=200.0)
    batch = {
        "dpo_beta_override": torch.tensor([0.0, 50.0, 800.0]),
        "image_path": ["on_policy/chosen/a", "off_policy/chosen/b", "on_policy/chosen/c"],
    }
    assert torch.allclose(_resolve(setup, batch), torch.tensor([400.0, 25.0, 1600.0]))


def test_a_sample_under_no_adaptive_concept_follows_the_pooled_controller():
    # Pooled controller at 1.5x; on_policy's own controller at 2x.
    setup = _setup(ratios={"on_policy": 2.0}, runtime_beta=300.0)
    batch = {
        "dpo_beta_override": torch.tensor([0.0, 0.0, 0.0]),
        "image_path": ["on_policy/chosen/a", "elsewhere/x", "off_policy/chosen/b"],
    }
    # off_policy has a key but no ratio yet: unscaled until its controller updates.
    assert torch.allclose(_resolve(setup, batch), torch.tensor([400.0, 300.0, 200.0]))


def test_opted_out_samples_keep_their_configured_beta():
    setup = _setup(ratios={"on_policy": 2.0, "off_policy": 3.0}, runtime_beta=400.0)
    batch = {
        "dpo_beta_override": torch.tensor([0.0, 50.0, 0.0]),
        "dpo_adaptive_beta": torch.tensor([True, False, False]),
        "image_path": ["on_policy/chosen/a", "off_policy/chosen/b", "elsewhere/x"],
    }
    assert torch.allclose(_resolve(setup, batch), torch.tensor([400.0, 50.0, 200.0]))


def test_without_ratios_the_pooled_path_is_unchanged():
    batch = {
        "dpo_beta_override": torch.tensor([0.0, 50.0]),
        "image_path": ["on_policy/chosen/a", "off_policy/chosen/b"],
    }
    assert torch.equal(_resolve(_setup(), batch), torch.tensor([200.0, 50.0]))
    assert torch.allclose(_resolve(_setup(runtime_beta=400.0), batch), torch.tensor([400.0, 100.0]))
    # The no-override scalar path never looks at the keys.
    assert (
        _resolve(_setup(ratios={"on_policy": 2.0}, runtime_beta=137.5), {"image_path": ["on_policy/chosen/a"]}) == 137.5
    )
    assert _resolve(_setup(ratios={"on_policy": 2.0}), {"image_path": ["on_policy/chosen/a"]}) == 200.0


def test_a_batch_without_paths_falls_back_to_the_pooled_ratio():
    setup = _setup(ratios={"on_policy": 2.0}, runtime_beta=400.0)
    beta = _resolve(setup, {"dpo_beta_override": torch.tensor([0.0, 50.0])})
    assert torch.allclose(beta, torch.tensor([400.0, 100.0]))


# --------------------------------------------------------------------------- #
# controllers
# --------------------------------------------------------------------------- #


def test_controllers_on_different_margin_scales_adapt_independently():
    # Two concepts whose margins differ by ~6x, each with a ratio controller
    # (beta0 = 1). A jump relative to a concept's own history moves its ratio,
    # while the other concept, steady on its own scale, stays put.
    big, small = DPOBetaController(1.0, warmup_steps=5), DPOBetaController(1.0, warmup_steps=5)
    for step in range(30):
        wobble = 1e-5 * (-1) ** step
        big.update(0.0006 + wobble)
        small.update(0.0001 + wobble / 6)
    big_ratio = big.update(0.0012)  # twice its usual margin
    small_ratio = small.update(0.0001)
    assert big_ratio > 1.2
    assert small_ratio == pytest.approx(1.0, abs=0.05)
    # The small concept's own jump is invisible on the big concept's scale, but
    # not to its own controller.
    assert small.update(0.0002) > 1.2
    assert 0.25 <= big_ratio <= 4.0
