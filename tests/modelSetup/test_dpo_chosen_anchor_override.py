"""Per-concept DPOP chosen anchor (ConceptConfig.dpo_chosen_anchor_override):
how a batch resolves its per-pair coefficients against the global
rlhf_dpo_chosen_anchor, the objective with mixed per-pair anchors (still beta
scaled), the metric gate when only concepts turn the anchor on, and the column's
trip through the concept config, the data loader and the collate fill for rows
cached before it existed. The light and batched DPO paths with overrides are
covered in test_dpo_light_reorder."""

from types import MethodType, SimpleNamespace

from modules.dataLoader.BaseDataLoader import _shape_safe_collate
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_light_util import (
    dpo_chosen_anchor_active,
    dpo_chosen_anchor_penalties,
    dpo_objective,
    dpo_per_sample_losses,
)
from modules.util.enum.DataType import DataType
from modules.util.enum.DPOObjective import DPOObjective

from mgds.OutputPipelineModule import OutputPipelineModule

import torch
import torch.nn.functional as F

import pytest


def _config(global_anchor: float, objective=DPOObjective.SIGMOID) -> TrainConfig:
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_beta = 200.0
    config.rlhf_dpo_objective = objective
    config.rlhf_dpo_chosen_anchor = global_anchor
    config.rlhf_dpo_concept_logging = False
    config.rlhf_dpo_live_saturation = False
    config.rlhf_dpo_timestep_margin_logging = False
    return config


def _resolve(global_anchor: float, override=None):
    batch = {} if override is None else {"dpo_chosen_anchor_override": torch.tensor(override)}
    return BaseModelSetup._resolve_chosen_anchor(_config(global_anchor), batch)


def _inputs():
    pc = torch.tensor([-1.0, -2.0, -0.5, -0.9])
    pr = torch.tensor([-1.5, -1.0, -0.9, -2.0])
    rc = torch.tensor([-1.2, -1.5, -0.4, -0.9])  # pairs 1 and 2 fit the chosen worse than the reference
    rr = torch.tensor([-1.4, -1.1, -0.7, -1.8])
    return pc, pr, rc, rr


def _kwargs(beta, objective=DPOObjective.SIGMOID):
    return {"beta": beta, "objective": objective, "ipo_tau": 10.0, "label_smoothing": 0.0}


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_without_the_column_the_global_comes_back_untouched():
    assert _resolve(0.5) == 0.5
    assert _resolve(0.0) == 0.0


def test_negative_inherits_and_mixed_pairs_resolve_per_pair():
    anchor = _resolve(0.5, [-1.0, 40.0, 0.0, -3.0])
    assert isinstance(anchor, torch.Tensor)
    assert torch.equal(anchor, torch.tensor([0.5, 40.0, 0.0, 0.5]))


def test_an_override_turns_the_anchor_on_while_the_global_is_off():
    anchor = _resolve(0.0, [-1.0, 25.0])
    assert torch.equal(anchor, torch.tensor([0.0, 25.0]))
    assert dpo_chosen_anchor_active(anchor)
    # every pair under the overriding concept: one scalar
    assert _resolve(0.0, [25.0, 25.0]) == 25.0


def test_uniform_batches_collapse_to_a_scalar():
    # every pair inherits: the global itself, so runs without overrides are unchanged
    assert _resolve(0.5, [-1.0, -1.0]) == 0.5
    assert _resolve(0.0, [-1.0, -1.0]) == 0.0
    # a concept switching the anchor off for all of the batch's pairs
    assert _resolve(0.5, [0.0, 0.0]) == 0.0
    assert not dpo_chosen_anchor_active(_resolve(0.5, [0.0, 0.0]))


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #


def test_mixed_per_pair_anchors_keep_each_pairs_beta_scaling():
    pc, pr, rc, rr = _inputs()
    beta = torch.tensor([200.0, 2.0, 200.0, 2.0])
    # pair 1 (beta 2) overridden to 50, pair 2 (beta 200) inherits 0.5
    anchor = torch.tensor([0.5, 50.0, 0.5, 50.0])
    loss, dpo_loss, chosen_ratio, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(beta), chosen_anchor=anchor)
    per_pair = anchor * beta * F.relu(-(pc - rc))
    assert torch.allclose(loss - dpo_loss, per_pair.mean())
    assert per_pair[1].item() == pytest.approx(50.0 * 2.0 * 0.5)
    assert per_pair[2].item() == pytest.approx(0.5 * 200.0 * 0.1)
    assert torch.allclose(
        dpo_chosen_anchor_penalties(chosen_ratio, beta=beta, objective=DPOObjective.SIGMOID, chosen_anchor=anchor),
        per_pair,
    )
    # the per-pair losses the gradient diagnostics read still average to the loss
    losses = dpo_per_sample_losses(chosen_ratio, pr - rr, **_kwargs(beta), chosen_anchor=anchor)
    assert torch.allclose(losses.mean(), loss)


def test_an_override_with_the_global_off_adds_only_its_own_pairs_penalty():
    pc, pr, rc, rr = _inputs()
    beta = 2.0
    anchor = torch.tensor([0.0, 25.0, 0.0, 0.0])
    loss, dpo_loss, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(beta), chosen_anchor=anchor)
    assert (loss - dpo_loss).item() == pytest.approx(25.0 * 2.0 * 0.5 / 4)
    # the inherited global 0 everywhere is the plain objective, bit for bit
    plain, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(beta))
    off, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(beta), chosen_anchor=_resolve(0.0, [-1.0] * 4))
    assert torch.equal(plain, off)


def test_gradient_reaches_only_the_anchored_pairs_chosen_half():
    pc, pr, rc, rr = _inputs()
    pc.requires_grad_(True)
    beta = torch.tensor([200.0, 2.0, 200.0, 2.0])
    anchor = torch.tensor([0.5, 50.0, 0.0, 50.0])
    loss, dpo_loss, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(beta), chosen_anchor=anchor)
    (grad,) = torch.autograd.grad(loss - dpo_loss, pc)
    # pair 1 is below the reference and anchored; pair 2 is below it but its
    # concept switched the anchor off; pairs 0 and 3 are above the reference
    assert grad[1].item() == pytest.approx(-50.0 * 2.0 / 4)
    assert grad[0].item() == 0.0 and grad[2].item() == 0.0 and grad[3].item() == 0.0


def test_ipo_anchor_is_unscaled_and_kto_ignores_overrides():
    pc, pr, rc, rr = _inputs()
    anchor = torch.tensor([0.0, 3.0, 1.0, 0.0])
    loss, dpo_loss, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(5.0, DPOObjective.IPO), chosen_anchor=anchor)
    assert torch.allclose(loss - dpo_loss, (anchor * F.relu(-(pc - rc))).mean())
    loss, dpo_loss, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(5.0, DPOObjective.KTO), chosen_anchor=anchor)
    assert torch.equal(loss, dpo_loss)


# --------------------------------------------------------------------------- #
# Metrics gate
# --------------------------------------------------------------------------- #


def _record(config: TrainConfig, chosen_anchor, overridden: bool) -> dict:
    setup = SimpleNamespace(
        _dpo_concept_keys=[],
        _dpo_concept_key_of_path=None,
        _dpo_concept_index_cache={},
        _dpo_chosen_anchor_overridden=overridden,
    )
    setup._dpo_concept_indices = MethodType(BaseModelSetup._dpo_concept_indices, setup)
    chosen = torch.tensor([0.1, -0.5])
    rejected = torch.tensor([-0.2, -0.9])
    BaseModelSetup._record_dpo_metrics(
        setup,
        None,
        config,
        torch.tensor(0.5),
        torch.tensor(0.5),
        chosen,
        rejected,
        chosen - rejected,
        None,
        2,
        torch.tensor([200.0, 2.0]),
        chosen_anchor=chosen_anchor,
    )
    return setup._last_dpo_metrics


def test_anchor_metrics_use_the_resolved_per_pair_coefficients():
    metrics = _record(_config(0.5), torch.tensor([0.5, 40.0]), overridden=True)
    assert metrics["chosen_anchor_loss"] == pytest.approx(40.0 * 2.0 * 0.5 / 2)
    assert metrics["chosen_below_ref"] == pytest.approx(0.5)


def test_anchor_metrics_stay_on_every_batch_when_only_concepts_override():
    # global 0, a concept overrides: a batch without that concept's pairs
    # still logs the metric (as 0) so the window average divides alike
    metrics = _record(_config(0.0), 0.0, overridden=True)
    assert metrics["chosen_anchor_loss"] == 0.0
    # global 0 and no overrides anywhere: no anchor metrics, as before
    assert "chosen_anchor_loss" not in _record(_config(0.0), 0.0, overridden=False)
    # a batch whose resolved anchor is live logs it even without the run flag
    assert "chosen_anchor_loss" in _record(_config(0.0), torch.tensor([0.0, 25.0]), overridden=False)
    # never under KTO
    assert "chosen_anchor_loss" not in _record(_config(0.5, DPOObjective.KTO), 0.5, overridden=True)


# --------------------------------------------------------------------------- #
# Concept config, data loader column, collate fill
# --------------------------------------------------------------------------- #


def test_concept_defaults_to_inheriting_the_global_anchor():
    concept = ConceptConfig.default_values()
    assert concept.dpo_chosen_anchor_override == -1.0
    assert concept.to_dict()["dpo_chosen_anchor_override"] == -1.0
    concept.dpo_chosen_anchor_override = 40.0
    assert ConceptConfig.default_values().from_dict(concept.to_dict()).dpo_chosen_anchor_override == 40.0
    # concept files saved before the field existed load as "inherit"
    legacy = ConceptConfig.default_values().from_dict({"name": "legacy", "dpo_beta_override": 2.0})
    assert legacy.dpo_chosen_anchor_override == -1.0


class _DataLoaderHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


def _output_module(rlhf_enabled: bool) -> OutputPipelineModule:
    config = TrainConfig.default_values()
    config.rlhf_enabled = rlhf_enabled
    modules = _DataLoaderHarness()._output_modules_from_out_names(
        model=None, model_setup=None, output_names=["latent_image"], config=config, train_dtype=DataType.FLOAT_32
    )
    return next(module for module in modules if isinstance(module, OutputPipelineModule))


def test_dpo_loader_carries_the_column_per_sample():
    columns = list(zip(_output_module(True).input_names, _output_module(True).output_names, strict=True))
    assert ("concept.dpo_chosen_anchor_override", "dpo_chosen_anchor_override") in columns
    assert "dpo_chosen_anchor_override" not in _output_module(False).output_names


def test_collate_fills_rows_whose_snapshot_predates_the_field():
    # sourceless rows serve the concept snapshot stamped at cache time; an old
    # snapshot has no key, mgds yields None and the output module drops it
    new = {"latent_image": torch.zeros(2), "dpo_chosen_anchor_override": torch.tensor(40.0)}
    old = {"latent_image": torch.zeros(2)}
    batch = _shape_safe_collate([new, old])
    assert torch.equal(batch["dpo_chosen_anchor_override"], torch.tensor([40.0, -1.0]))
    batch = _shape_safe_collate([old, new])
    assert torch.equal(batch["dpo_chosen_anchor_override"], torch.tensor([-1.0, 40.0]))
    # no row has it: the column stays absent and the global anchor applies
    assert "dpo_chosen_anchor_override" not in _shape_safe_collate([old, old])
