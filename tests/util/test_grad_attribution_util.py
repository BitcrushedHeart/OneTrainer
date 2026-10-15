"""Gradient attribution primitives: snapshot-diff accumulators, cosines, the
rest-cosine conflict reading, the timestep-bin normaliser and the window stats."""

import math

from modules.util.grad_attribution_util import (
    ConceptWindowStats,
    GradAttribution,
    TimestepBinNormalizer,
    concept_groups,
    concept_tag,
    cosine,
    sample_mask,
)

import torch

import pytest


def _params():
    return [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2, 2))]


def _add_grad(params, values):
    for p, v in zip(params, values, strict=True):
        v = torch.as_tensor(v, dtype=torch.float32).reshape(p.shape)
        p.grad = v.clone() if p.grad is None else p.grad + v


def _flat(vectors):
    return torch.cat([v.reshape(-1) for v in vectors])


def test_accumulators_hold_each_segment_and_sum_to_the_total():
    params = _params()
    attribution = GradAttribution(params)
    a = ([1.0, 2.0, 3.0], [[1.0, 0.0], [0.0, 1.0]])
    b = ([-1.0, 0.5, 0.0], [[2.0, 2.0], [2.0, 2.0]])
    attribution.begin()
    _add_grad(params, a)
    attribution.end("sft")
    _add_grad(params, b)
    attribution.end("dpo")  # end() re-snapshots, so the second delta is b alone

    assert attribution.keys == ["sft", "dpo"]
    assert torch.equal(_flat(attribution.vector("sft")), _flat([torch.tensor(x) for x in a]))
    assert torch.equal(_flat(attribution.vector("dpo")), _flat([torch.tensor(x) for x in b]))
    total = _flat([p.grad for p in params])
    assert torch.equal(_flat(attribution.vector("sft")) + _flat(attribution.vector("dpo")), total)

    scalars = attribution.scalars()
    assert scalars["norm_sft"] == pytest.approx(_flat(attribution.vector("sft")).norm().item())
    assert scalars["norm_dpo"] == pytest.approx(_flat(attribution.vector("dpo")).norm().item())
    assert scalars["norm_total"] == pytest.approx(total.norm().item())
    expected_cos = cosine(attribution.vector("sft"), attribution.vector("dpo"))
    assert scalars["cos_sft_dpo"] == pytest.approx(expected_cos)
    assert set(scalars) == {"norm_sft", "norm_dpo", "cos_sft_dpo", "norm_total"}


def test_cosine_on_known_vectors():
    x = [torch.tensor([1.0, 0.0]), torch.tensor([[0.0]])]
    y = [torch.tensor([0.0, 1.0]), torch.tensor([[0.0]])]
    assert cosine(x, x) == pytest.approx(1.0)
    assert cosine(x, y) == pytest.approx(0.0)
    assert cosine(x, [-v for v in x]) == pytest.approx(-1.0)
    z = [torch.tensor([1.0, 1.0]), torch.tensor([[0.0]])]
    assert cosine(x, z) == pytest.approx(1 / math.sqrt(2))

    params = _params()
    attribution = GradAttribution(params)
    attribution.begin()
    _add_grad(params, ([1.0, 0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]))
    attribution.end("a")
    _add_grad(params, ([1.0, 1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]))
    attribution.end("b")
    _add_grad(params, ([-1.0, 0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]))
    attribution.end("c")
    scalars = attribution.scalars()
    assert scalars["cos_a_b"] == pytest.approx(1 / math.sqrt(2))
    assert scalars["cos_a_c"] == pytest.approx(-1.0)
    assert scalars["cos_b_c"] == pytest.approx(-1 / math.sqrt(2))
    assert scalars["norm_total"] == pytest.approx(math.sqrt(2))  # (1,0,0) + (1,1,0) + (-1,0,0) = (1,1,0)


def test_rest_cosines_compare_each_key_against_the_sum_of_the_others():
    params = _params()
    attribution = GradAttribution(params)
    attribution.begin()
    vectors = {
        "A": ([1.0, 0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]),
        "B": ([0.0, 1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]),
        "C": ([-2.0, 0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]),
    }
    for key, v in vectors.items():
        _add_grad(params, v)
        attribution.end(key)
    readings = attribution.rest_cosines()
    flat = {key: _flat(attribution.vector(key)) for key in vectors}
    for key in vectors:
        rest = sum((flat[k] for k in vectors if k != key), torch.zeros(7))
        expected = torch.dot(flat[key], rest) / (flat[key].norm() * rest.norm())
        assert readings[f"cos_rest_{key}"] == pytest.approx(expected.item(), abs=1e-6), key
        assert readings[f"norm_{key}"] == pytest.approx(flat[key].norm().item())
    assert readings["cos_rest_A"] == pytest.approx(-2 / math.sqrt(5))  # A against B + C = (-2, 1, 0)


def test_missing_grads_count_as_zero_and_reset_keeps_the_buffers():
    params = _params()
    attribution = GradAttribution(params)
    attribution.begin()
    params[0].grad = torch.tensor([1.0, 1.0, 1.0])  # params[1] never gets a gradient
    attribution.end("sft")
    assert attribution.scalars()["norm_sft"] == pytest.approx(math.sqrt(3))
    buffers_before = attribution.vector("sft")
    attribution.reset()
    assert attribution.keys == []
    assert attribution.scalars() == {}
    assert attribution.vector("sft") is buffers_before  # reused, not reallocated
    assert all(torch.count_nonzero(b) == 0 for b in buffers_before)
    # a fresh window after zero_grad(set_to_none=True)
    for p in params:
        p.grad = None
    attribution.begin()
    params[1].grad = torch.ones(2, 2)
    attribution.end("dpo")
    assert attribution.scalars()["norm_dpo"] == pytest.approx(2.0)
    assert attribution.buffer_bytes() == 7 * 4 * 3  # snapshot + two buffers


def test_scale_divides_norms_but_not_cosines():
    params = _params()
    attribution = GradAttribution(params)
    attribution.begin()
    _add_grad(params, ([2.0, 0.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]))
    attribution.end("a")
    _add_grad(params, ([2.0, 2.0, 0.0], [[0.0, 0.0], [0.0, 0.0]]))
    attribution.end("b")
    scaled = attribution.scalars(scale=2.0)
    assert scaled["norm_a"] == pytest.approx(1.0)
    assert scaled["cos_a_b"] == pytest.approx(1 / math.sqrt(2))


def test_timestep_bin_normaliser_uses_the_running_bin_mean():
    normaliser = TimestepBinNormalizer(bins=10, decay=0.5)
    # first batch: each value over its own bin's batch mean
    out = normaliser.normalize("loss", [2.0, 4.0, 10.0], [0.05, 0.05, 0.95])
    assert out == pytest.approx([2 / 3, 4 / 3, 1.0])
    assert normaliser.means["loss"][0] == pytest.approx(3.0)
    assert normaliser.means["loss"][9] == pytest.approx(10.0)
    # second batch: against the means as they stood, then the EMA moves
    out = normaliser.normalize("loss", [6.0, 5.0], [0.0, 0.99])
    assert out == pytest.approx([2.0, 0.5])
    assert normaliser.means["loss"][0] == pytest.approx(4.5)
    assert normaliser.means["loss"][9] == pytest.approx(7.5)
    # keys are independent; NaN values and missing timesteps are tolerated
    out = normaliser.normalize("out", [float("nan"), 1.0], [None, None])
    assert math.isnan(out[0]) and out[1] == pytest.approx(1.0)
    assert normaliser.means["loss"][0] == pytest.approx(4.5)


def test_window_stats_average_per_concept_and_skip_nan():
    stats = ConceptWindowStats()
    stats.add("A", "loss_tnorm", 1.0)
    stats.add("A", "loss_tnorm", 3.0)
    stats.add("A", "out_norm", float("nan"))
    stats.add("B", "out_norm", 0.5)
    assert stats.scalars() == {"A/loss_tnorm": 2.0, "B/out_norm": 0.5}
    assert stats
    stats.reset()
    assert not stats and stats.scalars() == {}


def test_concept_groups_and_sample_mask():
    groups = concept_groups(["b", None, "a", "b"])
    assert groups == {"b": [0, 3], "other": [1], "a": [2]}
    like = torch.zeros(4, 2, 3)
    mask = sample_mask([0, 3], like)
    assert mask.shape == (4, 1, 1)
    assert mask.flatten().tolist() == [1.0, 0.0, 0.0, 1.0]
    assert concept_tag("Off Policy RLHF!") == "Off_Policy_RLHF"
