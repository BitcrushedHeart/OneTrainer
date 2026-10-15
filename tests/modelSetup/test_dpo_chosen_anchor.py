"""DPOP chosen anchor (Pal et al. 2024, "Smaug"; config rlhf_dpo_chosen_anchor)
at the objective level: the formula and its per-pair beta scaling, the one-sided
gradient, IPO (unscaled) and KTO (not applied), the per-pair losses the
gradient diagnostics read, the light path's surrogate weights, the per-concept
window, and that the default 0 is bit-for-bit the plain objective."""

from modules.util.dpo_light_util import (
    dpo_chosen_anchor_penalty,
    dpo_objective,
    dpo_per_sample_losses,
    dpo_surrogate_weights,
)
from modules.util.dpo_live_saturation import DPOConceptInfo, DPOConceptWindow
from modules.util.enum.DPOObjective import DPOObjective

import torch
import torch.nn.functional as F

import pytest


def _kwargs(objective=DPOObjective.SIGMOID, beta=5.0, ipo_tau=10.0, label_smoothing=0.0):
    return {"beta": beta, "objective": objective, "ipo_tau": ipo_tau, "label_smoothing": label_smoothing}


def _inputs():
    pc = torch.tensor([-1.0, -2.0, -0.5, -0.9])
    pr = torch.tensor([-1.5, -1.0, -0.9, -2.0])
    rc = torch.tensor([-1.2, -1.5, -0.4, -0.9])  # pairs 1 and 2 fit the chosen worse than the reference
    rr = torch.tensor([-1.4, -1.1, -0.7, -1.8])
    return pc, pr, rc, rr


def _toy_logp(theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -((x @ theta) - y).pow(2)


def test_anchor_matches_the_formula_and_scales_with_each_pairs_beta():
    pc, pr, rc, rr = _inputs()
    beta = torch.tensor([3.0, 7.0, 0.5, 2.0])
    anchor = 0.6
    loss, dpo_loss, chosen_ratio, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(beta=beta), chosen_anchor=anchor)
    expected = anchor * (beta * F.relu(-(pc - rc))).mean()
    assert torch.allclose(loss - dpo_loss, expected)
    assert torch.allclose(
        dpo_chosen_anchor_penalty(chosen_ratio, beta=beta, objective=DPOObjective.SIGMOID, chosen_anchor=anchor),
        expected,
    )
    # pair 1 is anchored at beta 7, pair 2 at beta 0.5: each pair's own beta scales its penalty
    per_pair = anchor * beta * F.relu(-(pc - rc))
    assert per_pair[1].item() == pytest.approx(0.6 * 7.0 * 0.5)
    assert per_pair[2].item() == pytest.approx(0.6 * 0.5 * 0.1)
    assert per_pair[0].item() == 0.0 and per_pair[3].item() == 0.0
    # a scalar beta scales every pair alike
    loss_s, dpo_loss_s, *_ = dpo_objective(pc, pr, rc, rr, **_kwargs(beta=4.0), chosen_anchor=anchor)
    assert torch.allclose(loss_s - dpo_loss_s, anchor * (4.0 * F.relu(-(pc - rc))).mean())


def test_anchor_gradient_is_zero_unless_the_chosen_half_is_below_the_reference():
    pc, pr, rc, rr = _inputs()
    pc_anchor = pc.clone().requires_grad_(True)
    pc_plain = pc.clone().requires_grad_(True)
    dpo_objective(pc_anchor, pr, rc, rr, **_kwargs(beta=5.0), chosen_anchor=2.0)[0].backward()
    dpo_objective(pc_plain, pr, rc, rr, **_kwargs(beta=5.0))[0].backward()
    extra = pc_anchor.grad - pc_plain.grad
    # pairs 0 and 3 (chosen_ratio >= 0): nothing; pairs 1 and 2: the chosen log-prob is pushed up
    assert extra[0].item() == 0.0 and extra[3].item() == 0.0
    assert extra[1].item() == pytest.approx(-2.0 * 5.0 / 4)
    assert extra[2].item() == pytest.approx(-2.0 * 5.0 / 4)
    # the rejected half never feels the anchor
    pr_anchor = pr.clone().requires_grad_(True)
    pr_plain = pr.clone().requires_grad_(True)
    dpo_objective(pc, pr_anchor, rc, rr, **_kwargs(beta=5.0), chosen_anchor=2.0)[0].backward()
    dpo_objective(pc, pr_plain, rc, rr, **_kwargs(beta=5.0))[0].backward()
    assert torch.equal(pr_anchor.grad, pr_plain.grad)


def test_anchor_is_unscaled_under_ipo_and_not_applied_under_kto():
    pc, pr, rc, rr = _inputs()
    beta = torch.tensor([3.0, 7.0, 0.5, 2.0])
    ipo = dpo_objective(
        pc, pr, rc, rr, **_kwargs(objective=DPOObjective.IPO, beta=beta, ipo_tau=3.0), chosen_anchor=2.0
    )
    assert torch.allclose(ipo[0] - ipo[1], 2.0 * F.relu(-(pc - rc)).mean())
    # KTO's desirable term already anchors the chosen half; the DPOP term is left out
    kto_kwargs = _kwargs(objective=DPOObjective.KTO, beta=beta)
    plain = dpo_objective(pc, pr, rc, rr, **kto_kwargs)
    anchored = dpo_objective(pc, pr, rc, rr, **kto_kwargs, chosen_anchor=2.0)
    for a, b in zip(plain, anchored, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(
        dpo_per_sample_losses(pc - rc, pr - rr, **kto_kwargs, chosen_anchor=2.0),
        dpo_per_sample_losses(pc - rc, pr - rr, **kto_kwargs),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        _kwargs(),
        _kwargs(label_smoothing=0.2),
        _kwargs(objective=DPOObjective.IPO),
        _kwargs(beta=torch.tensor([1.0, 2.0, 3.0, 4.0])),
        _kwargs(objective=DPOObjective.KTO),
    ],
)
def test_anchor_off_is_bit_for_bit_the_plain_objective(kwargs):
    pc, pr, rc, rr = _inputs()
    plain = dpo_objective(pc, pr, rc, rr, **kwargs)
    off = dpo_objective(pc, pr, rc, rr, **kwargs, chosen_anchor=0.0)
    for a, b in zip(plain, off, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(
        dpo_per_sample_losses(pc - rc, pr - rr, **kwargs),
        dpo_per_sample_losses(pc - rc, pr - rr, **kwargs, chosen_anchor=0.0),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {**_kwargs(), "chosen_anchor": 1.5},
        {**_kwargs(beta=torch.tensor([2.0, 5.0, 9.0, 0.5])), "chosen_anchor": 0.7},
        {**_kwargs(objective=DPOObjective.IPO, ipo_tau=3.0), "chosen_anchor": 2.0},
        {**_kwargs(label_smoothing=0.2), "chosen_anchor": 0.3},
    ],
)
def test_per_pair_losses_include_the_anchor_and_average_to_the_loss(kwargs):
    # What the gradient diagnostics' per-concept loss trend reads: each pair's
    # share of the objective's loss, anchor included.
    pc, pr, rc, rr = _inputs()
    loss, *_ = dpo_objective(pc, pr, rc, rr, **kwargs)
    per_pair = dpo_per_sample_losses(pc - rc, pr - rr, **kwargs)
    assert per_pair.shape == (4,)
    assert torch.allclose(per_pair.mean(), loss.detach())
    without = dpo_per_sample_losses(pc - rc, pr - rr, **{**kwargs, "chosen_anchor": 0.0})
    assert torch.all(per_pair >= without)
    assert per_pair[0] == without[0] and per_pair[1] > without[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {**_kwargs(), "chosen_anchor": 1.5},
        {**_kwargs(beta=torch.tensor([2.0, 5.0, 9.0])), "chosen_anchor": 0.7},
        {**_kwargs(objective=DPOObjective.IPO, ipo_tau=3.0), "chosen_anchor": 2.0},
        {**_kwargs(label_smoothing=0.2), "chosen_anchor": 0.3},
    ],
)
def test_surrogate_weights_carry_the_anchor_onto_the_chosen_half(kwargs):
    """The light path backwards each half with dL/dlogp; the anchor sits in
    ``loss``, so g_chosen includes its derivative and the split gradient equals
    the fused one."""
    torch.manual_seed(0)
    x_c, x_r = torch.randn(3, 4), torch.randn(3, 4)
    y_c, y_r = torch.randn(3), torch.randn(3)
    ref_c, ref_r = torch.randn(3), torch.randn(3)
    theta0 = torch.randn(4)

    theta = theta0.clone().requires_grad_(True)
    loss_fused, *_ = dpo_objective(_toy_logp(theta, x_c, y_c), _toy_logp(theta, x_r, y_r), ref_c, ref_r, **kwargs)
    loss_fused.backward()
    grad_fused = theta.grad.clone()

    theta = theta0.clone().requires_grad_(True)
    with torch.no_grad():
        pr_value = _toy_logp(theta, x_r, y_r)
    pc_live = _toy_logp(theta, x_c, y_c)
    g_c, g_r, outputs = dpo_surrogate_weights(pc_live, pr_value, ref_c, ref_r, **kwargs)
    (g_c * pc_live).sum().backward()
    pr_live = _toy_logp(theta, x_r, y_r)
    (g_r * pr_live).sum().backward()

    assert torch.allclose(theta.grad, grad_fused, atol=1e-6)
    assert torch.allclose(outputs[0], loss_fused.detach())
    # without the anchor the chosen weight differs wherever the anchor is live
    g_c_plain, _, _ = dpo_surrogate_weights(pc_live, pr_value, ref_c, ref_r, **{**kwargs, "chosen_anchor": 0.0})
    live = (pc_live.detach() - ref_c) < 0
    assert torch.all((g_c != g_c_plain) == live)


def test_concept_window_reports_the_anchor_active_fraction_only_with_the_anchor_on():
    info = DPOConceptInfo(label="RLHF_5", tag="RLHF_5", path="RLHF_5", chosen_dir="RLHF_5/chosen", beta_override=120.0)

    def record(path, chosen):
        return {"path": path, "margin": 0.01, "chosen": chosen, "rejected": chosen - 0.01, "beta": 120.0, "t": 0.5}

    window = DPOConceptWindow(with_anchor=True)
    for path, chosen in (("a", -0.02), ("b", 0.03), ("c", -0.01), ("d", 0.0)):
        window.add(info, record(path, chosen))
    assert window.scalars()["dpo/concept/RLHF_5/chosen_below_ref"] == pytest.approx(0.5)

    plain = DPOConceptWindow()
    plain.add(info, record("a", -0.02))
    assert not any(key.endswith("chosen_below_ref") for key in plain.scalars())
