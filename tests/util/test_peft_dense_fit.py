"""Tests for peft_dense_fit -- dfit warm starts for LoRA / DoRA / LoKr adapters.

The merged-form builders are validated against the REAL training modules
(LoRAModule / DoRAModule / LoKrModule forwards with an identity input batch:
``F.linear(I, W) == W^T``), so any drift in the modules' math fails here
instead of silently corrupting a bake. Fit tests check exact recovery of
representable targets, ALS monotonicity, the dora_scale write-back invariant,
and the keep-best guarantee of the generic refinement.
"""

from modules.module.LoRAModule import DoRAModule, LoKrModule, LoRAModule
from modules.util.peft_dense_fit import (
    LoKrStemParams,
    LoRAStemParams,
    dense_warmstart_for_peft_tensor,
    detect_adapter_kind,
    dora_scale_from_axis_scale,
    fit_student_dora,
    fit_student_lokr,
    fit_student_lora,
    lokr_params_from_tensors,
    lora_params_from_tensors,
    merged_lokr,
    merged_lora,
    refine_peft_fit,
)

import torch
from torch import nn

import pytest

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Detection


def _keys(*suffixes):
    return [f"transformer.layers.30.attention.to_q.{s}" for s in suffixes]


def test_detect_all_kinds():
    assert detect_adapter_kind(_keys("oft_R.weight")) == "oft"
    assert detect_adapter_kind(_keys("oft_R.weight", "dora_multiplier")) == "dora_oft"
    assert detect_adapter_kind(_keys("lora_down.weight", "lora_up.weight", "alpha")) == "lora"
    assert detect_adapter_kind(_keys("lora_down.weight", "lora_up.weight", "alpha", "dora_scale")) == "dora"
    assert detect_adapter_kind(_keys("lokr_w1", "lokr_w2_a", "lokr_w2_b", "alpha")) == "lokr"
    assert detect_adapter_kind(_keys("lokr_w1_a", "lokr_w1_b", "lokr_w2", "alpha", "dora_scale")) == "dora_lokr"


def test_detect_rejects_loha_mixed_empty():
    with pytest.raises(NotImplementedError):
        detect_adapter_kind(_keys("hada_w1_a", "hada_w1_b"))
    with pytest.raises(ValueError):
        detect_adapter_kind(_keys("oft_R.weight", "lora_down.weight"))
    with pytest.raises(ValueError):
        detect_adapter_kind(_keys("alpha"))


# ---------------------------------------------------------------------------
# Merged forms vs the real modules


def _module_merged_weight(module, orig: nn.Linear) -> torch.Tensor:
    """The deployed weight the hooked module actually computes:
    forward(I_in) == I @ W^T == W^T."""
    module.hook_to_module()
    try:
        with torch.no_grad():
            out = orig(torch.eye(orig.in_features))
    finally:
        module.remove_hook_from_module()
    return out.transpose(0, 1)


def test_merged_lora_matches_module():
    torch.manual_seed(0)
    orig = nn.Linear(8, 6, bias=False)
    module = LoRAModule("p", orig, rank=3, alpha=1.5)
    with torch.no_grad():
        module.lora_down.weight.copy_(torch.randn(3, 8))
        module.lora_up.weight.copy_(torch.randn(6, 3))

    p = LoRAStemParams(down=module.lora_down.weight.detach(), up=module.lora_up.weight.detach(), alpha=1.5)
    ours = merged_lora(orig.weight.detach(), p)
    theirs = _module_merged_weight(module, orig)
    assert float((ours - theirs).norm()) < 1e-5


@pytest.mark.parametrize("output_axis", [False, True])
def test_merged_dora_matches_module(output_axis):
    torch.manual_seed(1)
    orig = nn.Linear(8, 6, bias=False)
    module = DoRAModule(
        "p", orig, 3, 2.0, norm_epsilon=True, decompose_output_axis=output_axis, train_device=torch.device("cpu")
    )
    with torch.no_grad():
        module.lora_down.weight.copy_(torch.randn(3, 8))
        module.lora_up.weight.copy_(torch.randn(6, 3))
        module.dora_scale.mul_(1.0 + 0.2 * torch.rand_like(module.dora_scale))

    p = LoRAStemParams(
        down=module.lora_down.weight.detach(),
        up=module.lora_up.weight.detach(),
        alpha=2.0,
        dora_scale=module.dora_scale.detach().reshape(-1, 1)
        if output_axis
        else module.dora_scale.detach().reshape(1, -1),
        norm_eps=True,
    )
    assert p.output_axis == output_axis
    ours = merged_lora(orig.weight.detach(), p)
    theirs = _module_merged_weight(module, orig)
    assert float((ours - theirs).norm()) < 1e-5


@pytest.mark.parametrize("weight_decompose", [False, True])
def test_merged_lokr_matches_module(weight_decompose):
    torch.manual_seed(2)
    orig = nn.Linear(64, 36, bias=False)  # factorizes 64 -> (8, 8), 36 -> (6, 6)
    module = LoKrModule(
        "p",
        orig,
        dim=2,
        alpha=1.0,
        decompose_both=True,
        decompose_factor=-1,
        use_tucker=False,
        weight_decompose=weight_decompose,
        dora_on_output=False,
        full_matrix=False,
        train_device=torch.device("cpu"),
    )
    with torch.no_grad():
        for name in ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b"):
            t = getattr(module, name, None)
            if t is not None:
                t.copy_(0.1 * torch.randn_like(t))
        if weight_decompose:
            module.dora_scale.mul_(1.0 + 0.2 * torch.rand_like(module.dora_scale))

    def opt(name):
        t = getattr(module, name, None)
        return t.detach() if t is not None else None

    p = LoKrStemParams(
        alpha=float(module.alpha),
        w1=opt("lokr_w1"),
        w1_a=opt("lokr_w1_a"),
        w1_b=opt("lokr_w1_b"),
        w2=opt("lokr_w2"),
        w2_a=opt("lokr_w2_a"),
        w2_b=opt("lokr_w2_b"),
        dora_scale=module.dora_scale.detach().reshape(1, -1) if weight_decompose else None,
        norm_eps=True,
    )
    ours = merged_lokr(orig.weight.detach(), p)
    theirs = _module_merged_weight(module, orig)
    assert float((ours - theirs).norm()) < 1e-4


# ---------------------------------------------------------------------------
# Student fits


def test_fit_student_lora_exact_on_representable_target():
    gen = torch.Generator().manual_seed(3)
    w_v = torch.randn(10, 12, generator=gen, dtype=torch.float64)
    p_true = LoRAStemParams(
        down=torch.randn(4, 12, generator=gen, dtype=torch.float64),
        up=torch.randn(10, 4, generator=gen, dtype=torch.float64),
        alpha=2.0,
    )
    target = merged_lora(w_v, p_true)
    fit = fit_student_lora(w_v, target, rank=4, scale=p_true.scale)
    err = float((merged_lora(w_v, LoRAStemParams(down=fit.down, up=fit.up, alpha=2.0)) - target).norm())
    assert err < 1e-9


@pytest.mark.parametrize("output_axis", [False, True])
def test_fit_student_dora_recovers_and_is_monotone(output_axis):
    gen = torch.Generator().manual_seed(4)
    w_v = torch.randn(10, 12, generator=gen, dtype=torch.float64)
    down = torch.randn(3, 12, generator=gen, dtype=torch.float64)
    up = 0.1 * torch.randn(10, 3, generator=gen, dtype=torch.float64)
    c_true = 1.0 + 0.1 * torch.randn(10 if output_axis else 12, generator=gen, dtype=torch.float64)
    wp = w_v + 0.5 * (up @ down)
    target = wp * (c_true.view(-1, 1) if output_axis else c_true.view(1, -1))

    fit = fit_student_dora(w_v, target, rank=3, scale=0.5, output_axis=output_axis, iters=8)
    hist = fit.objective_history
    assert all(b <= a + 1e-12 for a, b in zip(hist, hist[1:], strict=False))
    # ALS is a stationary-point method, not exact recovery; the closed-form
    # scale init must land it well under 1% of the target here (the ones-init
    # stalled ~100x higher). refine_peft_fit closes the rest in production.
    assert hist[-1] < 1e-2 * float(target.norm())

    # write-back invariant: the stored dora_scale reproduces the effective
    # axis scale EXACTLY through the module's decode (independent of how close
    # the fit itself got)
    wp_fit = w_v + 0.5 * (fit.up @ fit.down)
    scale_t = dora_scale_from_axis_scale(fit.axis_scale, wp_fit, output_axis, norm_eps=True)
    p_new = LoRAStemParams(down=fit.down, up=fit.up, alpha=0.5 * 3, dora_scale=scale_t, norm_eps=True)
    effective = wp_fit * (fit.axis_scale.view(-1, 1) if output_axis else fit.axis_scale.view(1, -1))
    err = float((merged_lora(w_v, p_new) - effective).norm())
    assert err < 1e-9 * float(target.norm())


def test_fit_student_lokr_exact_full_factors():
    gen = torch.Generator().manual_seed(5)
    w_v = torch.randn(6, 8, generator=gen, dtype=torch.float64)
    p_true = LoKrStemParams(
        alpha=2.0,  # both factors full: scale is 1 regardless of alpha
        w1=torch.randn(2, 2, generator=gen, dtype=torch.float64),
        w2=torch.randn(3, 4, generator=gen, dtype=torch.float64),
    )
    target = merged_lokr(w_v, p_true)
    fit = fit_student_lokr(w_v, target, p_true, iters=4)
    p_new = LoKrStemParams(alpha=2.0, w1=fit.w1, w2=fit.w2)
    err = float((merged_lokr(w_v, p_new) - target).norm())
    assert err < 1e-9


def test_refine_keep_best_never_regresses():
    gen = torch.Generator().manual_seed(6)
    w_v = torch.randn(6, 8, generator=gen, dtype=torch.float64)
    target = torch.randn(6, 8, generator=gen, dtype=torch.float64)  # unreachable
    p = LoRAStemParams(
        down=torch.randn(2, 8, generator=gen, dtype=torch.float64),
        up=torch.randn(6, 2, generator=gen, dtype=torch.float64),
        alpha=2.0,
    )
    tensors = {"lora_down.weight": p.down, "lora_up.weight": p.up}

    def build(leaves):
        q = LoRAStemParams(down=leaves["lora_down.weight"], up=leaves["lora_up.weight"], alpha=2.0)
        return merged_lora(w_v, q)

    res = refine_peft_fit(tensors, build, target, iters=20)
    manual = float((build(dict(tensors)) - target).norm())
    assert abs(res.err_init - manual) < 1e-9
    assert res.err_final <= res.err_init + 1e-12
    out_err = float((build(res.tensors) - target).norm())
    assert abs(res.err_final - out_err) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end warmstart


def _rotation(n: int, angle: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    skew = torch.randn(n, n, generator=gen, dtype=torch.float64)
    skew = skew - skew.transpose(0, 1)
    return torch.matrix_exp(angle * skew / skew.norm())


@pytest.mark.parametrize("kind", ["lora", "dora", "lokr", "dora_lokr"])
def test_dense_warmstart_recovers_transported_education(kind):
    """Variant = exact dense rotation of Base; the fitted adapter's deployed
    education on the variant must approximate the transported education."""
    gen = torch.Generator().manual_seed(7)
    out_f, in_f = 6, 8
    w_b = torch.randn(out_f, in_f, generator=gen, dtype=torch.float64)
    w_v = _rotation(out_f, 0.3, 8) @ w_b @ _rotation(in_f, 0.3, 9)

    if kind in ("lora", "dora"):
        p = LoRAStemParams(
            down=0.05 * torch.randn(3, in_f, generator=gen, dtype=torch.float64),
            up=0.05 * torch.randn(out_f, 3, generator=gen, dtype=torch.float64),
            alpha=3.0,
            dora_scale=(1.0 + 0.02 * torch.randn(1, in_f, generator=gen, dtype=torch.float64))
            if kind == "dora"
            else None,
        )
    else:
        p = LoKrStemParams(
            alpha=1.0,
            w1=0.2 * torch.randn(2, 2, generator=gen, dtype=torch.float64),
            w2_a=0.2 * torch.randn(3, 1, generator=gen, dtype=torch.float64),
            w2_b=0.2 * torch.randn(1, 4, generator=gen, dtype=torch.float64),
            dora_scale=(1.0 + 0.02 * torch.randn(out_f, 1, generator=gen, dtype=torch.float64))
            if kind == "dora_lokr"
            else None,
        )

    ws = dense_warmstart_for_peft_tensor(kind, p, w_b, w_v, refine_iters=40)
    education = float((ws.target - w_v).norm())
    assert education > 0.0
    assert ws.deployed_err <= ws.deployed_err_pre_refine + 1e-12
    # the deployed form can express the bulk of the transported education
    assert ws.deployed_err < 0.35 * education
    if kind == "lora":
        assert ws.refine is None  # SVD is already optimal; refinement skipped
        assert ws.deployed_err < 1e-9 * max(education, 1.0) + 1e-9


def test_params_from_tensors_roundtrip():
    prefix = "transformer.layers.30.attention.to_q."
    gen = torch.Generator().manual_seed(10)
    tensors = {
        prefix + "lora_down.weight": torch.randn(3, 8, generator=gen),
        prefix + "lora_up.weight": torch.randn(6, 3, generator=gen),
        prefix + "alpha": torch.tensor(2.0),
        prefix + "dora_scale": torch.randn(1, 8, generator=gen),
    }
    p = lora_params_from_tensors(tensors, prefix)
    assert p.rank == 3 and p.alpha == 2.0 and not p.output_axis

    tensors_k = {
        prefix + "lokr_w1": torch.randn(2, 2, generator=gen),
        prefix + "lokr_w2_a": torch.randn(3, 2, generator=gen),
        prefix + "lokr_w2_b": torch.randn(2, 4, generator=gen),
        prefix + "alpha": torch.tensor(2.0),
    }
    k = lokr_params_from_tensors(tensors_k, prefix)
    assert k.dim == 2 and k.scale == 1.0 and k.w1 is not None and k.w2 is None
