"""Unit tests for modules/util/transfer_util.py (pure/CPU pieces).

Covers family parsing/matching, late-module collection (including the
``.checkpoint.`` name normalization gradient checkpointing introduces), the
weight-swap context manager's exact-restore guarantee, the forward trap, the
late-stack runner, the per-block activation loss, and the stop criterion's
state machine. The full fork-teacher-student step is exercised in
tests/modelSetup/test_transfer_loss_step.py.
"""

from modules.util.transfer_util import (
    TransferStopCriterion,
    collect_late_weight_modules,
    is_late_family_key,
    parse_transfer_families,
    run_late_stack,
    run_prefix_and_capture,
    swap_module_weights,
    transfer_activation_loss,
)

import torch
from torch import nn

import pytest


def test_parse_transfer_families():
    fams = parse_transfer_families("qkv,mlp")
    assert set(fams) == {"qkv", "mlp"}
    assert "attention.to_q" in fams["qkv"]
    assert "feed_forward.w2" in fams["mlp"]
    assert set(parse_transfer_families("qkv")) == {"qkv"}
    with pytest.raises(ValueError):
        parse_transfer_families("qkv,nonsense")
    with pytest.raises(ValueError):
        parse_transfer_families("")


def test_is_late_family_key():
    fams = parse_transfer_families("qkv,mlp")
    assert is_late_family_key("layers.22.attention.to_q", 22, fams) == "qkv"
    assert is_late_family_key("transformer.layers.29.feed_forward.w2", 22, fams) == "mlp"
    assert is_late_family_key("layers.21.attention.to_q", 22, fams) is None  # early
    assert is_late_family_key("layers.22.attention.to_out.0", 22, fams) is None  # not a family
    assert is_late_family_key("noise_refiner.0.attention.to_q", 22, fams) is None
    # gradient checkpointing inserts a wrapper level into module names
    assert is_late_family_key("layers.22.checkpoint.attention.to_q", 22, fams) == "qkv"
    # qkv-only selection must not match mlp stems
    qkv_only = parse_transfer_families("qkv")
    assert is_late_family_key("layers.22.feed_forward.w1", 22, qkv_only) is None


class _Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attention = nn.Module()
        self.attention.to_q = nn.Linear(hidden, hidden, bias=False)
        self.attention.to_out = nn.ModuleList([nn.Linear(hidden, hidden, bias=False)])
        self.feed_forward = nn.Module()
        self.feed_forward.w1 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, offset=0.0):
        return self.attention.to_q(x) + self.feed_forward.w1(x) + offset


class _Stack(nn.Module):
    def __init__(self, hidden: int = 8, layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([_Block(hidden) for _ in range(layers)])

    def forward(self, x, offset=0.0):
        for layer in self.layers:
            x = layer(x, offset=offset)
        return x


def test_collect_late_weight_modules():
    fams = parse_transfer_families("qkv,mlp")
    stack = _Stack(layers=4)
    late = collect_late_weight_modules(stack, 2, fams)
    assert set(late) == {
        "layers.2.attention.to_q",
        "layers.2.feed_forward.w1",
        "layers.3.attention.to_q",
        "layers.3.feed_forward.w1",
    }
    assert all(isinstance(m, nn.Linear) for m in late.values())
    # to_out.0 is a Linear but not a family member
    assert not any("to_out" in k for k in late)


def test_swap_module_weights_restores_exactly():
    fams = parse_transfer_families("qkv,mlp")
    stack = _Stack(layers=3)
    late = collect_late_weight_modules(stack, 1, fams)
    snapshot = {k: m.weight.data.clone() for k, m in late.items()}
    replacements = {k: torch.full_like(m.weight.data, 7.0) for k, m in late.items()}

    with swap_module_weights(late, replacements):
        for k, m in late.items():
            assert torch.equal(m.weight.data, replacements[k])
    for k, m in late.items():
        assert torch.equal(m.weight.data, snapshot[k])

    # Restore must also run when the body raises.
    with pytest.raises(RuntimeError, match="boom"), swap_module_weights(late, replacements):
        raise RuntimeError("boom")
    for k, m in late.items():
        assert torch.equal(m.weight.data, snapshot[k])


def test_swap_module_weights_missing_key_raises_and_restores():
    fams = parse_transfer_families("qkv,mlp")
    stack = _Stack(layers=2)
    late = collect_late_weight_modules(stack, 0, fams)
    snapshot = {k: m.weight.data.clone() for k, m in late.items()}
    partial = {next(iter(late)): torch.zeros_like(next(iter(late.values())).weight)}

    with pytest.raises(KeyError), swap_module_weights(late, partial):
        pass  # pragma: no cover - swap raises before the body runs
    for k, m in late.items():
        assert torch.equal(m.weight.data, snapshot[k])


def test_run_prefix_and_capture_and_late_stack():
    torch.manual_seed(0)
    stack = _Stack(hidden=8, layers=4)
    x = torch.randn(2, 8)
    fork_index = 2

    args, kwargs = run_prefix_and_capture(lambda: stack(x, offset=0.5), stack.layers[fork_index])
    # The trap must restore the block's forward
    full = stack(x, offset=0.5)

    outs = run_late_stack(list(stack.layers[fork_index:]), args[0], args[1:], kwargs)
    assert len(outs) == 2
    torch.testing.assert_close(outs[-1], full, rtol=1e-6, atol=1e-7)

    # kwargs travel through the trap
    assert kwargs.get("offset") == 0.5


def test_run_prefix_and_capture_requires_fork_hit():
    stack = _Stack(layers=2)
    other = _Block(8)
    with pytest.raises(RuntimeError, match="fork"):
        run_prefix_and_capture(lambda: stack(torch.randn(2, 8)), other)


def test_transfer_activation_loss_values_and_metrics():
    student = [torch.full((2, 3, 4), 2.0), torch.full((2, 3, 4), 1.0)]
    teacher = [torch.full((2, 3, 4), 1.0), torch.full((2, 3, 4), 1.0)]

    loss, metrics = transfer_activation_loss(student, teacher, normalize=True)
    # block 0: mse 1.0 / mean(t^2) 1.0 = 1.0; block 1: 0.0 -> mean 0.5
    assert loss.dtype == torch.float32
    assert float(loss) == pytest.approx(0.5)
    assert metrics["residual"] == pytest.approx(0.5)
    assert metrics["block_0"] == pytest.approx(1.0)
    assert metrics["block_1"] == pytest.approx(0.0)

    loss_raw, _ = transfer_activation_loss(student, teacher, normalize=False)
    assert float(loss_raw) == pytest.approx(0.5)  # teacher RMS^2 is 1 here

    # Masking: zero out the mismatching tokens -> loss 0
    mask = torch.zeros(2, 3, dtype=torch.bool)
    loss_masked, metrics_masked = transfer_activation_loss([teacher[0] + torch.randn(2, 3, 4)], [teacher[0]], mask=mask)
    assert float(loss_masked) == pytest.approx(0.0)
    assert metrics_masked["residual"] == pytest.approx(0.0)

    # Partial mask: only masked-in tokens count
    student_p = teacher[0].clone()
    student_p[:, 0] += 1.0  # perturb token 0 only
    mask = torch.ones(2, 3, dtype=torch.bool)
    mask[:, 0] = False
    loss_partial, _ = transfer_activation_loss([student_p], [teacher[0]], mask=mask)
    assert float(loss_partial) == pytest.approx(0.0)


def test_transfer_activation_loss_grad_flows():
    student = [torch.randn(2, 3, 4, requires_grad=True)]
    teacher = [torch.randn(2, 3, 4)]
    loss, _ = transfer_activation_loss(student, teacher)
    loss.backward()
    assert student[0].grad is not None
    assert float(student[0].grad.abs().sum()) > 0.0


def test_stop_criterion_patience_and_best():
    crit = TransferStopCriterion(patience=3, min_relative_improvement=0.005)
    series = [1.0, 0.8, 0.7, 0.75, 0.72, 0.74, 0.76]
    stops = [crit.update(v, tag=i) for i, v in enumerate(series)]
    assert stops == [False, False, False, False, False, True, True]
    assert crit.best_residual == pytest.approx(0.7)
    assert crit.best_tag == 2


def test_stop_criterion_min_delta_and_numeric_best():
    """Sub-threshold numeric improvements still update the best snapshot but do
    not reset patience -- the final adapter is the numeric best, while the run
    stops when meaningful progress dries up."""
    crit = TransferStopCriterion(patience=3, min_relative_improvement=0.01)
    series = [1.0, 0.999, 0.998, 0.997]
    stops = [crit.update(v, tag=i) for i, v in enumerate(series)]
    assert stops == [False, False, False, True]
    assert crit.best_residual == pytest.approx(0.997)
    assert crit.best_tag == 3


def test_stop_criterion_reset_on_improvement():
    crit = TransferStopCriterion(patience=2, min_relative_improvement=0.005)
    assert not crit.update(1.0)
    assert not crit.update(1.1)  # counter 1
    assert not crit.update(0.5)  # meaningful -> reset
    assert not crit.update(0.51)  # counter 1
    assert crit.update(0.52)  # counter 2 -> stop


def test_stop_criterion_target_residual():
    crit = TransferStopCriterion(patience=0, min_relative_improvement=0.005, target_residual=0.5)
    assert not crit.update(0.6)
    assert crit.update(0.5)  # at/below target -> stop
    # patience=0 means patience never fires on its own
    crit2 = TransferStopCriterion(patience=0, min_relative_improvement=0.005)
    assert not any(crit2.update(1.0 + i) for i in range(10))
