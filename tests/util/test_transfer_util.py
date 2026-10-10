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
    add_distillation_offset,
    collect_late_weight_modules,
    high_frequency_ratio,
    is_late_family_key,
    parse_transfer_families,
    run_late_stack,
    run_prefix_and_capture,
    swap_module_weights,
    transfer_activation_loss,
    transfer_education_gaps,
    transfer_turbo_metrics,
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


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("warm_compile", [False, True])
def test_run_prefix_and_capture_compiled_block(checkpointed, warm_compile):
    """The capture trap must bypass Module.compile, including compiled checkpoints."""
    stack = _Stack(hidden=8, layers=4)
    if checkpointed:
        from modules.util.checkpointing_util import CheckpointLayer

        for i, block in enumerate(stack.layers):
            stack.layers[i] = CheckpointLayer(block, None, torch.device("cpu"))
    for block in stack.layers:
        block.compile(backend="eager", fullgraph=True)
    fork = stack.layers[2]
    compiled_call = fork._compiled_call_impl
    x = torch.randn(2, 8)

    with torch.no_grad():
        if warm_compile:
            stack(x, offset=0.5)
        args, kwargs = run_prefix_and_capture(lambda: stack(x, offset=0.5), fork)
        expected_prefix = stack.layers[1](stack.layers[0](x, offset=0.5), offset=0.5)
        torch.testing.assert_close(args[0], expected_prefix)
        assert kwargs == {"offset": 0.5}
        assert fork._compiled_call_impl is compiled_call
        assert "forward" not in fork.__dict__
        expected = stack(x, offset=0.5)

    outs = run_late_stack(list(stack.layers[2:]), args[0], args[1:], kwargs)
    torch.testing.assert_close(outs[-1], expected)
    outs[-1].square().mean().backward()
    assert all(p.grad is None for block in stack.layers[:2] for p in block.parameters())
    for block in stack.layers[2:]:
        original = block.checkpoint if checkpointed else block
        assert original.attention.to_q.weight.grad is not None
        assert original.feed_forward.w1.weight.grad is not None


@pytest.mark.parametrize("failure", ["missing_fork", "forward_error"])
def test_run_prefix_and_capture_restores_compiled_block_on_failure(failure):
    fork = _Block(8)
    original_forward = fork.forward
    fork.forward = original_forward  # Preserve an existing instance override too.
    fork.compile(backend="eager", fullgraph=True)
    compiled_call = fork._compiled_call_impl

    def call_forward():
        if failure == "forward_error":
            raise ValueError("prefix failed")

    error = ValueError if failure == "forward_error" else RuntimeError
    with pytest.raises(error, match="prefix failed|without reaching the fork"):
        run_prefix_and_capture(call_forward, fork)
    assert fork.forward is original_forward
    assert fork._compiled_call_impl is compiled_call


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


def test_stop_criterion_guard_blocks_best_and_counts_patience():
    """A guard-tripped validation never becomes the best, even with the lowest
    residual, counts against patience and cannot satisfy the target."""
    crit = TransferStopCriterion(patience=2, min_relative_improvement=0.005, target_residual=0.1)
    assert not crit.update(1.0, tag=0)
    assert not crit.update(0.05, tag=1, guard_ok=False)  # below target and best, but guarded
    assert crit.best_residual == pytest.approx(1.0)
    assert crit.best_tag == 0
    assert crit.counter == 1
    assert crit.update(0.04, tag=2, guard_ok=False)  # patience exhausted
    assert crit.best_tag == 0


def test_add_distillation_offset():
    teacher = [torch.full((2, 3, 4), 1.0, dtype=torch.bfloat16)]
    variant = [torch.full((2, 3, 4), 0.75)]
    base = [torch.full((2, 3, 4), 0.25)]
    (target,) = add_distillation_offset(teacher, variant, base)
    assert target.dtype == torch.float32
    assert torch.allclose(target, torch.full((2, 3, 4), 1.5))
    with pytest.raises(ValueError, match="blocks"):
        add_distillation_offset(teacher, variant, [])


def test_transfer_turbo_metrics_retention_and_split():
    torch.manual_seed(0)
    teacher = torch.randn(2, 5, 8)
    base = torch.randn(2, 5, 8)
    direction = torch.randn(2, 5, 8)
    variant = base + direction
    # A component orthogonal to the distillation direction: pure education error.
    noise = torch.randn(2, 5, 8)
    perp = noise - (noise * direction).sum() / direction.pow(2).sum() * direction

    def run(student):
        return transfer_turbo_metrics([student], [teacher], [variant], [base], block_index_offset=22)

    full = run(teacher + direction)
    assert full["turbo_retention"] == pytest.approx(1.0)
    assert full["turbo_retention_block_22"] == pytest.approx(1.0)
    assert full["education_residual"] == pytest.approx(0.0, abs=1e-9)
    assert full["turbo_gap"] == pytest.approx(float(direction.pow(2).sum() / teacher.pow(2).sum()), rel=1e-5)
    assert full["teacher_residual"] == pytest.approx(full["turbo_gap"], rel=1e-5)

    collapsed = run(teacher.clone())
    assert collapsed["turbo_retention"] == pytest.approx(0.0, abs=1e-9)
    assert collapsed["teacher_residual"] == pytest.approx(0.0, abs=1e-9)

    half = run(teacher + 0.5 * direction + perp)
    assert half["turbo_retention"] == pytest.approx(0.5, abs=1e-5)
    assert half["education_residual"] == pytest.approx(float(perp.pow(2).sum() / teacher.pow(2).sum()), rel=1e-4)


def test_transfer_turbo_metrics_mask_and_block_mean():
    teacher = torch.ones(1, 2, 4)
    base = torch.zeros(1, 2, 4)
    variant = torch.ones(1, 2, 4)
    student = teacher.clone()
    student[:, 0] += 1.0  # token 0 keeps the full offset, token 1 none
    mask = torch.tensor([[True, False]])
    masked = transfer_turbo_metrics([student], [teacher], [variant], [base], mask=mask)
    assert masked["turbo_retention"] == pytest.approx(1.0)
    unmasked = transfer_turbo_metrics([student], [teacher], [variant], [base])
    assert unmasked["turbo_retention"] == pytest.approx(0.5)

    two = transfer_turbo_metrics([student, teacher], [teacher, teacher], [variant, variant], [base, base], mask=mask)
    assert two["turbo_retention_block_0"] == pytest.approx(1.0)
    assert two["turbo_retention_block_1"] == pytest.approx(0.0, abs=1e-9)
    assert two["turbo_retention"] == pytest.approx(0.5)

    # No distillation offset at all (variant == base): finite, zero retention.
    flat = transfer_turbo_metrics([student], [teacher], [base], [base])
    assert flat["turbo_retention"] == 0.0 and flat["turbo_gap"] == 0.0


def test_high_frequency_ratio():
    torch.manual_seed(0)
    sharp = torch.randn(2, 4, 16, 16)
    blurred = torch.nn.functional.avg_pool2d(sharp, kernel_size=3, stride=1, padding=1)
    assert high_frequency_ratio(sharp, sharp) == pytest.approx(1.0)
    assert high_frequency_ratio(blurred, sharp) < 0.5
    assert high_frequency_ratio(sharp, blurred) > 2.0


def test_education_gap_and_missing_fraction():
    """education_missing reads the education residual as a fraction of the
    education's own effect (teacher minus raw Base), per block."""
    torch.manual_seed(0)
    base = torch.randn(2, 5, 8)
    education = 0.1 * torch.randn(2, 5, 8)
    teacher = base + education
    direction = torch.randn(2, 5, 8)
    variant = base + direction
    # Orthogonal to the distillation direction, so it all counts as education error.
    perp = education - (education * direction).sum() / direction.pow(2).sum() * direction

    (gap,) = transfer_education_gaps([teacher], [base])
    assert gap == pytest.approx(float(education.pow(2).sum() / teacher.pow(2).sum()), rel=1e-5)

    def run(student):
        return transfer_turbo_metrics([student], [teacher], [variant], [base], education_gaps=[gap])

    exact = run(teacher + direction)
    assert exact["education_gap"] == pytest.approx(gap)
    assert exact["education_missing"] == pytest.approx(0.0, abs=1e-6)

    half = run(teacher + direction - 0.5 * perp)
    assert half["education_missing"] == pytest.approx(
        0.25 * float(perp.pow(2).sum() / education.pow(2).sum()), rel=1e-3
    )

    assert "education_missing" not in transfer_turbo_metrics([teacher], [teacher], [variant], [base])
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 0] = True
    (masked,) = transfer_education_gaps([teacher], [base], mask=mask)
    assert masked == pytest.approx(float(education[:, 0].pow(2).sum() / teacher[:, 0].pow(2).sum()), rel=1e-5)
    with pytest.raises(ValueError, match="education gaps"):
        transfer_turbo_metrics([teacher], [teacher], [variant], [base], education_gaps=[gap, gap])
