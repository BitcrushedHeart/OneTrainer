"""bf16 stochastic rounding helpers: the add_stochastic_ alpha fix, explicit generators,
chunking, and the accumulation error bounds the OFT rotation cache relies on.

CPU-only. The accumulation fixture mimics dL/dR of an OFT rotation over a 70-micro-step
window: per-step sum over tokens of x^T g, heavy-tailed activation channels (including
x30 outliers), a per-micro-step loss scale drawn lognormally, arriving in bf16 and
already divided by the accumulation count.
"""

from __future__ import annotations

import math

from modules.util import bf16_stochastic_rounding as sr

import torch

import pytest


def _legacy_copy_stochastic_(target, source):
    """copy_stochastic_ before the generator argument, verbatim."""
    result = torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=sr.generator,
    )
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))


def _legacy_add_stochastic_(input, other, alpha=1.0):
    """add_stochastic_ before the alpha fix, verbatim."""
    result = other.clone() if other.dtype == torch.float32 else other.to(dtype=torch.float32)
    result.add_(input, alpha=alpha)
    _legacy_copy_stochastic_(input, result)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


# --------------------------------------------------------------------------- #
# alpha fix, CAME call patterns, generators
# --------------------------------------------------------------------------- #


def test_add_stochastic_scales_other_not_input():
    torch.manual_seed(0)
    a = torch.randn(4096).to(torch.bfloat16)
    o = torch.randn(4096)
    want = a.double() + 0.5 * o.double()

    fixed = a.clone()
    sr.add_stochastic_(fixed, o, alpha=0.5, generator=torch.Generator().manual_seed(1))
    buggy = a.clone()
    sr.set_seed(1, torch.device("cpu"))
    _legacy_add_stochastic_(buggy, o, alpha=0.5)

    assert _rel(fixed, want) < 1e-2
    assert _rel(buggy, want) > 0.3  # the old helper computed other + 0.5 * input


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_came_call_patterns_bit_identical_to_legacy(dtype):
    """CAME calls add_stochastic_(p, p, alpha=-wd*lr) and add_stochastic_(p, -update):
    both must round exactly as before the fix, from the same global generator state."""
    torch.manual_seed(2)
    p = torch.randn(3, 257).to(dtype)
    update = torch.randn(3, 257) * 1e-3

    new, old = p.clone(), p.clone()
    sr.set_seed(123, torch.device("cpu"))
    sr.add_stochastic_(new, new, alpha=-0.01 * 1e-3)
    sr.add_stochastic_(new, -update)
    sr.set_seed(123, torch.device("cpu"))
    _legacy_add_stochastic_(old, old, alpha=-0.01 * 1e-3)
    _legacy_add_stochastic_(old, -update)
    assert torch.equal(new, old)


def test_explicit_generator_leaves_global_generator_untouched():
    sr.set_seed(7, torch.device("cpu"))
    before = sr.generator.get_state().clone()
    acc = torch.zeros(64, 64, dtype=torch.bfloat16)
    own = torch.Generator().manual_seed(0x5EED0F7)
    for _ in range(5):
        sr.add_stochastic_(acc, torch.randn(64, 64) * 1e-3, generator=own, chunk_numel=512)
    assert torch.equal(sr.generator.get_state(), before)


def test_default_generator_is_the_global_one():
    x = torch.randn(1000) * 1e-3
    a, b = torch.zeros(1000, dtype=torch.bfloat16), torch.zeros(1000, dtype=torch.bfloat16)
    sr.set_seed(5, torch.device("cpu"))
    sr.copy_stochastic_(a, x)
    sr.set_seed(5, torch.device("cpu"))
    sr.copy_stochastic_(b, x, torch.Generator().manual_seed(999))
    c = torch.zeros(1000, dtype=torch.bfloat16)
    sr.set_seed(5, torch.device("cpu"))
    _legacy_copy_stochastic_(c, x)
    assert torch.equal(a, c)
    assert not torch.equal(a, b)


def test_chunked_add_matches_unchunked_draws_on_cpu_and_bounds_slices(monkeypatch):
    torch.manual_seed(3)
    acc = (torch.randn(10, 32, 32) * 1e-2).to(torch.bfloat16)
    grad = (torch.randn(10, 32, 32) * 1e-4).to(torch.bfloat16)
    full, chunked = acc.clone(), acc.clone()
    sr.add_stochastic_(full, grad, generator=torch.Generator().manual_seed(4))

    seen = []
    original = sr.copy_stochastic_
    monkeypatch.setattr(sr, "copy_stochastic_", lambda t, s, g=None: (seen.append(t.numel()), original(t, s, g)))
    sr.add_stochastic_(chunked, grad, generator=torch.Generator().manual_seed(4), chunk_numel=3 * 32 * 32)

    assert seen == [3 * 1024, 3 * 1024, 3 * 1024, 1024]
    # CPU draws are sequential, so slicing consumes the same random stream.
    assert torch.equal(full, chunked)


def test_chunk_smaller_than_a_row_still_processes_whole_rows(monkeypatch):
    acc = torch.zeros(4, 8, 8, dtype=torch.bfloat16)
    seen = []
    original = sr.copy_stochastic_
    monkeypatch.setattr(sr, "copy_stochastic_", lambda t, s, g=None: (seen.append(t.shape[0]), original(t, s, g)))
    sr.add_stochastic_(acc, torch.ones(4, 8, 8), generator=torch.Generator(), chunk_numel=10)
    assert seen == [1, 1, 1, 1]
    assert torch.equal(acc, torch.ones(4, 8, 8, dtype=torch.bfloat16))


# --------------------------------------------------------------------------- #
# accumulation error over a 70-micro-step window
# --------------------------------------------------------------------------- #

STEPS = 70
# Measured on this fixture: single-window SR 1.1-1.6%, nearest rounding ~1%,
# mean of 16 SR runs ~0.3-0.4%; minibatch noise between windows is 120-260%.
SR_SINGLE_WINDOW_BOUND = 3e-2


def _window_grads(seed: int, r: int = 4, b: int = 64, tokens: int = 128, sigma: float = 1.0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    chan = torch.exp(torch.randn(r, b, generator=g))
    chan[:, :3] *= 30.0
    out = []
    for _ in range(STEPS):
        scale = math.exp(sigma * torch.randn((), generator=g).item())
        x = torch.randn(r, tokens, b, generator=g) * chan[:, None, :]
        up = torch.randn(r, tokens, b, generator=g) * 1e-4 * scale
        out.append((torch.einsum("rtk,rtc->rkc", x, up) / STEPS).to(torch.bfloat16))
    return out


def _accumulate(grads, dtype, generator=None) -> torch.Tensor:
    acc = torch.zeros(grads[0].shape, dtype=dtype)
    for grad in grads:
        if generator is None:
            acc.add_(grad)
        else:
            sr.add_stochastic_(acc, grad, generator=generator, chunk_numel=2048)
    return acc


@pytest.mark.parametrize("sigma", [1.0, 2.5])
def test_sr_accumulation_error_is_bounded_and_unbiased(sigma):
    grads = _window_grads(11, sigma=sigma)
    exact = sum(grad.double() for grad in grads)

    assert _rel(_accumulate(grads, torch.float32), exact) <= 1e-6

    runs = [_accumulate(grads, torch.bfloat16, torch.Generator().manual_seed(100 + i)) for i in range(16)]
    singles = [_rel(run, exact) for run in runs]
    assert max(singles) <= SR_SINGLE_WINDOW_BOUND
    mean_of_runs = torch.stack([run.double() for run in runs]).mean(0)
    # Unbiased: averaging 16 independent rounding streams shrinks the error ~4x.
    assert _rel(mean_of_runs, exact) <= 0.5 * (sum(singles) / len(singles))


def test_sr_keeps_increments_below_half_ulp_that_nearest_rounding_drops():
    n = 20000
    increment = torch.full((n,), 0.003).to(torch.bfloat16)  # < half ulp of 1.0 in bf16 (0.0039)
    nearest = torch.ones(n, dtype=torch.bfloat16)
    stochastic = torch.ones(n, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(0)
    for _ in range(69):
        nearest.add_(increment)
        sr.add_stochastic_(stochastic, increment, generator=gen)
    exact = 1.0 + 69 * increment[0].double().item()

    assert nearest.double().mean().item() == 1.0  # every increment rounded away: -17%
    assert abs(stochastic.double().mean().item() - exact) / exact <= 5e-3
