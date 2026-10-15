"""The INDEPENDENT add-on schedule: exact per-kind counts, even interleave,
determinism, resume bookkeeping, GA-window mixing, and the loader winding."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

from modules.util.rlhf_schedule_util import (
    KIND_ORDER,
    IndependentSchedule,
    MicroStepKind,
    interleave,
    wind_data_loader,
)

import pytest

SFT, DPO, KTO, TEXT = MicroStepKind.SFT, MicroStepKind.DPO, MicroStepKind.KTO, MicroStepKind.TEXT

#: The live run that motivated the schedule (batch 15, GA 2).
LIVE_COUNTS = {SFT: 4857, DPO: 5306, TEXT: 2565}


def _kinds(schedule: IndependentSchedule) -> list[MicroStepKind]:
    return list(schedule.kinds_from(0))


def test_every_kind_appears_exactly_its_count_times():
    schedule = IndependentSchedule(LIVE_COUNTS)
    assert len(schedule) == 4857 + 5306 + 2565 == 12728
    assert Counter(_kinds(schedule)) == LIVE_COUNTS
    assert schedule.optimizer_steps(2) == 6364
    assert schedule.counts == LIVE_COUNTS


def test_zero_and_missing_kinds_are_left_out():
    schedule = IndependentSchedule({SFT: 3, DPO: 0, TEXT: 2})
    assert schedule.counts == {SFT: 3, TEXT: 2}
    assert Counter(_kinds(schedule)) == {SFT: 3, TEXT: 2}
    assert len(IndependentSchedule({})) == 0


def test_single_kind_is_just_that_kind():
    assert _kinds(IndependentSchedule({SFT: 4})) == [SFT] * 4


def test_interleave_is_deterministic_and_order_independent():
    a = interleave({SFT: 4857, DPO: 5306, TEXT: 2565})
    b = interleave({TEXT: 2565, SFT: 4857, DPO: 5306})
    assert a == b
    assert list(a) == list(interleave(LIVE_COUNTS))


@pytest.mark.parametrize(
    "counts",
    [
        LIVE_COUNTS,
        {SFT: 100, DPO: 10},
        {SFT: 10, DPO: 100},
        {SFT: 7, DPO: 7, KTO: 7, TEXT: 7},
        {SFT: 1000, DPO: 1, TEXT: 3},
        {SFT: 96000, DPO: 20000, TEXT: 1000},
    ],
)
def test_each_kind_is_spread_evenly(counts):
    """No clumping: the gap between consecutive micro-steps of one kind never
    strays further than a handful of positions from ``total / count``; the
    merge can only displace an entry by the number of other kinds that share
    its neighbourhood."""
    schedule = IndependentSchedule(counts)
    kinds = _kinds(schedule)
    total = len(kinds)
    slack = len(counts)
    for kind, count in counts.items():
        positions = [i for i, k in enumerate(kinds) if k is kind]
        assert len(positions) == count
        spacing = total / count
        gaps = [b - a for a, b in zip(positions, positions[1:], strict=False)]
        assert all(abs(gap - spacing) <= slack for gap in gaps), (kind, min(gaps), max(gaps), spacing)
        # the first / last occurrence sit near the ends, not bunched at one
        assert positions[0] <= spacing + slack
        assert positions[-1] >= total - spacing - slack


def test_ga_windows_mix_kinds_as_far_as_the_shares_allow():
    """With GA 2 on the live counts, SFT holds 38 % of the micro-steps, so at
    most 76 % of the windows can contain one; an even spread gets there. No
    window is ever two text rollouts (20 % share), and the per-window kind mix
    never repeats a fixed pattern aligned with the window size."""
    schedule = IndependentSchedule(LIVE_COUNTS)
    kinds = _kinds(schedule)
    windows = [tuple(kinds[i : i + 2]) for i in range(0, len(kinds) - 1, 2)]
    with_sft = sum(1 for w in windows if SFT in w)
    share = LIVE_COUNTS[SFT] / len(kinds)
    assert with_sft / len(windows) >= min(1.0, 2 * share) - 0.02
    assert (TEXT, TEXT) not in windows
    assert (SFT, SFT) not in windows
    # Every pair of kinds that can share a window does, many times.
    mixes = Counter(frozenset(w) for w in windows)
    assert mixes[frozenset({SFT, DPO})] > 1000
    assert mixes[frozenset({SFT, TEXT})] > 500
    assert mixes[frozenset({DPO, TEXT})] > 500


def test_no_long_runs_of_the_same_kind():
    schedule = IndependentSchedule(LIVE_COUNTS)
    kinds = _kinds(schedule)
    longest = run = 1
    for a, b in zip(kinds, kinds[1:], strict=False):
        run = run + 1 if a is b else 1
        longest = max(longest, run)
    # DPO holds 42 %: an even spread never puts three in a row.
    assert longest <= 2


def test_ties_follow_the_fixed_kind_order():
    # Equal counts tie at every position; the merge order is SFT, DPO, KTO, TEXT.
    schedule = IndependentSchedule({TEXT: 2, KTO: 2, DPO: 2, SFT: 2})
    assert _kinds(schedule) == list(KIND_ORDER) * 2


def test_consumed_before_counts_each_kind_up_to_the_position():
    schedule = IndependentSchedule(LIVE_COUNTS)
    kinds = _kinds(schedule)
    for position in (0, 1, 2, 1000, 6363, 6364, 12727, 12728):
        expected = Counter(kinds[:position])
        assert schedule.consumed_before(position) == {kind: expected.get(kind, 0) for kind in LIVE_COUNTS}
    # a mid-epoch resume: the first 1000 micro-steps hold each kind in proportion
    consumed = schedule.consumed_before(1000)
    for kind, count in LIVE_COUNTS.items():
        assert abs(consumed[kind] - 1000 * count / len(schedule)) <= 2


def test_resume_position_is_clamped_to_the_epoch():
    schedule = IndependentSchedule({SFT: 5, DPO: 3})
    assert schedule.clamp(-1) == 0
    assert schedule.clamp(4) == 4
    assert schedule.clamp(8) == 8
    assert schedule.clamp(50) == 8  # a PIGGYBACK backup past this epoch's end
    assert list(schedule.kinds_from(50)) == []
    assert schedule.consumed_before(50) == {SFT: 5, DPO: 3}


def test_kinds_from_continues_the_same_sequence():
    schedule = IndependentSchedule(LIVE_COUNTS)
    kinds = _kinds(schedule)
    assert list(schedule.kinds_from(5000)) == kinds[5000:]
    assert schedule.kind_at(5000) is kinds[5000]


def test_describe_reports_counts_and_optimizer_steps():
    text = IndependentSchedule(LIVE_COUNTS).describe(2)
    assert text == (
        "4857 standard + 5306 DPO + 2565 text reward = 12728 micro-steps per epoch, "
        "6364 optimizer steps (gradient accumulation 2)"
    )
    assert IndependentSchedule({SFT: 10, KTO: 4}).optimizer_steps(4) == 3


def test_wind_data_loader_rewrites_the_mgds_start_offset():
    pipeline = SimpleNamespace(_LoadingPipeline__initial_index=0, _LoadingPipeline__mark_reset_serial_modules=False)
    loader = SimpleNamespace(get_data_set=lambda: SimpleNamespace(loading_pipeline=pipeline))
    assert wind_data_loader(loader, batches=7, batch_size=15)
    assert pipeline._LoadingPipeline__initial_index == 105
    assert pipeline._LoadingPipeline__mark_reset_serial_modules is True


def test_wind_data_loader_ignores_non_mgds_loaders():
    loader = Mock()
    loader.get_data_set.return_value = SimpleNamespace()  # no loading_pipeline
    assert not wind_data_loader(loader, batches=3, batch_size=2)
    data_set = SimpleNamespace(loading_pipeline=SimpleNamespace())  # not an mgds pipeline
    assert not wind_data_loader(SimpleNamespace(get_data_set=lambda: data_set), batches=3, batch_size=2)
