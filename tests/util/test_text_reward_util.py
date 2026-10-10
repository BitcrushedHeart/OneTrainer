import random

from modules.util.text_reward_util import (
    BUILTIN_WORDS,
    DEFAULT_VOCAB,
    TextRewardBank,
    TextRewardScheduler,
    TextSpec,
    align_words,
    choose_grad_step,
    derive_text_reward_interval,
    filter_to_vocab,
    generate_text_spec,
    generate_text_specs,
    grad_step_candidates,
    grad_window,
    nearest_sigma_index,
    reading_order,
    text_reward_metrics,
)

import pytest


def test_generated_specs_stay_inside_vocab_and_cover_families():
    rng = random.Random(0)
    families = set()
    for _ in range(400):
        spec = generate_text_spec(rng, DEFAULT_VOCAB)
        families.add(spec.family)
        assert spec.text == " ".join(spec.words)
        assert spec.words
        for word in spec.words:
            assert word
            assert all(ch in DEFAULT_VOCAB for ch in word), word
        assert spec.prompt == f'{spec.surface} text reading "{spec.text}"'
    assert families == {"words", "price", "date", "time", "url", "abbrev", "number", "nonword"}


def test_vocab_filter_drops_unreadable_characters_but_keeps_prices():
    ascii_only = DEFAULT_VOCAB.replace("£", "").replace("€", "")
    assert filter_to_vocab("The Woollen Mill — £11.50", ascii_only) == "The Woollen Mill  11.50"
    assert filter_to_vocab("£11.50", DEFAULT_VOCAB) == "£11.50"
    rng = random.Random(4)
    for _ in range(200):
        spec = generate_text_spec(rng, ascii_only, family_weights={"price": 1.0, "words": 0.0})
        assert "£" not in spec.text and "€" not in spec.text


def test_generation_is_deterministic_per_seed():
    a = generate_text_specs(20, seed=7, vocab=DEFAULT_VOCAB)
    b = generate_text_specs(20, seed=7, vocab=DEFAULT_VOCAB)
    c = generate_text_specs(20, seed=8, vocab=DEFAULT_VOCAB)
    assert a == b
    assert a != c


def test_word_bounds_respected_for_plain_words():
    rng = random.Random(1)
    for _ in range(100):
        spec = generate_text_spec(
            rng,
            DEFAULT_VOCAB,
            family_weights={
                "words": 1.0,
                **dict.fromkeys(("price", "date", "time", "url", "abbrev", "number", "nonword"), 0.0),
            },
            min_words=2,
            max_words=3,
        )
        assert 2 <= len(spec.words) <= 3
        assert all(w.lower() in BUILTIN_WORDS for w in spec.words)


def test_align_words_skips_spurious_boxes_and_reports_missing():
    al = align_words(["The", "Woollen", "Mill", "-", "-", "£11.50"], ["The", "Woollen", "Mill", "-", "£11.50"])
    assert al.pairs == [(0, 0), (1, 1), (2, 2), (4, 3), (5, 4)] or al.pairs == [(0, 0), (1, 1), (2, 2), (3, 3), (5, 4)]
    assert al.missing == []

    al = align_words(["Tne", "Wollen", "MiII"], ["The", "Woollen", "Mill", "-", "£11.50"])
    assert al.pairs == [(0, 0), (1, 1), (2, 2)]
    assert al.missing == [3, 4]

    al = align_words(["zzzzzz"], ["Bakery"])
    assert al.pairs == []
    assert al.missing == [0]

    assert align_words([], ["a", "b"]).missing == [0, 1]
    assert align_words(["a"], []).pairs == []


def test_metrics_penalise_missing_and_reward_verbatim():
    targets = [["Open", "24h"]]
    exact = text_reward_metrics([["Open", "24h"]], targets, [align_words(["Open", "24h"], targets[0])])
    assert exact.exact == 1.0 and exact.word_ratio == 1.0 and exact.missing == 0.0

    partial = text_reward_metrics([["Opem"]], targets, [align_words(["Opem"], targets[0])])
    assert partial.exact == 0.0
    assert 0.0 < partial.word_ratio < 1.0
    assert partial.missing == 0.5


def test_reading_order_is_line_then_left_to_right():
    boxes = [[0.6, 0.1, 0.9, 0.2], [0.1, 0.12, 0.5, 0.22], [0.1, 0.5, 0.9, 0.6]]
    assert reading_order(boxes) == [1, 0, 2]


@pytest.mark.parametrize(
    ("primary", "samples", "expected"),
    [(96000, 1000, 96), (5000, 1000, 5), (500, 1000, 1), (0, 1000, 0), (1000, 0, 0)],
)
def test_interval_matches_budget(primary, samples, expected):
    assert derive_text_reward_interval(primary, samples) == expected


def test_scheduler_never_fires_on_a_blocked_step_but_keeps_the_budget():
    sched = TextRewardScheduler(5)
    fired = []
    for step in range(30):
        blocked = step % 10 == 0  # a DPO pair rides on every 10th step
        if sched.due(step, blocked):
            fired.append(step)
    # every 5th step is due; the blocked ones (0, 10, 20) carry over to the next free step
    assert fired == [1, 5, 11, 15, 21, 25]
    assert sched.pending == 0

    sched.start_epoch(0)
    assert not any(sched.due(step, False) for step in range(10))


def test_bank_cycles():
    specs = [TextSpec("Sign", f"w{i}", (f"w{i}",), "words") for i in range(3)]
    bank = TextRewardBank(specs, ["e0", "e1", "e2"], None)
    seen = [bank.next().spec.text for _ in range(5)]
    assert seen == ["w0", "w1", "w2", "w0", "w1"]
    assert bank.next().embedding == "e2"
    with pytest.raises(ValueError):
        TextRewardBank(specs, ["e0"], None)


# -- grad step selection on the sigma grid -------------------------------------

_GRID = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]  # 8 steps, shift 1


def test_default_window_is_the_original_second_half_rule():
    assert grad_step_candidates(_GRID, 8) == [4, 5, 6, 7]
    # bit-identical to the original rng.randint(steps // 2, steps - 1) draw
    for seed in range(50):
        legacy = random.Random(seed).randint(4, 7)
        assert choose_grad_step(_GRID, 8, random_grad_step=True, rng=random.Random(seed)) == legacy
    assert choose_grad_step(_GRID, 8, random_grad_step=False, rng=random.Random(0)) == 7


def test_sigma_window_selects_by_sigma_and_never_comes_back_empty():
    assert grad_step_candidates(_GRID, 8, sigma_min=0.4, sigma_max=0.7) == [3, 4]
    assert choose_grad_step(_GRID, 8, random_grad_step=False, rng=random.Random(0), sigma_min=0.4, sigma_max=0.7) == 4
    for seed in range(20):
        pick = choose_grad_step(_GRID, 8, random_grad_step=True, rng=random.Random(seed), sigma_min=0.4, sigma_max=0.7)
        assert pick in (3, 4)
    # a window between grid points falls back to the nearest index
    assert grad_step_candidates(_GRID, 8, sigma_min=0.66, sigma_max=0.70) == [3]


def test_hybrid_init_index_sits_below_the_grad_step():
    assert nearest_sigma_index(_GRID, 8, 0.8) == 2
    assert nearest_sigma_index(_GRID, 8, 0.99) == 0
    assert nearest_sigma_index(_GRID, 8, 0.0) == 6  # capped at steps - 2 so a grad step remains
    assert grad_step_candidates(_GRID, 8, sigma_min=0.4, sigma_max=0.7, after_index=2) == [3, 4]
    assert grad_step_candidates(_GRID, 8, sigma_min=0.4, sigma_max=0.7, after_index=4) == [5]
    assert grad_step_candidates(_GRID, 8, after_index=6) == [7]
    assert grad_step_candidates(_GRID, 8, after_index=7) == [7]


def test_grad_window_is_the_single_step_for_k1_and_clamps_to_the_grid():
    # K = 1: first == last == the drawn step (the original single forward)
    assert grad_window(4, 1, 8) == (4, 4)
    assert grad_window(4, 0, 8) == (4, 4)
    # K = 3 from index 3: the chain runs 3, 4 and scores the estimate at 5
    assert grad_window(3, 3, 8) == (3, 5)
    # a chain that would run off the grid stops at the last index
    assert grad_window(6, 3, 8) == (6, 7)
    assert grad_window(7, 2, 8) == (7, 7)
    # a drawn step past the grid is clamped before the chain is built
    assert grad_window(12, 2, 8) == (7, 7)
