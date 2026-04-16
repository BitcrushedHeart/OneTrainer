"""Unit tests for ELO ranking math in DPOService."""
import math

from web.backend.services.dpo_service import DPOService

import pytest


@pytest.fixture
def svc():
    s = DPOService()
    s._elo_init(["a.png", "b.png"])
    return s


def test_elo_init_assigns_base_rating_to_all(svc):
    assert svc._elo_ratings == {"a.png": 1500.0, "b.png": 1500.0}
    assert svc._elo_done == 0


def test_elo_vote_winner_a_increases_a_decreases_b(svc):
    svc._elo_vote("a.png", "b.png", winner="a")
    assert svc._elo_ratings["a.png"] > 1500.0
    assert svc._elo_ratings["b.png"] < 1500.0
    delta_a = svc._elo_ratings["a.png"] - 1500.0
    delta_b = svc._elo_ratings["b.png"] - 1500.0
    assert math.isclose(delta_a + delta_b, 0.0, abs_tol=1e-9), "ELO is zero-sum"


def test_elo_vote_winner_b_decreases_a_increases_b(svc):
    svc._elo_vote("a.png", "b.png", winner="b")
    assert svc._elo_ratings["a.png"] < 1500.0
    assert svc._elo_ratings["b.png"] > 1500.0


def test_elo_vote_tie_leaves_equal_ratings_unchanged(svc):
    svc._elo_vote("a.png", "b.png", winner="tie")
    # Equal starting ratings + tie + zero-sum -> no change
    assert svc._elo_ratings["a.png"] == 1500.0
    assert svc._elo_ratings["b.png"] == 1500.0


def test_elo_vote_increments_done(svc):
    assert svc._elo_done == 0
    svc._elo_vote("a.png", "b.png", winner="a")
    svc._elo_vote("a.png", "b.png", winner="tie")
    assert svc._elo_done == 2


def test_elo_suggested_count_minimum_15():
    s = DPOService()
    assert s._elo_suggested_count(2) >= 15
    assert s._elo_suggested_count(0) >= 15


def test_elo_suggested_count_scales_with_n():
    s = DPOService()
    # n * log2(n) for n=10 is ~33; for n=50 is ~282
    assert s._elo_suggested_count(50) > s._elo_suggested_count(10)


def test_elo_next_pair_picks_consecutive_when_sorted():
    s = DPOService()
    s._elo_init(["a.png", "b.png", "c.png"])
    s._elo_ratings = {"a.png": 1600.0, "b.png": 1500.0, "c.png": 1400.0}
    pair = s._elo_next_pair()
    assert pair is not None
    p1, p2 = pair
    sorted_keys = sorted(s._elo_ratings, key=s._elo_ratings.get, reverse=True)
    i1, i2 = sorted_keys.index(p1), sorted_keys.index(p2)
    assert abs(i1 - i2) == 1, "ELO next-pair must be consecutive in sorted ranking"


def test_elo_next_pair_returns_none_for_singleton():
    s = DPOService()
    s._elo_init(["only.png"])
    assert s._elo_next_pair() is None


def test_elo_finish_round_returns_best_and_worst():
    s = DPOService()
    s._elo_init(["a.png", "b.png", "c.png"])
    s._elo_ratings = {"a.png": 1600.0, "b.png": 1500.0, "c.png": 1400.0}
    pair = s._elo_finish_round()
    assert pair == ("a.png", "c.png")


def test_elo_finish_round_returns_none_below_two():
    s = DPOService()
    s._elo_init(["only.png"])
    assert s._elo_finish_round() is None
