"""The DPO tool's "off" scoring setting.

The setting exists for one reason — the GPU is busy with something else — so
these tests assert the strong form of it: with scoring off the realism model is
never loaded, not merely that its scores stay hidden. Each one substitutes a
loader that raises, which is what "never reached" looks like from here.
"""

from modules.util.realism_detect_util import REALISM_DEVICE_OFF, REALISM_DEVICE_SETTINGS
from web.backend.services.dpo_service import DPOService

import pytest


@pytest.fixture
def service() -> DPOService:
    """A bare service instance, bypassing the singleton so tests don't leak the
    device setting into each other."""
    return DPOService.__new__(DPOService)


def _explode(*args, **kwargs):
    raise AssertionError("scoring is off — the model must not be loaded")


def test_off_is_an_accepted_setting_but_not_a_device():
    assert REALISM_DEVICE_OFF in REALISM_DEVICE_SETTINGS
    assert REALISM_DEVICE_OFF == "off"


def test_scoring_off_returns_no_scores_without_touching_the_model(service, monkeypatch):
    import modules.util.realism_detect_util as rdu

    monkeypatch.setattr(rdu, "load_realism_model", _explode)
    service._realism_device = REALISM_DEVICE_OFF

    result = service.realism_scores(["anything.png"])

    # Not a failure: the grid simply keeps source order.
    assert result == {"ok": True, "scores": {}, "disabled": True}


def test_scoring_off_refuses_to_start_a_backfill(service, monkeypatch):
    import modules.util.realism_detect_util as rdu

    monkeypatch.setattr(rdu, "load_realism_model", _explode)
    service._realism_device = REALISM_DEVICE_OFF

    result = service.start_realism_backfill()

    assert result["ok"] is False
    assert "off" in result["error"].lower()


class _FakeCache:
    """A fully-populated scan cache: enough to reorder, so a missing gate is
    visible as a changed order rather than as a no-op."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def peek_realism(self, path: str, model_version=None):
        return self._scores.get(path)


def test_scoring_off_leaves_group_order_alone(service, monkeypatch):
    """Cached scores would cost no GPU, but a sort the user cannot see the
    reason for — no badges, no model — is not what "off" was asked for."""
    import modules.util.realism_detect_util as rdu

    monkeypatch.setattr(rdu, "realism_model_version", lambda: "v1")
    service._hash_cache = _FakeCache({"b.png": 0.1, "a.png": 0.9})
    groups = [{"prompt": "b", "images": ["b.png"]}, {"prompt": "a", "images": ["a.png"]}]

    service._realism_device = "auto"
    service._group_order = "most_realistic"
    assert [g["prompt"] for g in service._order_by_realism(list(groups))] == ["b", "a"]

    # Same cache, same scores — off is what changes the answer.
    service._realism_device = REALISM_DEVICE_OFF
    reversed_groups = list(reversed(groups))
    assert service._order_by_realism(reversed_groups) == reversed_groups


def test_group_order_selects_the_ranking_direction(service, monkeypatch):
    import modules.util.realism_detect_util as rdu

    monkeypatch.setattr(rdu, "realism_model_version", lambda: "v1")
    service._hash_cache = _FakeCache({"b.png": 0.1, "a.png": 0.9})
    service._realism_device = "auto"
    groups = [{"prompt": "a", "images": ["a.png"]}, {"prompt": "b", "images": ["b.png"]}]

    service._group_order = "most_realistic"
    assert [g["prompt"] for g in service._order_by_realism(list(groups))] == ["b", "a"]
    service._group_order = "least_realistic"
    assert [g["prompt"] for g in service._order_by_realism(list(groups))] == ["a", "b"]
    # Random keeps the caller's shuffle even though every group is scored.
    service._group_order = "random"
    assert service._order_by_realism(list(groups)) == groups


def test_largest_gap_order_ranks_on_the_spread_within_each_group(service, monkeypatch):
    import modules.util.realism_detect_util as rdu

    monkeypatch.setattr(rdu, "realism_model_version", lambda: "v1")
    service._hash_cache = _FakeCache({"a1.png": 0.1, "a2.png": 0.2, "b1.png": 0.1, "b2.png": 0.9})
    service._realism_device = "auto"
    groups = [{"prompt": "a", "images": ["a1.png", "a2.png"]}, {"prompt": "b", "images": ["b1.png", "b2.png"]}]

    # Both groups share the same best image; only the spread separates them.
    service._group_order = "largest_gap"
    assert [g["prompt"] for g in service._order_by_realism(list(groups))] == ["b", "a"]


def test_start_session_rejects_an_unknown_group_order(service, tmp_path):
    res = service.start_session(str(tmp_path), str(tmp_path), 1, "selection", group_order="bogus")
    assert res["ok"] is False
    assert "group_order" in res["error"]
    # Rejected before the session was claimed, so nothing is left active.
    assert getattr(service, "_session_active", False) is False


def test_caption_filter_narrows_groups_before_ordering(service, monkeypatch):
    """The keyword filter drops non-matching groups and the realism order is
    applied to what survives."""
    import modules.util.realism_detect_util as rdu
    from modules.util.dpo_curation_util import group_matches_keywords, parse_caption_keywords

    monkeypatch.setattr(rdu, "realism_model_version", lambda: "v1")
    service._hash_cache = _FakeCache({"a.png": 0.9, "b.png": 0.1, "c.png": 0.5})
    service._realism_device = "auto"
    service._group_order = "most_realistic"
    service._caption_keywords = parse_caption_keywords("Red Dress")
    groups = [
        {"prompt": "a red dress at dawn", "images": ["a.png"]},
        {"prompt": "blue jeans", "images": ["b.png"]},
        {"prompt": "RED DRESS on a beach", "images": ["c.png"]},
    ]
    kept = [g for g in groups if group_matches_keywords(g, service._caption_keywords)]
    ordered = service._order_by_realism(kept)
    assert [g["prompt"] for g in ordered] == ["RED DRESS on a beach", "a red dress at dawn"]


def test_turning_scoring_off_stops_a_backfill_in_flight(service):
    import threading

    service._backfill_stop = threading.Event()
    service._realism_device = "auto"

    assert service.set_realism_device(REALISM_DEVICE_OFF) == {"ok": True, "device": "off"}
    assert service._backfill_stop.is_set()


def test_switching_back_on_is_still_accepted(service):
    import threading

    service._backfill_stop = threading.Event()
    service._realism_device = REALISM_DEVICE_OFF

    assert service.set_realism_device("cpu")["device"] == "cpu"
    assert service._scoring_disabled() is False


def test_an_unknown_setting_is_still_rejected(service):
    service._realism_device = "auto"

    result = service.set_realism_device("tpu")

    assert result["ok"] is False
    assert result["device"] == "auto"
