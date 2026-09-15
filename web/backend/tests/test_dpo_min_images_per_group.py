"""``min_images_per_group`` skips thinly-generated prompts.

The off-policy builder leaves hundreds of captions with exactly two images;
raising the floor above the 2 a pair needs filters them out of a curation
session without touching the source tree."""

import time

from web.backend.services.dpo_service import DPOService

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(path, prompt: str, color) -> str:
    info = PngInfo()
    info.add_text("prompt", prompt)
    Image.new("RGB", (64, 64), color).save(path, pnginfo=info)
    return str(path)


@pytest.fixture
def curation(monkeypatch):
    svc = DPOService.__new__(DPOService)
    svc.__init__()
    monkeypatch.setattr(DPOService, "get_instance", classmethod(lambda cls: svc))
    svc._realism_device = "off"
    monkeypatch.setattr(DPOService, "_resolve_concepts", lambda self: [])
    yield svc
    svc.cancel_session()


def _wait(svc, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if svc.get_session_status()["worker_finished"]:
            return
        time.sleep(0.05)
    raise AssertionError("scan did not finish")


def _drain_prompts(svc) -> set[str]:
    prompts = set()
    while True:
        res = svc.next_group()
        if res.get("done"):
            return prompts
        if res.get("waiting"):
            time.sleep(0.05)
            continue
        prompts.add(res["group"]["prompt"])


def _tree(tmp_path):
    source = tmp_path / "generated"
    source.mkdir()
    for i in range(2):
        _png(source / f"thin_{i}.png", "thin prompt", (10 + i, 20, 30))
    for i in range(4):
        _png(source / f"thick_{i}.png", "thick prompt", (40 + i, 50, 60))
    return source, tmp_path / "out"


def test_default_floor_of_two_shows_every_group(curation, tmp_path):
    source, out = _tree(tmp_path)
    assert curation.start_session(str(source), str(out), mode="selection", group_order="random")["ok"]
    _wait(curation)
    assert _drain_prompts(curation) == {"thin prompt", "thick prompt"}


def test_min_images_per_group_skips_thin_groups(curation, tmp_path):
    source, out = _tree(tmp_path)
    assert curation.start_session(
        str(source), str(out), mode="selection", group_order="random", min_images_per_group=3
    )["ok"]
    _wait(curation)
    assert _drain_prompts(curation) == {"thick prompt"}


def test_threshold_counts_deduped_images(curation, tmp_path):
    """A group padded out with pixel-identical copies is still thin."""
    source = tmp_path / "generated"
    source.mkdir()
    _png(source / "a.png", "dupe prompt", (1, 2, 3))
    _png(source / "b.png", "dupe prompt", (1, 2, 3))  # pixel-identical
    _png(source / "c.png", "dupe prompt", (9, 9, 9))
    assert curation.start_session(
        str(source), str(tmp_path / "out"), mode="selection", group_order="random", min_images_per_group=3
    )["ok"]
    _wait(curation)
    assert _drain_prompts(curation) == set()


def test_below_the_pair_floor_is_clamped(curation, tmp_path):
    source, out = _tree(tmp_path)
    assert curation.start_session(
        str(source), str(out), mode="selection", group_order="random", min_images_per_group=0
    )["ok"]
    assert curation._min_images_per_group == 2
    _wait(curation)
    assert _drain_prompts(curation) == {"thin prompt", "thick prompt"}
