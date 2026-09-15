"""Replace Pairs end to end: a concept on disk, a fake SwarmUI that writes real
files into the generated tree, and a stubbed realism head."""

import hashlib
import json
import os
import time

from web.backend.services.dpo_replace_service import DPOReplacePairsService
from web.backend.services.dpo_service import DPOService

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(path, size=(64, 48), params=None, color=(120, 90, 60)):
    info = PngInfo()
    if params is not None:
        info.add_text("parameters", json.dumps({"sui_image_params": params}))
    Image.new("RGB", size, color).save(path, pnginfo=info)
    return str(path)


def _jpg(path, size=(64, 48)):
    Image.new("RGB", size, (30, 140, 200)).save(path, quality=90)
    return str(path)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _wait(service, key, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.status()
        if key(status):
            return status
        time.sleep(0.02)
    raise AssertionError(f"timed out: {service.status()}")


NEW_PARAMS = {"model": "Models/new.safetensors", "loras": ["V41"], "loraweights": ["0.8"], "steps": 8}


class FakeClient:
    calls: list[dict] = []
    output_root: str = ""
    fail_times: int = 0

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt, width, height, model, images=1, **kwargs):
        FakeClient.calls.append({"prompt": prompt, "width": width, "height": height, "model": model, **kwargs})
        if FakeClient.fail_times > 0:
            FakeClient.fail_times -= 1
            from modules.util.swarm_client import SwarmError

            raise SwarmError("boom")
        views = []
        for index in range(images):
            name = f"fresh-{len(FakeClient.calls)}-{index}.png"
            written = os.path.join(FakeClient.output_root, "2026-09-04", name)
            os.makedirs(os.path.dirname(written), exist_ok=True)
            _png(written, (width, height), {**NEW_PARAMS, "prompt": prompt}, color=(index * 40, 10, 10))
            views.append(f"View/local/raw/2026-09-04/{name}")
        return views


@pytest.fixture
def scores(monkeypatch):
    """P(AI) by basename; anything unknown (the fresh generations) gets
    ``table['*']``."""
    import modules.util.realism_detect_util as rdu

    table: dict[str, float] = {"*": 0.2}

    def fake_score(paths, cache=None, device="auto", **_k):
        return {p: table.get(os.path.basename(p), table["*"]) for p in paths}

    monkeypatch.setattr(rdu, "score_images_realism", fake_score)
    monkeypatch.setattr(rdu, "realism_model_version", lambda: "test")
    return table


@pytest.fixture
def service(monkeypatch):
    import web.backend.services.dpo_replace_service as module

    FakeClient.calls = []
    FakeClient.fail_times = 0
    monkeypatch.setattr(module, "SwarmClient", FakeClient)
    monkeypatch.setattr(DPOService, "realism_device", lambda self: {"ok": True, "device": "cpu"})
    svc = DPOReplacePairsService.__new__(DPOReplacePairsService)
    svc.__init__()
    monkeypatch.setattr(DPOReplacePairsService, "get_instance", classmethod(lambda cls: svc))
    yield svc
    svc.cancel()


@pytest.fixture
def dirs(tmp_path):
    root = tmp_path / "concept"
    (root / "chosen").mkdir(parents=True)
    (root / "rejected").mkdir()
    real = tmp_path / "real"
    real.mkdir()
    gen = tmp_path / "gen"
    gen.mkdir()
    FakeClient.output_root = str(gen)
    # The user's most recent generation: where the settings come from.
    _png(gen / "last.png", params={**NEW_PARAMS, "prompt": "whatever"})

    photo = _jpg(real / "photo.jpg", (2048, 1536))
    (real / "photo.txt").write_text("a woman in a garden\n", encoding="utf-8")
    old = {"model": "old.safetensors"}
    _jpg(root / "chosen" / "pair_0000.jpg", (2048, 1536))
    (root / "chosen" / "pair_0000.txt").write_text("a woman in a garden", encoding="utf-8")
    _png(root / "rejected" / "pair_0000.png", (2048, 1536), params=old)
    os.utime(root / "rejected" / "pair_0000.png", (1_000, 1_000))
    _png(root / "chosen" / "pair_0001.png", (2048, 1536), params=old, color=(1, 2, 3))
    (root / "chosen" / "pair_0001.txt").write_text("a cat on a sofa", encoding="utf-8")
    _png(root / "rejected" / "pair_0001.png", (2048, 1536), params=old, color=(4, 5, 6))
    os.utime(root / "rejected" / "pair_0001.png", (2_000, 2_000))
    manifest = {
        "pairs": [
            {
                "pair_id": 0,
                "prompt": "a woman in a garden",
                "aspectratio": "4:3",
                "chosen_file": "pair_0000.jpg",
                "rejected_file": "pair_0000.png",
                "chosen_source": photo,
                "rejected_source": str(gen / "x.png"),
            },
            {
                "pair_id": 1,
                "prompt": "a cat on a sofa",
                "aspectratio": "4:3",
                "chosen_file": "pair_0001.png",
                "rejected_file": "pair_0001.png",
                "chosen_source": str(gen / "c.png"),
                "rejected_source": str(gen / "r.png"),
            },
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, gen, real


def _start(service, dirs, **kwargs):
    root, gen, _real = dirs
    res = service.start(str(root), str(gen), **kwargs)
    assert res["ok"], res
    return _wait(service, lambda s: s["scan_finished"])


def test_scan_ranks_and_backfills_provenance(service, dirs, scores):
    root, _gen, _real = dirs
    status = _start(service, dirs)
    assert status["pairs"] == 2 and status["error"] is None
    listing = service.pairs()
    assert [p["pair_id"] for p in listing["pairs"]] == [0, 1]  # oldest rejected first
    assert listing["models"][0]["label"] == "old.safetensors" and listing["models"][0]["pairs"] == 2
    saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert saved["pairs"][0]["rejected_provenance"]["model"] == "old.safetensors"
    assert saved["pairs"][1]["chosen_provenance"]["model"] == "old.safetensors"
    params = service.params()
    assert params["found"] and params["model"] == "Models/new.safetensors" and params["params"]["steps"] == 8


def _by_role(proposal, role):
    return [i for i in proposal["images"] if i["role"] == role]


def test_generate_builds_groups_and_applies_selections(service, dirs, scores):
    root, _gen, _real = dirs
    scores.update({"pair_0000.jpg": 0.5, "pair_0000.png": 0.6, "pair_0001.png": 0.4, "*": 0.1})
    _start(service, dirs)
    res = service.generate(count=2, images_per_pair=2)
    assert res["ok"] and res["planned"] == 2 and res["model"] == "Models/new.safetensors"
    _wait(service, lambda s: not s["running"])
    status = service.status()
    assert status["run_phase"] == "done" and status["failed"] == 0 and status["images"] == 4
    # Settings of the last generation were reused, sized to the chosen image's bucket.
    call = FakeClient.calls[0]
    assert call["extra_params"]["steps"] == 8 and call["extra_params"]["loras"] == ["V41"]
    assert call["width"] % 64 == 0 and call["height"] % 64 == 0 and call["width"] > call["height"]

    seen = set()
    for _ in range(2):
        proposal = service.next()["proposal"]
        seen.add(proposal["pair_id"])
        fresh = _by_role(proposal, "fresh")
        assert len(fresh) == 2 and proposal["scored"] and proposal["new_model"].startswith("new.safetensors")
        assert [i["role"] for i in proposal["images"][:2]] == ["chosen", "rejected"]
        if proposal["pair_id"] == 0:
            # Fresh 0.1 beats the chosen too, so the suggestion promotes; the
            # reviewer keeps the chosen here and just takes a harder negative.
            assert not proposal["chosen_locked"]
            assert proposal["suggested_chosen"] in {f["path"] for f in fresh}
            old_sha = _sha(root / "rejected" / "pair_0000.png")
            assert service.apply(proposal["chosen_path"], fresh[0]["path"])["ok"]
            assert _sha(root / "rejected" / "pair_0000.png") == _sha(fresh[0]["path"])
            assert _sha(root / "chosen" / "pair_0000.jpg") != _sha(fresh[0]["path"])
            assert any(f.startswith("pair_0000__") for f in os.listdir(root / ".replaced"))
            assert service.undo()["ok"]
            assert _sha(root / "rejected" / "pair_0000.png") == old_sha
            # Pick the other fresh image by hand.
            assert service.apply(proposal["chosen_path"], fresh[1]["path"])["ok"]
            assert _sha(root / "rejected" / "pair_0000.png") == _sha(fresh[1]["path"])
        else:
            assert not proposal["chosen_locked"]
            assert proposal["suggested_chosen"] in {f["path"] for f in fresh}  # 0.1 beats chosen 0.4
            # Decline the promotion: keep the chosen, take a fresh negative.
            assert service.apply(proposal["chosen_path"], fresh[0]["path"])["ok"]
            assert _sha(root / "rejected" / "pair_0001.png") == _sha(fresh[0]["path"])
            assert _sha(root / "chosen" / "pair_0001.png") != _sha(fresh[0]["path"])
    assert seen == {0, 1}
    assert service.next()["done"]
    status = service.status()
    assert status["applied"] == 2 and status["promoted"] == 0 and status["can_undo"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pairs"][0]["rejected_provenance"]["loras"] == ["V41"]
    assert len(manifest["pairs"][0]["history"]) == 1


def test_promotion_counts_and_keep_undo(service, dirs, scores):
    root, _gen, _real = dirs
    scores.update({"pair_0001.png": 0.4, "*": 0.1})
    _start(service, dirs)
    assert service.generate(pair_ids=[1], images_per_pair=1)["ok"]
    _wait(service, lambda s: not s["running"])
    proposal = service.next()["proposal"]
    fresh = _by_role(proposal, "fresh")[0]["path"]
    assert proposal["suggested_chosen"] == fresh and proposal["suggested_rejected"] == proposal["chosen_path"]
    assert service.apply()["ok"]
    assert service.status()["promoted"] == 1
    assert _sha(root / "chosen" / "pair_0001.png") == _sha(fresh)
    assert service.undo()["ok"] and service.status()["promoted"] == 0
    assert service.keep()["ok"] and service.status()["kept"] == 1
    assert service.undo()["ok"] and service.status()["kept"] == 0
    # Re-applying the pair as it already is is refused.
    res = service.apply(proposal["chosen_path"], proposal["rejected_path"])
    assert not res["ok"] and "keep" in res["error"]


def test_auto_apply_applies_rejected_swaps_but_queues_chosen_moves(service, dirs, scores):
    root, _gen, _real = dirs
    # Pair 0's chosen (0.05) stays ahead of the fresh 0.1 -> rejected-only swap, auto-applied.
    scores.update({"pair_0000.jpg": 0.05, "pair_0000.png": 0.6, "pair_0001.png": 0.4, "*": 0.1})
    _start(service, dirs)
    assert service.generate(count=2, images_per_pair=1, auto_apply=True)["ok"]
    _wait(service, lambda s: not s["running"])
    status = service.status()
    assert status["auto_applied"] == 1 and status["ready"] == 1
    proposal = service.next()["proposal"]
    assert proposal["pair_id"] == 1 and proposal["suggested_chosen"] != proposal["chosen_path"]
    assert service.apply()["ok"]
    assert _sha(root / "chosen" / "pair_0001.png") == _sha(proposal["suggested_chosen"])
    assert service.status()["promoted"] == 1


def test_corrected_caption_ranks_first_and_is_regenerated(service, dirs, scores):
    root, _gen, _real = dirs
    (root / "chosen" / "pair_0001.txt").write_text("a tabby cat asleep on a sofa", encoding="utf-8")
    scores.update({"pair_0001.png": 0.6, "*": 0.1})
    status = _start(service, dirs)
    assert status["stale_captions"] == 1
    assert service.pairs()["pairs"][0]["pair_id"] == 1  # corrected caption first, despite being newer
    assert service.generate(count=1)["ok"]
    _wait(service, lambda s: not s["running"])
    assert FakeClient.calls[0]["prompt"] == "a tabby cat asleep on a sofa"
    proposal = service.next()["proposal"]
    assert proposal["caption_changed"] and proposal["caption"] == "a tabby cat asleep on a sofa"
    assert service.apply()["ok"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pairs"][1]["prompt"] == "a tabby cat asleep on a sofa"


def test_unscored_when_head_unavailable(service, dirs, monkeypatch):
    import modules.util.realism_detect_util as rdu

    def missing(*_a, **_k):
        raise RuntimeError("no model")

    monkeypatch.setattr(rdu, "score_images_realism", missing)
    monkeypatch.setattr(rdu, "realism_model_version", lambda: None)
    _start(service, dirs)
    assert service.generate(count=1, images_per_pair=1)["ok"]
    _wait(service, lambda s: not s["running"])
    proposal = service.next()["proposal"]
    assert proposal["scored"] is False and proposal["suggestion_is_keep"]
    assert "pick by eye" in (service.status()["warning"] or "")
    # The reviewer can still pick a fresh negative by hand.
    fresh = _by_role(proposal, "fresh")[0]["path"]
    assert service.apply(proposal["chosen_path"], fresh)["ok"]


def test_swarm_failures_are_counted_and_no_model_is_refused(service, dirs, scores, tmp_path):
    root, gen, _real = dirs
    FakeClient.fail_times = 1
    _start(service, dirs)
    assert service.generate(count=2, images_per_pair=1)["ok"]
    _wait(service, lambda s: not s["running"])
    status = service.status()
    assert status["failed"] == 1 and status["done"] == 2 and status["run_phase"] == "done"
    service.cancel()

    # No previous generation to take settings from.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert service.start(str(root), str(empty))["ok"]
    _wait(service, lambda s: s["scan_finished"])
    res = service.generate(count=1)
    assert not res["ok"] and "No model" in res["error"]


def test_start_rejects_folders_without_a_manifest(service, tmp_path):
    (tmp_path / "chosen").mkdir()
    res = service.start(str(tmp_path), str(tmp_path))
    assert not res["ok"] and "manifest" in res["error"]
