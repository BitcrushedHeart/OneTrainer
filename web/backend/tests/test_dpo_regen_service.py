"""Fix & Regenerate through a stubbed SwarmUI, against a live curation group.

Real PNGs carrying SwarmUI parameter blocks form a selection group; the fake
client records what it was asked for, writes the "generated" file the way
SwarmUI would, and the results must land in the group without displacing
anything."""

import json
import os
import threading
import time

from modules.util.swarm_client import SwarmError
from web.backend.services.dpo_regen_service import DPORegenService
from web.backend.services.dpo_service import DPOService

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

PROMPT = "A wide shot of a young woman standing in a bustling city street."
MODEL = "SoReal/V14"


def _png(path, prompt: str, color, seed: int | None = 7, size=(64, 48), block: bool = True) -> str:
    info = PngInfo()
    if block:
        params = {
            "prompt": prompt,
            "model": MODEL,
            "seed": seed,
            "steps": 8,
            "cfgscale": 1.0,
            "sampler": "euler",
            "width": size[0],
            "height": size[1],
            "swarm_version": "0.9.6",
        }
        info.add_text("parameters", json.dumps({"sui_image_params": params}))
    else:
        info.add_text("prompt", prompt)
    Image.new("RGB", size, color).save(path, pnginfo=info)
    return str(path)


class FakeClient:
    calls: list[dict] = []
    fail_with: Exception | None = None
    fail_times: int = 0
    output_root: str = ""
    write_files: bool = True
    color = (200, 100, 50)
    gate: threading.Event | None = None
    view_bytes: bytes = b""

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt, width, height, model, images=1, **kwargs):
        FakeClient.calls.append(
            {"prompt": prompt, "width": width, "height": height, "model": model, "images": images, **kwargs}
        )
        if FakeClient.gate is not None:
            FakeClient.gate.wait(10)
        if FakeClient.fail_with is not None and FakeClient.fail_times > 0:
            FakeClient.fail_times -= 1
            raise FakeClient.fail_with
        name = f"{len(FakeClient.calls)}-0-{kwargs.get('seed', 0)}.png"
        if FakeClient.write_files:
            written = os.path.join(FakeClient.output_root, "2026-09-15", name)
            os.makedirs(os.path.dirname(written), exist_ok=True)
            _png(written, prompt, FakeClient.color, seed=kwargs.get("seed"), size=(width, height))
        return [f"View/local/raw/2026-09-15/{name}"]

    def fetch_view(self, view_path, timeout=120):
        if not FakeClient.view_bytes:
            raise SwarmError("no download")
        return FakeClient.view_bytes


@pytest.fixture
def curation(monkeypatch):
    svc = DPOService.__new__(DPOService)
    svc.__init__()
    monkeypatch.setattr(DPOService, "get_instance", classmethod(lambda cls: svc))
    svc._realism_device = "off"
    monkeypatch.setattr(DPOService, "_resolve_concepts", lambda self: [])
    yield svc
    svc.cancel_session()


@pytest.fixture
def regen(monkeypatch):
    import web.backend.services.dpo_regen_service as module

    FakeClient.calls = []
    FakeClient.fail_with = None
    FakeClient.fail_times = 0
    FakeClient.output_root = ""
    FakeClient.write_files = True
    FakeClient.color = (200, 100, 50)
    FakeClient.gate = None
    FakeClient.view_bytes = b""
    monkeypatch.setattr(module, "SwarmClient", FakeClient)
    instance = DPORegenService.__new__(DPORegenService)
    instance.__init__()
    monkeypatch.setattr(DPORegenService, "get_instance", classmethod(lambda cls: instance))
    yield instance
    instance.cancel()


def _wait_scan(svc, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if svc.get_session_status()["worker_finished"]:
            return
        time.sleep(0.05)
    raise AssertionError("scan did not finish")


def _next(svc, timeout=20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = svc.next_group()
        if res.get("group"):
            return res["group"]
        if res.get("done"):
            raise AssertionError("no group")
        time.sleep(0.05)
    raise AssertionError("no group in time")


def _wait_run(service, timeout=20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = service.status()
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError(f"run did not finish: {service.status()}")


def _session(curation, tmp_path, count=3, mode="selection", second_group=False):
    source = tmp_path / "raw"
    source.mkdir(parents=True)
    for i in range(count):
        _png(source / f"src_{i}.png", PROMPT, (10 + i, 20, 30), seed=100 + i)
    if second_group:
        for i in range(2):
            _png(source / f"other_{i}.png", "another prompt", (60 + i, 20, 30), seed=200 + i)
    FakeClient.output_root = str(source)
    assert curation.start_session(str(source), str(tmp_path / "out"), mode=mode, group_order="random")["ok"]
    _wait_scan(curation)
    group = _next(curation)
    return source, group


def _fix(target="face", **kw):
    return {"target": target, **kw}


def test_happy_path_resends_seed_and_adds_to_group(curation, regen, tmp_path):
    source, group = _session(curation, tmp_path)
    target = sorted(group["images"])[0]
    before = list(curation._remaining_images)
    res = regen.start(
        [target],
        [_fix("face"), _fix("text")],
        {"segmentmaskblur": 12},
        base_url="http://x",
        concurrency=1,
    )
    assert res["ok"], res
    state = _wait_run(regen)
    assert state["phase"] == "done", state
    assert len(FakeClient.calls) == 1
    call = FakeClient.calls[0]
    assert call["seed"] == 100
    assert call["model"] == MODEL
    assert (call["width"], call["height"]) == (64, 48)
    assert call["prompt"] == f"{PROMPT} <segment:face,0.95,0.3> <segment:text,0.95,0.3>"
    assert call["extra_params"]["steps"] == 8
    assert call["extra_params"]["segmentmaskblur"] == 12
    assert "swarm_version" not in call["extra_params"]
    added = state["added"]
    assert len(added) == 1
    assert os.path.dirname(added[0]) == os.path.join(str(source), "2026-09-15")
    assert curation._remaining_images == before + added
    assert added[0] in curation._current_group["images"]
    assert state["items"][target]["status"] == "done"


def test_multi_select_adds_every_result(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path)
    targets = sorted(group["images"])[:2]
    assert regen.start(targets, [_fix("hands|feet")], {}, concurrency=2)["ok"]
    state = _wait_run(regen)
    assert len(FakeClient.calls) == 2
    assert sorted(c["seed"] for c in FakeClient.calls) == [100, 101]
    assert len(state["added"]) == 2
    assert all(state["items"][t]["status"] == "done" for t in targets)
    assert set(state["added"]) <= set(curation._remaining_images)


def test_locates_under_generated_folder_when_source_is_elsewhere(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path)
    generated = tmp_path / "swarm_out"
    generated.mkdir()
    FakeClient.output_root = str(generated)
    target = sorted(group["images"])[0]
    assert regen.start([target], [_fix()], {}, generated_folder=str(generated))["ok"]
    state = _wait_run(regen)
    assert state["phase"] == "done", state
    assert state["added"] and state["added"][0].startswith(str(generated))


def test_downloads_beside_source_as_last_resort(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path)
    FakeClient.write_files = False
    scratch = tmp_path / "dl.png"
    _png(scratch, PROMPT, (250, 250, 250))
    FakeClient.view_bytes = scratch.read_bytes()
    target = sorted(group["images"])[0]
    assert regen.start([target], [_fix()], {})["ok"]
    state = _wait_run(regen)
    assert state["phase"] == "done", state
    expected = os.path.splitext(target)[0] + "_fix1.png"
    assert state["added"] == [expected]
    assert os.path.isfile(expected)


def test_image_without_parameter_block_fails_without_a_call(curation, regen, tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    _png(source / "a.png", PROMPT, (1, 2, 3), block=False)
    _png(source / "b.png", PROMPT, (4, 5, 6), block=False)
    FakeClient.output_root = str(source)
    assert curation.start_session(str(source), str(tmp_path / "out"), mode="selection", group_order="random")["ok"]
    _wait_scan(curation)
    group = _next(curation)
    target = sorted(group["images"])[0]
    preview = regen.preview([target], [_fix()], {})
    assert preview["ok"] and preview["items"][0]["ok"] is False
    assert "parameter block" in preview["items"][0]["error"]
    res = regen.start([target], [_fix()], {})
    assert not res["ok"]
    assert FakeClient.calls == []


def test_refuses_paths_outside_group_and_non_selection_modes(curation, regen, tmp_path):
    _source, _group = _session(curation, tmp_path)
    assert regen.start([str(tmp_path / "nope.png")], [_fix()], {})["ok"] is False
    assert regen.preview([], [_fix()], {})["ok"] is False
    curation.cancel_session()
    _source, group = _session(curation, tmp_path / "swiss", mode="swiss")
    target = sorted(group["images"])[0]
    res = regen.start([target], [_fix()], {})
    assert not res["ok"] and "selection" in res["error"]


def test_result_after_group_change_is_dropped(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path, second_group=True)
    FakeClient.gate = threading.Event()
    target = sorted(group["images"])[0]
    assert regen.start([target], [_fix()], {}, concurrency=1)["ok"]
    time.sleep(0.2)
    curation.skip_group()
    other = _next(curation)
    assert other["prompt"] != group["prompt"]
    before = list(curation._remaining_images)
    FakeClient.gate.set()
    state = _wait_run(regen)
    assert state["added"] == []
    assert len(state["dropped"]) == 1
    assert state["items"][target]["status"] == "dropped"
    assert state["warning"]
    assert curation._remaining_images == before


def test_pixel_identical_to_exported_is_skipped(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path, count=4)
    images = sorted(group["images"])
    # Export a pair whose chosen has the colour the fake will paint.
    FakeClient.color = (10, 20, 30)  # == src_0
    curation.select_image(images[0])
    res = curation.select_image(images[1])
    assert res.get("pair_pending")
    assert curation.confirm_pair(True)["ok"]
    target = images[2]
    assert regen.start([target], [_fix()], {})["ok"]
    state = _wait_run(regen)
    assert state["added"] == []
    assert state["items"][target]["status"] == "skipped"
    assert "Identical" in state["items"][target]["error"]


def test_pending_pair_survives_an_addition(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path, count=4)
    images = sorted(group["images"])
    curation.select_image(images[0])
    assert curation.select_image(images[1]).get("pair_pending")
    assert regen.start([images[2]], [_fix()], {})["ok"]
    state = _wait_run(regen)
    assert len(state["added"]) == 1
    res = curation.confirm_pair(True)
    assert res["ok"] and res["continue_group"]
    assert state["added"][0] in res["remaining_images"]
    assert images[0] not in res["remaining_images"]


def test_consecutive_failures_stop_the_run_and_cancel(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path, count=4)
    FakeClient.fail_with = SwarmError("boom")
    FakeClient.fail_times = 10
    assert regen.start(sorted(group["images"]), [_fix()], {}, concurrency=1)["ok"]
    state = _wait_run(regen)
    assert state["phase"] == "error"
    assert "consecutive" in state["error"]
    assert state["failed"] >= 3
    assert regen.cancel()["ok"]
    assert regen.status()["running"] is False


def test_second_start_while_running_is_refused_and_preview_warns(curation, regen, tmp_path):
    _source, group = _session(curation, tmp_path)
    images = sorted(group["images"])
    preview = regen.preview([images[0]], [_fix("face", sub_prompt="detailed skin")], {"segmentmaskgrow": 4})
    assert preview["ok"]
    item = preview["items"][0]
    assert item["ok"] and item["seed"] == 100 and item["model"] == MODEL
    assert item["prompt"].endswith("<segment:face,0.95,0.3> detailed skin")
    assert any("sub-prompt" in w.lower() for w in item["warnings"])
    assert preview["segment_params"] == {"segmentmaskgrow": 4}
    FakeClient.gate = threading.Event()
    assert regen.start([images[0]], [_fix()], {})["ok"]
    res = regen.start([images[1]], [_fix()], {})
    assert not res["ok"] and "in progress" in res["error"]
    FakeClient.gate.set()
    _wait_run(regen)
