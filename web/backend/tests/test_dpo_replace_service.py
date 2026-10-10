"""Blind replacement with a real temporary dataset and controllable fake SwarmUI."""

import copy
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from modules.util import dpo_replace_session as state
from web.backend.services import dpo_replace_service as module

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

REAL_RESOLUTION = module.DPOReplacePairsService._resolution
PARAMS = {
    "model": "new-model",
    "steps": 9,
    "loras": ["adapter"],
    "loraweights": [0.7],
    "sampler": "euler",
    "negativeprompt": "blur",
    "prompt": "WRONG",
    "seed": 42,
    "width": 999,
    "height": 999,
    "images": 8,
}
CAPTION = "  A corrected caption, <detail:1.2>\r\nsecond line.\n"


def png(path, color=(25, 60, 90), params=None):
    info = PngInfo()
    if params is not None:
        info.add_text("parameters", json.dumps({"sui_image_params": params}))
    Image.new("RGB", (64, 48), color).save(path, pnginfo=info)


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out")


@pytest.fixture
def env(tmp_path, monkeypatch):
    concept = tmp_path / "dataset"
    (concept / "chosen").mkdir(parents=True)
    (concept / "rejected").mkdir()
    output = tmp_path / "swarm"
    output.mkdir()
    png(output / "last.png", params=PARAMS)
    entries = []
    for index in range(4):
        name = f"pair_{index}.png"
        png(concept / "chosen" / name, color=(index, 60, 90))
        png(concept / "rejected" / name, color=(index, 20, 90))
        state.write_caption((concept / "chosen" / name).with_suffix(".txt"), CAPTION)
        state.write_caption((concept / "rejected" / name).with_suffix(".txt"), "old rejected caption")
        entries.append({"pair_id": index, "chosen_file": name, "rejected_file": name, "prompt": "OLD PROMPT"})
    (concept / "manifest.json").write_text(json.dumps({"pairs": entries}), encoding="utf-8")

    class Client:
        calls = []
        entered = threading.Event()
        release = threading.Event()
        release.set()
        fail = False
        block_at = None

        def __init__(self, *args):
            pass

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == self.block_at:
                self.release.clear()
            self.entered.set()
            assert self.release.wait(10)
            if self.fail:
                raise RuntimeError("Swarm unavailable")
            name = f"candidate-{len(self.calls)}.png"
            info = {**kwargs["extra_params"], "model": kwargs["model"], "prompt": kwargs["prompt"]}
            png(output / name, color=(len(self.calls), 40, 90), params=info)
            return [name]

    monkeypatch.setattr(module, "SwarmClient", Client)
    monkeypatch.setattr(module.DPOReplacePairsService, "_resolution", lambda *args: 1024)
    services = []

    def create(start=True, **options):
        service = module.DPOReplacePairsService()
        services.append(service)
        if start:
            result = service.start(str(concept), str(output), **options)
            assert result["ok"], result
        return service

    yield concept, output, Client, create
    Client.release.set()
    for service in services:
        service.cancel()
        if service._run_thread:
            service._run_thread.join(10)


def ready(service):
    wait_for(lambda: not service.status()["running"])
    result = service.current()
    assert result["ok"] and result["group"]
    return result["group"]


def candidates(service, group):
    return service._ledger["items"][group["group_id"]]["candidates"]


def choose(service, group, best, worst, override=False):
    assert service.select_best(group["session_id"], group["group_id"], best)["ok"]
    return service.apply(group["session_id"], group["group_id"], best, worst, override)


@pytest.mark.parametrize("count", [1, 2, 3, 10])
def test_exact_caption_settings_size_and_anonymous_candidates(env, count):
    _, _, client, create = env
    service = create(images_per_pair=count)
    group = ready(service)
    assert len(client.calls) == 4 * count
    for call in client.calls:
        assert call["prompt"] == CAPTION
        assert (call["width"], call["height"]) == (64, 48)
        assert call["images"] == 1 and call["seed"] == -1
        assert call["model"] == PARAMS["model"]
        assert call["extra_params"] == {
            **{k: PARAMS[k] for k in ("steps", "loras", "loraweights", "sampler", "negativeprompt")},
            "donotsave": True,
        }
    assert len(group["images"]) == count + 2
    for candidate in group["images"]:
        assert set(candidate) == {"id", "url", "current_rejected"}
        assert "chosen" not in candidate["url"] and "rejected" not in candidate["url"]
    # Only the current rejected is revealed; the current chosen stays blind among the fresh images.
    assert sum(c["current_rejected"] for c in group["images"]) == 1
    for candidate in candidates(service, group):
        assert state.caption_text(Path(candidate["path"]).with_suffix(".txt")) == CAPTION


def test_pause_finishes_one_image_and_persists_partial_group(env):
    _, _, client, create = env
    client.release.clear()
    service = create(images_per_pair=3)
    assert client.entered.wait(5)
    sid = service.status()["session_id"]
    service.pause(sid)
    client.release.set()
    wait_for(lambda: not service.status()["running"])
    assert len(client.calls) == 1
    assert service.status()["images"] == 1
    assert service.current()["group"] is None
    order = service._ledger["order"][:]
    service.cancel()
    restored = create(restore_only=True)
    assert restored.status()["paused"] and not restored.status()["running"]
    assert restored._ledger["order"] == order
    assert restored.status()["images"] == 1
    assert restored.resume(restored.status()["session_id"])["ok"]
    ready(restored)
    assert len(client.calls) == 12


def test_review_and_best_selection_survive_restart(env):
    _, _, _, create = env
    service = create()
    group = ready(service)
    selected = group["images"][1]["id"]
    assert service.select_best(group["session_id"], group["group_id"], selected)["ok"]
    service.cancel()
    restored = create(restore_only=True)
    current = restored.current()["group"]
    assert current["images"] == group["images"]
    assert current["best_id"] == selected
    assert current["session_id"] == group["session_id"]


@pytest.mark.parametrize(
    "best_role,worst_role",
    [
        ("chosen", "rejected"),
        ("rejected", "chosen"),
        ("fresh", "chosen"),
        ("chosen", "fresh"),
        ("rejected", "fresh"),
        ("fresh", "rejected"),
        ("fresh", "fresh"),
    ],
)
def test_combinations_captions_reversal_and_undo(env, best_role, worst_role):
    concept, _, _, create = env
    service = create()
    group = ready(service)
    item = service._ledger["items"][group["group_id"]]
    before = {str(p): p.read_bytes() for side in ("chosen", "rejected") for p in (concept / side).iterdir()}
    manifest_before = json.loads((concept / "manifest.json").read_text())
    best = next(c for c in candidates(service, group) if c["role"] == best_role)
    worst = next(c for c in candidates(service, group) if c["role"] == worst_role and c != best)
    result = choose(service, group, best["id"], worst["id"])
    if best_role == "rejected" and worst_role == "chosen":
        assert result == {"ok": False, "requires_reversal_confirmation": True}
        assert {p: Path(p).read_bytes() for p in before} == before
        result = service.apply(group["session_id"], group["group_id"], best["id"], worst["id"], True)
    assert result["ok"], result
    for side, candidate in (("chosen", best), ("rejected", worst)):
        assert Path(item["originals"][side]).read_bytes() == Path(candidate["path"]).read_bytes()
        assert state.caption_text(Path(item["originals"][side]).with_suffix(".txt")) == CAPTION
        if candidate["role"] == "fresh":
            after_entry = next(
                e
                for e in json.loads((concept / "manifest.json").read_text())["pairs"]
                if e["pair_id"] == item["pair_id"]
            )
            assert after_entry[side + "_source"] == candidate["source"]
    assert service.undo(group["session_id"])["ok"]
    assert {p: Path(p).read_bytes() for p in before} == before
    assert json.loads((concept / "manifest.json").read_text()) == manifest_before
    assert service.current()["group"]["images"] == group["images"]


def test_missing_and_empty_captions_never_fall_back(env):
    concept, _, client, create = env
    (concept / "chosen/pair_0.txt").unlink()
    (concept / "chosen/pair_1.txt").write_text(" \n")
    service = create()
    ready(service)
    assert len(client.calls) == 4
    assert service.status()["counts"]["failed"] == 2
    service.pause(service.status()["session_id"])
    failed = service.status()["errors"][0]
    state.write_caption(concept / f"chosen/pair_{failed['pair_id']}.txt", "corrected")
    assert service.retry(service.status()["session_id"], failed["group_id"])["ok"]
    assert service.resume(service.status()["session_id"])["ok"]
    wait_for(lambda: not service.status()["running"])
    assert [c["prompt"] for c in client.calls[-2:]] == ["corrected", "corrected"]


@pytest.mark.parametrize("at_commit", [False, True])
def test_changed_caption_requeues_without_overwriting_pair(env, at_commit):
    _, _, _, create = env
    service = create()
    group = ready(service)
    service.pause(group["session_id"])
    item = service._ledger["items"][group["group_id"]]
    best, worst = group["images"][:2]
    service.select_best(group["session_id"], group["group_id"], best["id"])
    state.write_caption(Path(item["originals"]["chosen"]).with_suffix(".txt"), "Edited after generation")
    if at_commit:
        result = service.apply(group["session_id"], group["group_id"], best["id"], worst["id"])
        assert not result["ok"] and "Caption changed" in result["error"]
    else:
        assert service.current()["group"]["group_id"] != group["group_id"]
    assert item["state"] == "queued" and item["candidates"] == []


def test_caption_changed_during_request_discards_output(env):
    _, _, client, create = env
    client.release.clear()
    service = create()
    assert client.entered.wait(5)
    item = next(i for i in service._ledger["items"].values() if i["state"] == "generating")
    state.write_caption(Path(item["originals"]["chosen"]).with_suffix(".txt"), "changed in flight")
    service.pause(service.status()["session_id"])
    client.release.set()
    wait_for(lambda: not service.status()["running"])
    assert item["state"] == "queued" and item["candidates"] == []
    assert service.status()["images"] == 0


def test_settings_refresh_only_on_resume(env):
    _, output, client, create = env
    client.release.clear()
    service = create()
    assert client.entered.wait(5)
    sid = service.status()["session_id"]
    service.pause(sid)
    client.release.set()
    wait_for(lambda: not service.status()["running"])
    time.sleep(0.02)
    png(output / "changed.png", params={**PARAMS, "model": "changed-model", "steps": 77})
    service.resume(sid)
    ready(service)
    assert client.calls[0]["model"] == "new-model"
    assert all(c["model"] == "changed-model" and c["extra_params"]["steps"] == 77 for c in client.calls[1:])


def test_three_failures_pause_and_leave_work_retryable(env):
    _, _, client, create = env
    client.fail = True
    service = create()
    wait_for(lambda: not service.status()["running"])
    assert len(client.calls) == 3
    assert service.status()["counts"]["failed"] == 3
    assert service.status()["counts"]["queued"] == 1
    assert "three consecutive" in service.status()["error"]


def test_reject_stale_duplicate_foreign_and_external_changes(env):
    _, _, _, create = env
    service = create()
    group = ready(service)
    a, b = [c["id"] for c in group["images"][:2]]
    assert not service.select_best("wrong-session", group["group_id"], a)["ok"]
    assert not service.select_best(group["session_id"], "wrong-group", a)["ok"]
    assert not service.select_best(group["session_id"], group["group_id"], "foreign")["ok"]
    service.select_best(group["session_id"], group["group_id"], a)
    assert not service.apply(group["session_id"], group["group_id"], a, a)["ok"]
    item = service._ledger["items"][group["group_id"]]
    png(item["originals"]["chosen"], color=(255, 0, 0))
    result = service.apply(group["session_id"], group["group_id"], a, b)
    assert not result["ok"] and "changed outside" in result["error"]


def test_commit_failure_rolls_back_images_captions_manifest_and_ledger(env, monkeypatch):
    concept, _, _, create = env
    service = create()
    group = ready(service)
    images = candidates(service, group)
    best = next(c["id"] for c in images if c["role"] == "fresh")
    worst = next(c["id"] for c in images if c["role"] == "rejected")
    service.select_best(group["session_id"], group["group_id"], best)
    before = {str(p): p.read_bytes() for side in ("chosen", "rejected") for p in (concept / side).iterdir()}
    manifest = (concept / "manifest.json").read_bytes()
    ledger = copy.deepcopy(service._ledger)
    monkeypatch.setattr(module, "save_manifest", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    result = service.apply(group["session_id"], group["group_id"], best, worst)
    assert not result["ok"] and "disk full" in result["error"]
    assert {p: Path(p).read_bytes() for p in before} == before
    assert (concept / "manifest.json").read_bytes() == manifest
    assert service._ledger == ledger


def test_skip_undo_and_new_pass(env):
    _, _, _, create = env
    service = create()
    group = ready(service)
    assert service.skip(group["session_id"], group["group_id"])["ok"]
    assert service.current()["group"]["group_id"] != group["group_id"]
    assert service.undo(group["session_id"])["ok"]
    assert service.current()["group"]["group_id"] == group["group_id"]
    while (group := service.current()["group"]) is not None:
        service.skip(group["session_id"], group["group_id"])
    assert service.status()["complete"]
    old_id = service.status()["session_id"]
    service.cancel()
    fresh = create(new_pass=True)
    assert fresh.status()["session_id"] != old_id
    ready(fresh)


def test_review_saves_while_another_image_is_generating(env):
    _, _, client, create = env
    client.block_at = 3
    service = create()
    wait_for(lambda: len(client.calls) == 3)
    group = service.current()["group"]
    assert group and service.status()["running"]
    fresh = next(c for c in candidates(service, group) if c["role"] == "fresh")
    chosen = next(c for c in candidates(service, group) if c["role"] == "chosen")
    assert choose(service, group, fresh["id"], chosen["id"])["ok"]
    assert service.status()["counts"]["applied"] == 1
    service.pause(group["session_id"])
    client.release.set()
    wait_for(lambda: not service.status()["running"])


def test_retry_keeps_successful_partial_outputs(env):
    _, _, client, create = env
    client.block_at = 2
    service = create(images_per_pair=3)
    wait_for(lambda: len(client.calls) == 2)
    item = next(i for i in service._ledger["items"].values() if i["state"] == "generating")
    saved_id = next(c["id"] for c in item["candidates"] if c["role"] == "fresh")
    client.fail = True
    client.release.set()
    wait_for(lambda: not service.status()["running"])
    assert item["state"] == "failed"
    client.fail = False
    sid = service.status()["session_id"]
    assert service.retry(sid, item["id"])["ok"]
    assert service.resume(sid)["ok"]
    wait_for(lambda: not service.status()["running"])
    assert item["state"] == "ready"
    assert saved_id in [c["id"] for c in item["candidates"]]
    assert len(item["candidates"]) == 5


def test_caption_change_restarts_a_naturally_drained_queue(env):
    _, _, client, create = env
    service = create()
    group = ready(service)
    item = service._ledger["items"][group["group_id"]]
    state.write_caption(Path(item["originals"]["chosen"]).with_suffix(".txt"), "new wording")
    service.current()
    wait_for(lambda: len(client.calls) == 10 and not service.status()["running"])
    assert item["state"] == "ready"
    assert all(call["prompt"] == "new wording" for call in client.calls[-2:])


def test_changed_caption_missing_candidate_and_duplicate_apply(env):
    _, _, _, create = env
    service = create()
    group = ready(service)
    item = service._ledger["items"][group["group_id"]]
    Path(item["candidates"][0]["path"]).unlink()
    assert service.current()["group"]["group_id"] != group["group_id"]
    assert item["state"] == "failed"
    next_group = service.current()["group"]
    fresh = next(c for c in candidates(service, next_group) if c["role"] == "fresh")
    chosen = next(c for c in candidates(service, next_group) if c["role"] == "chosen")
    assert choose(service, next_group, fresh["id"], chosen["id"])["ok"]
    assert not service.apply(next_group["session_id"], next_group["group_id"], fresh["id"], chosen["id"])["ok"]


def test_mixed_extensions_replace_and_undo_without_stray_images(env):
    concept, _, _, create = env
    for path in (concept / "chosen").glob("*.png"):
        with Image.open(path) as image:
            image.save(path.with_suffix(".jpg"))
        path.unlink()
    manifest = json.loads((concept / "manifest.json").read_text())
    for entry in manifest["pairs"]:
        entry["chosen_file"] = entry["chosen_file"].replace(".png", ".jpg")
    (concept / "manifest.json").write_text(json.dumps(manifest))
    service = create()
    group = ready(service)
    item = service._ledger["items"][group["group_id"]]
    old = Path(item["originals"]["chosen"])
    before = old.read_bytes()
    fresh = next(c for c in candidates(service, group) if c["role"] == "fresh")
    chosen = next(c for c in candidates(service, group) if c["role"] == "chosen")
    assert choose(service, group, fresh["id"], chosen["id"])["ok"]
    assert not old.exists() and old.with_suffix(".png").is_file()
    assert service.undo(group["session_id"])["ok"]
    assert old.read_bytes() == before and not old.with_suffix(".png").exists()


@pytest.mark.parametrize("failure", ["ledger", "marker"])
def test_transaction_failure_after_manifest_write_is_rolled_back(env, monkeypatch, failure):
    concept, _, _, create = env
    service = create()
    group = ready(service)
    fresh = next(c for c in candidates(service, group) if c["role"] == "fresh")
    chosen = next(c for c in candidates(service, group) if c["role"] == "chosen")
    service.select_best(group["session_id"], group["group_id"], fresh["id"])
    before = {str(path): path.read_bytes() for side in ("chosen", "rejected") for path in (concept / side).iterdir()}
    manifest = (concept / "manifest.json").read_bytes()
    ledger = copy.deepcopy(service._ledger)
    method = "save" if failure == "ledger" else "commit"
    monkeypatch.setattr(service._store, method, lambda *args: (_ for _ in ()).throw(OSError("write failed")))
    assert not service.apply(group["session_id"], group["group_id"], fresh["id"], chosen["id"])["ok"]
    assert service._ledger == ledger
    assert (concept / "manifest.json").read_bytes() == manifest
    assert {path: Path(path).read_bytes() for path in before} == before


def test_actual_config_resolution_override_and_multiple_resolutions(env, monkeypatch):
    concept, _, _, create = env
    from web.backend.services.config_service import ConfigService
    from web.backend.services.dpo_service import DPOService

    config = SimpleNamespace(resolution="512, 1024")
    image = SimpleNamespace(enable_resolution_override=False, resolution_override="768, 1536")
    monkeypatch.setattr(ConfigService, "get_instance", lambda: SimpleNamespace(get_config_for_training=lambda: config))
    monkeypatch.setattr(
        DPOService,
        "get_instance",
        lambda: SimpleNamespace(_resolve_concepts=lambda: [SimpleNamespace(path=str(concept), image=image)]),
    )
    service = create(start=False)
    assert REAL_RESOLUTION(service, str(concept)) == 1024
    image.enable_resolution_override = True
    assert REAL_RESOLUTION(service, str(concept)) == 1536


def test_router_validation_and_anonymous_image_endpoint(env, monkeypatch):
    from web.backend.routers.dpo_replace import router

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    _, _, _, create = env
    service = create()
    group = ready(service)
    monkeypatch.setattr(module.DPOReplacePairsService, "get_instance", lambda: service)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        assert (
            client.post("/dpo/replace/start", json={"concept_dir": "anything", "images_per_pair": 11}).status_code
            == 422
        )
        response = client.get(group["images"][0]["url"])
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        assert "content-disposition" not in response.headers
        assert client.get("/dpo/replace/image/bad/group/candidate").status_code == 404


@pytest.mark.parametrize("format_name,suffix", [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp")])
def test_unavailable_local_output_downloads_and_stages_candidate(env, monkeypatch, format_name, suffix):
    _, _, client, create = env
    downloaded = []
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (42, 63, 84)).save(buffer, format=format_name)
    data = buffer.getvalue()

    def fetch_view(self, view):
        downloaded.append(view)
        return data

    monkeypatch.setattr(module, "local_path_for_view", lambda *args: None)
    monkeypatch.setattr(client, "fetch_view", fetch_view, raising=False)
    service = create(images_per_pair=1)
    group = ready(service)
    assert len(downloaded) == 4
    assert service.status()["counts"]["failed"] == 0
    fresh = next(c for c in candidates(service, group) if c["role"] == "fresh")
    assert Path(fresh["path"]).suffix == suffix
    assert Path(fresh["path"]).read_bytes() == data
    assert fresh["source"] == fresh["path"]
    assert state.caption_text(Path(fresh["path"]).with_suffix(".txt")) == CAPTION
    chosen = next(c for c in candidates(service, group) if c["role"] == "chosen")
    assert choose(service, group, fresh["id"], chosen["id"])["ok"]


def test_available_local_output_does_not_download(env, monkeypatch):
    _, _, client, create = env

    def fetch_view(self, view):
        pytest.fail("A local output should not be downloaded")

    monkeypatch.setattr(client, "fetch_view", fetch_view, raising=False)
    ready(create())


def test_download_failure_is_retryable_and_explains_remote_failure(env, monkeypatch):
    _, _, client, create = env

    def fetch_view(self, view):
        raise module.SwarmError("SwarmUI returned 404")

    monkeypatch.setattr(module, "local_path_for_view", lambda *args: None)
    monkeypatch.setattr(client, "fetch_view", fetch_view, raising=False)
    service = create()
    wait_for(lambda: not service.status()["running"])
    assert service.status()["counts"]["failed"] == 3
    assert service.status()["paused"]
    assert "downloading it from SwarmUI failed" in service.status()["errors"][0]["error"]
    assert "404" in service.status()["errors"][0]["error"]


def test_invalid_download_is_not_added_to_review(env, monkeypatch):
    _, _, client, create = env
    monkeypatch.setattr(module, "local_path_for_view", lambda *args: None)
    monkeypatch.setattr(client, "fetch_view", lambda *args: b"<html>Login required</html>", raising=False)
    service = create()
    wait_for(lambda: not service.status()["running"])
    assert service.status()["counts"]["ready"] == 0
    assert service.status()["counts"]["failed"] == 3


# --------------------------------------------------------------------------- #
# Several pairs from one group, and retiring what the review drops
# --------------------------------------------------------------------------- #


def by_role(service, group, role):
    return [c for c in candidates(service, group) if c["role"] == role]


def manifest_of(concept):
    return json.loads((concept / "manifest.json").read_text())


def test_one_best_image_can_penalise_both_old_images(env):
    concept, _, _, create = env
    service = create()
    group = ready(service)
    item = service._ledger["items"][group["group_id"]]
    before = {str(p): p.read_bytes() for side in ("chosen", "rejected") for p in (concept / side).iterdir()}
    manifest_before = manifest_of(concept)
    fresh = by_role(service, group, "fresh")[0]
    old_chosen = by_role(service, group, "chosen")[0]
    old_rejected = by_role(service, group, "rejected")[0]
    result = service.apply_pairs(
        group["session_id"], group["group_id"], [(fresh["id"], old_rejected["id"]), (fresh["id"], old_chosen["id"])]
    )
    assert result == {"ok": True, "pairs": 2}
    pairs = {e["pair_id"]: e for e in manifest_of(concept)["pairs"]}
    assert len(pairs) == 5
    # The reviewed pair keeps its id; the old chosen becomes the negative of a new pair.
    assert Path(item["originals"]["chosen"]).read_bytes() == Path(fresh["path"]).read_bytes()
    assert Path(item["originals"]["rejected"]).read_bytes() == Path(old_rejected["path"]).read_bytes()
    added = pairs[4]
    assert added["chosen_file"] == "pair_0004.png" and added["rejected_file"] == "pair_0004.png"
    assert (concept / "chosen" / "pair_0004.png").read_bytes() == Path(fresh["path"]).read_bytes()
    assert (concept / "rejected" / "pair_0004.png").read_bytes() == Path(old_chosen["path"]).read_bytes()
    assert state.caption_text(concept / "chosen" / "pair_0004.txt") == CAPTION
    assert not (concept / "rejected" / "pair_0004.txt").exists()
    assert added["chosen_source"] == fresh["source"] and added["prompt"] == CAPTION
    assert added["history"][0]["from_pair"] == item["pair_id"]
    # Both old images are still in a pair, so nothing was retired.
    assert "retired" not in manifest_of(concept) or all(
        r["reason"] != "replace_pairs_dropped" for r in manifest_of(concept)["retired"]
    )
    assert service.undo(group["session_id"])["ok"]
    assert {str(p): p.read_bytes() for side in ("chosen", "rejected") for p in (concept / side).iterdir()} == before
    assert manifest_of(concept) == manifest_before


def test_pairs_have_no_cap_but_an_image_takes_one_side(env):
    concept, _, _, create = env
    service = create(images_per_pair=4)
    group = ready(service)
    ids = [image["id"] for image in group["images"]]
    best, rest = ids[0], ids[1:]
    both = service.apply_pairs(group["session_id"], group["group_id"], [(best, rest[0]), (rest[0], rest[1])])
    assert not both["ok"] and "cannot be chosen in one pair and rejected" in both["error"]
    assert not service.apply_pairs(group["session_id"], group["group_id"], [(best, best)])["ok"]
    assert not service.apply_pairs(group["session_id"], group["group_id"], [])["ok"]
    twice = service.apply_pairs(group["session_id"], group["group_id"], [(best, rest[0]), (best, rest[0])])
    assert not twice["ok"]
    # Two chosen images against shared negatives is fine.
    pairs = [(best, other) for other in rest[1:]] + [(rest[0], rest[1])]
    result = service.apply_pairs(group["session_id"], group["group_id"], pairs, override_reversal=True)
    assert result["ok"] and result["pairs"] == 5, result
    assert len(manifest_of(concept)["pairs"]) == 4 + 4


def test_reversed_old_pair_inside_several_pairs_asks_first(env):
    _, _, _, create = env
    service = create()
    group = ready(service)
    old_chosen = by_role(service, group, "chosen")[0]["id"]
    old_rejected = by_role(service, group, "rejected")[0]["id"]
    fresh = by_role(service, group, "fresh")[0]["id"]
    pairs = [(fresh, old_chosen), (old_rejected, old_chosen)]
    result = service.apply_pairs(group["session_id"], group["group_id"], pairs)
    assert result == {"ok": False, "requires_reversal_confirmation": True}
    assert service.apply_pairs(group["session_id"], group["group_id"], pairs, override_reversal=True)["ok"]


def test_dropped_and_unused_images_are_retired_from_every_tool(env):
    from modules.util.dpo_curation_util import (
        load_manifest,
        manifest_used_fingerprints,
        manifest_used_sources,
        retired_pixel_hashes,
    )

    concept, output, _, create = env
    service = create(images_per_pair=3)
    group = ready(service)
    manifest_before = manifest_of(concept)
    fresh = by_role(service, group, "fresh")
    old = by_role(service, group, "chosen") + by_role(service, group, "rejected")
    assert service.apply_pairs(group["session_id"], group["group_id"], [(fresh[0]["id"], fresh[1]["id"])])["ok"]
    manifest = load_manifest(str(concept))
    retired = manifest["retired"]
    # Both old images were dropped; the unused fresh one sits in SwarmUI's output tree.
    assert sorted(r["reason"] for r in retired) == [
        "replace_pairs_dropped",
        "replace_pairs_dropped",
        "replace_pairs_unused",
    ]
    unused = next(r for r in retired if r["reason"] == "replace_pairs_unused")
    assert unused["source"] and Path(unused["source"]).parent == output
    assert any(Path(unused["source"]).samefile(Path(p)) for p in manifest_used_sources(manifest))
    fingerprints = manifest_used_fingerprints(manifest)
    for candidate in old:
        size = Path(candidate["path"]).stat().st_size
        assert state.fingerprint(candidate["path"]) in fingerprints[size]
    assert len(retired_pixel_hashes([str(concept)])) == 3
    assert service.undo(group["session_id"])["ok"]
    assert manifest_of(concept) == manifest_before


def test_skip_retires_unused_images_in_the_output_tree_and_undo_restores(env):
    concept, _, _, create = env
    service = create()
    group = ready(service)
    manifest_before = manifest_of(concept)
    assert service.skip(group["session_id"], group["group_id"])["ok"]
    retired = manifest_of(concept)["retired"]
    assert len(retired) == 2 and {r["reason"] for r in retired} == {"replace_pairs_unused"}
    assert service.undo(group["session_id"])["ok"]
    assert manifest_of(concept) == manifest_before


def test_fresh_images_come_back_as_data_and_never_touch_the_output_tree(env, monkeypatch):
    import base64

    concept, output, client, create = env

    def generate(self, **kwargs):
        client.calls.append(kwargs)
        assert kwargs["extra_params"]["donotsave"] is True
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), (len(client.calls), 1, 2)).save(buffer, format="PNG")
        return ["data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()]

    def no_fetch(self, *args, **kwargs):
        raise AssertionError("a data URI needs no download")

    monkeypatch.setattr(client, "generate", generate)
    monkeypatch.setattr(client, "fetch_view", no_fetch, raising=False)
    before = sorted(p.name for p in output.iterdir())
    service = create()
    group = ready(service)
    assert sorted(p.name for p in output.iterdir()) == before
    fresh = by_role(service, group, "fresh")
    assert all(Path(c["source"]) == Path(c["path"]) for c in fresh)
    assert service.skip(group["session_id"], group["group_id"])["ok"]
    # Nothing of it is anywhere a curation scan looks, so nothing needs retiring.
    assert "retired" not in manifest_of(concept)


def test_router_applies_several_pairs(env, monkeypatch):
    from web.backend.routers.dpo_replace import router

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    concept, _, _, create = env
    service = create()
    group = ready(service)
    monkeypatch.setattr(module.DPOReplacePairsService, "get_instance", lambda: service)
    fresh = by_role(service, group, "fresh")
    old = by_role(service, group, "chosen") + by_role(service, group, "rejected")
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        body = {
            "session_id": group["session_id"],
            "group_id": group["group_id"],
            "pairs": [{"chosen_id": fresh[0]["id"], "rejected_id": c["id"]} for c in old],
        }
        assert client.post("/dpo/replace/apply-pairs", json={**body, "pairs": []}).status_code == 422
        assert client.post("/dpo/replace/apply-pairs", json=body).json() == {"ok": True, "pairs": 2}
    assert len(manifest_of(concept)["pairs"]) == 5
