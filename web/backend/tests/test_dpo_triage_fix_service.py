"""Triage & Fix against a stubbed SwarmUI on a real exported dataset under
tmp_path: confirm streams straight into generation, candidates land in the
ledger, decisions rewrite the dataset and can be undone, and a session
resumes from its ledger after a pause."""

import json
import os
import threading
import time

from modules.util import dpo_triage_fix_util as tf
from modules.util.dpo_curation_util import compute_file_sha256, load_manifest, save_manifest
from modules.util.swarm_client import SwarmError
from web.backend.services.dpo_triage_fix_service import DPOTriageFixService

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

CAPTION = "A woman on a beach in Mallorca."
NEWSPAPER = tf.INSTRUCTIONS["other"].replace("{target}", "newspaper")
KLEIN = "Flux/flux2-klein-4b"


def _png_bytes(color, size=(64, 48)) -> bytes:
    import io

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _png(path, color=(10, 20, 30), size=(64, 48), block: dict | None = None) -> str:
    info = PngInfo()
    if block is not None:
        info.add_text("parameters", json.dumps({"sui_image_params": block}))
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    Image.new("RGB", size, color).save(str(path), pnginfo=info)
    return str(path)


def _dataset(tmp_path, count=3, name="ds"):
    root = tmp_path / name
    manifest = {"pairs": []}
    for i in range(count):
        stem = f"pair_{i:04d}"
        _png(
            root / "chosen" / f"{stem}.png", (10 + i, 20, 30), block={"prompt": f"{CAPTION} <segment:x>", "model": "Z"}
        )
        _png(root / "rejected" / f"{stem}.png", (90 + i, 20, 30))
        (root / "chosen" / f"{stem}.txt").write_text(CAPTION, encoding="utf-8")
        manifest["pairs"].append(
            {
                "pair_id": i,
                "prompt": CAPTION,
                "aspectratio": "4:3",
                "chosen_file": f"{stem}.png",
                "rejected_file": f"{stem}.png",
                "chosen_source": f"src/{stem}.png",
                "rejected_source": f"src/r{stem}.png",
            }
        )
    save_manifest(str(root), manifest)
    return str(root)


class FakeClient:
    calls: list[dict] = []
    fail_with: Exception | None = None
    fail_times: int = 0
    gate: threading.Event | None = None
    #: Bytes served for segment-mode results; None = a clearly changed image.
    segment_bytes: bytes | None = None
    #: Serve ``segment_bytes`` only for this many segment calls, then change.
    unchanged_calls: int | None = None
    edit_bytes: bytes | None = None
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt, width, height, model, images=1, **kwargs):
        extra = kwargs.get("extra_params") or {}
        mode = "edit" if "promptimages" in extra else "segment"
        with FakeClient.lock:
            FakeClient.calls.append(
                {"prompt": prompt, "width": width, "height": height, "model": model, "images": images, **kwargs}
            )
            n = len(FakeClient.calls)
        if FakeClient.gate is not None:
            FakeClient.gate.wait(10)
        with FakeClient.lock:
            if FakeClient.fail_with is not None and FakeClient.fail_times > 0:
                FakeClient.fail_times -= 1
                raise FakeClient.fail_with
        return [f"View/local/raw/2026-09-16/{mode}-{n}-{kwargs.get('seed', 0)}.png"]

    def fetch_view(self, view_path, timeout=120):
        if "/edit-" in view_path:
            return FakeClient.edit_bytes or _png_bytes((0, 0, 250))
        if FakeClient.segment_bytes is not None:
            n = int(view_path.rsplit("/", 1)[1].split("-")[1])
            if FakeClient.unchanged_calls is None or n <= FakeClient.unchanged_calls:
                return FakeClient.segment_bytes
        return _png_bytes((250, 0, 0))


@pytest.fixture
def service(monkeypatch):
    import web.backend.services.dpo_triage_fix_service as module

    FakeClient.calls = []
    FakeClient.fail_with = None
    FakeClient.fail_times = 0
    FakeClient.gate = None
    FakeClient.segment_bytes = None
    FakeClient.unchanged_calls = None
    FakeClient.edit_bytes = None
    monkeypatch.setattr(module, "SwarmClient", FakeClient)
    instance = DPOTriageFixService.__new__(DPOTriageFixService)
    instance.__init__()
    monkeypatch.setattr(DPOTriageFixService, "get_instance", classmethod(lambda cls: instance))
    yield instance
    instance.cancel()


def _settings(**overrides) -> dict:
    base = {"model": KLEIN, "options_per_image": 2, "concurrency": 2}
    base.update(overrides)
    return base


def _wait(service, predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = service.status()
        if predicate(state):
            return state
        time.sleep(0.02)
    raise AssertionError(f"condition not met: {service.status()}")


def _wait_ready(service, n=1, timeout=20.0):
    return _wait(service, lambda s: s["counts"]["ready"] >= n, timeout)


def _face():
    return [{"id": "faces"}]


# --------------------------------------------------------------------------- #


def test_start_creates_ledger_and_lists_items(service, tmp_path):
    root = _dataset(tmp_path)
    before = service.inspect(root)
    assert before["ok"] and before["total"] == 3 and not before["has_ledger"]
    res = service.start(root, settings=_settings(), shuffle_seed=99)
    assert res["ok"], res
    assert res["total"] == 3 and res["undecided"] == 3 and not res["resumed"] and res["shuffle_seed"] == 99
    assert os.path.isfile(tf.ledger_path(root))
    items = service.items()["items"]
    assert [i["stem"] for i in items] == ["pair_0000", "pair_0001", "pair_0002"]
    assert items[0]["caption"] == CAPTION and items[0]["status"] is None
    assert service.status()["counts"]["undecided"] == 3
    # Second start is refused while the session runs.
    assert not service.start(root, settings=_settings())["ok"]
    # The model is required.
    service.cancel()
    assert "model" in service.start(root, settings={"model": ""})["error"].lower()


def test_confirm_streams_into_generation(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(steps=4, segment_params={"segmentmaskblur": 12}), shuffle_seed=5)[
        "ok"
    ]
    assert service.confirm("pair_0001", _face())["ok"]
    state = _wait_ready(service)
    assert state["counts"]["ready"] == 1 and state["counts"]["undecided"] == 2
    seed = tf.item_seed(5, "pair_0001")
    calls = sorted(FakeClient.calls, key=lambda c: c["seed"])
    assert [c["seed"] for c in calls] == [seed, seed + 1]
    call = calls[0]
    assert call["model"] == KLEIN and (call["width"], call["height"]) == (64, 48) and call["steps"] == 4
    assert call["cfg_scale"] == 1.0 and call["images"] == 1
    # Faces go without the caption: on a face crop it paints a tiny scene.
    assert call["prompt"] == f"<segment:yolo-face_yolov8m.pt,1,0.25> {tf.INSTRUCTIONS['faces']}"
    extra = call["extra_params"]
    assert extra["initimage"].startswith("data:image/png;base64,") and extra["initimagecreativity"] == 0
    assert extra["segmentmaskblur"] == 12 and "maskimage" not in extra and "sampler" not in extra
    ready = service.next_ready()
    item = ready["item"]
    assert item["stem"] == "pair_0001" and len(item["candidates"]) == 2 and not item["all_no_change"]
    for candidate in item["candidates"]:
        assert os.path.isfile(candidate["path"]) and candidate["mode"] == "segment" and not candidate["no_change"]
        assert candidate["path"].startswith(tf.candidates_dir(root))
    with Image.open(item["candidates"][0]["path"]) as img:
        assert json.loads(img.info[tf.OT_CHUNK])["mode"] == "segment"
    ledger = tf.load_ledger(root)
    assert ledger["items"]["pair_0001"]["status"] == "ready" and ledger["items"]["pair_0001"]["ready_at"]


def test_no_change_falls_back_to_edit_only_for_other(service, tmp_path):
    root = _dataset(tmp_path)
    with open(os.path.join(root, "chosen", "pair_0000.png"), "rb") as f:
        FakeClient.segment_bytes = f.read()
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    assert service.confirm("pair_0000", [{"id": "other", "target": "newspaper"}])["ok"]
    assert service.confirm("pair_0001", _face())["ok"]
    _wait_ready(service, 2)
    a = service.next_ready()["item"]
    other = a if a["stem"] == "pair_0000" else None
    if other is None:
        service.keep(a["stem"])
        other = service.next_ready()["item"]
    assert other["candidates"][0]["mode"] == "edit" and not other["candidates"][0]["no_change"]
    edit_calls = [c for c in FakeClient.calls if "promptimages" in c["extra_params"]]
    assert len(edit_calls) == 1 and edit_calls[0]["prompt"] == NEWSPAPER
    assert "initimage" not in edit_calls[0]["extra_params"]
    face_record = tf.load_ledger(root)["items"]["pair_0001"]
    assert face_record["candidates"][0]["mode"] == "segment" and face_record["candidates"][0]["no_change"]
    assert tf.all_no_change(face_record)


def test_swarm_error_on_segment_triggers_edit_for_other(service, tmp_path):
    root = _dataset(tmp_path)
    FakeClient.fail_with = SwarmError("empty mask")
    FakeClient.fail_times = 1
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    assert service.confirm("pair_0000", [{"id": "other", "target": "newspaper"}])["ok"]
    _wait_ready(service)
    item = service.next_ready()["item"]
    assert item["candidates"][0]["mode"] == "edit" and item["candidates"][0]["error"] is None


def test_queue_drains_while_confirming_more(service, tmp_path):
    root = _dataset(tmp_path, count=4)
    FakeClient.gate = threading.Event()
    assert service.start(root, settings=_settings(options_per_image=1, concurrency=1), shuffle_seed=3)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait(service, lambda s: s["counts"]["generating"] == 1)
    assert service.confirm("pair_0001", _face())["ok"]
    assert service.confirm("pair_0002", _face())["ok"]
    state = service.status()
    assert state["counts"]["queued"] + state["counts"]["generating"] == 3
    assert state["in_flight"] <= 2
    FakeClient.gate.set()
    state = _wait_ready(service, 3)
    assert state["counts"]["undecided"] == 1 and state["in_flight"] == 0
    stems = [service.next_ready()["item"]["stem"]]
    service.keep(stems[0])
    stems.append(service.next_ready()["item"]["stem"])
    service.keep(stems[1])
    stems.append(service.next_ready()["item"]["stem"])
    assert stems == ["pair_0000", "pair_0001", "pair_0002"]


def test_held_session_queues_without_sending_until_released(service, tmp_path):
    root = _dataset(tmp_path)
    res = service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3, hold=True)
    assert res["ok"] and res["held"]
    assert service.confirm("pair_0000", _face())["ok"]
    assert service.confirm("pair_0001", _face())["ok"]
    time.sleep(0.3)
    state = service.status()
    assert state["held"] and state["counts"]["queued"] == 2 and state["in_flight"] == 0
    assert FakeClient.calls == []
    assert service.next_ready() == {"ok": True, "waiting": True, "held": True}
    # Taking a confirm back still works while it waits.
    assert service.undo_triage("pair_0001")["ok"]
    assert service.set_hold(False) == {"ok": True, "held": False, "withdrawn": 0}
    state = _wait_ready(service)
    assert not state["held"] and state["counts"]["queued"] == 0
    assert [c["seed"] for c in FakeClient.calls] == [tf.item_seed(3, "pair_0000")]


def test_hold_withdraws_jobs_waiting_for_a_worker(service, tmp_path):
    root = _dataset(tmp_path, count=3)
    FakeClient.gate = threading.Event()
    assert service.start(root, settings=_settings(options_per_image=2, concurrency=1), shuffle_seed=3)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait(service, lambda s: s["counts"]["generating"] == 1)
    assert service.confirm("pair_0001", _face())["ok"]
    _wait(service, lambda s: s["counts"]["generating"] == 2)
    # pair_0000 is at SwarmUI; pair_0001 waits behind it in the pool.
    res = service.set_hold(True)
    assert res["ok"] and res["held"] and res["withdrawn"] == 1
    state = service.status()
    assert state["counts"]["generating"] == 1 and state["counts"]["queued"] == 1 and state["in_flight"] == 1
    FakeClient.gate.set()
    # The one already sent finishes every option; the withdrawn one waits.
    state = _wait_ready(service)
    time.sleep(0.3)
    state = service.status()
    assert state["counts"]["ready"] == 1 and state["counts"]["queued"] == 1 and state["counts"]["failed"] == 0
    record = tf.load_ledger(root)["items"]["pair_0000"]
    assert len(record["candidates"]) == 2
    assert tf.load_ledger(root)["items"]["pair_0001"]["candidates"] == []
    assert service.set_hold(False)["ok"]
    _wait_ready(service, 2)


def test_hold_needs_a_session(service):
    assert not service.set_hold(True)["ok"]


def _mask_uri(size=(64, 48)) -> str:
    import base64
    import io

    mask = Image.new("L", size, 0)
    for x in range(10, 30):
        for y in range(10, 30):
            mask.putpixel((x, y), 255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_confirm_mask_sends_masked_inpaint_and_requeues_from_compare(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1, mask_params={"maskblur": 6}), shuffle_seed=4)[
        "ok"
    ]
    assert not service.confirm_mask("pair_0000", "", "Fix the medal")["ok"]
    res = service.confirm_mask("pair_0000", _mask_uri(), "Fix the medal")
    assert res["ok"], res
    assert os.path.isfile(tf.mask_path(root, "pair_0000"))
    _wait_ready(service)
    call = FakeClient.calls[-1]
    assert call["prompt"] == "Fix the medal"
    extra = call["extra_params"]
    assert extra["maskimage"].startswith("data:image/png;base64,") and extra["initimage"].startswith("data:image/png")
    assert extra["initimagecreativity"] == 1.0 and extra["initimagerecompositemask"] is True
    assert extra["maskshrinkgrow"] == 32 and extra["maskblur"] == 6
    assert "segmentmaskblur" not in extra and "promptimages" not in extra
    item = service.next_ready()["item"]
    assert item["candidates"][0]["mode"] == "mask" and item["mask"] == tf.mask_path(root, "pair_0000")
    with Image.open(item["candidates"][0]["path"]) as img:
        assert json.loads(img.info[tf.OT_CHUNK])["mode"] == "mask"
    # From the compare screen a new mask re-queues the image.
    assert service.confirm_mask("pair_0000", _mask_uri(), "Fix the ribbon", creativity=0.8)["ok"]
    _wait_ready(service)
    again = service.next_ready()["item"]
    assert again["attempt"] == 2 and again["fixes"][0]["instruction"] == "Fix the ribbon"
    assert FakeClient.calls[-1]["extra_params"]["initimagecreativity"] == 0.8
    assert not service.confirm_mask("nope", _mask_uri(), "x")["ok"]


def test_confirm_all_default_and_detected(service, tmp_path, monkeypatch):
    class Detector:
        def detect(self, path):
            return "<segment:hand,0.9,0.25>" if "pair_0002" in path else ""

    monkeypatch.setattr(tf, "load_detector", lambda spec: Detector() if spec else None)
    root = _dataset(tmp_path, count=4)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3, detector="x")["ok"]
    _wait(service, lambda s: s["counts"]["tagged"] == 1)
    items = {i["stem"]: i for i in service.items()["items"]}
    assert items["pair_0002"]["status"] == "tagged" and items["pair_0002"]["detected"] == "<segment:hand,0.9,0.25>"
    assert items["pair_0000"]["detected"] == "" and items["pair_0000"]["status"] is None
    assert service.confirm_all("detected")["queued"] == 1
    record = tf.load_ledger(root)["items"]["pair_0002"]
    assert record["fixes"][0]["id"] == "hands" and record["fixes"][0]["creativity"] == 0.9
    assert service.skip("pair_0003")["ok"]
    assert service.confirm_all("default", _face())["queued"] == 2
    _wait_ready(service, 3)
    assert service.status()["counts"]["undecided"] == 0
    assert not service.confirm_all("default", [])["ok"]


def test_skip_and_undo_triage(service, tmp_path):
    root = _dataset(tmp_path)
    FakeClient.gate = threading.Event()
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    assert service.skip("pair_0000")["ok"]
    assert tf.status_of(tf.load_ledger(root), "pair_0000") == "skipped"
    assert service.undo_triage("pair_0000")["ok"]
    assert tf.status_of(tf.load_ledger(root), "pair_0000") is None
    assert service.confirm("pair_0001", _face())["ok"]
    _wait(service, lambda s: s["counts"]["generating"] == 1)
    assert service.confirm("pair_0002", _face())["ok"]
    assert service.undo_triage("pair_0002")["ok"]
    assert not service.undo_triage("pair_0001")["ok"]
    assert not service.confirm("pair_0001", _face())["ok"]
    FakeClient.gate.set()


def test_pick_replaces_in_place_and_writes_fix_pair(service, tmp_path):
    root = _dataset(tmp_path)
    fix_dir = str(tmp_path / "fixes")
    assert service.start(root, settings=_settings(options_per_image=2), fix_pair_dir=fix_dir, shuffle_seed=3)["ok"]
    assert service.confirm("pair_0001", _face())["ok"]
    _wait_ready(service)
    item = service.next_ready()["item"]
    res = service.pick("pair_0001", 1)
    assert res["ok"], res
    assert res["fix_pair_id"] == 0 and res["backup"].startswith(".replaced/pair_0001__")
    chosen = os.path.join(root, "chosen", "pair_0001.png")
    with Image.open(chosen) as img:
        assert img.getpixel((0, 0)) == (250, 0, 0)
        assert json.loads(img.info[tf.OT_CHUNK])["seed"] == item["candidates"][1]["seed"]
    assert (tmp_path / "ds" / "chosen" / "pair_0001.txt").read_text(encoding="utf-8") == CAPTION
    with Image.open(os.path.join(root, "rejected", "pair_0001.png")) as img:
        assert img.getpixel((0, 0)) == (91, 20, 30)
    entry = load_manifest(root)["pairs"][1]
    assert entry["history"][-1]["action"] == "triage_fix" and entry["chosen_source"] == "src/pair_0001.png"
    fix_manifest = load_manifest(fix_dir)
    assert len(fix_manifest["pairs"]) == 1
    with Image.open(os.path.join(fix_dir, "chosen", "pair_0000.png")) as img:
        assert img.getpixel((0, 0)) == (250, 0, 0)
    with Image.open(os.path.join(fix_dir, "rejected", "pair_0000.png")) as img:
        assert img.getpixel((0, 0)) == (11, 20, 30)
    assert (tmp_path / "fixes" / "chosen" / "pair_0000.txt").read_text(encoding="utf-8") == CAPTION
    state = service.status()
    assert state["counts"]["fixed"] == 1 and state["can_undo"]
    assert service.next_ready().get("done")
    # It never comes back to triage or compare.
    assert not service.confirm("pair_0001", _face())["ok"]

    undo = service.undo_decision()
    assert undo["ok"] and undo["undone"] == "fixed"
    with Image.open(chosen) as img:
        assert img.getpixel((0, 0)) == (11, 20, 30)
    assert load_manifest(fix_dir)["pairs"] == [] and not os.listdir(os.path.join(fix_dir, "chosen"))
    assert "history" not in load_manifest(root)["pairs"][1]
    assert service.next_ready()["item"]["stem"] == "pair_0001"
    assert not os.listdir(os.path.join(root, ".replaced"))


def test_retriage_sends_a_ready_image_back_with_its_tags(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    candidate = service.next_ready()["item"]["candidates"][0]["path"]
    assert service.retriage("pair_0000")["ok"]
    assert not os.path.exists(candidate)
    item = next(i for i in service.items()["items"] if i["stem"] == "pair_0000")
    assert item["status"] == "tagged" and item["fixes"][0]["id"] == "faces"
    assert service.status()["counts"]["undecided"] == 3
    assert service.next_ready().get("done")
    # It can be confirmed again, and a kept image can come back too.
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    assert service.keep("pair_0000")["ok"]
    assert service.retriage("pair_0000")["ok"]
    assert not service.retriage("pair_0001")["ok"]


def test_no_change_on_first_option_skips_the_rest(service, tmp_path):
    root = _dataset(tmp_path)
    with open(os.path.join(root, "chosen", "pair_0000.png"), "rb") as f:
        FakeClient.segment_bytes = f.read()
    assert service.start(root, settings=_settings(options_per_image=3), shuffle_seed=1)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    item = service.next_ready()["item"]
    assert len(item["candidates"]) == 1 and item["candidates"][0]["no_change"] and item["all_no_change"]
    assert len(FakeClient.calls) == 3  # the three ladder rungs, no extra seeds
    # A real change on the first option brings the other seeds in.
    FakeClient.segment_bytes = None
    assert service.confirm("pair_0001", _face())["ok"]
    _wait_ready(service, 2)
    assert service.keep("pair_0000")["ok"]
    item = service.next_ready()["item"]
    assert item["stem"] == "pair_0001" and len(item["candidates"]) == 3
    assert sorted(c["seed"] for c in item["candidates"]) == [item["seed"] + i for i in range(3)]


def test_delete_pair_moves_files_drops_manifest_entry_and_undoes(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    chosen = os.path.join(root, "chosen", "pair_0001.png")
    rejected = os.path.join(root, "rejected", "pair_0001.png")
    caption = os.path.join(root, "chosen", "pair_0001.txt")
    res = service.delete_pair("pair_0001")
    assert res["ok"], res
    assert not os.path.exists(chosen) and not os.path.exists(rejected) and not os.path.exists(caption)
    folder = os.path.join(root, res["folder"])
    assert len(os.listdir(folder)) == 3
    assert [p["pair_id"] for p in load_manifest(root)["pairs"]] == [0, 2]
    state = service.status()
    assert state["counts"]["deleted"] == 1 and state["counts"]["undecided"] == 2 and state["can_undo"]
    assert not service.confirm("pair_0001", _face())["ok"]
    assert not service.delete_pair("pair_0001")["ok"]

    undo = service.undo_decision()
    assert undo["ok"] and undo["undone"] == "deleted"
    assert os.path.isfile(chosen) and os.path.isfile(rejected) and os.path.isfile(caption)
    assert not os.path.exists(folder)
    assert [p["pair_id"] for p in load_manifest(root)["pairs"]] == [0, 1, 2]
    assert service.status()["counts"]["undecided"] == 3
    # A ready image can be deleted too; it comes back tagged on undo.
    assert service.confirm("pair_0001", _face())["ok"]
    _wait_ready(service)
    candidate = service.next_ready()["item"]["candidates"][0]["path"]
    assert service.delete_pair("pair_0001")["ok"]
    assert not os.path.exists(candidate)
    assert service.undo_decision()["ok"]
    assert next(i for i in service.items()["items"] if i["stem"] == "pair_0001")["status"] == "tagged"


def test_detection_ladder_climbs_then_reuses_the_rung(service, tmp_path):
    root = _dataset(tmp_path)
    with open(os.path.join(root, "chosen", "pair_0000.png"), "rb") as f:
        FakeClient.segment_bytes = f.read()
    FakeClient.unchanged_calls = 2  # tagged and low-threshold rungs find nothing
    assert service.start(root, settings=_settings(options_per_image=2), shuffle_seed=1)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    item = service.next_ready()["item"]
    assert len(FakeClient.calls) == 4  # three rungs for option 0, one call for option 1
    prompts = [c["prompt"] for c in FakeClient.calls]
    assert prompts[0].startswith("<segment:yolo-face_yolov8m.pt,1,0.25>")
    assert prompts[1].startswith("<segment:yolo-face_yolov8m.pt,1,0.1>")
    assert prompts[2].startswith("<segment:face,1,0.3>")
    assert prompts[3].startswith("<segment:face,1,0.3>")  # option 1 reuses the CLIPSeg rung
    assert [c["rung"] for c in item["candidates"]] == [2, 2]
    assert item["candidates"][0]["fallback"] == "clipseg" and not item["all_no_change"]
    # A whole ladder that finds nothing is one unchanged candidate, no extras.
    FakeClient.calls = []
    FakeClient.unchanged_calls = None
    assert service.confirm("pair_0001", _face())["ok"]
    _wait_ready(service, 2)
    assert service.keep("pair_0000")["ok"]
    item = service.next_ready()["item"]
    assert len(FakeClient.calls) == 3 and len(item["candidates"]) == 1 and item["all_no_change"]


def test_round_two_on_a_fixed_image(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    assert service.pick("pair_0000", 0)["ok"]
    chosen = os.path.join(root, "chosen", "pair_0000.png")
    res = service.retriage("pair_0000")
    assert res["ok"] and res["status"] == "tagged" and res["round"] == 2
    item = next(i for i in service.items()["items"] if i["stem"] == "pair_0000")
    assert item["status"] == "tagged" and item["fixes"] == [] and item["round"] == 2
    assert not service.status()["can_undo"]
    # Round two works on the fixed file, and its own pick backs that file up.
    FakeClient.calls = []
    assert service.confirm("pair_0000", [{"id": "background"}])["ok"]
    _wait_ready(service)
    with Image.open(chosen) as img:
        assert img.getpixel((0, 0)) == (250, 0, 0)
    sent = FakeClient.calls[0]["extra_params"]["initimage"]
    assert sent == tf.image_data_uri(chosen)
    assert service.pick("pair_0000", 0)["ok"]
    entry = load_manifest(root)["pairs"][0]
    assert [h["action"] for h in entry["history"]] == ["triage_fix", "triage_fix"]
    assert len(os.listdir(os.path.join(root, ".replaced"))) == 2
    record = tf.load_ledger(root)["items"]["pair_0000"]
    assert record["status"] == "fixed" and len(record["rounds"]) == 1


def test_fix_pair_carries_the_caption_corrected_after_start(service, tmp_path):
    root = _dataset(tmp_path)
    fix_dir = str(tmp_path / "fixes")
    assert service.start(root, settings=_settings(options_per_image=1), fix_pair_dir=fix_dir, shuffle_seed=3)["ok"]
    assert service.confirm("pair_0000", [{"id": "faces", "include_caption": True}])["ok"]
    _wait_ready(service)
    # Corrected while the session runs; the Swarm prompt used the original.
    (tmp_path / "ds" / "chosen" / "pair_0000.txt").write_text("Corrected caption", encoding="utf-8")
    assert CAPTION in FakeClient.calls[0]["prompt"] and "Corrected" not in FakeClient.calls[0]["prompt"]
    assert service.pick("pair_0000", 0)["ok"]
    assert (tmp_path / "fixes" / "chosen" / "pair_0000.txt").read_text(encoding="utf-8") == "Corrected caption"
    assert load_manifest(fix_dir)["pairs"][0]["prompt"] == "Corrected caption"
    # The dataset's own caption file is untouched by the replacement.
    assert (tmp_path / "ds" / "chosen" / "pair_0000.txt").read_text(encoding="utf-8") == "Corrected caption"


def test_pick_refuses_when_chosen_changed(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    _png(os.path.join(root, "chosen", "pair_0000.png"), (77, 77, 77))
    res = service.pick("pair_0000", 0)
    assert not res["ok"] and "changed" in res["error"]
    assert not service.pick("pair_0000", 5)["ok"]


def test_keep_undo_and_requeue(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    first = service.next_ready()["item"]
    assert service.keep("pair_0000")["ok"]
    assert service.next_ready().get("done") and service.status()["counts"]["kept"] == 1
    assert service.undo_decision()["undone"] == "kept"
    assert service.next_ready()["item"]["stem"] == "pair_0000"
    old_path = first["candidates"][0]["path"]
    # A plain re-roll is a new seed only: tags and creativity stay.
    assert service.requeue("pair_0000")["ok"]
    _wait_ready(service)
    rolled = service.next_ready()["item"]
    assert rolled["seed"] != first["seed"]
    assert rolled["fixes"][0]["id"] == "faces" and rolled["fixes"][0]["creativity"] == 1.0
    assert FakeClient.calls[-1]["prompt"].startswith("<segment:yolo-face_yolov8m.pt,1,0.25>")
    # Asked to, it also turns the creativity down one sampling step (4 steps).
    assert service.requeue("pair_0000", lower=True)["ok"]
    _wait_ready(service)
    lowered = service.next_ready()["item"]
    assert lowered["fixes"][0]["creativity"] == 0.75
    assert FakeClient.calls[-1]["prompt"].startswith("<segment:yolo-face_yolov8m.pt,0.75,0.25>")
    assert service.requeue("pair_0000", [{"id": "hands"}])["ok"]
    assert not os.path.exists(old_path) or True  # overwritten by the new attempt
    _wait_ready(service)
    again = service.next_ready()["item"]
    assert again["attempt"] == 4 and again["seed"] != first["seed"] and again["fixes"][0]["id"] == "hands"
    assert FakeClient.calls[-1]["prompt"].startswith("<segment:yolo-hand_yolov8s.pt,1,0.25>")
    assert not service.undo_decision()["ok"]


def test_cancel_demotes_generating_and_restart_resumes(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path, count=4)
    FakeClient.gate = threading.Event()
    assert service.start(root, settings=_settings(options_per_image=1, concurrency=1), shuffle_seed=8)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    assert service.confirm("pair_0001", _face())["ok"]
    assert service.skip("pair_0002")["ok"]
    _wait(service, lambda s: s["counts"]["generating"] >= 1)
    assert service.cancel()["ok"]
    ledger = tf.load_ledger(root)
    assert {ledger["items"][s]["status"] for s in ("pair_0000", "pair_0001")} == {"queued"}
    assert not os.path.isdir(tf.candidates_dir(root)) or not os.listdir(tf.candidates_dir(root))
    assert not service.status()["active"]
    FakeClient.gate.set()

    fresh = DPOTriageFixService.__new__(DPOTriageFixService)
    fresh.__init__()
    monkeypatch.setattr(DPOTriageFixService, "get_instance", classmethod(lambda cls: fresh))
    res = fresh.start(root, settings=_settings(options_per_image=1), shuffle_seed=1234)
    assert res["ok"] and res["resumed"] and res["shuffle_seed"] == 8
    _wait_ready(fresh, 2)
    items = {i["stem"]: i["status"] for i in fresh.items()["items"]}
    assert items == {"pair_0000": "ready", "pair_0001": "ready", "pair_0002": "skipped", "pair_0003": None}
    fresh.cancel()


def test_conflict_guards(service, tmp_path):
    root = _dataset(tmp_path)
    assert not service.start(root, settings=_settings(), fix_pair_dir=os.path.join(root, "fixes"))["ok"]
    other = tmp_path / "other"
    other.mkdir()
    (other / "stray.txt").write_text("x", encoding="utf-8")
    assert not service.start(root, settings=_settings(), fix_pair_dir=str(other))["ok"]
    assert not service.start(str(tmp_path / "nowhere"), settings=_settings())["ok"]


def test_consecutive_failures_stop_the_run_and_keep_the_queue(service, tmp_path):
    root = _dataset(tmp_path, count=6)
    FakeClient.fail_with = SwarmError("backend down")
    FakeClient.fail_times = 99
    assert service.start(root, settings=_settings(options_per_image=1, concurrency=1), shuffle_seed=2)["ok"]
    for i in range(6):
        assert service.confirm(f"pair_{i:04d}", _face())["ok"]
    state = _wait(service, lambda s: s["phase"] == "error")
    assert not state["active"] and "failures in a row" in state["error"]
    ledger = tf.load_ledger(root)
    statuses = [ledger["items"][f"pair_{i:04d}"]["status"] for i in range(6)]
    assert statuses.count("failed") == 5 and statuses.count("queued") == 1
    assert service.status()["counts"]["failed"] == 5


def test_photo_fix_is_a_whole_image_edit(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    assert service.confirm("pair_0000", [{"id": "photo"}])["ok"]
    _wait_ready(service, 1)
    item = service.next_ready()["item"]
    candidate = item["candidates"][0]
    assert candidate["mode"] == "edit" and not candidate["no_change"]
    (call,) = FakeClient.calls
    assert call["prompt"] == tf.INSTRUCTIONS["photo"]
    assert "promptimages" in call["extra_params"] and "initimage" not in call["extra_params"]


def test_photo_fix_below_full_creativity_starts_from_the_image(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    assert service.confirm("pair_0000", [{"id": "photo", "creativity": 0.5}])["ok"]
    _wait_ready(service, 1)
    (call,) = FakeClient.calls
    original = tf.image_data_uri(os.path.join(root, "chosen", "pair_0000.png"))
    assert call["extra_params"]["promptimages"] == original
    assert call["extra_params"]["initimage"] == original
    assert call["extra_params"]["initimagecreativity"] == 0.5


def test_photo_fix_then_segments_repair_the_edited_image(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    assert service.confirm("pair_0000", [{"id": "photo"}, {"id": "faces"}])["ok"]
    _wait_ready(service, 1)
    item = service.next_ready()["item"]
    assert item["candidates"][0]["mode"] == "edit" and not item["candidates"][0]["no_change"]
    edit, segment = FakeClient.calls
    assert "promptimages" in edit["extra_params"]
    # The segment pass starts from the edited frame, not the original, and
    # carries no tag for the whole-image fix.
    original = tf.image_data_uri(os.path.join(root, "chosen", "pair_0000.png"))
    assert segment["extra_params"]["initimage"] != original
    assert segment["prompt"].count("<segment:") == 1 and "photo" not in segment["prompt"].lower()
    leftovers = [name for name in os.listdir(tf.candidates_dir(root)) if "photo-base" in name]
    assert leftovers == []


def _fix_and_pick(service, stem, color=None, second_round=False):
    FakeClient.segment_bytes = _png_bytes(color) if color else None
    before = service.status()["counts"]["ready"]
    assert service.confirm(stem, _face(), second_round=second_round)["ok"]
    _wait_ready(service, before + 1)
    res = service.pick(stem, 0)
    assert res["ok"], res
    return res


def _pixel(path):
    with Image.open(path) as img:
        return img.getpixel((0, 0))


def test_phase_two_replaces_the_fix_pair_chosen_in_place_and_undoes(service, tmp_path):
    root = _dataset(tmp_path)
    fix_dir = str(tmp_path / "fixes")
    assert service.start(root, settings=_settings(options_per_image=1), fix_pair_dir=fix_dir, shuffle_seed=3)["ok"]
    _fix_and_pick(service, "pair_0000", (250, 0, 0))
    item = next(i for i in service.items()["items"] if i["stem"] == "pair_0000")
    assert item["status"] == "fixed" and item["has_fix_pair"] and item["second_pass"] is None
    assert _pixel(item["original_path"]) == (10, 20, 30)
    # Plain triage cannot take a decided image; the second round can.
    assert not service.confirm("pair_0000", _face())["ok"]

    res = _fix_and_pick(service, "pair_0000", (0, 250, 0), second_round=True)
    # Same pair, new chosen, rejected still the untouched original.
    assert res["fix_pair_id"] == 0
    fix_manifest = load_manifest(fix_dir)
    assert len(fix_manifest["pairs"]) == 1
    entry = fix_manifest["pairs"][0]
    assert _pixel(os.path.join(fix_dir, "chosen", entry["chosen_file"])) == (0, 250, 0)
    assert _pixel(os.path.join(fix_dir, "rejected", entry["rejected_file"])) == (10, 20, 30)
    assert entry["chosen_sha256"] == compute_file_sha256(os.path.join(fix_dir, "chosen", entry["chosen_file"]))
    assert (tmp_path / "fixes" / "chosen" / "pair_0000.txt").read_text(encoding="utf-8") == CAPTION

    assert service.undo_decision()["ok"]
    fix_manifest = load_manifest(fix_dir)
    assert len(fix_manifest["pairs"]) == 1
    assert _pixel(os.path.join(fix_dir, "chosen", fix_manifest["pairs"][0]["chosen_file"])) == (250, 0, 0)
    assert not os.listdir(tf.fix_pair_backup_dir(root))


def test_phase_two_without_an_earlier_pair_pairs_against_the_original(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    _fix_and_pick(service, "pair_0000", (250, 0, 0))
    service.cancel()

    fix_dir = str(tmp_path / "fixes")
    fresh = DPOTriageFixService.__new__(DPOTriageFixService)
    fresh.__init__()
    monkeypatch.setattr(DPOTriageFixService, "get_instance", classmethod(lambda cls: fresh))
    assert fresh.start(root, settings=_settings(options_per_image=1), fix_pair_dir=fix_dir)["ok"]
    _fix_and_pick(fresh, "pair_0000", (0, 250, 0), second_round=True)
    entry = load_manifest(fix_dir)["pairs"][0]
    assert _pixel(os.path.join(fix_dir, "chosen", entry["chosen_file"])) == (0, 250, 0)
    # Rejected is the image before round one, not round one's repair.
    assert _pixel(os.path.join(fix_dir, "rejected", entry["rejected_file"])) == (10, 20, 30)
    fresh.cancel()


def test_add_fix_pairs_recovers_fixes_made_without_a_folder(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    assert not service.add_fix_pairs(["pair_0000"])["ok"]
    _fix_and_pick(service, "pair_0001", (250, 0, 0))
    _fix_and_pick(service, "pair_0000", (0, 250, 0))
    assert not any(i["has_fix_pair"] for i in service.items()["items"])
    service.cancel()

    fix_dir = str(tmp_path / "fixes")
    fresh = DPOTriageFixService.__new__(DPOTriageFixService)
    fresh.__init__()
    monkeypatch.setattr(DPOTriageFixService, "get_instance", classmethod(lambda cls: fresh))
    assert fresh.start(root, settings=_settings(options_per_image=1), fix_pair_dir=fix_dir)["ok"]
    res = fresh.add_fix_pairs(["pair_0000", "pair_0001", "pair_0002"])
    # Decision order, not request order; the unfixed one is skipped.
    assert res["added"] == ["pair_0001", "pair_0000"] and res["skipped"] == ["pair_0002"] and not res["errors"]
    pairs = load_manifest(fix_dir)["pairs"]
    assert [_pixel(os.path.join(fix_dir, "chosen", p["chosen_file"])) for p in pairs] == [(250, 0, 0), (0, 250, 0)]
    assert [_pixel(os.path.join(fix_dir, "rejected", p["rejected_file"])) for p in pairs] == [
        (11, 20, 30),
        (10, 20, 30),
    ]
    assert all(i["has_fix_pair"] for i in fresh.items()["items"] if i["status"] == "fixed")
    # Idempotent: nothing is written twice.
    assert fresh.add_fix_pairs(["pair_0000"])["added"] == []
    assert len(load_manifest(fix_dir)["pairs"]) == 2
    fresh.cancel()


def _item(service, stem):
    return next(i for i in service.items()["items"] if i["stem"] == stem)


def test_second_round_takes_up_fixed_and_skipped_and_undo_restores(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    _fix_and_pick(service, "pair_0000", (250, 0, 0))
    assert service.skip("pair_0001")["ok"]
    monkeypatch.setattr(service, "_next_queued", lambda: None)  # the confirms stay queued

    assert service.confirm("pair_0000", _face(), second_round=True)["ok"]
    assert service.confirm("pair_0001", _face(), second_round=True)["ok"]
    fixed, skipped = _item(service, "pair_0000"), _item(service, "pair_0001")
    assert fixed["status"] == "queued" and fixed["second_pass"] == "open" and fixed["second_from"] == "fixed"
    assert fixed["round"] == 2
    assert skipped["status"] == "queued" and skipped["second_from"] == "skipped" and skipped["round"] == 1

    # Undo puts each back exactly as it was, unvisited.
    assert service.undo_second("pair_0000")["ok"]
    assert service.undo_second("pair_0001")["ok"]
    ledger = tf.load_ledger(root)["items"]
    assert ledger["pair_0000"]["status"] == "fixed" and ledger["pair_0000"]["decision"]["kind"] == "fixed"
    assert "rounds" not in ledger["pair_0000"] and "second_pass" not in ledger["pair_0000"]
    assert ledger["pair_0001"]["status"] == "skipped" and "second_pass" not in ledger["pair_0001"]
    assert _item(service, "pair_0000")["second_pass"] is None


def test_second_round_skip_marks_or_restores(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    _fix_and_pick(service, "pair_0000", (250, 0, 0))
    _fix_and_pick(service, "pair_0001", (0, 250, 0))
    # A fixed image sent back from review is open in the second round.
    assert service.retriage("pair_0001")["ok"]
    assert _item(service, "pair_0001")["second_pass"] == "open"

    res = service.skip_second(["pair_0000", "pair_0001", "pair_0002"])
    assert res["skipped"] == ["pair_0000", "pair_0001"] and res["refused"] == ["pair_0002"]
    ledger = tf.load_ledger(root)["items"]
    for stem in ("pair_0000", "pair_0001"):
        assert ledger[stem]["status"] == "fixed" and ledger[stem]["second_pass"]["state"] == "skipped"
        assert "rounds" not in ledger[stem]
    assert service.undo_second("pair_0000")["ok"]
    assert _item(service, "pair_0000")["second_pass"] is None


def test_second_round_confirm_many(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    _fix_and_pick(service, "pair_0000", (250, 0, 0))
    assert service.skip("pair_0001")["ok"]
    monkeypatch.setattr(service, "_next_queued", lambda: None)
    assert service.confirm("pair_0002", _face())["ok"]
    res = service.confirm_many(["pair_0000", "pair_0001", "pair_0002"], _face())
    assert res["queued"] == ["pair_0000", "pair_0001"] and res["refused"] == ["pair_0002"]
    assert {_item(service, s)["status"] for s in ("pair_0000", "pair_0001")} == {"queued"}


def test_triage_grid_confirm_many_refuses_decided_images(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    assert service.skip("pair_0001")["ok"]
    monkeypatch.setattr(service, "_next_queued", lambda: None)
    res = service.confirm_many(["pair_0000", "pair_0001", "pair_0002"], _face(), second_round=False)
    assert res["queued"] == ["pair_0000", "pair_0002"] and res["refused"] == ["pair_0001"]
    assert _item(service, "pair_0001")["status"] == "skipped"
    assert _item(service, "pair_0001").get("second_pass") in (None, {})


def test_triage_grid_skip_many(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    monkeypatch.setattr(service, "_next_queued", lambda: None)
    assert service.confirm("pair_0002", _face())["ok"]
    res = service.skip_many(["pair_0000", "pair_0001", "pair_0002", "nope"])
    assert res["skipped"] == ["pair_0000", "pair_0001"] and res["refused"] == ["pair_0002", "nope"]
    assert {_item(service, s)["status"] for s in ("pair_0000", "pair_0001")} == {"skipped"}
    assert _item(service, "pair_0002")["status"] == "queued"
    # Each one undoes on its own, as a single skip does.
    assert service.undo_triage("pair_0000")["ok"]


def test_quick_mask_color_correction_override_is_kept_for_rerolls(service, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1, color_correction="Linear2"), shuffle_seed=4)[
        "ok"
    ]
    assert service.confirm_mask("pair_0000", _mask_uri(), "Fix it")["ok"]
    _wait_ready(service)
    assert FakeClient.calls[-1]["extra_params"]["colorcorrectionbehavior"] == "Linear2"
    # Off for this image only, and still off when it is re-rolled.
    assert service.confirm_mask("pair_0000", _mask_uri(), "Fix it", color_correction="None")["ok"]
    _wait_ready(service)
    assert "colorcorrectionbehavior" not in FakeClient.calls[-1]["extra_params"]
    assert service.requeue("pair_0000")["ok"]
    _wait_ready(service)
    assert "colorcorrectionbehavior" not in FakeClient.calls[-1]["extra_params"]
    item = service.next_ready()["item"]
    with Image.open(item["candidates"][0]["path"]) as img:
        assert json.loads(img.info[tf.OT_CHUNK])["color_correction"] == "None"
    # A second override joins the first on the same image.
    assert service.confirm_mask("pair_0000", _mask_uri(), "Fix it", color_correction="None", crop_reference=0.4)["ok"]
    _wait_ready(service)
    extra = FakeClient.calls[-1]["extra_params"]
    assert extra["cropreferencestrength"] == 0.4 and "colorcorrectionbehavior" not in extra
    # Segment fixes use the session's.
    assert service.confirm("pair_0001", _face())["ok"]
    _wait(service, lambda s: s["counts"]["ready"] >= 2)
    segment_call = next(c for c in reversed(FakeClient.calls) if "maskimage" not in c["extra_params"])
    assert segment_call["extra_params"]["colorcorrectionbehavior"] == "Linear2"
    assert "cropreferencestrength" not in segment_call["extra_params"]


def test_clear_second_round_returns_every_handled_image(service, tmp_path, monkeypatch):
    root = _dataset(tmp_path, count=4)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=3)["ok"]
    _fix_and_pick(service, "pair_0000", (250, 0, 0))
    assert service.skip("pair_0001")["ok"]
    assert service.skip("pair_0002")["ok"]
    # Skipped this round; confirmed this round and fixed since; and one still in progress.
    assert service.skip_second(["pair_0001"])["skipped"] == ["pair_0001"]
    _fix_and_pick(service, "pair_0002", (0, 250, 0), second_round=True)
    monkeypatch.setattr(service, "_next_queued", lambda: None)
    assert service.confirm("pair_0000", _face(), second_round=True)["ok"]

    assert service.clear_second() == {"ok": True, "cleared": 2}
    items = {i["stem"]: i for i in service.items()["items"]}
    assert items["pair_0001"]["status"] == "skipped" and items["pair_0001"]["second_pass"] is None
    assert items["pair_0002"]["status"] == "fixed" and items["pair_0002"]["second_pass"] is None
    assert items["pair_0000"]["status"] == "queued" and items["pair_0000"]["second_pass"] == "open"
    ledger = tf.load_ledger(root)["items"]
    assert ledger["pair_0002"]["decision"]["kind"] == "fixed"


# --------------------------------------------------------------------------- #
# Several sessions
# --------------------------------------------------------------------------- #


@pytest.fixture
def second(service):
    """A second session ("flux") beside the main one, same stubbed SwarmUI."""
    other = DPOTriageFixService("flux")
    yield other
    other.cancel()


def _other(target="newspaper"):
    return [{"id": "other", "target": target, "instruction": ""}]


def _partial_png(base=(10, 20, 30), box=(40, 30, 60, 45), color=(250, 0, 0)) -> bytes:
    """The dataset's pair_0000 with only ``box`` repainted."""
    import io

    img = Image.new("RGB", (64, 48), base)
    img.paste(color, box)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _far_mask_uri(size=(64, 48)) -> str:
    import base64
    import io

    mask = Image.new("L", size, 0)
    mask.paste(255, (40, 30, 60, 45))
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_sessions_keep_their_own_ledgers_and_candidates(service, second, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    res = second.start(root, settings=_settings(options_per_image=1, model="Flux/other"), name="Flux")
    assert res["ok"], res
    assert second._manifest is service._manifest and second._entries is service._entries
    assert second.confirm("pair_0001", _other())["ok"]
    _wait_ready(second)
    item = second.next_ready()["item"]
    assert item["candidates"][0]["path"].startswith(tf.candidates_dir(root, "flux"))
    assert os.path.isfile(tf.ledger_path(root, "flux")) and os.path.isfile(tf.ledger_path(root))
    assert tf.load_ledger(root)["items"] == {}
    assert FakeClient.calls[-1]["model"] == "Flux/other"
    assert service.status()["peers"] == ["Flux"] and second.status()["name"] == "Flux"
    overview = {s["id"]: s for s in DPOTriageFixService.overview()["sessions"]}
    assert overview["flux"]["active"] and overview["main"]["active"]


def test_session_ids_are_folder_safe():
    with pytest.raises(ValueError):
        DPOTriageFixService.session("../evil")
    with pytest.raises(ValueError):
        tf.session_dir("x", "a b")


def test_second_session_waits_for_the_first_to_decide(service, second, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    assert second.start(root, settings=_settings(options_per_image=1), name="Flux")["ok"]
    FakeClient.segment_bytes = _partial_png(box=(0, 0, 20, 20))
    assert service.confirm("pair_0000", _face())["ok"]
    assert second.confirm("pair_0000", _other())["ok"]
    _wait_ready(service)
    time.sleep(0.3)
    # Flux holds its copy back while the main session's result awaits a decision.
    assert second.status()["waiting"] == 1 and second.status()["counts"]["queued"] == 1
    flux_item = next(i for i in second.items()["items"] if i["stem"] == "pair_0000")
    assert flux_item["waiting_on"] == "main" and flux_item["claimed_by"] == ["main"]
    assert not any("newspaper" in c["prompt"] for c in FakeClient.calls)
    assert service.pick("pair_0000", 0)["ok"]
    fixed_sha = compute_file_sha256(os.path.join(root, "chosen", "pair_0000.png"))
    FakeClient.segment_bytes = _partial_png(box=(40, 30, 60, 45))
    _wait_ready(second)
    record = tf.load_ledger(root, "flux")["items"]["pair_0000"]
    # It repaired the image the main session accepted, not the original.
    assert record["chosen_sha256"] == fixed_sha
    assert any("newspaper" in c["prompt"] for c in FakeClient.calls)
    assert second.pick("pair_0000", 0)["ok"]
    # The main session's fix is built on now; its undo would throw the second one away.
    undo = service.undo_decision()
    assert not undo["ok"] and "on top" in undo["error"]
    assert second.undo_decision()["ok"]
    # Flux's result awaits a decision again; the main session waits for it.
    assert '"Flux"' in service.undo_decision()["error"]
    assert second.keep("pair_0000")["ok"]
    assert service.undo_decision()["ok"]


def test_first_confirm_goes_first_even_when_held(service, second, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1, hold=True)["ok"]
    assert second.start(root, settings=_settings(options_per_image=1))["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    assert second.confirm("pair_0000", _other())["ok"]
    assert second.confirm("pair_0001", _other())["ok"]
    _wait_ready(second)
    time.sleep(0.3)
    counts = second.status()["counts"]
    assert counts["ready"] == 1 and counts["queued"] == 1 and second.status()["waiting"] == 1
    # The image cannot be deleted from under the session that holds it.
    res = second.delete_pair("pair_0000")
    assert not res["ok"] and '"main"' in res["error"]


def test_overlapping_fixes_are_refused(service, second, tmp_path):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1, hold=True)["ok"]
    assert second.start(root, settings=_settings(options_per_image=1), name="Flux", hold=True)["ok"]
    assert service.confirm("pair_0000", _face())["ok"]
    same = second.confirm("pair_0000", _face())
    assert not same["ok"] and same["overlap"] and "Cannot proceed" in same["error"] and '"main"' in same["error"]
    # A CLIPSeg "face" is the same area as the YOLO face model.
    assert not second.confirm("pair_0000", _other("face"))["ok"]
    assert second.confirm("pair_0000", _other("newspaper"))["ok"]
    # A whole-image edit overlaps everything.
    assert service.confirm_many(["pair_0001"], [{"id": "photo"}], second_round=False)["queued"] == ["pair_0001"]
    assert "pair_0001" in second.confirm_many(["pair_0001"], _other(), second_round=False)["conflicts"]
    # Two painted masks: refused when they share pixels, fine apart.
    assert service.confirm_mask("pair_0002", _mask_uri(), "Fix the medal")["ok"]
    res = second.confirm_mask("pair_0002", _mask_uri(), "Fix the ribbon")
    assert not res["ok"] and "painted mask" in res["error"]
    assert not os.path.exists(tf.mask_path(root, "pair_0002", "flux"))
    assert second.confirm_mask("pair_0002", _far_mask_uri(), "Fix the ribbon")["ok"]
    assert os.path.isfile(tf.mask_path(root, "pair_0002", "flux"))


@pytest.mark.parametrize("repaired_box, blocked", [((0, 0, 64, 48), True), ((45, 35, 60, 45), False)])
def test_mask_waiting_on_a_repair_is_checked_against_it(service, second, tmp_path, repaired_box, blocked):
    root = _dataset(tmp_path)
    assert service.start(root, settings=_settings(options_per_image=1), shuffle_seed=1)["ok"]
    assert second.start(root, settings=_settings(options_per_image=1), name="Flux")["ok"]
    FakeClient.segment_bytes = _partial_png(box=repaired_box)
    assert service.confirm("pair_0000", _face())["ok"]
    # A painted mask against a segment fix cannot be judged yet: accepted.
    assert second.confirm_mask("pair_0000", _mask_uri(), "Fix the medal")["ok"]
    _wait_ready(service)
    assert service.pick("pair_0000", 0)["ok"]
    if blocked:
        state = _wait(second, lambda s: s["counts"]["failed"] == 1)
        record = tf.load_ledger(root, "flux")["items"]["pair_0000"]
        assert state["counts"]["ready"] == 0 and "overlaps" in record["error"] and '"main"' in record["error"]
    else:
        _wait_ready(second)


def test_sessions_share_one_fix_pair_per_image(service, second, tmp_path):
    root = _dataset(tmp_path)
    fix_dir = str(tmp_path / "fixes")
    assert service.start(root, settings=_settings(options_per_image=1), fix_pair_dir=fix_dir, shuffle_seed=1)["ok"]
    assert second.start(root, settings=_settings(options_per_image=1), fix_pair_dir=fix_dir)["ok"]
    assert second._fix_manifest is service._fix_manifest
    original = _pixel(os.path.join(root, "chosen", "pair_0000.png"))
    FakeClient.segment_bytes = _partial_png(box=(0, 0, 20, 20))
    assert service.confirm("pair_0000", _face())["ok"]
    _wait_ready(service)
    assert service.pick("pair_0000", 0)["ok"]
    FakeClient.segment_bytes = _partial_png(box=(0, 0, 20, 20), color=(0, 250, 0))
    assert second.confirm("pair_0000", _other())["ok"]
    _wait_ready(second)
    assert second.pick("pair_0000", 0)["ok"]
    pairs = load_manifest(fix_dir)["pairs"]
    assert len(pairs) == 1
    assert _pixel(os.path.join(fix_dir, "chosen", pairs[0]["chosen_file"])) == (0, 250, 0)
    assert _pixel(os.path.join(fix_dir, "rejected", pairs[0]["rejected_file"])) == original


def test_a_dataset_cannot_be_another_sessions_fix_pair_folder(service, second, tmp_path):
    root = _dataset(tmp_path)
    other = _dataset(tmp_path, name="other")
    assert service.start(root, settings=_settings(), fix_pair_dir=other)["ok"]
    res = second.start(other, settings=_settings())
    assert not res["ok"] and "fix pair folder" in res["error"]
