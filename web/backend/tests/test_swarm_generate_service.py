"""End-to-end batch generation through a stubbed SwarmUI.

Real files on disk (photos with caption sidecars, PNG "generations" carrying a
prompt), a fake SwarmUI client, and the same caches the off-policy builder
uses — so the pool logic, the bucket sizing and the run loop are all exercised
without a server or a GPU.
"""

import json
import os
import threading
import time

from modules.util.swarm_client import SwarmError
from web.backend.services.swarm_generate_service import SwarmGenerateService

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _jpg(path: str, size: tuple[int, int]) -> str:
    Image.new("RGB", size, (30, 140, 200)).save(path, quality=90)
    return path


def _png(path: str, size: tuple[int, int], prompt: str, params: dict | None = None, color=(120, 90, 60)) -> str:
    info = PngInfo()
    info.add_text("prompt", prompt)
    if params is not None:
        # SwarmUI stamps its full parameter set into every image it writes.
        info.add_text("parameters", json.dumps({"prompt": prompt, **params}))
    Image.new("RGB", size, color).save(path, pnginfo=info)
    return path


def _black_png(path: str, size: tuple[int, int], prompt: str, params: dict | None = None) -> str:
    """What a generation that failed inside SwarmUI's backend looks like on
    disk: every pixel zero, the parameter block intact."""
    return _png(path, size, prompt, params, color=(0, 0, 0))


def _wait(service: SwarmGenerateService, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = service.status()
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError(f"run did not finish: {service.status()}")


class FakeClient:
    """Stands in for SwarmUI: records calls, writes real files into the
    generated tree the way the server would, and can be scripted to fail."""

    calls: list[dict] = []
    fail_with: Exception | None = None
    fail_times: int = 0
    #: Where the "server" writes; set by the dirs fixture.
    output_root: str = ""
    #: Simulate a backend that fails mid-run: the file is written, but black.
    write_black: bool = False
    #: Black for only the next N calls, then healthy again.
    write_black_times: int = 0

    def __init__(self, *args, **kwargs):
        pass

    def presets(self):
        return []

    def list_models(self, subtype="Stable-Diffusion", depth=3):
        return []

    def list_loaded_models(self):
        return []

    def generate(self, prompt, width, height, model, images=1, **kwargs):
        FakeClient.calls.append(
            {"prompt": prompt, "width": width, "height": height, "model": model, "images": images, **kwargs}
        )
        if FakeClient.fail_with is not None and FakeClient.fail_times > 0:
            FakeClient.fail_times -= 1
            raise FakeClient.fail_with
        views = []
        for index in range(images):
            name = f"{len(FakeClient.calls)}-{index}-{abs(hash(prompt)) % 9999}.png"
            written = os.path.join(FakeClient.output_root, "2026-08-30", name)
            os.makedirs(os.path.dirname(written), exist_ok=True)
            black = FakeClient.write_black or FakeClient.write_black_times > 0
            if black:
                _black_png(written, (width, height), prompt, {"model": model, "width": width, "height": height})
            else:
                _png(written, (width, height), prompt)
            views.append(f"View/local/raw/2026-08-30/{name}")
        if FakeClient.write_black_times > 0:
            FakeClient.write_black_times -= 1
        return views


@pytest.fixture
def service(monkeypatch):
    import web.backend.services.swarm_generate_service as module

    FakeClient.calls = []
    FakeClient.fail_with = None
    FakeClient.fail_times = 0
    FakeClient.output_root = ""
    FakeClient.write_black = False
    FakeClient.write_black_times = 0
    monkeypatch.setattr(module, "SwarmClient", FakeClient)
    instance = SwarmGenerateService.__new__(SwarmGenerateService)
    instance.__init__()
    monkeypatch.setattr(SwarmGenerateService, "get_instance", classmethod(lambda cls: instance))
    yield instance
    instance.cancel()


@pytest.fixture
def dirs(tmp_path):
    real, gen, out = tmp_path / "real", tmp_path / "gen", tmp_path / "out"
    real.mkdir()
    gen.mkdir()
    out.mkdir()
    # Prose caption, no generation yet -> should be generated for, at 3:2.
    _jpg(str(real / "prose.jpg"), (2400, 1600))
    (real / "prose.txt").write_text("A woman stands in a sunlit garden holding a watering can.\n", encoding="utf-8")
    # Tag-style caption, no generation yet -> filtered out by default.
    _jpg(str(real / "tags.jpg"), (2400, 1600))
    (real / "tags.txt").write_text("blonde hair, blue eyes, 1girl, outdoors\n", encoding="utf-8")
    # Portrait prose caption that already has a usable generation -> skipped.
    _jpg(str(real / "done.jpg"), (1600, 2400))
    (real / "done.txt").write_text("A tall woman leans against a brick wall at dusk.\n", encoding="utf-8")
    _png(str(gen / "done.png"), (832, 1280), "A tall woman leans against a brick wall at dusk.")
    FakeClient.output_root = str(gen)
    return real, gen, out


def _start(service, dirs, **kwargs):
    real, gen, out = dirs
    options = {
        "real_folder": str(real),
        "generated_folder": str(gen),
        "output_dir": str(out),
        "model": "MyModels/test.safetensors",
        "captions": 10,
    }
    options.update(kwargs)
    return service.start(**options)


def test_dry_run_plans_bucket_sizes_without_calling_swarm(service, dirs):
    assert _start(service, dirs, dry_run=True)["ok"]
    state = _wait(service)
    assert state["phase"] == "done" and state["error"] is None
    assert FakeClient.calls == []
    assert state["skipped_tag_style"] == 1
    (planned,) = state["results"]
    assert os.path.basename(planned["real_path"]) == "prose.jpg"
    assert planned["size"] == [1280, 832]  # the photo's own trainer bucket
    assert planned["planned_only"] is True
    assert planned["caption_type"] == "Short"


def test_generation_requests_the_photos_bucket_and_records_results(service, dirs):
    assert _start(service, dirs, images_per_caption=2)["ok"]
    state = _wait(service)
    assert state["phase"] == "done" and state["error"] is None
    (call,) = FakeClient.calls
    assert call["prompt"] == "A woman stands in a sunlit garden holding a watering can."
    assert (call["width"], call["height"]) == (1280, 832)
    assert call["images"] == 2 and call["model"] == "MyModels/test.safetensors"
    assert state["planned"] == 1 and state["done"] == 1 and state["images"] == 2 and state["failed"] == 0
    (result,) = state["results"]
    assert result["ok"] is True and len(result["paths"]) == 2
    # The images are in the scanned tree straight away — the off-policy
    # builder's own confirm/skip screen is the only approval step.
    _real, gen, _out = dirs
    written = sorted(path.name for path in (gen / "2026-08-30").iterdir())
    assert len(written) == 2


def test_tag_style_filter_can_be_turned_off(service, dirs):
    assert _start(service, dirs, exclude_tag_style=False)["ok"]
    state = _wait(service)
    prompts = sorted(call["prompt"] for call in FakeClient.calls)
    assert prompts == [
        "A woman stands in a sunlit garden holding a watering can.",
        "blonde hair, blue eyes, 1girl, outdoors",
    ]
    assert state["skipped_tag_style"] == 0


def test_short_captions_can_be_excluded(service, dirs):
    assert _start(service, dirs, min_caption_words=20)["ok"]
    state = _wait(service)
    assert FakeClient.calls == []
    assert state["skipped_short"] == 1 and state["planned"] == 0


def test_word_minimum_applies_per_caption_line_not_per_sidecar(service, dirs):
    real, _gen, _out = dirs
    long_line = "A woman in a red coat waits under a streetlight while rain drips from the awning behind her."
    _jpg(str(real / "mixed.jpg"), (2400, 1600))
    (real / "mixed.txt").write_text("A woman.\n" + long_line + "\n", encoding="utf-8")
    # Several runs: the line is chosen at random per photo, so a filter that
    # judged the draw instead of the pool would drop the photo some of the time.
    for _ in range(6):
        FakeClient.calls = []
        assert _start(service, dirs, min_caption_words=15)["ok"]
        state = _wait(service)
        assert [call["prompt"] for call in FakeClient.calls] == [long_line]
        assert state["planned"] == 1
        # prose.jpg's only line is too short; mixed.jpg is not counted.
        assert state["skipped_short"] == 1
        for path in (_gen / "2026-08-30").iterdir():
            path.unlink()


def test_keyword_filter_narrows_the_pool(service, dirs):
    assert _start(service, dirs, keywords="watering can", exclude_tag_style=False)["ok"]
    _wait(service)
    assert [call["prompt"] for call in FakeClient.calls] == [
        "A woman stands in a sunlit garden holding a watering can."
    ]


def test_photos_with_a_usable_generation_are_never_regenerated(service, dirs):
    assert _start(service, dirs, exclude_tag_style=False)["ok"]
    _wait(service)
    assert all("brick wall" not in call["prompt"] for call in FakeClient.calls)


def test_a_failing_caption_is_recorded_and_the_run_continues(service, dirs):
    FakeClient.fail_with = SwarmError("model not loaded")
    FakeClient.fail_times = 1
    assert _start(service, dirs, exclude_tag_style=False)["ok"]
    state = _wait(service)
    assert state["failed"] == 1 and state["done"] == 2
    assert state["phase"] == "done"
    failed = [r for r in state["results"] if not r.get("ok")]
    assert len(failed) == 1 and "model not loaded" in failed[0]["error"]


def test_a_dead_server_stops_the_run(service, dirs, tmp_path):
    real, gen, out = dirs
    for index in range(8):
        _jpg(str(real / f"extra{index}.jpg"), (2400, 1600))
        (real / f"extra{index}.txt").write_text(
            f"A woman number {index} stands in a sunlit garden holding a watering can.\n", encoding="utf-8"
        )
    FakeClient.fail_with = SwarmError("Could not reach SwarmUI")
    FakeClient.fail_times = 99
    assert _start(service, dirs)["ok"]
    state = _wait(service)
    assert state["phase"] == "error"
    assert "consecutive SwarmUI failures" in state["error"]
    assert state["failed"] == 5  # _MAX_CONSECUTIVE_FAILURES, then it gives up


def test_start_validates_its_inputs(service, dirs, tmp_path):
    real, gen, out = dirs
    assert not service.start(str(tmp_path / "nope"), str(gen), str(out), "m")["ok"]
    assert not service.start(str(real), str(gen), str(out), "", param_source="manual")["ok"]  # no model
    assert not service.start(str(real), str(gen), str(out), "m", captions=0)["ok"]
    assert service.start(str(real), str(gen), str(out), "", dry_run=True)["ok"]  # a plan needs no model


def test_only_one_run_at_a_time(service, dirs, monkeypatch):
    import web.backend.services.swarm_generate_service as module

    started = []

    class SlowClient(FakeClient):
        def generate(self, *args, **kwargs):
            started.append(1)
            time.sleep(0.4)
            return super().generate(*args, **kwargs)

    monkeypatch.setattr(module, "SwarmClient", SlowClient)
    assert _start(service, dirs)["ok"]
    deadline = time.time() + 5
    while not started and time.time() < deadline:
        time.sleep(0.01)
    # A second start while one runs is queued behind it, not refused.
    second = _start(service, dirs)
    assert second["ok"] and second["queued"] and second["position"] == 1
    assert service.status()["job_total"] == 2
    _wait(service)


def test_check_reports_models(service, monkeypatch):
    import web.backend.services.swarm_generate_service as module

    class Client(FakeClient):
        def ping(self):
            return {"version": "0.9.8", "user_id": "local"}

        def list_models(self, subtype="Stable-Diffusion", depth=3):
            return ["lora-a.safetensors"] if subtype == "LoRA" else ["a.safetensors"]

        def list_loaded_models(self):
            return ["SoReal!/SoReal!_Turbo_V1.3.safetensors"]

        def list_yolo_models(self):
            return ["face_yolov8m.pt"]

    monkeypatch.setattr(module, "SwarmClient", Client)
    assert service.check("localhost:7801") == {
        "ok": True,
        "version": "0.9.8",
        "user_id": "local",
        "models": ["a.safetensors"],
        "loras": ["lora-a.safetensors"],
        "yolo": ["face_yolov8m.pt"],
        "loaded": ["SoReal!/SoReal!_Turbo_V1.3.safetensors"],
        "presets": [],
    }

    class Dead(FakeClient):
        def ping(self):
            raise SwarmError("connection refused")

        def list_models(self, subtype="Stable-Diffusion", depth=3):
            raise SwarmError("connection refused")

    monkeypatch.setattr(module, "SwarmClient", Dead)
    result = service.check("localhost:9999")
    assert result["ok"] is False and "connection refused" in result["error"]


def test_loras_are_passed_through_with_weights(service, dirs):
    assert _start(service, dirs, loras=[{"name": "Distillers/late.safetensors", "weight": 0.8}])["ok"]
    _wait(service)
    (call,) = FakeClient.calls
    assert call["extra_params"]["loras"] == ["Distillers/late.safetensors"]
    assert call["extra_params"]["loraweights"] == ["0.8"]


# --- reusing the settings the user already generates with --------------------

LIVE_PARAMS = {
    "model": "SoReal!/09Aug/SoReal!_Turbo_V1.3",
    "steps": 10,
    "cfgscale": 1.0,
    "sigmashift": 1.5,
    "usetcfg": True,
    "zeronegative": True,
    "loras": ["SoReal!_V43_OFT"],
    "loraweights": ["1"],
    "negativeprompt": "blurry",
    # Ours to decide, and must never be taken from the source.
    "width": 1024,
    "height": 1024,
    "aspectratio": "1:1",
    "seed": 1239869429,
    "images": 3,
}


def test_last_generation_settings_are_reused_verbatim(service, dirs):
    """The whole point: sigma shift, samplers and LoRAs come from what the user
    is actually generating with, not from this tool's defaults."""
    real, gen, out = dirs
    _png(str(gen / "recent.png"), (832, 1216), "some earlier prompt", params=LIVE_PARAMS)

    assert _start(service, dirs, model="", images_per_caption=2)["ok"]
    state = _wait(service)
    assert state["phase"] == "done", state["error"]
    assert state["param_source"] == "last generation"
    (call,) = FakeClient.calls
    # Model came from the source; the size and prompt did not.
    assert call["model"] == "SoReal!/09Aug/SoReal!_Turbo_V1.3"
    assert (call["width"], call["height"]) == (1280, 832)
    assert call["images"] == 2
    extra = call["extra_params"]
    assert extra["sigmashift"] == 1.5 and extra["usetcfg"] is True and extra["zeronegative"] is True
    assert extra["steps"] == 10 and extra["cfgscale"] == 1.0
    assert extra["loras"] == ["SoReal!_V43_OFT"] and extra["negativeprompt"] == "blurry"
    for owned in ("prompt", "width", "height", "aspectratio", "seed", "images"):
        assert owned not in extra, owned


def test_explicit_fields_override_the_reused_settings(service, dirs):
    real, gen, out = dirs
    _png(str(gen / "recent.png"), (832, 1216), "earlier", params=LIVE_PARAMS)

    assert _start(service, dirs, model="Other/model.safetensors", steps=25)["ok"]
    _wait(service)
    (call,) = FakeClient.calls
    assert call["model"] == "Other/model.safetensors"
    assert call["extra_params"]["steps"] == 25
    assert call["extra_params"]["sigmashift"] == 1.5  # untouched by the override


def test_a_preset_can_supply_the_settings(service, dirs, monkeypatch):
    import web.backend.services.swarm_generate_service as module

    class Presetting(FakeClient):
        def presets(self):
            return [
                {"title": "default", "params": {"steps": 4, "sigmashift": 3.0, "model": "Preset/model.safetensors"}},
                {"title": "other", "params": {"steps": 99}},
            ]

    monkeypatch.setattr(module, "SwarmClient", Presetting)
    assert _start(service, dirs, model="", param_source="preset", preset_name="default")["ok"]
    state = _wait(service)
    assert state["param_source"] == "preset 'default'"
    (call,) = FakeClient.calls
    assert call["model"] == "Preset/model.safetensors"
    assert call["extra_params"]["sigmashift"] == 3.0 and call["extra_params"]["steps"] == 4

    FakeClient.calls = []
    assert _start(service, dirs, model="", param_source="preset", preset_name="nope")["ok"]
    state = _wait(service)
    assert state["phase"] == "error" and "nope" in state["error"]


def test_manual_source_ignores_the_last_generation(service, dirs):
    real, gen, out = dirs
    _png(str(gen / "recent.png"), (832, 1216), "earlier", params=LIVE_PARAMS)

    assert _start(service, dirs, param_source="manual", steps=7)["ok"]
    _wait(service)
    (call,) = FakeClient.calls
    assert call["model"] == "MyModels/test.safetensors"
    assert call["extra_params"] == {"steps": 7}


def test_a_run_without_any_model_anywhere_reports_why(service, dirs):
    assert _start(service, dirs, model="")["ok"]
    state = _wait(service)
    assert state["phase"] == "error"
    assert "No model" in state["error"]
    assert FakeClient.calls == []


def test_start_validates_the_parameter_source(service, dirs):
    assert not _start(service, dirs, param_source="nonsense")["ok"]
    assert not _start(service, dirs, param_source="preset", preset_name="")["ok"]
    # Manual with no model is refused up front; "last" is allowed to find one.
    assert not _start(service, dirs, param_source="manual", model="")["ok"]


def test_preview_params_reports_what_would_be_reused(service, dirs):
    real, gen, out = dirs
    assert service.preview_params(str(gen))["params"] == {}
    _png(str(gen / "recent.png"), (832, 1216), "earlier", params=LIVE_PARAMS)
    params = service.preview_params(str(gen))["params"]
    assert params["sigmashift"] == 1.5 and params["model"].endswith("Turbo_V1.3")
    assert "seed" not in params and "width" not in params


# --- settings from a reference folder, minus its adapters ---------------------


def _folder_options(service, dirs, param_folder: str, **kwargs) -> dict:
    """The options dict ``start`` would build, for calling ``_resolve_params``
    directly with a fake client."""
    import web.backend.services.swarm_generate_service as module

    captured: dict = {}

    def grab(*jobs):
        captured.update(jobs[0])
        return {"ok": True, "queued": False, "position": 1}

    kwargs.setdefault("model", "")
    original = service._enqueue
    service._enqueue = grab
    try:
        assert _start(service, dirs, param_source="folder", param_folder=param_folder, **kwargs)["ok"]
    finally:
        service._enqueue = original
    assert captured, "start did not reach the queue"
    assert module.PARAM_SOURCES == ("last", "preset", "manual", "folder")
    return captured


def test_folder_source_reads_the_reference_folders_last_generation(service, dirs, tmp_path):
    reference = tmp_path / "Reference Shots"
    reference.mkdir()
    _png(str(reference / "ref.png"), (832, 1216), "a reference prompt", params=LIVE_PARAMS)

    options = _folder_options(service, dirs, str(reference))
    params = service._resolve_params(FakeClient(), options)
    assert service.status()["param_source"] == "last generation in Reference Shots"
    assert params["model"] == LIVE_PARAMS["model"] and params["sigmashift"] == 1.5
    # The reference folder's own LoRAs ride along unless asked to drop them.
    assert params["loras"] == ["SoReal!_V43_OFT"] and params["loraweights"] == ["1"]
    for owned in ("prompt", "width", "height", "seed", "images"):
        assert owned not in params


def test_folder_source_can_drop_the_sources_loras_and_apply_its_own(service, dirs, tmp_path):
    reference = tmp_path / "ref"
    reference.mkdir()
    source = LIVE_PARAMS | {"lorasectionconfs": ["x"], "loratencweights": ["0.5"]}
    _png(str(reference / "ref.png"), (832, 1216), "p", params=source)

    options = _folder_options(
        service, dirs, str(reference), drop_source_loras=True, loras=[{"name": "Chars/rosie", "weight": 0.9}]
    )
    params = service._resolve_params(FakeClient(), options)
    assert params["loras"] == ["Chars/rosie"] and params["loraweights"] == ["0.9"]
    assert not any(key for key in params if key.startswith("lora") and key not in ("loras", "loraweights"))
    assert params["model"] == LIVE_PARAMS["model"] and params["usetcfg"] is True

    # Dropping without an override leaves no adapter at all.
    bare = service._resolve_params(FakeClient(), _folder_options(service, dirs, str(reference), drop_source_loras=True))
    assert not any(key.startswith("lora") for key in bare)


def test_folder_source_with_no_generation_falls_back_to_manual(service, dirs, tmp_path):
    empty = tmp_path / "Empty"
    empty.mkdir()
    options = _folder_options(service, dirs, str(empty), model="Other/model.safetensors")
    params = service._resolve_params(FakeClient(), options)
    assert service.status()["param_source"] == "manual (no generation found in Empty)"
    assert params == {"model": "Other/model.safetensors"}


def test_folder_source_runs_end_to_end_with_the_label_on_the_job(service, dirs, tmp_path):
    reference = tmp_path / "ref"
    reference.mkdir()
    _png(str(reference / "ref.png"), (832, 1216), "p", params=LIVE_PARAMS)
    assert _start(
        service,
        dirs,
        model="",
        param_source="folder",
        param_folder=str(reference),
        drop_source_loras=True,
        loras=[{"name": "Chars/rosie", "weight": 1.0}],
        label="Rosie Butler",
    )["ok"]
    state = _wait(service)
    assert state["phase"] == "done", state["error"]
    assert state["param_source"] == "last generation in ref"
    (call,) = FakeClient.calls
    assert call["model"] == LIVE_PARAMS["model"]
    assert call["extra_params"]["loras"] == ["Chars/rosie"] and call["extra_params"]["loraweights"] == ["1.0"]


def test_start_validates_the_parameter_folder(service, dirs, tmp_path):
    assert not _start(service, dirs, param_source="folder", param_folder="")["ok"]
    assert not _start(service, dirs, param_source="folder", param_folder=str(tmp_path / "missing"))["ok"]


def test_job_summary_carries_the_label():
    from web.backend.services.swarm_generate_service import _job_summary

    assert _job_summary({"label": "Rosie Butler", "captions": 3})["label"] == "Rosie Butler"
    assert _job_summary({})["label"] == ""


def test_queued_jobs_report_their_label(service, dirs, monkeypatch):
    """The label reaches the status the UI polls, for the run in flight and
    the ones waiting behind it."""
    import web.backend.services.swarm_generate_service as module

    gate = threading.Event()

    class Slow(FakeClient):
        def generate(self, *args, **kwargs):
            gate.wait(5)
            return super().generate(*args, **kwargs)

    monkeypatch.setattr(module, "SwarmClient", Slow)
    assert _start(service, dirs, label="first")["ok"]
    deadline = time.time() + 5
    while service.status()["phase"] != "generating" and time.time() < deadline:
        time.sleep(0.02)
    assert _start(service, dirs, label="second")["queued"]
    status = service.status()
    assert status["current_job"]["label"] == "first"
    assert [job["label"] for job in status["queued"]] == ["second"]
    gate.set()
    _wait(service)


def test_preview_params_for_can_drop_the_loras(service, dirs):
    real, gen, out = dirs
    assert service.preview_params_for(str(gen), True) == {"ok": True, "params": {}, "source_file": None}
    _png(str(gen / "recent.png"), (832, 1216), "earlier", params=LIVE_PARAMS)
    kept = service.preview_params_for(str(gen), False)
    assert kept["params"]["loras"] == ["SoReal!_V43_OFT"] and kept["source_file"] == "recent.png"
    dropped = service.preview_params_for(str(gen), True)["params"]
    assert "loras" not in dropped and "loraweights" not in dropped
    assert dropped["sigmashift"] == 1.5 and dropped["model"] == kept["params"]["model"]
    # A missing folder is simply empty, not an error.
    assert service.preview_params_for(str(gen / "nope"))["source_file"] is None


def test_last_generation_source_names_the_newest_image_with_a_parameter_block(service, dirs):
    from web.backend.services.swarm_generate_service import last_generation_source

    real, gen, out = dirs
    (gen / "sub").mkdir()
    (gen / ".pending").mkdir()
    older = _png(str(gen / "older.png"), (832, 1216), "p", params=LIVE_PARAMS)
    os.utime(older, (1_700_000_000, 1_700_000_000))
    # Newest, but without a parameter block: skipped in favour of the older one.
    bare = _png(str(gen / "sub" / "newest-no-params.png"), (832, 1216), "p")
    os.utime(bare, (1_800_000_000, 1_800_000_000))
    # Newest with a block, inside a dot-folder: never used.
    hidden = _png(str(gen / ".pending" / "hidden.png"), (832, 1216), "p", params=LIVE_PARAMS)
    os.utime(hidden, (1_900_000_000, 1_900_000_000))
    path, params = last_generation_source(str(gen))
    assert os.path.basename(path) == "older.png" and params["sigmashift"] == 1.5
    assert last_generation_source(str(gen / "missing")) == (None, {})


def test_last_params_route_accepts_drop_loras(service, dirs):
    from web.backend.routers.swarm_generate import router

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    real, gen, out = dirs
    _png(str(gen / "recent.png"), (832, 1216), "earlier", params=LIVE_PARAMS)
    app = FastAPI()
    app.include_router(router, prefix="/api")
    client = TestClient(app)
    kept = client.get("/api/swarm/last-params", params={"generated_folder": str(gen)}).json()
    assert kept["ok"] and kept["params"]["loras"] == ["SoReal!_V43_OFT"] and kept["source_file"] == "recent.png"
    dropped = client.get("/api/swarm/last-params", params={"generated_folder": str(gen), "drop_loras": True}).json()
    assert dropped["ok"] and "loras" not in dropped["params"] and dropped["source_file"] == "recent.png"


def test_lora_and_loaded_model_listing_failures_are_not_fatal(service, monkeypatch):
    """A server that cannot list LoRAs is still usable for generation."""
    import web.backend.services.swarm_generate_service as module

    class Partial(FakeClient):
        def ping(self):
            return {"version": "0.9.8", "user_id": "local"}

        def list_models(self, subtype="Stable-Diffusion", depth=3):
            if subtype == "LoRA":
                raise SwarmError("no lora folder")
            return ["a.safetensors"]

        def list_loaded_models(self):
            raise SwarmError("unsupported")

        def list_yolo_models(self):
            raise SwarmError("unsupported")

    monkeypatch.setattr(module, "SwarmClient", Partial)
    result = service.check("localhost:7801")
    assert result["ok"] and result["models"] == ["a.safetensors"]
    assert result["loras"] == [] and result["loaded"] == [] and result["presets"] == []
    assert result["yolo"] == []


# --- keeping SwarmUI's queue fed --------------------------------------------


def _many_captions(real, count=6):
    """Enough distinct prose captions to fill a few concurrent slots."""
    for index in range(count):
        _jpg(str(real / f"many{index}.jpg"), (2400, 1600))
        (real / f"many{index}.txt").write_text(
            f"A woman number {index} stands in a sunlit garden holding a watering can.\n", encoding="utf-8"
        )


class _ConcurrencyProbe(FakeClient):
    """Measures how many generate() calls are in flight at once."""

    active = 0
    peak = 0
    probe_lock = threading.Lock()

    def generate(self, *args, **kwargs):
        with _ConcurrencyProbe.probe_lock:
            _ConcurrencyProbe.active += 1
            _ConcurrencyProbe.peak = max(_ConcurrencyProbe.peak, _ConcurrencyProbe.active)
        try:
            time.sleep(0.15)
            return super().generate(*args, **kwargs)
        finally:
            with _ConcurrencyProbe.probe_lock:
                _ConcurrencyProbe.active -= 1


def _run_with_probe(service, dirs, monkeypatch, concurrency):
    import web.backend.services.swarm_generate_service as module

    real, _gen, _out = dirs
    _many_captions(real)
    _ConcurrencyProbe.active = 0
    _ConcurrencyProbe.peak = 0
    monkeypatch.setattr(module, "SwarmClient", _ConcurrencyProbe)
    assert _start(service, dirs, concurrency=concurrency, captions=6)["ok"]
    state = _wait(service, timeout=40)
    assert state["phase"] == "done", state["error"]
    return state


def test_requests_overlap_so_the_gpu_never_waits(service, dirs, monkeypatch):
    state = _run_with_probe(service, dirs, monkeypatch, concurrency=3)
    assert state["done"] >= 3
    assert _ConcurrencyProbe.peak >= 2, "requests were issued one at a time"
    assert _ConcurrencyProbe.peak <= 3, "more requests in flight than asked for"


def test_concurrency_one_keeps_the_old_serial_behaviour(service, dirs, monkeypatch):
    _run_with_probe(service, dirs, monkeypatch, concurrency=1)
    assert _ConcurrencyProbe.peak == 1


def test_concurrency_is_clamped_to_something_sane(service, dirs):
    assert _start(service, dirs, concurrency=999)["ok"]
    _wait(service)
    # Clamped rather than refused: an absurd value should not fail a run.
    assert service.status()["phase"] == "done"


def test_cancelling_a_concurrent_run_stops_it(service, dirs, monkeypatch):
    import web.backend.services.swarm_generate_service as module

    real, _gen, _out = dirs
    _many_captions(real, count=12)

    class Slow(FakeClient):
        def generate(self, *args, **kwargs):
            time.sleep(0.3)
            return super().generate(*args, **kwargs)

    monkeypatch.setattr(module, "SwarmClient", Slow)
    assert _start(service, dirs, concurrency=2, captions=12)["ok"]
    deadline = time.time() + 5
    while not FakeClient.calls and time.time() < deadline:
        time.sleep(0.01)
    service.cancel()
    state = _wait(service, timeout=30)
    assert state["phase"] == "cancelled"
    assert state["done"] < 12


# ---------------------------------------------------------------------------
# Black (failed) generations
# ---------------------------------------------------------------------------

_BLACK_PARAMS = {
    "model": "MyModels/test.safetensors",
    "width": 1280,
    "height": 832,
    "steps": 8,
    "cfgscale": 1.0,
    "seed": 12345,
    "sampler": "euler",
    "loras": ["style"],
    "loraweights": ["0.8"],
    "swarm_version": "0.9.6",
    "date": "2026-09-17",
}


def test_is_black_image_only_matches_all_zero_pixels(tmp_path):
    from modules.util.swarm_generate_util import is_black_image

    black = _black_png(str(tmp_path / "black.png"), (64, 48), "x")
    dark = _png(str(tmp_path / "dark.png"), (64, 48), "x", color=(20, 20, 20))
    normal = _png(str(tmp_path / "normal.png"), (64, 48), "x")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    assert is_black_image(black)
    assert not is_black_image(dark)
    assert is_black_image(dark, max_level=20)
    assert not is_black_image(normal)
    # Unreadable is not black: nothing that cannot be looked at is retired.
    assert not is_black_image(str(tmp_path / "broken.png"))


def test_scan_finds_black_images_and_skips_dot_folders(service, dirs):
    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    _black_png(str(gen / "2026-09-17" / "a.png"), (1280, 832), "A woman on a pier at dawn.", _BLACK_PARAMS)
    _black_png(str(gen / "2026-09-17" / "b.png"), (832, 1280), "A man on a pier at dawn.")
    (gen / ".black").mkdir()
    _black_png(str(gen / ".black" / "old.png"), (1280, 832), "Already retired.", _BLACK_PARAMS)
    result = service.scan_black(str(gen))
    assert result["ok"] and result["count"] == 2
    by_name = {os.path.basename(image["path"]): image for image in result["images"]}
    assert set(by_name) == {"a.png", "b.png"}
    assert by_name["a.png"]["prompt"] == "A woman on a pier at dawn."
    assert by_name["a.png"]["size"] == [1280, 832] and by_name["a.png"]["has_params"]
    # The metadata-only "prompt" chunk is not a parameter block: retire only.
    assert not by_name["b.png"]["has_params"]
    assert not service.scan_black(str(gen / "missing"))["ok"]


def test_requeue_retires_black_images_and_regenerates_from_their_own_params(service, dirs):
    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    black = _black_png(str(gen / "2026-09-17" / "a.png"), (1280, 832), "A woman on a pier at dawn.", _BLACK_PARAMS)
    no_params = _black_png(str(gen / "2026-09-17" / "b.png"), (832, 1280), "A man on a pier at dawn.")
    fine = _png(str(gen / "2026-09-17" / "c.png"), (1280, 832), "A dog on a pier at dawn.", _BLACK_PARAMS)

    result = service.requeue_black(str(gen), model="")
    assert result["ok"] and result["found"] == 2 and result["moved"] == 2 and result["requeued"] == 1
    assert [os.path.basename(entry["path"]) for entry in result["unresendable"]] == ["b.png"]
    state = _wait(service)
    assert state["phase"] == "done" and state["error"] is None

    # Both black files are out of the scanned tree, the good one untouched.
    assert not os.path.exists(black) and not os.path.exists(no_params) and os.path.exists(fine)
    retired = sorted(path.name for path in (gen / "2026-09-17" / ".black").iterdir())
    assert retired == ["a.png", "b.png"]
    assert service.scan_black(str(gen))["count"] == 0

    # The resend carries the image's own block: prompt, size, model, sampler,
    # LoRAs — with a fresh seed and without the output-only keys.
    (call,) = FakeClient.calls
    assert call["prompt"] == "A woman on a pier at dawn."
    assert (call["width"], call["height"]) == (1280, 832)
    assert call["model"] == "MyModels/test.safetensors" and call["images"] == 1
    assert call["seed"] == -1
    assert call["loras"] == ["style"] and call["lora_weights"] == [0.8]
    assert call["extra_params"]["steps"] == 8 and call["extra_params"]["sampler"] == "euler"
    assert "swarm_version" not in call["extra_params"] and "date" not in call["extra_params"]

    assert state["planned"] == 1 and state["done"] == 1 and state["images"] == 1 and state["failed"] == 0
    (record,) = state["results"]
    assert record["ok"] and record["requeued_from"] == black and record["real_path"] is None
    assert state["param_source"] == "each image's own parameters"
    assert state["current_job"] is None and state["job_total"] == 1


def test_requeue_model_override_replaces_the_recorded_model(service, dirs):
    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    _black_png(str(gen / "2026-09-17" / "a.png"), (1280, 832), "A woman on a pier at dawn.", _BLACK_PARAMS)
    assert service.requeue_black(str(gen), model="Other/new.safetensors")["requeued"] == 1
    _wait(service)
    (call,) = FakeClient.calls
    assert call["model"] == "Other/new.safetensors"


def test_requeue_never_retires_an_image_that_is_not_black(service, dirs):
    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    fine = _png(str(gen / "2026-09-17" / "c.png"), (1280, 832), "A dog on a pier at dawn.", _BLACK_PARAMS)
    outside = _black_png(str(gen.parent / "elsewhere.png"), (64, 64), "Not in the tree.", _BLACK_PARAMS)
    result = service.requeue_black(str(gen), paths=[fine, outside])
    assert result["ok"] and result["found"] == 0 and result["requeued"] == 0
    assert os.path.exists(fine) and os.path.exists(outside)
    assert FakeClient.calls == []
    assert not service.status()["running"]


def test_requeue_waits_behind_a_running_generation(service, dirs, monkeypatch):
    import web.backend.services.swarm_generate_service as module

    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    _black_png(str(gen / "2026-09-17" / "a.png"), (1280, 832), "A woman on a pier at dawn.", _BLACK_PARAMS)

    class SlowClient(FakeClient):
        def generate(self, *args, **kwargs):
            time.sleep(0.3)
            return super().generate(*args, **kwargs)

    monkeypatch.setattr(module, "SwarmClient", SlowClient)
    # The run's own start-of-run scan is off, so the manual requeue is what
    # picks the black image up.
    assert _start(service, dirs, auto_requeue_black=False)["ok"]
    result = service.requeue_black(str(gen))
    assert result["ok"] and result["queued"] and result["requeued"] == 1
    status = service.status()
    assert status["job_total"] == 2 and status["queued"][0]["mode"] == "requeue" and status["queued"][0]["images"] == 1
    state = _wait(service)
    assert state["phase"] == "done" and state["job_index"] == 2
    prompts = sorted(call["prompt"] for call in FakeClient.calls)
    assert prompts == ["A woman on a pier at dawn.", "A woman stands in a sunlit garden holding a watering can."]


def test_a_black_output_from_swarm_is_retired_and_counted_as_a_failure(service, dirs):
    _real, gen, _out = dirs
    FakeClient.write_black = True
    assert _start(service, dirs)["ok"]
    state = _wait(service)
    assert state["phase"] == "done"
    assert state["failed"] == 1 and state["images"] == 0
    (record,) = state["results"]
    # One attempt plus _BLACK_RETRIES, every output retired.
    assert record["ok"] is False and "black" in record["error"] and record["black"] == 3
    assert len(FakeClient.calls) == 3 and state["black_retried"] == 2
    # The black files never claim their caption: they are in .black, not the tree.
    assert not [p for p in (gen / "2026-08-30").iterdir() if p.is_file()]
    assert len(list((gen / "2026-08-30" / ".black").iterdir())) == 3
    assert service.scan_black(str(gen))["count"] == 0
    # ... so the next ordinary run generates for the same caption again.
    FakeClient.write_black = False
    FakeClient.calls = []
    assert _start(service, dirs)["ok"]
    state = _wait(service)
    assert state["failed"] == 0 and [call["prompt"] for call in FakeClient.calls] == [
        "A woman stands in a sunlit garden holding a watering can."
    ]


def test_a_backend_writing_only_black_images_stops_the_run(service, dirs):
    real, _gen, _out = dirs
    for index in range(8):
        _jpg(str(real / f"extra{index}.jpg"), (2400, 1600))
        (real / f"extra{index}.txt").write_text(
            f"A woman number {index} stands in a sunlit garden holding a watering can.\n", encoding="utf-8"
        )
    FakeClient.write_black = True
    assert _start(service, dirs)["ok"]
    state = _wait(service)
    assert state["phase"] == "error" and "consecutive SwarmUI failures" in state["error"]
    assert state["failed"] == 5


def test_a_transient_black_output_is_retried_with_a_fresh_seed(service, dirs):
    _real, gen, _out = dirs
    FakeClient.write_black_times = 1
    assert _start(service, dirs)["ok"]
    state = _wait(service)
    assert state["phase"] == "done" and state["failed"] == 0 and state["images"] == 1
    assert len(FakeClient.calls) == 2 and state["black_retried"] == 1
    (record,) = state["results"]
    assert record["ok"] and record["black"] == 1 and len(record["paths"]) == 1
    assert len(list((gen / "2026-08-30" / ".black").iterdir())) == 1
    assert service.scan_black(str(gen))["count"] == 0


def test_black_retries_can_be_turned_off(service, dirs):
    FakeClient.write_black = True
    assert _start(service, dirs, auto_requeue_black=False)["ok"]
    state = _wait(service)
    assert state["failed"] == 1 and len(FakeClient.calls) == 1 and state["black_retried"] == 0


def test_a_run_retires_and_regenerates_black_images_before_new_captions(service, dirs):
    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    # A black generation of prose.jpg's caption: it claims that caption, and
    # once retired the caption is free again — it must be generated exactly
    # once, from the image's own block, not drawn a second time as new.
    prose = "A woman stands in a sunlit garden holding a watering can."
    black = _black_png(str(gen / "2026-09-17" / "a.png"), (1280, 832), prose, _BLACK_PARAMS)
    orphan = _black_png(str(gen / "2026-09-17" / "b.png"), (832, 1280), "A caption no photo has.", _BLACK_PARAMS)
    assert _start(service, dirs)["ok"]
    state = _wait(service)
    assert state["phase"] == "done" and state["error"] is None
    assert state["black_retired"] == 2 and state["planned"] == 2 and state["failed"] == 0
    assert not os.path.exists(black) and not os.path.exists(orphan)
    assert sorted(p.name for p in (gen / "2026-09-17" / ".black").iterdir()) == ["a.png", "b.png"]
    calls = {call["prompt"]: call for call in FakeClient.calls}
    assert set(calls) == {prose, "A caption no photo has."}
    # Each came from its own block (steps recorded there), with a fresh seed.
    assert calls[prose]["seed"] == -1 and calls[prose]["extra_params"]["steps"] == 8
    assert {r["requeued_from"] for r in state["results"]} == {black, orphan}


def test_auto_requeue_can_be_turned_off(service, dirs):
    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    prose = "A woman stands in a sunlit garden holding a watering can."
    black = _black_png(str(gen / "2026-09-17" / "a.png"), (1280, 832), prose, _BLACK_PARAMS)
    assert _start(service, dirs, auto_requeue_black=False)["ok"]
    state = _wait(service)
    # The black image still claims the caption, so nothing is generated.
    assert FakeClient.calls == [] and state["black_retired"] == 0 and os.path.exists(black)


def test_dry_run_lists_black_images_without_touching_them(service, dirs):
    _real, gen, _out = dirs
    (gen / "2026-09-17").mkdir()
    black = _black_png(str(gen / "2026-09-17" / "a.png"), (1280, 832), "A caption no photo has.", _BLACK_PARAMS)
    assert _start(service, dirs, dry_run=True)["ok"]
    state = _wait(service)
    assert FakeClient.calls == [] and os.path.exists(black) and state["black_retired"] == 0
    planned = {r["caption"]: r for r in state["results"]}
    assert planned["A caption no photo has."]["requeued_from"] == black
    assert planned["A caption no photo has."]["planned_only"]
    assert "prose" in os.path.basename(
        planned["A woman stands in a sunlit garden holding a watering can."]["real_path"]
    )


def test_split_keyword_jobs_cleans_entries():
    from web.backend.services.swarm_generate_service import split_keyword_jobs

    assert split_keyword_jobs([" garden ", "", "brick wall + dusk", "Garden", "a,b"]) == [
        "garden",
        "brick wall, dusk",
        "a, b",
    ]
    assert split_keyword_jobs(None) == []


def test_bulk_keywords_queue_one_run_each_without_repeating_a_caption(service, dirs):
    real, _gen, _out = dirs
    _many_captions(real, count=3)
    (real / "many0.txt").write_text("A man repairs a bicycle beside a canal.\n", encoding="utf-8")
    result = _start(service, dirs, captions=5, keywords="ignored", keyword_jobs=["bicycle", "garden", "woman+garden"])
    assert result["ok"] and result["jobs"] == 3 and result["position"] == 1
    state = _wait(service)
    assert state["phase"] == "done" and state["job_total"] == 3 and state["job_index"] == 3
    prompts = [call["prompt"] for call in FakeClient.calls]
    # bicycle: 1 caption; garden: prose + many1 + many2; the third run finds them all done.
    assert len(prompts) == 4 and len(set(prompts)) == 4
    assert "bicycle" in prompts[0] and all("garden" in prompt for prompt in prompts[1:])
