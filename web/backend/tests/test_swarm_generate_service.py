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


def _png(path: str, size: tuple[int, int], prompt: str, params: dict | None = None) -> str:
    info = PngInfo()
    info.add_text("prompt", prompt)
    if params is not None:
        # SwarmUI stamps its full parameter set into every image it writes.
        info.add_text("parameters", json.dumps({"prompt": prompt, **params}))
    Image.new("RGB", size, (120, 90, 60)).save(path, pnginfo=info)
    return path


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
            _png(written, (width, height), prompt)
            views.append(f"View/local/raw/2026-08-30/{name}")
        return views


@pytest.fixture
def service(monkeypatch):
    import web.backend.services.swarm_generate_service as module

    FakeClient.calls = []
    FakeClient.fail_with = None
    FakeClient.fail_times = 0
    FakeClient.output_root = ""
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
    second = _start(service, dirs)
    assert not second["ok"] and "already in progress" in second["error"]
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

    monkeypatch.setattr(module, "SwarmClient", Client)
    assert service.check("localhost:7801") == {
        "ok": True,
        "version": "0.9.8",
        "user_id": "local",
        "models": ["a.safetensors"],
        "loras": ["lora-a.safetensors"],
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

    monkeypatch.setattr(module, "SwarmClient", Partial)
    result = service.check("localhost:7801")
    assert result["ok"] and result["models"] == ["a.safetensors"]
    assert result["loras"] == [] and result["loaded"] == [] and result["presets"] == []


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
