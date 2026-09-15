"""The DPO tool's per-image score breakdown (Selection side panel).

No model is loaded: every scorer is replaced at its module boundary. The
tests pin the contract the panel relies on — scoring off never touches a
model, each scorer degrades on its own, the age head is consulted only when
the caption claims an age, the OCR reader only when the caption mentions
on-image text, the SigLIP adherence scorer is never loaded (it is switched
off; the ``adherence`` axis stays in the output as ``None``), and the
combined score follows the default weights the UI will re-weight client-side.
"""

import os
import threading

from modules.util import adherence_score_util as asu
from modules.util import age_estimate_util as aeu
from modules.util import realism_detect_util as rdu
from modules.util import scalar_quality_util as sq
from modules.util import text_adherence_util as tau
from modules.util.realism_detect_util import REALISM_DEVICE_OFF
from web.backend.services.dpo_service import DPOService

import numpy as np
import pytest


class FakeReader:
    """An OCR backend that reads the image's file name: ``g0`` -> "OPEN",
    ``g1`` -> "OPEM" (one letter wrong), ``g2`` -> nothing."""

    model_version = "fake-ocr/1"
    readings = {"g0": "OPEN", "g1": "OPEM", "g2": ""}
    unloaded = 0

    def __init__(self):
        self.current: str | None = None

    def read(self, pil):
        return self.readings.get(self.current or "", "")

    def unload_model(self):
        type(self).unloaded += 1


class FakeEmbedder:
    model_version = "fake/1"
    scale = 10.0
    bias = -10.0

    @staticmethod
    def _vec(key: str) -> np.ndarray:
        rng = np.random.default_rng(sum(ord(c) for c in key))
        v = rng.standard_normal(8)
        return v / np.linalg.norm(v)

    def embed_images(self, paths):
        return np.stack([self._vec(p) for p in paths])

    def embed_texts(self, texts):
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def images(tmp_path):
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"g{i}.png"
        Image.new("RGB", (16, 24), (i * 60, 30, 30)).save(p)
        paths.append(str(p))
    return paths


@pytest.fixture
def service(images):
    svc = DPOService.__new__(DPOService)
    svc._lock = threading.Lock()
    svc._realism_device = "cpu"
    svc._hash_cache = None
    svc._current_group = {"prompt": "A low angle medium shot of a 34-year-old woman.", "images": list(images)}
    return svc


def _explode(*_args, **_kwargs):
    raise AssertionError("must not be reached")


@pytest.fixture
def fakes(monkeypatch):
    calls = {"age": 0, "ocr": 0}

    def fake_realism(paths, **_kw):
        return {p: 0.1 * (i + 1) for i, p in enumerate(paths)}

    def fake_quality(paths, **_kw):
        return {p: 50.0 + i for i, p in enumerate(paths)}

    def fake_ages(paths, **_kw):
        calls["age"] += 1
        return {p: {"age": 34.0 + 5 * i, "p_none": 0.05} for i, p in enumerate(paths)}

    def fake_read_image_text(paths, *, backend, cache=None, progress=None):
        calls["ocr"] += 1
        out = {}
        for p in paths:
            backend.current = os.path.splitext(os.path.basename(p))[0]
            out[p] = backend.read(None)
        return out

    monkeypatch.setattr(rdu, "score_images_realism", fake_realism)
    monkeypatch.setattr(sq, "score_images_quality", fake_quality)
    monkeypatch.setattr(aeu, "estimate_ages", fake_ages)
    monkeypatch.setattr(asu, "load_siglip_embedder", _explode)
    monkeypatch.setattr(tau, "load_ocr_backend", lambda **_kw: FakeReader())
    monkeypatch.setattr(tau, "read_image_text", fake_read_image_text)
    return calls


def test_scoring_off_touches_no_model(service, images, monkeypatch):
    monkeypatch.setattr(rdu, "load_realism_model", _explode)
    monkeypatch.setattr(asu, "load_siglip_embedder", _explode)
    monkeypatch.setattr(aeu, "load_age_model", _explode)
    monkeypatch.setattr(tau, "load_ocr_backend", _explode)
    service._current_group["prompt"] = 'A sign reading "OPEN" beside a 34-year-old woman.'
    service._realism_device = REALISM_DEVICE_OFF
    assert service.score_breakdown(images) == {"ok": True, "scores": {}, "disabled": True}


def test_scoring_off_releases_a_resident_ocr_reader(service, images, fakes):
    service._current_group["prompt"] = 'A sign reading "OPEN".'
    service.score_breakdown(images)
    assert service._ocr_backend_obj is not None
    before = FakeReader.unloaded
    service._realism_device = REALISM_DEVICE_OFF
    service.score_breakdown(images)
    assert service._ocr_backend_obj is None
    assert FakeReader.unloaded == before + 1


def test_no_group_is_an_error(service, images, fakes):
    service._current_group = None
    assert service.score_breakdown(images)["ok"] is False


def test_paths_outside_the_group_are_refused(service, images, fakes):
    assert service.score_breakdown([*images, "/elsewhere.png"])["ok"] is False


def test_breakdown_has_every_axis_and_the_combined_score(service, images, fakes):
    res = service.score_breakdown(images)
    assert res["ok"] and res["errors"] == {}
    assert set(res["scores"]) == set(images)
    for s in res["scores"].values():
        assert 0 <= s["realism"] <= 1
        assert s["adherence"] is None  # SigLIP is switched off; the key is kept
        assert s["age"] is not None and s["estimated_age"] is not None
        assert s["quality"] >= 50
        assert isinstance(s["combined"], float)
        assert 1 <= s["gate_rank"] <= 3
        assert any(p["axis"] == asu.AGE_MODEL_AXIS for p in s["propositions"])
        assert s["text"] is None  # the caption mentions no on-image text
    assert sorted(s["gate_rank"] for s in res["scores"].values()) == [1, 2, 3]
    assert set(res["default_weights"]) == {"text", "adherence", "age", "realism", "quality"}


def test_ocr_is_skipped_when_the_caption_mentions_no_text(service, images, fakes, monkeypatch):
    monkeypatch.setattr(tau, "load_ocr_backend", _explode)
    res = service.score_breakdown(images)
    assert fakes["ocr"] == 0
    assert res["errors"] == {}
    assert all(s["text"] is None for s in res["scores"].values())


def test_text_axis_reads_the_image_and_gates_the_ranking(service, images, fakes):
    service._current_group["prompt"] = 'A low angle medium shot of a woman beside a sign reading "OPEN".'
    res = service.score_breakdown(images)
    assert res["ok"] and res["errors"] == {}
    assert fakes["ocr"] == 1
    by_name = {os.path.splitext(os.path.basename(p))[0]: s for p, s in res["scores"].items()}
    assert by_name["g0"]["text"] == pytest.approx(1.0)
    assert by_name["g2"]["text"] == tau.ABSENT_SCORE
    assert 0.0 < by_name["g1"]["text"] < by_name["g2"]["text"]
    assert by_name["g0"]["ocr_text"] == "OPEN"
    # The verbatim sign wins the gate even though the realism fake gives it
    # the highest P(AI) of the three; the empty sign beats the misspelled one.
    assert by_name["g0"]["gate_rank"] == 1
    assert by_name["g2"]["gate_rank"] == 2
    assert by_name["g1"]["gate_rank"] == 3
    span = next(p for p in by_name["g1"]["propositions"] if p["axis"] == tau.TEXT_OCR_AXIS)
    assert span["label"] == "OPEN" and span["read"] == "OPEM"
    assert by_name["g1"]["propositions"][0]["axis"] == tau.TEXT_OCR_AXIS  # text spans lead the list


def test_ocr_failing_leaves_the_other_scorers_standing(service, images, fakes, monkeypatch):
    def broken(**_kw):
        raise RuntimeError("no OCR model")

    monkeypatch.setattr(tau, "load_ocr_backend", broken)
    service._current_group["prompt"] = 'A sign reading "OPEN".'
    res = service.score_breakdown(images)
    assert res["ok"] and "text" in res["errors"]
    for s in res["scores"].values():
        assert s["text"] is None
        assert s["realism"] is not None and s["quality"] is not None


def test_ocr_reader_is_kept_across_calls_and_rebuilt_on_device_change(service, images, fakes, monkeypatch):
    built = []

    def loader(**kw):
        built.append(kw.get("device"))
        return FakeReader()

    monkeypatch.setattr(tau, "load_ocr_backend", loader)
    service._current_group["prompt"] = 'A sign reading "OPEN".'
    service.score_breakdown(images)
    service.score_breakdown(images)
    assert built == ["cpu"]
    service._realism_device = "auto"
    service.score_breakdown(images)
    assert built == ["cpu", "auto"]


def test_age_head_is_skipped_when_the_caption_claims_no_age(service, images, fakes):
    service._current_group["prompt"] = "A low angle medium shot of a woman."
    res = service.score_breakdown(images)
    assert fakes["age"] == 0
    assert all(s["age"] is None for s in res["scores"].values())


def test_one_scorer_failing_leaves_the_others_standing(service, images, fakes, monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("no realism model")

    monkeypatch.setattr(rdu, "score_images_realism", broken)
    res = service.score_breakdown(images)
    assert res["ok"]
    assert "realism" in res["errors"]
    for s in res["scores"].values():
        assert s["realism"] is None
        assert s["age"] is not None and s["quality"] is not None


# --------------------------------------------------------------------------- #
# The whole-tree backfill
# --------------------------------------------------------------------------- #
#
# The button's promise is that after it runs, curating loads no model at all.
# The ONNX heads are exercised for real against fake sessions (their cache
# short-circuit reads the sidecar version, which is what these pin), the OCR
# reader through a counting fake. The SigLIP embedder is switched off and must
# never be built.

from modules.util.dpo_curation_util import DpoScanCache  # noqa: E402
from modules.util.scalar_quality_util import QUALITY_CACHE_FIELD, quality_version  # noqa: E402

BACKFILL_PROMPT = 'A low angle medium shot of a 34-year-old woman beside a sign reading "OPEN".'

REALISM_SPEC = {
    "model_version": "realism-test/1",
    "aspect_ratio_buckets": [[512, 512], [576, 448], [448, 576]],
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "positive_class_index": 1,
    "staging_area_px": 800_000,
    "tta": {
        "draws": 1,
        "seed": 1,
        "codecs": [{"fmt": "PNG", "suffix": ".png", "weight": 1.0, "qmin": None, "qmax": None}],
    },
}
AGE_SPEC = {
    "model_version": "age-test/1",
    "aspect_ratio_buckets": [[512, 512], [576, 448], [448, 576]],
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "class_names": ["__none__", "30-year-old", "34-year-old", "40-year-old"],
    "none_index": 0,
    "min_age": 0,
}


class FakeOnnxSession:
    def __init__(self, width: int):
        self.width = width
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        n = feeds["pixels"].shape[0]
        return [np.tile(np.linspace(0.0, 1.0, self.width, dtype=np.float32), (n, 1))]


class ConstReader:
    model_version = "ocr-test/1"

    def __init__(self, text="OPEN"):
        self.text = text
        self.calls = 0
        self.unloaded = 0

    def read(self, pil):
        self.calls += 1
        return self.text

    def unload_model(self):
        self.unloaded += 1


@pytest.fixture
def backfill_service(tmp_path, monkeypatch):
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"b{i}.png"
        Image.new("RGB", (64, 96), (i * 60, 30, 30)).save(p)
        paths.append(str(p))

    svc = DPOService.__new__(DPOService)
    svc._lock = threading.Lock()
    svc._realism_device = "cpu"
    svc._session_active = True
    svc._source_folder = str(tmp_path)
    svc._hash_cache = DpoScanCache(str(tmp_path / ".dpo_hash_cache.json"))
    svc._backfill_thread = None
    svc._backfill_stop = threading.Event()
    svc._backfill_state = {"running": False}
    svc._current_group = {"prompt": BACKFILL_PROMPT, "images": list(paths)}
    monkeypatch.setattr(DPOService, "_extract_group_key", staticmethod(lambda path: (BACKFILL_PROMPT, "1:1")))

    built = {"reader": None}
    monkeypatch.setattr(rdu, "realism_model_version", lambda: REALISM_SPEC["model_version"])
    monkeypatch.setattr(rdu, "load_realism_model", lambda device="auto": (FakeOnnxSession(2), REALISM_SPEC))
    monkeypatch.setattr(aeu, "age_model_version", lambda: AGE_SPEC["model_version"])
    monkeypatch.setattr(aeu, "load_age_model", lambda device="auto": (FakeOnnxSession(4), AGE_SPEC))

    def make_reader(**_kw):
        built["reader"] = ConstReader()
        return built["reader"]

    monkeypatch.setattr(asu, "load_siglip_embedder", _explode)
    monkeypatch.setattr(tau, "load_ocr_backend", make_reader)
    return svc, paths, built


def _run_backfill(svc):
    res = svc.start_score_backfill()
    assert res["ok"], res
    svc._backfill_thread.join(60)
    assert not svc._backfill_thread.is_alive()
    return svc.score_backfill_status()


def test_backfill_covers_every_axis_and_fills_the_cache(backfill_service):
    svc, paths, built = backfill_service
    status = _run_backfill(svc)

    assert status["finished"] and status["error"] is None and status["errors"] == {}
    assert status["done"] == status["total"] == 3
    assert status["axes"] == {"realism": 3, "quality": 3, "age": 3, "adherence": 3, "text": 3}

    cache = svc._hash_cache
    for p in paths:
        assert cache.peek_realism(p, REALISM_SPEC["model_version"]) is not None
        assert cache.peek_versioned(p, "age_estimate", AGE_SPEC["model_version"])["age"] > 0
        assert cache.peek_versioned(p, tau.OCR_CACHE_FIELD, "ocr-test/1") == "OPEN"
        assert cache.peek_versioned(p, QUALITY_CACHE_FIELD, quality_version()) is not None
        # SigLIP is off: no image vectors are written, and none are needed.
        assert cache.peek_adherence_embedding(p, "fake/1") is None


def test_backfill_releases_the_resident_models_when_done(backfill_service):
    svc, _paths, built = backfill_service
    _run_backfill(svc)
    assert built["reader"].unloaded == 1
    assert getattr(svc, "_adherence_embedder_obj", None) is None
    assert getattr(svc, "_ocr_backend_obj", None) is None


def test_after_a_backfill_selection_loads_no_model(backfill_service, monkeypatch):
    """The promise behind the button: with the cache full, every scorer
    answers from it and a busy GPU is never asked for anything."""
    svc, paths, _built = backfill_service
    _run_backfill(svc)

    class NeverReads:
        model_version = "ocr-test/1"

        def read(self, pil):
            raise AssertionError("readings are cached — the OCR model must not load")

    monkeypatch.setattr(rdu, "load_realism_model", _explode)
    monkeypatch.setattr(aeu, "load_age_model", _explode)
    monkeypatch.setattr(asu, "load_siglip_embedder", _explode)
    monkeypatch.setattr(tau, "load_ocr_backend", lambda **_kw: NeverReads())

    res = svc.score_breakdown(paths)
    assert res["ok"] and res["errors"] == {}
    for s in res["scores"].values():
        assert s["realism"] is not None and s["quality"] is not None
        assert s["age"] is not None and s["adherence"] is None
        assert s["text"] == pytest.approx(1.0)


def test_backfill_skips_the_axes_the_caption_does_not_ask_for(backfill_service, monkeypatch):
    svc, _paths, built = backfill_service
    monkeypatch.setattr(DPOService, "_extract_group_key", staticmethod(lambda path: ("A woman in a red dress.", "1:1")))
    monkeypatch.setattr(aeu, "load_age_model", _explode)
    monkeypatch.setattr(tau, "load_ocr_backend", _explode)
    status = _run_backfill(svc)
    assert status["errors"] == {}
    assert status["axes"]["age"] == 0 and status["axes"]["text"] == 0
    assert status["axes"]["realism"] == 3 and status["axes"]["adherence"] == 3
    assert built["reader"] is None


def test_backfill_reports_a_missing_scorer_and_finishes_the_rest(backfill_service, monkeypatch):
    svc, paths, _built = backfill_service
    monkeypatch.setattr(rdu, "load_realism_model", lambda device="auto": (None, None))
    status = _run_backfill(svc)
    assert status["finished"] and status["error"] is None
    assert "realism" in status["errors"] and "Realism scoring needs" in status["errors"]["realism"]
    assert status["axes"]["realism"] == 0
    assert status["axes"]["text"] == 3 and status["axes"]["adherence"] == 3
    assert svc._hash_cache.peek_versioned(paths[0], tau.OCR_CACHE_FIELD, "ocr-test/1") == "OPEN"


def test_siglip_is_never_loaded_and_stale_vectors_are_left_alone(backfill_service, monkeypatch):
    """SigLIP is switched off: a breakdown neither builds the embedder nor
    touches the image vectors an earlier session may have cached, and the
    other cache fields stay valid."""
    svc, paths, _built = backfill_service
    cache = svc._hash_cache
    stale = np.ones(4, dtype=np.float32)
    cache.store_adherence_embedding(paths[0], stale, "old-siglip/1")
    monkeypatch.setattr(asu, "load_siglip_embedder", _explode)
    monkeypatch.setattr(asu, "score_images_adherence", _explode)

    res = svc.score_breakdown(paths)
    assert res["ok"] and res["errors"] == {}
    assert all(s["adherence"] is None for s in res["scores"].values())
    assert all(s["age"] is not None for s in res["scores"].values())
    assert np.array_equal(cache.peek_adherence_embedding(paths[0], "old-siglip/1"), stale)
