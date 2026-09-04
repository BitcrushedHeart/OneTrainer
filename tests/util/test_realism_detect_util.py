"""Tests for the DPO realism scorer in modules.util.realism_detect_util.

No model is ever loaded: ``score_images_realism`` takes an injectable
``loader``, so the ONNX session is replaced with a fake that records what it was
handed. That lets the tests assert the things that actually matter and would
otherwise only fail silently in production:

* the codec choice reproduces the training-time selection bit for bit — get this
  wrong and the scorer quietly leaves the distribution the head was fitted on;
* several codec draws are averaged, because scoring a single arbitrary encoding
  measurably collapses the usable spread of the score;
* cache hits skip the model entirely, and a missing model degrades to an error
  the UI can show rather than a crash.
"""

import json
from pathlib import Path

from modules.util import realism_detect_util as rdu

import numpy as np
import pytest
from PIL import Image

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

SPEC = {
    "aspect_ratio_buckets": [[512, 512], [576, 448], [448, 576], [672, 384], [384, 672]],
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "positive_class_index": 1,
    "threshold": 0.59,
    "staging_area_px": 800_000,
    "tta": {
        "draws": 4,
        "seed": 20260810,
        "codecs": [
            {"fmt": "JPEG", "suffix": ".jpg", "weight": 0.45, "qmin": 82, "qmax": 96},
            {"fmt": "WEBP", "suffix": ".webp", "weight": 0.45, "qmin": 80, "qmax": 95},
            {"fmt": "PNG", "suffix": ".png", "weight": 0.10, "qmin": None, "qmax": None},
        ],
    },
}


class FakeSession:
    """Stands in for an onnxruntime session; returns fixed logits per call."""

    def __init__(self, logit_pairs=None):
        self.calls = 0
        self.batch_shapes = []
        self._logit_pairs = logit_pairs

    def run(self, _outputs, feeds):
        pixels = feeds["pixels"]
        self.calls += 1
        self.batch_shapes.append(pixels.shape)
        n = pixels.shape[0]
        if self._logit_pairs is not None:
            pair = self._logit_pairs[min(self.calls - 1, len(self._logit_pairs) - 1)]
            return [np.tile(np.asarray(pair, dtype=np.float32), (n, 1))]
        return [np.zeros((n, 2), dtype=np.float32)]


def _loader(session, spec=None):
    return lambda: (session, spec if spec is not None else SPEC)


@pytest.fixture()
def image(tmp_path: Path) -> str:
    rng = np.random.default_rng(0)
    path = tmp_path / "gen.webp"
    Image.fromarray(rng.integers(0, 256, (1216, 832, 3), dtype=np.uint8)).save(path, "WEBP")
    return str(path)


# --------------------------------------------------------------------------- #
# Codec selection must reproduce the training-time draw
# --------------------------------------------------------------------------- #


def test_choose_codec_is_deterministic():
    assert rdu._choose_codec("a/b.webp#tta0", SPEC) == rdu._choose_codec("a/b.webp#tta0", SPEC)


def test_choose_codec_varies_across_draws():
    # Flatten to a hashable form — the second element is a params dict.
    picks = {
        (fmt, params.get("quality")) for fmt, params in (rdu._choose_codec(f"a/b.webp#tta{i}", SPEC) for i in range(8))
    }
    assert len(picks) > 1


def test_choose_codec_uses_every_configured_codec():
    seen = {rdu._choose_codec(f"img{i}#tta0", SPEC)[0] for i in range(2000)}
    assert seen == {"JPEG", "WEBP", "PNG"}


def test_choose_codec_quality_stays_in_range():
    for i in range(500):
        fmt, params = rdu._choose_codec(f"img{i}#tta0", SPEC)
        if fmt == "JPEG":
            assert 82 <= params["quality"] <= 96
        elif fmt == "WEBP":
            assert 80 <= params["quality"] <= 95
        else:
            assert params == {}


def test_choose_codec_matches_the_build_tools_selection():
    """Bit-for-bit against the staging tool's own algorithm.

    Re-implemented here rather than imported — the build tool lives in another
    repo. If either side changes its hashing, this goes red, which is the point:
    a silent divergence would move serving off the training distribution.
    """
    import hashlib

    def reference(key: str, seed: int) -> str:
        def unit(salt: str) -> float:
            digest = hashlib.blake2b(f"{salt}\x00{seed}\x00{key}".encode(), digest_size=8).digest()
            return int.from_bytes(digest, "big") / float(1 << 64)

        codecs = SPEC["tta"]["codecs"]
        roll = unit("codec") * sum(c["weight"] for c in codecs)
        upto = 0.0
        for candidate in codecs:
            upto += candidate["weight"]
            if roll < upto:
                return candidate["fmt"]
        return codecs[-1]["fmt"]

    for i in range(200):
        key = f"E:/gen/{i}.webp#tta{i % 4}"
        assert rdu._choose_codec(key, SPEC)[0] == reference(key, 20260810)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_nearest_bucket_picks_the_closest_ratio():
    assert rdu._nearest_bucket(832, 1216, SPEC["aspect_ratio_buckets"]) == (448, 576)
    assert rdu._nearest_bucket(1024, 1024, SPEC["aspect_ratio_buckets"]) == (512, 512)
    assert rdu._nearest_bucket(1920, 1080, SPEC["aspect_ratio_buckets"]) == (672, 384)


def test_crop_to_ratio_returns_the_source_when_already_close():
    img = Image.new("RGB", (1000, 1000))
    assert rdu._crop_to_ratio(img, 1.0) is img


def test_crop_to_ratio_produces_the_target_ratio():
    img = Image.new("RGB", (832, 1216))
    out = rdu._crop_to_ratio(img, 448 / 576)
    assert abs(out.width / out.height - 448 / 576) < 0.01


def test_tensor_for_draw_has_bucket_shape_and_is_normalised(image: str):
    arr = rdu._tensor_for_draw(image, 0, SPEC)
    assert arr.shape == (3, 576, 448)
    assert arr.dtype == np.float32
    assert -5.0 < float(arr.mean()) < 5.0


def test_tensor_for_draw_differs_between_draws(image: str):
    """Different codec per draw is the whole mechanism; identical tensors would
    mean the averaging is doing nothing."""
    tensors = [rdu._tensor_for_draw(image, d, SPEC) for d in range(4)]
    assert any(not np.allclose(tensors[0], t) for t in tensors[1:])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_score_averages_over_the_configured_draws(image: str):
    # Alternate saturated logits so the draws give P ~= 0, 1, 0, 1 and the mean
    # is unambiguously 0.5 — a scorer that returned any single draw instead of
    # the average would land on 0 or 1 and fail loudly.
    session = FakeSession(logit_pairs=[(100.0, 0.0), (0.0, 100.0), (100.0, 0.0), (0.0, 100.0)])
    scores = rdu.score_images_realism([image], loader=_loader(session))
    assert session.calls == SPEC["tta"]["draws"]
    assert scores[image] == pytest.approx(0.5, abs=1e-3)


def test_score_is_a_probability(image: str):
    session = FakeSession(logit_pairs=[(0.0, 2.0)])
    value = rdu.score_images_realism([image], loader=_loader(session))[image]
    assert 0.0 <= value <= 1.0


def test_cache_hits_never_touch_the_model(image: str):
    class Cache:
        def peek_realism(self, _path, _version=None):
            return 0.25

        def store_realism(self, _path, _score, _version=None):
            raise AssertionError("should not store a cache hit")

    session = FakeSession()
    scores = rdu.score_images_realism([image], loader=_loader(session), cache=Cache())
    assert scores == {image: 0.25}
    assert session.calls == 0


def test_misses_are_written_back_to_the_cache(image: str):
    stored = {}

    class Cache:
        def peek_realism(self, _path, _version=None):
            return None

        def store_realism(self, path, score, version=None):
            stored[path] = (score, version)

    rdu.score_images_realism([image], loader=_loader(FakeSession()), cache=Cache())
    assert list(stored) == [image]
    score, version = stored[image]
    assert 0.0 <= score <= 1.0
    assert version == SPEC.get("model_version")


def test_a_new_model_version_invalidates_cached_scores(image: str):
    """How a retrained head reaches every session cache at once.

    These caches live in each DPO session's output folder, so there is no single
    file to go and clear, and a stale score looks exactly like a fresh one. The
    model version is therefore part of the validity check: bump it and the old
    score is simply a miss.
    """
    entries: dict = {}

    class Cache:
        def peek_realism(self, path, version=None):
            hit = entries.get(path)
            return hit[0] if hit and hit[1] == version else None

        def store_realism(self, path, score, version=None):
            entries[path] = (score, version)

    cache = Cache()
    spec_v1 = {**SPEC, "model_version": "aaaa1111"}
    spec_v2 = {**SPEC, "model_version": "bbbb2222"}

    first = FakeSession(logit_pairs=[(0.0, 100.0)])
    rdu.score_images_realism([image], loader=_loader(first, spec_v1), cache=cache)
    assert first.calls > 0

    # Same model again -> served from cache, no forward pass.
    again = FakeSession(logit_pairs=[(0.0, 100.0)])
    rdu.score_images_realism([image], loader=_loader(again, spec_v1), cache=cache)
    assert again.calls == 0

    # Retrained model -> recomputed, and the new value replaces the old.
    retrained = FakeSession(logit_pairs=[(100.0, 0.0)])
    scores = rdu.score_images_realism([image], loader=_loader(retrained, spec_v2), cache=cache)
    assert retrained.calls > 0
    assert scores[image] == pytest.approx(0.0, abs=1e-3)
    assert entries[image][1] == "bbbb2222"


def test_scores_from_an_unversioned_spec_still_cache(image: str):
    """A sidecar predating versioning must not thrash the cache forever."""
    entries: dict = {}

    class Cache:
        def peek_realism(self, path, version=None):
            hit = entries.get(path)
            return hit[0] if hit and hit[1] == version else None

        def store_realism(self, path, score, version=None):
            entries[path] = (score, version)

    cache = Cache()
    rdu.score_images_realism([image], loader=_loader(FakeSession()), cache=cache)
    second = FakeSession()
    rdu.score_images_realism([image], loader=_loader(second), cache=cache)
    assert second.calls == 0


def test_missing_model_raises_a_reportable_error(image: str):
    with pytest.raises(RuntimeError, match="onnxruntime"):
        rdu.score_images_realism([image], loader=lambda: (None, None))


def test_unreadable_files_are_skipped_not_fatal(tmp_path: Path, image: str):
    broken = tmp_path / "broken.webp"
    broken.write_bytes(b"not an image")
    scores = rdu.score_images_realism([image, str(broken)], loader=_loader(FakeSession()))
    assert image in scores
    assert str(broken) not in scores


def test_empty_input_does_not_load_the_model():
    session = FakeSession()
    assert rdu.score_images_realism([], loader=_loader(session)) == {}
    assert session.calls == 0


def test_progress_is_reported_and_ends_complete(image: str):
    seen = []
    rdu.score_images_realism([image], loader=_loader(FakeSession()), progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1] == (1, 1)


def test_batches_are_shape_homogeneous(tmp_path: Path):
    """Aspect buckets have different dims, so a mixed batch cannot be stacked."""
    rng = np.random.default_rng(1)
    paths = []
    for i, (w, h) in enumerate([(832, 1216), (1216, 832), (1024, 1024)]):
        path = tmp_path / f"g{i}.webp"
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path, "WEBP")
        paths.append(str(path))
    session = FakeSession()
    rdu.score_images_realism(paths, loader=_loader(session))
    assert session.batch_shapes
    for shape in session.batch_shapes:
        assert len(shape) == 4


def test_single_draw_spec_still_works(image: str):
    spec = json.loads(json.dumps(SPEC))
    spec["tta"]["draws"] = 1
    session = FakeSession()
    rdu.score_images_realism([image], loader=_loader(session, spec))
    assert session.calls == 1


# --------------------------------------------------------------------------- #
# Preprocessing pool size
# --------------------------------------------------------------------------- #


def test_worker_count_scales_with_the_machine(monkeypatch):
    """The old flat cap of 8 left a 32-core box mostly idle: preprocessing is
    the bottleneck (measured ~4 img/s against a GPU that was not saturated), so
    the pool has to grow with the machine."""
    monkeypatch.setattr(rdu.os, "cpu_count", lambda: 32)
    assert rdu._preprocess_worker_count() == 24


def test_worker_count_leaves_headroom_on_the_box(monkeypatch):
    """Generation runs on the same machine. Taking every core would make the
    scorer win a fight it should not be in."""
    monkeypatch.setattr(rdu.os, "cpu_count", lambda: 32)
    assert rdu._preprocess_worker_count() < 32


def test_worker_count_stays_sane_on_small_machines(monkeypatch):
    monkeypatch.setattr(rdu.os, "cpu_count", lambda: 4)
    assert rdu._preprocess_worker_count() == 1
    monkeypatch.setattr(rdu.os, "cpu_count", lambda: 1)
    assert rdu._preprocess_worker_count() == 1
    monkeypatch.setattr(rdu.os, "cpu_count", lambda: None)
    assert rdu._preprocess_worker_count() >= 1


def test_an_explicit_worker_count_wins(monkeypatch):
    monkeypatch.setattr(rdu.os, "cpu_count", lambda: 32)
    assert rdu._preprocess_worker_count(3) == 3
    # Nonsense is clamped rather than handed to ThreadPoolExecutor, which
    # raises on zero and negative counts.
    assert rdu._preprocess_worker_count(0) == 1
    assert rdu._preprocess_worker_count(-5) == 1


def test_score_uses_the_requested_worker_count(image: str, monkeypatch):
    """The parameter has to reach the pool — a plumbing break here is invisible,
    because the scores come out identical either way, just slower."""
    seen: list[int] = []
    real_pool = rdu.ThreadPoolExecutor

    class RecordingPool(real_pool):
        def __init__(self, *args, max_workers=None, **kwargs):
            seen.append(max_workers)
            super().__init__(*args, max_workers=max_workers, **kwargs)

    monkeypatch.setattr(rdu, "ThreadPoolExecutor", RecordingPool)
    rdu.score_images_realism([image], loader=_loader(FakeSession()), workers=5)
    assert seen and set(seen) == {5}


# --------------------------------------------------------------------------- #
# realism_model_version — the cache key, without paying for a session
# --------------------------------------------------------------------------- #


def test_model_version_is_read_without_creating_a_session(tmp_path: Path, monkeypatch):
    """Ordering groups needs the version to consult the cache, and nothing else.

    Loading the model for it would take ~2 GB of VRAM and eight seconds during a
    scan that must never touch the GPU — the whole point of ordering from cached
    scores is that it is free.
    """
    sidecar = tmp_path / "realism.json"
    sidecar.write_text(json.dumps({"model_version": "abc123"}), encoding="utf-8")
    monkeypatch.setattr(rdu, "model_paths", lambda: (str(tmp_path / "realism.onnx"), str(sidecar)))

    def _explode():
        raise AssertionError("the model must not be loaded to read a version string")

    monkeypatch.setattr(rdu, "load_realism_model", _explode)
    assert rdu.realism_model_version() == "abc123"


def test_model_version_is_none_when_the_sidecar_is_missing(tmp_path: Path, monkeypatch):
    """No model published yet. Ordering then has no scores to order by and must
    fall back, not raise mid-scan."""
    monkeypatch.setattr(rdu, "model_paths", lambda: (str(tmp_path / "nope.onnx"), str(tmp_path / "nope.json")))
    assert rdu.realism_model_version() is None


def test_model_version_is_none_when_the_sidecar_is_corrupt(tmp_path: Path, monkeypatch):
    sidecar = tmp_path / "realism.json"
    sidecar.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(rdu, "model_paths", lambda: (str(tmp_path / "realism.onnx"), str(sidecar)))
    assert rdu.realism_model_version() is None


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #


class FakeOnnxRuntime:
    """Enough of the onnxruntime module surface to see which providers we ask for."""

    def __init__(self, available=("CUDAExecutionProvider", "CPUExecutionProvider")):
        self._available = list(available)
        self.requested_providers = None

    def get_available_providers(self):
        return list(self._available)

    def InferenceSession(self, _path, providers=None):  # noqa: N802 — mirrors ORT's name
        self.requested_providers = providers
        return FakeSession()


@pytest.fixture()
def published_model(tmp_path: Path, monkeypatch):
    """A model that exists on disk, so load_realism_model gets as far as choosing
    providers. The .onnx is never parsed — the fake session stands in for it."""
    onnx = tmp_path / "realism_femto.onnx"
    onnx.write_bytes(b"")
    sidecar = tmp_path / "realism_femto.json"
    sidecar.write_text(json.dumps(SPEC), encoding="utf-8")
    monkeypatch.setattr(rdu, "model_paths", lambda: (str(onnx), str(sidecar)))
    return onnx, sidecar


def _install_runtime(monkeypatch, runtime) -> list[int]:
    """Point the lazy ``import onnxruntime`` at *runtime* and count DLL preloads."""
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    preloads: list[int] = []
    monkeypatch.setattr(rdu, "_preload_cuda_dlls", lambda: preloads.append(1))
    return preloads


def test_auto_device_prefers_cuda_when_it_is_available(published_model, monkeypatch):
    runtime = FakeOnnxRuntime()
    _install_runtime(monkeypatch, runtime)
    session, spec = rdu.load_realism_model()
    assert session is not None and spec["positive_class_index"] == 1
    assert runtime.requested_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_cpu_device_never_offers_the_gpu_provider(published_model, monkeypatch):
    """The whole point of the option: CUDA is available and must still go unused,
    so the GPU a training run is holding stays untouched."""
    runtime = FakeOnnxRuntime()
    _install_runtime(monkeypatch, runtime)
    rdu.load_realism_model(device="cpu")
    assert runtime.requested_providers == ["CPUExecutionProvider"]


def test_cpu_device_skips_the_cuda_dll_preload(published_model, monkeypatch):
    """Preloading pulls torch's CUDA DLLs into the process; on the CPU path that
    is cost with no benefit."""
    preloads = _install_runtime(monkeypatch, FakeOnnxRuntime())
    rdu.load_realism_model(device="cpu")
    assert preloads == []
    rdu.load_realism_model(device="auto")
    assert preloads == [1]


def test_off_is_pinned_to_the_cpu_as_a_backstop(published_model, monkeypatch):
    """Callers gate on "off" and never get here. If one ever forgets, the failure
    has to be a slow score rather than the VRAM the setting was meant to spare —
    which is exactly what the unknown-device fallback below would not give."""
    runtime = FakeOnnxRuntime()
    preloads = _install_runtime(monkeypatch, runtime)
    rdu.load_realism_model(device=rdu.REALISM_DEVICE_OFF)
    assert runtime.requested_providers == ["CPUExecutionProvider"]
    assert preloads == []


def test_an_unknown_device_falls_back_to_auto(published_model, monkeypatch):
    runtime = FakeOnnxRuntime()
    _install_runtime(monkeypatch, runtime)
    rdu.load_realism_model(device="tpu")
    assert runtime.requested_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_score_passes_the_device_to_the_default_loader(image: str, monkeypatch):
    seen = []

    def _record(device="auto"):
        seen.append(device)
        return FakeSession(), SPEC

    monkeypatch.setattr(rdu, "load_realism_model", _record)
    rdu.score_images_realism([image], device="cpu")
    assert seen == ["cpu"]


def test_an_explicit_loader_still_wins(image: str, monkeypatch):
    """Every test above injects a loader; the device argument must not break that."""

    def _explode(device="auto"):
        raise AssertionError("an explicit loader owns session creation")

    monkeypatch.setattr(rdu, "load_realism_model", _explode)
    scores = rdu.score_images_realism([image], loader=_loader(FakeSession()), device="cpu")
    assert scores[image] == pytest.approx(0.5)
