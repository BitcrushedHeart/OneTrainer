"""Realism scoring for DPO curation: "how AI-generated does this look?"

Scores an image with a small ConvNeXt-V2 femto head trained to separate this
project's own generations from real photographs. Selection mode uses it to sort
a group most-photorealistic-first, so the *ordering* is the product — absolute
calibration does not matter.

The model ships as ONNX plus a JSON sidecar under ``external/models/realism/``.
Everything about how an image must be prepared lives in that sidecar (bucket
table, normalisation constants, downscale area, codec draws), so this module
reads the recipe rather than restating it and the two cannot drift apart.

Two non-obvious pieces of that recipe, both measured rather than assumed:

* **Downscale before bucketing.** The head was fitted on images normalised to
  ~0.8 MP. Skipping that step is not fatal on its own, but it is part of making
  a raw generation look like what the model was trained on.
* **Average over several codec round-trips.** Training randomised the codec so
  the model could not use compression artifacts as a shortcut. Scoring one
  arbitrary encoding therefore samples that nuisance variable exactly once:
  measured on held-out data it dropped the usable spread of the score from 0.107
  to 0.068 IQR, i.e. the ranker lost about 40% of its resolving power. Averaging
  four draws restores it (0.109). The draws are keyed on the file path, so the
  score is deterministic and safe to cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import numpy as np
from PIL import Image

_MODEL_DIR = os.path.join("external", "models", "realism")
_MODEL_NAME = "realism_femto"


def model_paths() -> tuple[str, str]:
    """(onnx path, sidecar path) for the realism head."""
    base = os.path.join(_MODEL_DIR, _MODEL_NAME)
    return base + ".onnx", base + ".json"


#: Ceiling on preprocessing threads, and the cores left for everything else.
#: The pool decodes, resizes and re-encodes; Pillow drops the GIL throughout, so
#: throughput tracks core count until the GPU becomes the bottleneck. Measured
#: on a 32-core box: 8 threads gave ~4 img/s, 24 gave ~8.6. The headroom exists
#: because image generation runs on the same machine — the scorer should get
#: most of the box, not all of it.
_MAX_PREPROCESS_WORKERS = 24
_PREPROCESS_HEADROOM_CORES = 8


def _preprocess_worker_count(requested: int | None = None) -> int:
    """Threads for the preprocessing pool, at least 1.

    An explicit *requested* count wins (clamped to a usable minimum, since
    ThreadPoolExecutor rejects zero and negatives); otherwise it scales with the
    machine, capped and with cores left over for the generator.
    """
    if requested is not None:
        return max(1, int(requested))
    cores = os.cpu_count() or 4
    return max(1, min(_MAX_PREPROCESS_WORKERS, cores - _PREPROCESS_HEADROOM_CORES))


def realism_model_version() -> str | None:
    """The published model's version string, read straight from the sidecar.

    Cached scores are keyed on this, so anything that wants to *consult* the
    cache needs it — but creating an InferenceSession just to read a string
    would cost seconds and ~2 GB of VRAM. Group ordering runs during a scan that
    must not touch the GPU at all, so it reads the sidecar instead.

    ``None`` when no model is published or the sidecar is unreadable: the caller
    then finds no valid cached scores and falls back, rather than failing.
    """
    _onnx_path, sidecar_path = model_paths()
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            spec = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    version = spec.get("model_version") if isinstance(spec, dict) else None
    return version if isinstance(version, str) else None


_CUDA_PRELOADED = False


def _preload_cuda_dlls() -> None:
    """Load the CUDA/cuDNN DLLs the CUDA execution provider depends on.

    Without this, scoring silently runs on CPU. ``get_available_providers()``
    lists ``CUDAExecutionProvider`` because it is compiled in, so the provider
    list below looks correct — but ORT loads ``onnxruntime_providers_cuda.dll``
    through the plain Windows loader, and that loader does not consult
    ``os.add_dll_directory`` when resolving *that* DLL's own imports
    (``cublasLt64_13.dll``, ``cudnn64_9.dll``, …). Session creation then falls
    back to CPU with nothing louder than a warning.

    torch bundles a version-matched CUDA set (this venv: torch 2.12+cu130
    against ORT 1.23), so pointing ``preload_dlls`` at its ``lib`` directory
    resolves them by absolute path; afterwards the loader finds them by base
    name. Best-effort and idempotent — a missing ``preload_dlls`` (ORT < 1.21),
    an absent torch, or a load failure all leave the CPU path intact.

    Mirrors ``bitcrush_pyshared.inference.onnx_runtime.preload_cuda_dlls``
    (Bitcrush ISSUE-0704 — the same failure wearing a different hat).
    """
    global _CUDA_PRELOADED
    if _CUDA_PRELOADED:
        return
    _CUDA_PRELOADED = True
    try:
        import onnxruntime
    except ImportError:
        return
    preload = getattr(onnxruntime, "preload_dlls", None)
    if preload is None:
        return
    directory = None
    try:
        import torch

        lib = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
        if os.path.isdir(lib):
            directory = lib
    except Exception:
        directory = None
    with suppress(Exception):
        preload(directory=directory) if directory else preload()


#: Devices the head can be pinned to. ``"auto"`` prefers CUDA and falls back to
#: CPU when it is not usable; ``"cpu"`` never touches the GPU. The CPU option
#: exists because the scorer shares a box with generation and training — a
#: curation pass is worth trading throughput for when the ~2 GB of VRAM and the
#: GPU time it wants are already spoken for.
REALISM_DEVICES = ("auto", "cpu")

#: Setting value meaning "score nothing at all". Not a device — it is the third
#: position of the same control, because it answers the same question the CPU
#: option does (what can this box spare right now?) and putting it anywhere else
#: would give every scoring entry point two flags to consult instead of one.
#: The policy is enforced by the caller, which never reaches the model; the
#: ``cpu_only`` treatment below is only a backstop, so a path that forgets the
#: check degrades to the CPU rather than quietly taking the VRAM.
REALISM_DEVICE_OFF = "off"

#: Everything the scoring control can be set to: a real device, or off.
REALISM_DEVICE_SETTINGS = (*REALISM_DEVICES, REALISM_DEVICE_OFF)


def load_realism_model(device: str = "auto"):
    """Lazy-load the ONNX session and its preprocessing spec.

    ``device`` is one of :data:`REALISM_DEVICES`; anything unrecognised behaves
    as ``"auto"``, so a stale caller degrades to the old behaviour rather than
    failing. The one exception is :data:`REALISM_DEVICE_OFF`, which is pinned to
    the CPU — a caller that reaches here with scoring off has already skipped a
    check, and erring toward "slow" beats erring toward "took the VRAM".

    Returns ``(None, None)`` when onnxruntime or the weights are missing, so the
    DPO tool degrades to "no scores" instead of crashing — same contract as
    ``load_dino_model``.
    """
    onnx_path, sidecar_path = model_paths()
    if not os.path.isfile(onnx_path) or not os.path.isfile(sidecar_path):
        return None, None
    try:
        import onnxruntime
    except ImportError:
        return None, None

    cpu_only = str(device).lower() in ("cpu", REALISM_DEVICE_OFF)
    # Preloading is only ever useful to the CUDA provider, and it is not free —
    # it pulls torch's whole CUDA DLL set into the process. Skip it when the
    # caller has already said the GPU is off limits.
    if not cpu_only:
        _preload_cuda_dlls()

    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            spec = json.load(handle)
        providers = ["CPUExecutionProvider"]
        if not cpu_only and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return session, spec
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Preprocessing — mirrors the training transform described by the sidecar
# ---------------------------------------------------------------------------


def _unit_hash(key: str, seed: int, salt: str) -> float:
    digest = hashlib.blake2b(f"{salt}\x00{seed}\x00{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _choose_codec(key: str, spec: dict) -> tuple[str, dict]:
    """The codec+quality staging would have picked for ``key``.

    Reproduces the build tool's selection bit for bit — same blake2b keying,
    same cumulative-weight walk — because the point is to land back inside the
    distribution the head was fitted on.
    """
    tta = spec["tta"]
    codecs = tta["codecs"]
    seed = int(tta["seed"])
    roll = _unit_hash(key, seed, "codec") * sum(c["weight"] for c in codecs)
    upto = 0.0
    chosen = codecs[-1]
    for candidate in codecs:
        upto += candidate["weight"]
        if roll < upto:
            chosen = candidate
            break

    params: dict = {}
    if chosen.get("qmin") is not None and chosen.get("qmax") is not None:
        span = int(chosen["qmax"]) - int(chosen["qmin"])
        quality = int(chosen["qmin"]) + int(_unit_hash(key, seed, "quality") * (span + 1))
        params["quality"] = min(int(chosen["qmax"]), quality)
    return chosen["fmt"], params


def _nearest_bucket(width: int, height: int, buckets) -> tuple[int, int]:
    ratio = width / height
    best = buckets[0]
    best_diff = abs(ratio - best[0] / best[1])
    for bucket in buckets[1:]:
        diff = abs(ratio - bucket[0] / bucket[1])
        if diff < best_diff:
            best_diff = diff
            best = bucket
    return int(best[0]), int(best[1])


def _crop_to_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    """Centre-crop to ``target_ratio``.

    The 0.01 tolerance and the ``int()`` truncation are deliberate: this mirrors
    BitTrainer's ``face_aware_crop`` no-face branch, and both details affect
    which pixels survive.
    """
    img_w, img_h = img.size
    img_ratio = img_w / img_h
    if abs(img_ratio - target_ratio) < 0.01:
        return img
    if target_ratio > img_ratio:
        new_h = max(1, int(img_w / target_ratio))
        new_w = img_w
    else:
        new_w = max(1, int(img_h * target_ratio))
        new_h = img_h
    left = max(0, min((img_w - new_w) // 2, img_w - new_w))
    top = max(0, min((img_h - new_h) // 2, img_h - new_h))
    return img.crop((left, top, left + new_w, top + new_h))


def _tensor_for_draw(path: str, draw: int, spec: dict) -> np.ndarray:
    """One TTA sample of ``path`` as a normalised CHW float32 array."""
    with Image.open(path) as handle:
        img = handle.convert("RGB")

    area = int(spec.get("staging_area_px") or 0)
    if area:
        w, h = img.size
        scale = (area / (w * h)) ** 0.5
        if scale < 1.0:  # never upscale — that invents detail the head never saw
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    fmt, params = _choose_codec(f"{path}#tta{draw}", spec)
    buffer = io.BytesIO()
    img.save(buffer, fmt, **params)
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        img = reopened.convert("RGB")

    bw, bh = _nearest_bucket(img.width, img.height, spec["aspect_ratio_buckets"])
    img = _crop_to_ratio(img, bw / bh)
    img = img.resize((bw, bh), Image.LANCZOS)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(spec["imagenet_mean"], dtype=np.float32)) / np.asarray(
        spec["imagenet_std"], dtype=np.float32
    )
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def _softmax_positive(logits: np.ndarray, index: int) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp[:, index] / exp.sum(axis=1)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_images_realism(
    paths: list[str],
    *,
    loader=None,
    cache=None,
    batch_size: int = 8,
    progress=None,
    workers: int | None = None,
    device: str = "auto",
    cache_only: bool = False,
) -> dict[str, float]:
    """P(AI-generated) in [0, 1] for each readable path.

    ``cache_only`` answers from the cache and leaves every miss unscored
    instead of loading the model: the caller knows a backfill has already
    scored these paths, so a miss is a file the model cannot read anyway.

    Cache hits (validated by size, mtime and model version) are served without
    a forward pass; only the misses go through the model. Unreadable files are
    skipped rather than raising — one bad file in a group should not cost the
    whole group its scores.

    The cache is consulted before the model is loaded: the model version that
    decides whether a cached score is still valid is read from the sidecar
    (:func:`realism_model_version`), so a group that is fully cached — after a
    backfill — never creates a session. That is the point of the backfill: no
    weights on a GPU that is busy with something else. Only a miss loads the
    model, and an empty ``paths`` short-circuits before anything.

    ``workers`` overrides the preprocessing pool size; the default scales with
    the machine (see :func:`_preprocess_worker_count`). It changes throughput
    only — scores are identical, because every TTA draw is keyed on the file
    path rather than on where it landed in a batch.

    ``device`` picks where the forward pass runs (:data:`REALISM_DEVICES`) and
    is likewise throughput-only: the same ONNX graph on the CPU provider
    produces the same ranking. It is ignored when an explicit ``loader`` is
    given, since that loader owns session creation.

    Raises ``RuntimeError`` when the model is unavailable, so the caller can
    surface a message; that mirrors ``embed_images_dino``.
    """
    total = len(paths)
    if not paths:
        return {}

    # The model has to be identified before the cache can be consulted: a score
    # cached by a previous model is stale, and version-keying is how a retrained
    # head invalidates every session cache without hunting for the files. With
    # the default loader the version comes from the sidecar, so a fully cached
    # call never builds a session; an explicit loader owns its version.
    missing = RuntimeError(
        f"Realism scoring needs onnxruntime and the model at {model_paths()[0]} (export it from Bitcrush Engine)."
    )
    session = spec = None
    if loader is not None:
        session, spec = loader()
        if session is None or spec is None:
            raise missing
        model_version = spec.get("model_version")
    else:
        model_version = realism_model_version() if cache is not None else None
        if model_version is None:
            session, spec = load_realism_model(device=device)
            if session is None or spec is None:
                raise missing
            model_version = spec.get("model_version")

    scores: dict[str, float] = {}
    misses: list[str] = []
    for path in paths:
        cached = cache.peek_realism(path, model_version) if cache is not None else None
        if cached is None:
            misses.append(path)
        else:
            scores[path] = cached

    done = len(scores)
    if progress is not None:
        progress(done, total)
    if not misses or cache_only:
        return scores
    if session is None:
        session, spec = load_realism_model(device=device)
        if session is None or spec is None:
            raise missing
        model_version = spec.get("model_version")
    draws = max(1, int(spec.get("tta", {}).get("draws", 1)))
    positive_index = int(spec.get("positive_class_index", 1))

    try:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        # Preprocessing is decode + resize + a codec round-trip per draw, all of
        # which drop the GIL inside Pillow. On a CPU-only onnxruntime it is
        # roughly half the wall clock, so the thread pool matters: a group of
        # sixteen images otherwise stalls the grid for ~15s on first view, and a
        # whole-tree backfill runs at half the rate the GPU could sustain.
        workers = _preprocess_worker_count(workers)
        for draw in range(draws):
            # Group by tensor shape: the aspect buckets have different dims, so
            # a single stacked batch is only possible within one shape.
            by_shape: dict[tuple, list[tuple[str, np.ndarray]]] = {}

            def _prepare(path: str, _draw: int = draw):
                try:
                    return path, _tensor_for_draw(path, _draw, spec)
                except Exception:
                    return path, None

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for path, arr in pool.map(_prepare, misses):
                    if arr is None:
                        continue
                    by_shape.setdefault(arr.shape, []).append((path, arr))

            for items in by_shape.values():
                for start in range(0, len(items), batch_size):
                    chunk = items[start : start + batch_size]
                    stacked = np.stack([arr for _, arr in chunk])
                    logits = session.run(["logits"], {"pixels": stacked})[0]
                    probs = _softmax_positive(np.asarray(logits, dtype=np.float32), positive_index)
                    for (path, _), prob in zip(chunk, probs, strict=True):
                        totals[path] = totals.get(path, 0.0) + float(prob)
                        counts[path] = counts.get(path, 0) + 1

            if progress is not None:
                progress(done + int(len(totals) * (draw + 1) / draws), total)

        for path, accumulated in totals.items():
            value = accumulated / counts[path]
            scores[path] = value
            if cache is not None:
                cache.store_realism(path, value, model_version)
    finally:
        del session

    if progress is not None:
        progress(total, total)
    return scores
