"""Apparent-age estimation for DPO prompt adherence — Bitcrush Engine's Age model.

SoReal! captions state an age ("a 34-year-old woman"), and a zero-shot
bi-encoder cannot check it beyond child/adult. Engine's Age group can: a
ConvNeXt-V2 base head over 102 ordinal classes (``__none__`` + 0..100) with
a validation QWK of 0.80. It is exported to ONNX with the serve-time prior
correction and softmax calibration baked into the classifier, so
``softmax(logits)`` here is exactly what Engine would compute, and shipped
under ``external/models/age/`` with a JSON sidecar that carries the
preprocessing recipe and class list (same contract as the realism head).

Decoding: the ordinal expected value over the real classes (the QWK-optimal
point estimate the trainer uses when the posterior is diffuse), reported with
``p_none`` — the mass the model puts on "no readable subject". Callers gate on
``p_none`` rather than on the estimate: a diffuse age posterior on a photo of
a cake is still a number, and only ``p_none`` says it is meaningless.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence

from modules.util.realism_detect_util import _crop_to_ratio, _nearest_bucket, _preload_cuda_dlls

import numpy as np
from PIL import Image

_MODEL_DIR = os.path.join("external", "models", "age")
_MODEL_NAME = "age_base"

AGE_DEVICES = ("auto", "cpu")
#: Above this much ``__none__`` mass the estimate is not trusted.
DEFAULT_NONE_GATE = 0.5


def model_paths() -> tuple[str, str]:
    base = os.path.join(_MODEL_DIR, _MODEL_NAME)
    return base + ".onnx", base + ".json"


def age_model_version() -> str | None:
    """The published model's version string from the sidecar, without loading
    a session — cache lookups key on it."""
    _onnx, sidecar = model_paths()
    try:
        with open(sidecar, encoding="utf-8") as handle:
            spec = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    version = spec.get("model_version") if isinstance(spec, dict) else None
    return version if isinstance(version, str) else None


def load_age_model(device: str = "auto"):
    """``(session, spec)`` or ``(None, None)`` when onnxruntime or the weights
    are missing — the caller degrades to "no ages" rather than crashing."""
    onnx_path, sidecar_path = model_paths()
    if not os.path.isfile(onnx_path) or not os.path.isfile(sidecar_path):
        return None, None
    try:
        import onnxruntime
    except ImportError:
        return None, None
    cpu_only = str(device).lower() == "cpu"
    if not cpu_only:
        _preload_cuda_dlls()
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            spec = json.load(handle)
        providers = ["CPUExecutionProvider"]
        if not cpu_only and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return onnxruntime.InferenceSession(onnx_path, providers=providers), spec
    except Exception:
        return None, None


def _tensor(path: str, spec: dict) -> np.ndarray:
    """Engine's serve-time geometry: centre-crop to the nearest bucket aspect,
    LANCZOS resize to the bucket, ImageNet-normalise, CHW."""
    with Image.open(path) as handle:
        img = handle.convert("RGB")
    bw, bh = _nearest_bucket(img.width, img.height, spec["aspect_ratio_buckets"])
    img = _crop_to_ratio(img, bw / bh).resize((bw, bh), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(spec["imagenet_mean"], dtype=np.float32)) / np.asarray(
        spec["imagenet_std"], dtype=np.float32
    )
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def class_ages(spec: dict) -> tuple[np.ndarray, int]:
    """``(ages per class index, none_index)``; ``__none__`` gets NaN."""
    names = spec.get("class_names") or []
    none_index = int(spec.get("none_index", -1))
    min_age = int(spec.get("min_age", 0))
    ages = np.full(len(names), np.nan, dtype=np.float64)
    real = 0
    for i, name in enumerate(names):
        if i == none_index:
            continue
        digits = "".join(ch for ch in str(name).split("-")[0] if ch.isdigit())
        ages[i] = int(digits) if digits else min_age + real
        real += 1
    return ages, none_index


def decode_ages(logits: np.ndarray, spec: dict) -> list[dict]:
    """Per row: ``{"age": expected age over real classes, "p_none": mass on
    __none__, "argmax_age": the mode's age or None when the mode is __none__}``."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    ages, none_index = class_ages(spec)
    real = ~np.isnan(ages)
    out: list[dict] = []
    for row in probs:
        mass = float(row[real].sum())
        ev = float((row[real] * ages[real]).sum() / mass) if mass > 0 else float("nan")
        p_none = float(row[none_index]) if none_index >= 0 else 0.0
        mode = int(row.argmax())
        out.append({"age": ev, "p_none": p_none, "argmax_age": None if mode == none_index else float(ages[mode])})
    return out


def estimate_ages(
    paths: Sequence[str],
    *,
    loader=None,
    cache=None,
    batch_size: int = 4,
    device: str = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict]:
    """Age estimate per readable path (see :func:`decode_ages` for the shape).

    Cache hits (size, mtime, model version) skip the model; unreadable files
    are skipped. Raises ``RuntimeError`` when the model is unavailable.
    """
    if not paths:
        return {}
    missing = RuntimeError(
        f"Age estimation needs onnxruntime and the model at {model_paths()[0]} (export it from Bitcrush Engine)."
    )
    # Same contract as ``score_images_realism``: with the default loader the
    # version comes from the sidecar, so a fully cached call never builds a
    # session; only a miss loads the model.
    session = spec = None
    if loader is not None:
        session, spec = loader()
        if session is None or spec is None:
            raise missing
        model_version = spec.get("model_version")
    else:
        model_version = age_model_version() if cache is not None else None
        if model_version is None:
            session, spec = load_age_model(device=device)
            if session is None or spec is None:
                raise missing
            model_version = spec.get("model_version")

    results: dict[str, dict] = {}
    misses: list[str] = []
    for path in paths:
        cached = cache.peek_versioned(path, "age_estimate", model_version) if cache is not None else None
        if isinstance(cached, dict) and "age" in cached:
            results[path] = dict(cached)
        else:
            misses.append(path)
    total = len(paths)
    if progress is not None:
        progress(len(results), total)
    if not misses:
        return results
    if session is None:
        session, spec = load_age_model(device=device)
        if session is None or spec is None:
            raise missing
        model_version = spec.get("model_version")

    try:
        by_shape: dict[tuple, list[tuple[str, np.ndarray]]] = {}
        for path in misses:
            try:
                arr = _tensor(path, spec)
            except Exception:
                continue
            by_shape.setdefault(arr.shape, []).append((path, arr))
        done = len(results)
        for items in by_shape.values():
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                stacked = np.stack([arr for _, arr in chunk])
                logits = session.run(["logits"], {"pixels": stacked})[0]
                for (path, _), decoded in zip(chunk, decode_ages(logits, spec), strict=True):
                    results[path] = decoded
                    if cache is not None:
                        cache.store_versioned(path, "age_estimate", decoded, model_version)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    finally:
        del session
    if progress is not None:
        progress(total, total)
    return results
