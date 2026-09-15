"""Scalar image quality for DPO curation — Bitcrush Studio's "Simple" score.

A port of the model-free formula behind Studio's quality tool
(``apps/studio/src/backend/backend/quality_router.py``, Simple mode): a 0–100
blend of sharpness (variance of Laplacian, penalised when the detail is
noise), contrast (grey-level spread), exposure (distance from mid-grey) and,
optionally, compression (bits per pixel of the file on disk).

The blend is kept identical to Studio's so the number means the same thing in
both tools, with one deliberate difference: **compression is off by default.**
Studio compares photos in one folder, where bits-per-pixel says something. The
DPO tool compares a PNG generation against a WebP generation or a JPEG photo,
where it says only which codec was used. The remaining weights renormalise so
the score still spans 0–100.

Deterministic, CPU-only and cheap (one decode plus two 3x3 filters). It is
still cached when a scan cache is offered — keyed on :data:`QUALITY_VERSION`,
so a change to the formula invalidates old numbers — because the backfill's
promise is that a group costs nothing on first view, not merely no model.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEFAULT_WEIGHTS = {"sharpness": 0.50, "contrast": 0.25, "exposure": 0.15, "compression": 0.10}
#: Bump when the blend changes: cached totals are stamped with it.
QUALITY_VERSION = "simple/1"
QUALITY_CACHE_FIELD = "scalar_quality"


def quality_version(include_compression: bool = False) -> str:
    """The cache stamp for a ``total`` — the formula plus the one option that
    changes the number."""
    return f"{QUALITY_VERSION}@compression={int(bool(include_compression))}"


def _decode_gray(path: str):
    """(gray uint8 array, file size) or ``None``. Reads the bytes ourselves so
    non-ASCII Windows paths decode (``cv2.imread`` cannot open them)."""
    try:
        import cv2
    except ImportError:
        return None
    try:
        size = os.path.getsize(path)
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif channels == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = np.clip(gray.astype(np.float64) / (gray.max() or 1) * 255.0, 0, 255).astype(np.uint8)
    return gray, size


def _sharpness(raw_var: float, noise_diff: float, bpp: float) -> float:
    """Studio's ``score_simple``: Laplacian variance on a 0–100 scale, halved
    when most of that variance disappears under a 3x3 blur (it was noise, not
    detail) and halved again when a sharp-looking file is starved of bits."""
    score = min(100.0, math.sqrt(raw_var) * 15.0)
    noise_ratio = noise_diff / raw_var if raw_var > 0 else 0.0
    if noise_ratio > 0.6 and noise_diff > 50:
        score *= 0.5
    if raw_var > 100 and bpp < 0.8:
        score *= 0.5
    return min(100.0, score)


def scalar_quality(path: str, *, include_compression: bool = False, weights: dict | None = None) -> dict | None:
    """Component scores and their blend for one image, or ``None`` if it
    cannot be decoded. Keys: ``total, sharpness, contrast, exposure,
    compression, laplacian_var, brightness_mean, contrast_std, bpp``."""
    decoded = _decode_gray(path)
    if decoded is None:
        return None
    gray, file_size = decoded
    import cv2

    h, w = gray.shape[:2]
    total_pixels = int(h) * int(w)
    if total_pixels == 0:
        return None

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clean_var = float(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F).var())
    noise_diff = max(0.0, laplacian_var - clean_var)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    bpp = (file_size * 8) / total_pixels

    score_contrast = min(100.0, contrast * 2.5)
    score_exposure = max(0.0, 100.0 - abs(brightness - 128.0) * 0.8)
    if brightness < 5 or brightness > 250:
        score_exposure = 0.0
    score_compression = min(100.0, bpp * 66.0)
    score_sharpness = _sharpness(laplacian_var, noise_diff, bpp)

    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if not include_compression:
        w.pop("compression", None)
    components = {
        "sharpness": score_sharpness,
        "contrast": score_contrast,
        "exposure": score_exposure,
        "compression": score_compression,
    }
    weight_sum = sum(w.get(k, 0.0) for k in components)
    total = sum(components[k] * w.get(k, 0.0) for k in components) / weight_sum if weight_sum > 0 else 0.0
    total = min(100.0, total)

    if laplacian_var < 10.0:
        total = min(total, 15.0)
    if brightness < 2.0:
        total = 0.0
    if contrast < 1.0:
        total = 0.0

    return {
        "total": float(total),
        **components,
        "laplacian_var": laplacian_var,
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "bpp": float(bpp),
    }


def score_images_quality(
    paths: Sequence[str], *, include_compression: bool = False, workers: int | None = None, cache=None
) -> dict[str, float]:
    """``total`` per readable path. Threaded — OpenCV releases the GIL. With a
    ``cache`` (a ``DpoScanCache``) hits skip the decode and misses are written
    back, keyed on :func:`quality_version`."""
    if not paths:
        return {}
    version = quality_version(include_compression)
    scores: dict[str, float] = {}
    misses: list[str] = []
    for path in paths:
        cached = cache.peek_versioned(path, QUALITY_CACHE_FIELD, version) if cache is not None else None
        if isinstance(cached, (int, float)):
            scores[path] = float(cached)
        else:
            misses.append(path)
    if not misses:
        return scores
    workers = workers or max(1, min(16, (os.cpu_count() or 4) // 2))

    def _one(path: str):
        try:
            result = scalar_quality(path, include_compression=include_compression)
        except Exception:
            result = None
        return path, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, result in pool.map(_one, misses):
            if result is not None:
                scores[path] = result["total"]
                if cache is not None:
                    cache.store_versioned(path, QUALITY_CACHE_FIELD, float(result["total"]), version)
    return scores
