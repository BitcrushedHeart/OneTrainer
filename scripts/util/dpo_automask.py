"""Propose, review and apply soft ``-masklabel.png`` masks for DPO pairs whose
chosen image is an in-place edit of the rejected image.

Without a mask a masked DPO run weighs the whole image, which dilutes a small
local fix. This tool finds where the pair really differs and turns that into
a generous, feathered mask: a multiscale CIELAB difference map (blurred at
several sigmas, each scale normalised by its own robust background level so
VAE round-trip noise and slight colour shifts do not light up the image),
thresholded, small blobs dropped, closed, dilated by a fraction of the short
edge and feathered with a smooth distance falloff. Masks are soft 0..1 and
written as 8-bit greyscale PNGs, exactly what the loader reads.

Pairing mirrors the trainer (FilterDPOChosenPaths + DeriveDPORejectedPath via
modules.util.dpo_pattern_util): the chosen pattern's ``{}`` stem is looked up
under the rejected pattern with any supported image extension, preferring the
chosen image's own. The mask path is the image path with its extension
replaced by ``-masklabel.png`` (ModifyPath), on BOTH halves.

Everything the tool writes before ``apply`` lives in ``<dataset>/.automask``,
a dot-folder the loader skips:

    proposals/<stem>-masklabel.png   proposed masks
    proposals.json                   per pair: flags, coverage, parameters
    decisions.json                   accept / reject per pair (crash-safe)
    previews/                        downscaled review previews
    applied.json, backup/            apply log and overwritten masks

Usage (from the repo root):

    venv/Scripts/python.exe scripts/util/dpo_automask.py propose "<dataset>" [--include-covered 0.95] [--workers 8]
    venv/Scripts/python.exe scripts/util/dpo_automask.py review  "<dataset>" [--port 8766]
    venv/Scripts/python.exe scripts/util/dpo_automask.py apply   "<dataset>" [--overwrite]
    venv/Scripts/python.exe scripts/util/dpo_automask.py undo    "<dataset>" [--run N]

``propose`` handles every pair whose chosen image has no mask (plus, with
``--include-covered X``, pairs whose existing mask covers more than X of the
image). Flags per pair: ``ok``, ``none`` (no meaningful difference; nothing to
apply), ``global`` (the difference covers most of the image; a full mask is
proposed), ``size_mismatch`` (rejected resized to the chosen size for the
diff), ``had_mask`` (an existing mask would be replaced; apply needs
``--overwrite``). ``review`` serves a local page on 127.0.0.1 with accept /
reject, keyboard shortcuts and parameter sliders. ``apply`` writes accepted
masks to both halves and logs every write so ``undo`` can reverse it. Only
aggregate numbers are printed; file names never are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from modules.util import path_util  # noqa: E402
from modules.util.dpo_pattern_util import build_rejected_index, match_chosen, resolve_rejected  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

MASK_POSTFIX = "-masklabel"
EXCLUDE_POSTFIX = ("-masklabel", "-condlabel")
AUTOMASK_DIR = ".automask"
PREVIEW_EDGE = 768
# Preview kinds the review server serves, by pair index only. The *_full kinds
# are native resolution for the full-screen viewer and are never cached to disk.
FULL_KINDS = ("chosen_full", "rejected_full", "overlay_full", "overlay_rejected_full", "heat_full")
PREVIEW_KINDS = ("chosen", "rejected", "thumb", "overlay", "heat") + FULL_KINDS
STATIC_KINDS = ("chosen", "rejected", "thumb", "chosen_full", "rejected_full")  # browser-cacheable
THUMB_EDGE = 256
DEFAULT_CHOSEN_PATTERN = "chosen/{}"
DEFAULT_REJECTED_PATTERN = "rejected/{}"


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------


@dataclass
class MaskParams:
    """Everything that shapes a proposed mask. Fractions are of the chosen
    image's short edge (close / dilate / feather) or of its area
    (min_component); ``threshold`` is in robust standard deviations of each
    scale's background difference."""

    threshold: float = 6.0
    abs_floor: float = 0.25
    chroma_weight: float = 0.5
    scales: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    min_component: float = 0.0002
    close: float = 0.015
    dilate: float = 0.03
    feather: float = 0.02
    global_cover: float = 0.6
    global_delta: float = 4.0

    # Slider-exposed fields, with (min, max, step) for the review page.
    RANGES = {
        "threshold": (1.0, 30.0, 0.5),
        "abs_floor": (0.0, 3.0, 0.05),
        "min_component": (0.0, 0.005, 0.0001),
        "close": (0.0, 0.06, 0.0025),
        "dilate": (0.0, 0.12, 0.0025),
        "feather": (0.0, 0.12, 0.0025),
        "global_cover": (0.1, 1.0, 0.05),
        "global_delta": (0.5, 20.0, 0.5),
    }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scales"] = list(self.scales)
        return d

    @classmethod
    def from_dict(cls, data: dict | None, base: MaskParams | None = None) -> MaskParams:
        params = MaskParams() if base is None else MaskParams(**base.to_dict())
        for field in fields(cls):
            if data and field.name in data and data[field.name] is not None:
                value = data[field.name]
                if field.name == "scales":
                    value = tuple(float(v) for v in value)
                else:
                    value = float(value)
                    low, high, _ = cls.RANGES.get(field.name, (-1e9, 1e9, 0))
                    value = min(max(value, low), high)
                setattr(params, field.name, value)
        return params

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:10]


def add_param_arguments(parser: argparse.ArgumentParser):
    defaults = MaskParams()
    for field in fields(MaskParams):
        if field.name == "scales":
            parser.add_argument("--scales", type=float, nargs="+", default=None, help="blur sigmas in px")
        else:
            parser.add_argument(
                f"--{field.name.replace('_', '-')}",
                type=float,
                default=None,
                help=f"default {getattr(defaults, field.name)}",
            )


def params_from_args(args) -> MaskParams:
    data = {field.name: getattr(args, field.name, None) for field in fields(MaskParams)}
    return MaskParams.from_dict(data)


# ----------------------------------------------------------------------------
# Pair listing (mirrors the trainer's DPO pattern pairing)
# ----------------------------------------------------------------------------


@dataclass
class Pair:
    key: str  # the '{}' stem, e.g. 'train/foo'
    chosen: str  # absolute path
    rejected: str  # absolute path


def mask_path_for(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + MASK_POSTFIX + ".png"


def _collect_images(root: str) -> list[str]:
    """Every supported image under root, dot-directories skipped, mask / cond
    sidecars excluded: what mgds CollectPaths hands the DPO filter."""
    found = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if not path_util.is_supported_image_extension(ext):
                continue
            if stem.endswith(EXCLUDE_POSTFIX):
                continue
            found.append(os.path.join(current, name))
    return found


def list_pairs(dataset: str, chosen_pattern: str, rejected_pattern: str) -> tuple[list[Pair], int]:
    """(pairs sorted by stem, number of chosen images without a rejected match)."""
    dataset = os.path.abspath(dataset)
    index = build_rejected_index(dataset, True)
    pairs = []
    missing = 0
    for image_path in _collect_images(dataset):
        stem = match_chosen(chosen_pattern, dataset, image_path)
        if stem is None:
            continue
        try:
            rejected = resolve_rejected(rejected_pattern, dataset, stem, os.path.splitext(image_path)[1], index)
        except FileNotFoundError:
            missing += 1
            continue
        pairs.append(Pair(key=stem.replace("\\", "/"), chosen=image_path, rejected=rejected))
    pairs.sort(key=lambda p: p.key)
    return pairs, missing


# ----------------------------------------------------------------------------
# Image maths
# ----------------------------------------------------------------------------


def load_rgb(path: str) -> np.ndarray:
    """float32 HxWx3 in 0..1, EXIF-oriented like the loader."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def write_mask(path: str, soft: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.rint(np.clip(soft, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


@dataclass
class Difference:
    score: np.ndarray  # float32 HxW robust z-score, max over scales
    size_mismatch: bool
    background: float  # median blurred delta-E at the coarsest scale


def difference_score(chosen: np.ndarray, rejected: np.ndarray, params: MaskParams) -> Difference:
    """Multiscale robust difference score, float32 HxW at the chosen size.
    Each scale's CIELAB difference magnitude is blurred at that scale and
    z-scored against its own median / MAD background (never below
    ``abs_floor`` delta-E); the result is the max over scales. The coarsest
    scale's median is kept as ``background``: the normalisation cannot see a
    change that covers the whole image, so that absolute level is what flags
    it as global."""
    size_mismatch = chosen.shape[:2] != rejected.shape[:2]
    if size_mismatch:
        rejected = cv2.resize(rejected, (chosen.shape[1], chosen.shape[0]), interpolation=cv2.INTER_AREA)
    lab_c = cv2.cvtColor(chosen, cv2.COLOR_RGB2LAB)
    lab_r = cv2.cvtColor(rejected, cv2.COLOR_RGB2LAB)
    diff = lab_c - lab_r
    weights = np.array([1.0, params.chroma_weight, params.chroma_weight], dtype=np.float32)
    score = None
    background = 0.0
    for sigma in params.scales:
        blurred = cv2.GaussianBlur(diff, (0, 0), sigma) if sigma > 0 else diff
        magnitude = np.sqrt(np.sum(np.square(blurred) * weights, axis=2))
        sample = magnitude[::2, ::2]
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        denominator = max(1.4826 * mad, params.abs_floor, 1e-6)
        z = (magnitude - median) / denominator
        score = z if score is None else np.maximum(score, z)
        background = median
    return Difference(score.astype(np.float32), size_mismatch, background)


def mask_from_score(
    score: np.ndarray, params: MaskParams, background: float = 0.0
) -> tuple[np.ndarray | None, float, float, list[str]]:
    """(soft mask or None, raw coverage, final coverage, flags). None with
    flag ``none`` when nothing survives the threshold and blob filter; a full
    mask with flag ``global`` when the processed mask covers more than
    ``global_cover`` of the image or the whole image differs by more than
    ``global_delta`` delta-E (``background``)."""
    height, width = score.shape
    short = min(height, width)
    area = height * width
    if background > params.global_delta:
        return np.ones((height, width), dtype=np.float32), 1.0, 1.0, ["global"]
    binary = (score > params.threshold).astype(np.uint8)

    min_pixels = int(round(params.min_component * area))
    if min_pixels > 1 and binary.any():
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_pixels
        keep[0] = False
        binary = keep[labels].astype(np.uint8)

    raw_coverage = float(binary.mean())
    if raw_coverage == 0.0:
        return None, 0.0, 0.0, ["none"]

    close_radius = int(round(params.close * short))
    if close_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    dilate_radius = int(round(params.dilate * short))
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
        binary = cv2.dilate(binary, kernel)

    feather_px = params.feather * short
    if feather_px >= 1.0 and not binary.all():
        distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        t = np.clip(1.0 - distance / feather_px, 0.0, 1.0)
        soft = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    else:
        soft = binary.astype(np.float32)

    final_coverage = float(soft.mean())
    if final_coverage > params.global_cover:
        return np.ones_like(soft), raw_coverage, final_coverage, ["global"]
    return soft, raw_coverage, final_coverage, ["ok"]


def propose_pair(chosen_path: str, rejected_path: str, params: MaskParams) -> tuple[np.ndarray | None, dict]:
    chosen = load_rgb(chosen_path)
    rejected = load_rgb(rejected_path)
    difference = difference_score(chosen, rejected, params)
    soft, raw_coverage, final_coverage, flags = mask_from_score(difference.score, params, difference.background)
    if difference.size_mismatch:
        flags.append("size_mismatch")
    info = {
        "flags": flags,
        "raw_coverage": round(raw_coverage, 5),
        "final_coverage": round(final_coverage, 5),
        "size": [int(chosen.shape[1]), int(chosen.shape[0])],
        "score_max": round(float(difference.score.max()), 2),
        "background_delta": round(difference.background, 3),
    }
    return soft, info


# ----------------------------------------------------------------------------
# Propose
# ----------------------------------------------------------------------------


def automask_dir(dataset: str) -> str:
    return os.path.join(os.path.abspath(dataset), AUTOMASK_DIR)


def proposal_mask_path(dataset: str, key: str) -> str:
    return os.path.join(automask_dir(dataset), "proposals", *key.split("/")) + MASK_POSTFIX + ".png"


def _atomic_write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _read_json(path: str, default):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_proposals(dataset: str) -> dict:
    return _read_json(os.path.join(automask_dir(dataset), "proposals.json"), {"pairs": {}})


def save_proposals(dataset: str, proposals: dict):
    _atomic_write_json(os.path.join(automask_dir(dataset), "proposals.json"), proposals)


def load_decisions(dataset: str) -> dict:
    return _read_json(os.path.join(automask_dir(dataset), "decisions.json"), {"decisions": {}})


def save_decisions(dataset: str, decisions: dict):
    _atomic_write_json(os.path.join(automask_dir(dataset), "decisions.json"), decisions)


def _propose_worker(job: tuple) -> tuple[str, dict]:
    key, chosen, rejected, params_dict, out_path = job
    params = MaskParams.from_dict(params_dict)
    soft, info = propose_pair(chosen, rejected, params)
    if soft is not None:
        write_mask(out_path, soft)
    elif os.path.isfile(out_path):
        os.remove(out_path)
    info["mask"] = soft is not None
    return key, info


def select_pairs(
    dataset: str, pairs: list[Pair], include_covered: float | None
) -> tuple[list[Pair], dict[str, list[str]]]:
    """Pairs to propose for: no chosen mask, or (with include_covered) an
    existing mask covering more than that fraction. Returns the pairs and
    extra flags per key."""
    selected = []
    extra_flags: dict[str, list[str]] = {}
    for pair in pairs:
        existing = mask_path_for(pair.chosen)
        if not os.path.isfile(existing):
            selected.append(pair)
            continue
        if include_covered is not None and load_mask(existing).mean() > include_covered:
            selected.append(pair)
            extra_flags[pair.key] = ["had_mask"]
    return selected, extra_flags


def run_propose(
    dataset: str,
    params: MaskParams,
    chosen_pattern: str = DEFAULT_CHOSEN_PATTERN,
    rejected_pattern: str = DEFAULT_REJECTED_PATTERN,
    include_covered: float | None = None,
    workers: int = 1,
    progress=None,
) -> dict:
    """Computes and stores proposals for every selected pair, replacing any
    earlier proposals.json. Returns aggregate stats (no names)."""
    dataset = os.path.abspath(dataset)
    pairs, missing = list_pairs(dataset, chosen_pattern, rejected_pattern)
    selected, extra_flags = select_pairs(dataset, pairs, include_covered)

    jobs = [(p.key, p.chosen, p.rejected, params.to_dict(), proposal_mask_path(dataset, p.key)) for p in selected]
    results: dict[str, dict] = {}
    started = time.time()
    if workers > 1 and jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, (key, info) in enumerate(pool.map(_propose_worker, jobs, chunksize=4), start=1):
                results[key] = info
                if progress:
                    progress(done, len(jobs))
    else:
        for done, job in enumerate(jobs, start=1):
            key, info = _propose_worker(job)
            results[key] = info
            if progress:
                progress(done, len(jobs))

    entries = {}
    for pair in selected:
        info = results[pair.key]
        flags = info["flags"] + extra_flags.get(pair.key, [])
        entries[pair.key] = {
            "chosen": os.path.relpath(pair.chosen, dataset).replace("\\", "/"),
            "rejected": os.path.relpath(pair.rejected, dataset).replace("\\", "/"),
            "flags": flags,
            "raw_coverage": info["raw_coverage"],
            "final_coverage": info["final_coverage"],
            "size": info["size"],
            "score_max": info["score_max"],
            "background_delta": info["background_delta"],
            "mask": info["mask"],
            "params_hash": params.digest(),
        }

    proposals = {
        "dataset": dataset,
        "chosen_pattern": chosen_pattern,
        "rejected_pattern": rejected_pattern,
        "params": params.to_dict(),
        "params_hash": params.digest(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pairs": entries,
    }
    save_proposals(dataset, proposals)
    return summarize_proposals(proposals, total_pairs=len(pairs), missing=missing, seconds=time.time() - started)


def summarize_proposals(
    proposals: dict, total_pairs: int | None = None, missing: int = 0, seconds: float = 0.0
) -> dict:
    entries = proposals["pairs"]
    flag_counts = Counter(flag for entry in entries.values() for flag in entry["flags"])
    global_delta = MaskParams.from_dict(proposals.get("params")).global_delta
    global_by_delta = sum(
        1 for e in entries.values() if "global" in e["flags"] and e.get("background_delta", 0.0) > global_delta
    )
    ok_final = [e["final_coverage"] for e in entries.values() if "ok" in e["flags"]]
    ok_raw = [e["raw_coverage"] for e in entries.values() if "ok" in e["flags"]]
    percentiles = [5, 25, 50, 75, 95]

    def pct(values):
        if not values:
            return {}
        return {
            str(p): round(float(v) * 100, 2)
            for p, v in zip(percentiles, np.percentile(values, percentiles), strict=True)
        }

    return {
        "pairs_total": total_pairs if total_pairs is not None else None,
        "chosen_without_rejected": missing,
        "proposed": len(entries),
        "flags": dict(flag_counts),
        "global_by_whole_image_delta": global_by_delta,
        "global_by_coverage": flag_counts["global"] - global_by_delta,
        "ok_final_coverage_pct": pct(ok_final),
        "ok_raw_coverage_pct": pct(ok_raw),
        "seconds": round(seconds, 1),
    }


def print_stats(stats: dict):
    for name, value in stats.items():
        print(f"  {name}: {value}")


# ----------------------------------------------------------------------------
# Apply / undo
# ----------------------------------------------------------------------------


def _applied_log_path(dataset: str) -> str:
    return os.path.join(automask_dir(dataset), "applied.json")


def run_apply(dataset: str, overwrite: bool = False) -> dict:
    """Writes every accepted proposal mask to both halves of its pair. Existing
    masks are skipped unless overwrite, in which case they are backed up under
    .automask/backup/<run>/ first. Every write goes into applied.json."""
    dataset = os.path.abspath(dataset)
    proposals = load_proposals(dataset)
    decisions = load_decisions(dataset)["decisions"]
    log = _read_json(_applied_log_path(dataset), {"runs": []})
    run_id = time.strftime("%Y%m%d-%H%M%S")
    backup_root = os.path.join(automask_dir(dataset), "backup", run_id)

    run = {
        "id": run_id,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overwrite": overwrite,
        "entries": [],
        "undone": False,
    }
    counts = Counter()
    try:
        for key, decision in sorted(decisions.items()):
            if decision.get("decision") != "accept":
                continue
            entry = proposals["pairs"].get(key)
            if entry is None or not entry.get("mask"):
                counts["accepted_without_mask"] += 1
                continue
            source = proposal_mask_path(dataset, key)
            if not os.path.isfile(source):
                counts["proposal_file_missing"] += 1
                continue
            targets = [
                ("chosen", mask_path_for(os.path.join(dataset, entry["chosen"]))),
                ("rejected", mask_path_for(os.path.join(dataset, entry["rejected"]))),
            ]
            if any(os.path.isfile(path) for _, path in targets) and not overwrite:
                counts["skipped_existing"] += 1
                continue
            for side, path in targets:
                backup = None
                if os.path.isfile(path):
                    backup = os.path.join(backup_root, side, os.path.relpath(path, dataset))
                    os.makedirs(os.path.dirname(backup), exist_ok=True)
                    shutil.copy2(path, backup)
                    counts["overwritten"] += 1
                shutil.copyfile(source, path)
                run["entries"].append({"key": key, "side": side, "path": path, "backup": backup})
            counts["pairs_written"] += 1
    finally:
        if run["entries"]:
            log["runs"].append(run)
            _atomic_write_json(_applied_log_path(dataset), log)
    counts["run_id"] = run_id if run["entries"] else None
    return dict(counts)


def run_undo(dataset: str, run_index: int | None = None) -> dict:
    """Reverses an apply run (the latest not yet undone by default): removes
    the written masks and restores any backups."""
    dataset = os.path.abspath(dataset)
    log = _read_json(_applied_log_path(dataset), {"runs": []})
    candidates = [i for i, run in enumerate(log["runs"]) if not run.get("undone")]
    if run_index is None:
        if not candidates:
            return {"undone_run": None, "removed": 0, "restored": 0}
        run_index = candidates[-1]
    run = log["runs"][run_index]
    if run.get("undone"):
        raise ValueError(f"run {run_index} is already undone")
    counts = Counter()
    for entry in reversed(run["entries"]):
        path = entry["path"]
        if entry.get("backup") and os.path.isfile(entry["backup"]):
            shutil.copy2(entry["backup"], path)
            counts["restored"] += 1
        elif os.path.isfile(path):
            os.remove(path)
            counts["removed"] += 1
    run["undone"] = True
    run["undone_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(_applied_log_path(dataset), log)
    return {"undone_run": run["id"], **counts}


# ----------------------------------------------------------------------------
# Review server
# ----------------------------------------------------------------------------


def _downscale(image: np.ndarray, edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = edge / max(height, width)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )


def _encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    bgr = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return data.tobytes()


def render_overlay(chosen_small: np.ndarray, soft: np.ndarray | None) -> np.ndarray:
    """Chosen image tinted red by the soft mask (the feather shows as a
    fading tint) with the mask's 50% contour drawn in yellow."""
    if soft is None:
        return chosen_small
    mask = cv2.resize(soft, (chosen_small.shape[1], chosen_small.shape[0]), interpolation=cv2.INTER_AREA)
    tint = np.array([1.0, 0.12, 0.12], dtype=np.float32)
    alpha = (0.6 * mask)[..., None]
    out = chosen_small * (1 - alpha) + tint * alpha
    contours, _ = cv2.findContours((mask >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.ascontiguousarray(out)
    cv2.drawContours(out, contours, -1, (1.0, 0.9, 0.1), 1)
    return out


def render_heat(score: np.ndarray, threshold: float, edge: int) -> np.ndarray:
    small = _downscale(score, edge)
    normalised = np.clip(small / max(2.0 * threshold, 1e-6), 0, 1)
    coloured = cv2.applyColorMap((normalised * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class ReviewState:
    """Everything the review handler touches, behind one lock. Pairs are
    addressed by their index in the sorted proposal order only, so the server
    never opens a path a client names."""

    def __init__(self, dataset: str, workers: int = 1):
        self.dataset = os.path.abspath(dataset)
        self.workers = workers
        self.lock = threading.RLock()
        self.proposals = load_proposals(self.dataset)
        self.decisions = load_decisions(self.dataset)
        self.keys = sorted(self.proposals["pairs"])
        self.default_params = MaskParams.from_dict(self.proposals.get("params"))
        self._scores: OrderedDict[int, tuple[Difference, np.ndarray]] = OrderedDict()
        self._live: OrderedDict[tuple[int, str], tuple[np.ndarray | None, dict]] = OrderedDict()
        self.rerun = {"running": False, "done": 0, "total": 0, "error": None, "stats": None}
        os.makedirs(os.path.join(automask_dir(self.dataset), "previews"), exist_ok=True)

    # -- pair access -----------------------------------------------------

    def entry(self, index: int) -> dict | None:
        if not isinstance(index, int) or index < 0 or index >= len(self.keys):
            return None
        return self.proposals["pairs"][self.keys[index]]

    def _paths(self, index: int) -> tuple[str, str]:
        entry = self.entry(index)
        return os.path.join(self.dataset, entry["chosen"]), os.path.join(self.dataset, entry["rejected"])

    def _preview_file(self, index: int, name: str) -> str:
        safe = self.keys[index].replace("/", "__")
        return os.path.join(automask_dir(self.dataset), "previews", f"{safe}-{name}.jpg")

    def _score(self, index: int, params: MaskParams) -> tuple[Difference, np.ndarray]:
        """(difference, chosen preview); the difference depends only on the
        diff parameters, so the cache ignores the morphology ones."""
        with self.lock:
            if index in self._scores:
                self._scores.move_to_end(index)
                return self._scores[index]
            chosen_path, rejected_path = self._paths(index)
            chosen = load_rgb(chosen_path)
            rejected = load_rgb(rejected_path)
            value = (difference_score(chosen, rejected, params), _downscale(chosen, PREVIEW_EDGE))
            self._scores[index] = value
            while len(self._scores) > 6:
                self._scores.popitem(last=False)
            return value

    def live(self, index: int, params: MaskParams) -> tuple[np.ndarray | None, dict]:
        """Mask + info for a pair under the given parameters."""
        cache_key = (index, params.digest())
        with self.lock:
            if cache_key in self._live:
                self._live.move_to_end(cache_key)
                return self._live[cache_key]
            difference, _ = self._score(index, params)
            soft, raw, final, flags = mask_from_score(difference.score, params, difference.background)
            if difference.size_mismatch:
                flags.append("size_mismatch")
            info = {
                "flags": flags,
                "raw_coverage": round(raw, 5),
                "final_coverage": round(final, 5),
                "background_delta": round(difference.background, 3),
            }
            self._live[cache_key] = (soft, info)
            while len(self._live) > 8:
                self._live.popitem(last=False)
            return soft, info

    # -- previews --------------------------------------------------------

    def preview(self, index: int, kind: str, params: MaskParams) -> bytes | None:
        if self.entry(index) is None:
            return None
        if kind in ("chosen", "rejected"):
            path = self._preview_file(index, kind)
            if not os.path.isfile(path):
                image = load_rgb(self._paths(index)[0 if kind == "chosen" else 1])
                with open(path, "wb") as f:
                    f.write(_encode_jpeg(_downscale(image, PREVIEW_EDGE)))
            with open(path, "rb") as f:
                return f.read()
        if kind == "thumb":
            entry = self.entry(index)
            path = self._preview_file(index, f"thumb-{entry['params_hash']}")
            if not os.path.isfile(path):
                chosen = _downscale(load_rgb(self._paths(index)[0]), THUMB_EDGE)
                soft = None
                if entry.get("mask"):
                    soft = load_mask(proposal_mask_path(self.dataset, self.keys[index]))
                with open(path, "wb") as f:
                    f.write(_encode_jpeg(render_overlay(chosen, soft), 80))
            with open(path, "rb") as f:
                return f.read()
        if kind == "overlay":
            soft, _ = self.live(index, params)
            _, chosen_small = self._score(index, params)
            return _encode_jpeg(render_overlay(chosen_small, soft))
        if kind == "heat":
            difference, _ = self._score(index, params)
            return _encode_jpeg(render_heat(difference.score, params.threshold, PREVIEW_EDGE))
        if kind in FULL_KINDS:
            # Native resolution for the full-screen viewer: rendered per request,
            # never written to disk (the score / mask caches above are reused).
            if kind == "heat_full":
                difference, _ = self._score(index, params)
                return _encode_jpeg(render_heat(difference.score, params.threshold, max(difference.score.shape)), 92)
            side = 1 if kind in ("rejected_full", "overlay_rejected_full") else 0
            image = load_rgb(self._paths(index)[side])
            if kind in ("overlay_full", "overlay_rejected_full"):
                soft, _ = self.live(index, params)
                image = render_overlay(image, soft)
            return _encode_jpeg(image, 92)
        return None

    # -- decisions -------------------------------------------------------

    def decide(self, index: int, decision: str | None, params: MaskParams) -> dict:
        """Records accept / reject (None clears). Accepting under parameters
        that differ from the stored proposal recomputes and rewrites that
        pair's proposal first, so apply uses what the reviewer saw."""
        with self.lock:
            key = self.keys[index]
            entry = self.proposals["pairs"][key]
            if decision == "accept" and params.digest() != entry["params_hash"]:
                self.restore_pair(index, params)
            if decision is None:
                self.decisions["decisions"].pop(key, None)
            else:
                self.decisions["decisions"][key] = {
                    "decision": decision,
                    "params_hash": entry["params_hash"],
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            save_decisions(self.dataset, self.decisions)
            return self.summary(index)

    def restore_pair(self, index: int, params: MaskParams):
        """Rewrites one pair's stored proposal under the given parameters."""
        with self.lock:
            key = self.keys[index]
            entry = self.proposals["pairs"][key]
            soft, info = self.live(index, params)
            out_path = proposal_mask_path(self.dataset, key)
            if soft is not None:
                write_mask(out_path, soft)
            elif os.path.isfile(out_path):
                os.remove(out_path)
            extra = [f for f in entry["flags"] if f == "had_mask"]
            entry.update(
                flags=info["flags"] + extra,
                raw_coverage=info["raw_coverage"],
                final_coverage=info["final_coverage"],
                mask=soft is not None,
                params_hash=params.digest(),
                params=params.to_dict(),
            )
            save_proposals(self.dataset, self.proposals)

    def summary(self, index: int) -> dict:
        key = self.keys[index]
        entry = self.proposals["pairs"][key]
        decision = self.decisions["decisions"].get(key)
        return {
            "index": index,
            "key": key,
            "flags": entry["flags"],
            "raw_coverage": entry["raw_coverage"],
            "final_coverage": entry["final_coverage"],
            "size": entry.get("size"),
            "decision": decision["decision"] if decision else None,
            "stale": bool(decision and decision.get("params_hash") != entry["params_hash"]),
            "params": entry.get("params") or self.proposals.get("params"),
        }

    def counts(self) -> dict:
        decided = Counter(
            d["decision"] for d in self.decisions["decisions"].values() if d["decision"] in ("accept", "reject")
        )
        return {"total": len(self.keys), "accepted": decided["accept"], "rejected": decided["reject"]}

    def listing(self, flag_filter: str) -> list[dict]:
        rows = []
        for index in range(len(self.keys)):
            row = self.summary(index)
            if (
                flag_filter in ("", "all")
                or flag_filter == "undecided"
                and row["decision"] is None
                or flag_filter in ("accept", "reject")
                and row["decision"] == flag_filter
                or flag_filter == "stale"
                and row["stale"]
                or flag_filter in row["flags"]
            ):
                rows.append(row)
        return rows

    # -- rerun -----------------------------------------------------------

    def start_rerun(self, params: MaskParams) -> bool:
        with self.lock:
            if self.rerun["running"]:
                return False
            self.rerun = {"running": True, "done": 0, "total": 0, "error": None, "stats": None}
        threading.Thread(target=self._rerun, args=(params,), daemon=True).start()
        return True

    def _rerun(self, params: MaskParams):
        def progress(done, total):
            self.rerun["done"], self.rerun["total"] = done, total

        try:
            include_covered = self.proposals.get("include_covered")
            stats = run_propose(
                self.dataset,
                params,
                self.proposals.get("chosen_pattern", DEFAULT_CHOSEN_PATTERN),
                self.proposals.get("rejected_pattern", DEFAULT_REJECTED_PATTERN),
                include_covered,
                self.workers,
                progress,
            )
            with self.lock:
                self.proposals = load_proposals(self.dataset)
                self.proposals["include_covered"] = include_covered
                self.keys = sorted(self.proposals["pairs"])
                self.default_params = params
                self._scores.clear()
                self._live.clear()
                self.rerun["stats"] = stats
        except Exception as e:
            self.rerun["error"] = str(e)
        finally:
            self.rerun["running"] = False


def _params_from_query(query: dict, base: MaskParams) -> MaskParams:
    data = {}
    for name in MaskParams.RANGES:
        if name in query:
            try:
                data[name] = float(query[name][0])
            except ValueError:
                continue
    return MaskParams.from_dict(data, base)


class ReviewHandler(BaseHTTPRequestHandler):
    state: ReviewState = None  # set by make_server

    def log_message(self, format, *args):
        pass

    def _send(self, status: int, body: bytes, content_type: str, cache: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200):
        self._send(status, json.dumps(data).encode(), "application/json")

    def _not_found(self):
        self._send(404, b"not found", "text/plain")

    def do_GET(self):
        url = urlsplit(self.path)
        parts = [p for p in url.path.split("/") if p]
        query = parse_qs(url.query)
        state = self.state
        if not parts:
            self._send(200, REVIEW_HTML.encode(), "text/html; charset=utf-8")
            return
        if parts[0] == "api":
            if parts[1:] == ["state"]:
                self._json(
                    {
                        "counts": state.counts(),
                        "params": state.default_params.to_dict(),
                        "ranges": MaskParams.RANGES,
                        "rerun": state.rerun,
                        "dataset": os.path.basename(state.dataset),
                    }
                )
                return
            if parts[1:] == ["pairs"]:
                self._json(state.listing(query.get("filter", [""])[0]))
                return
            if len(parts) == 3 and parts[1] == "pair" and parts[2].isdigit():
                index = int(parts[2])
                if state.entry(index) is None:
                    self._not_found()
                    return
                params = _params_from_query(query, state.default_params)
                _, live = state.live(index, params)
                self._json({**state.summary(index), "live": live, "live_hash": params.digest()})
                return
            self._not_found()
            return
        if len(parts) == 3 and parts[0] == "preview" and parts[1].isdigit():
            kind = parts[2].rsplit(".", 1)[0]
            if kind not in PREVIEW_KINDS:
                self._not_found()
                return
            params = _params_from_query(query, state.default_params)
            body = state.preview(int(parts[1]), kind, params)
            if body is None:
                self._not_found()
                return
            self._send(200, body, "image/jpeg", cache=kind in STATIC_KINDS)
            return
        self._not_found()

    def do_POST(self):
        url = urlsplit(self.path)
        parts = [p for p in url.path.split("/") if p]
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json({"error": "bad json"}, 400)
            return
        state = self.state
        if parts == ["api", "decision"]:
            index = payload.get("index")
            decision = payload.get("decision")
            if state.entry(index) is None or decision not in ("accept", "reject", None):
                self._not_found()
                return
            params = MaskParams.from_dict(payload.get("params"), state.default_params)
            self._json(state.decide(index, decision, params))
            return
        if parts == ["api", "rerun"]:
            params = MaskParams.from_dict(payload.get("params"), state.default_params)
            self._json({"started": state.start_rerun(params)})
            return
        self._not_found()


def make_server(dataset: str, port: int = 8766, workers: int = 1) -> ThreadingHTTPServer:
    state = ReviewState(dataset, workers)
    handler = type("BoundReviewHandler", (ReviewHandler,), {"state": state})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


REVIEW_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>DPO automask review</title>
<style>
body{margin:0;font:13px system-ui,sans-serif;background:#15171a;color:#ddd;display:grid;grid-template-columns:1fr 300px;grid-template-rows:auto 1fr;height:100vh}
header{grid-column:1/3;display:flex;gap:14px;align-items:center;padding:6px 10px;background:#22252a;flex-wrap:wrap}
header b{color:#fff}
main{overflow:auto;padding:8px}
aside{overflow:auto;background:#1b1d21;padding:8px;border-left:1px solid #333}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.grid figure{margin:0;background:#000;position:relative}
.grid img{width:100%;display:block}
.grid figcaption{position:absolute;left:4px;top:4px;background:#0009;padding:2px 6px;border-radius:3px;font-size:12px}
.sliders{display:grid;grid-template-columns:auto 1fr auto;gap:4px 8px;align-items:center;margin-top:8px}
.sliders input{width:100%}
button{background:#2d3138;color:#ddd;border:1px solid #444;border-radius:3px;padding:3px 9px;cursor:pointer}
button:hover{background:#3a3f48}
button.on{background:#3a5a3a;border-color:#6a9a6a}
.thumbs{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.thumbs div{position:relative;cursor:pointer;border:2px solid transparent}
.thumbs div.cur{border-color:#ffd54a}
.thumbs img{width:100%;display:block}
.thumbs span{position:absolute;right:2px;bottom:2px;font-size:10px;background:#000a;padding:1px 4px;border-radius:2px}
.acc{outline:2px solid #4caf50}.rej{outline:2px solid #e53935}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;background:#333;margin-right:4px}
.tag.ok{background:#2e5d34}.tag.global{background:#6d4c1e}.tag.none{background:#555}.tag.size_mismatch{background:#5a2d6b}.tag.had_mask{background:#1e4d6d}.tag.stale{background:#7a2e2e}
#status{color:#9ad}
kbd{background:#333;border-radius:3px;padding:0 4px;font-family:monospace}
.pager{display:flex;justify-content:space-between;align-items:center;margin:6px 0}
.grid img{cursor:zoom-in}
#fs{display:none;position:fixed;inset:0;background:#000;z-index:10}
#fs.on{display:block}
#fs img{width:100vw;height:100vh;object-fit:contain;display:block;cursor:pointer}
#fs-hud{position:absolute;left:10px;top:8px;background:#000a;color:#eee;padding:4px 8px;border-radius:4px;font-size:13px}
#fs-help{position:absolute;right:10px;bottom:8px;background:#000a;color:#bbb;padding:4px 8px;border-radius:4px;font-size:12px}
</style></head><body>
<header>
 <b id="title">DPO automask</b>
 <span id="counts"></span>
 <label>filter <select id="filter">
  <option value="all">all</option><option value="undecided">undecided</option><option value="accept">accepted</option>
  <option value="reject">rejected</option><option value="stale">stale</option><option value="ok">flag: ok</option>
  <option value="global">flag: global</option><option value="none">flag: none</option>
  <option value="size_mismatch">flag: size_mismatch</option><option value="had_mask">flag: had_mask</option>
 </select></label>
 <span><kbd>A</kbd> accept <kbd>R</kbd> reject <kbd>U</kbd> clear <kbd>&larr;</kbd><kbd>&rarr;</kbd> move <kbd>N</kbd> next undecided <kbd>O</kbd> toggle overlay <kbd>F</kbd> full screen</span>
 <span id="status"></span>
</header>
<main>
 <div id="info" style="margin-bottom:6px"></div>
 <div class="grid">
  <figure><img id="img-chosen"><figcaption>chosen</figcaption></figure>
  <figure><img id="img-rejected"><figcaption>rejected</figcaption></figure>
  <figure><img id="img-overlay"><figcaption id="cap-overlay">chosen + proposed mask</figcaption></figure>
  <figure><img id="img-heat"><figcaption>multiscale difference (0 .. 2x threshold)</figcaption></figure>
 </div>
 <div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap">
  <button id="btn-accept">Accept (A)</button><button id="btn-reject">Reject (R)</button><button id="btn-clear">Clear (U)</button>
  <button id="btn-reset">Reset sliders to stored</button>
  <button id="btn-rerun">Re-run ALL pairs with these sliders</button>
 </div>
 <div class="sliders" id="sliders"></div>
 <div id="live" style="margin-top:6px"></div>
</main>
<aside>
 <div class="pager"><button id="pg-prev">&lsaquo; page</button><span id="pg-label"></span><button id="pg-next">page &rsaquo;</button></div>
 <div class="thumbs" id="thumbs"></div>
</aside>
<div id="fs"><img id="fs-img"><div id="fs-hud"></div>
 <div id="fs-help"><kbd>Space</kbd> flip chosen/rejected <kbd>O</kbd> mask <kbd>H</kbd> heatmap <kbd>A</kbd>/<kbd>R</kbd>/<kbd>U</kbd> decide <kbd>&larr;</kbd><kbd>&rarr;</kbd> pair <kbd>Esc</kbd> exit</div>
</div>
<script>
const PAGE=48;let state=null,rows=[],pos=0,page=0,params={},storedParams={},overlayOn=true,liveTimer=null;
let fsOn=false,fsSide='chosen',fsMode='plain';
const $=id=>document.getElementById(id);
async function getJSON(u,o){const r=await fetch(u,o);return r.json();}
function qs(){return Object.entries(params).map(([k,v])=>k+'='+encodeURIComponent(v)).join('&');}
function fmtPct(v){return (v*100).toFixed(2)+'%';}
async function init(){
 state=await getJSON('/api/state');params={...state.params};storedParams={...state.params};
 $('title').textContent='DPO automask: '+state.dataset;
 buildSliders();await loadRows();render();pollRerun();
}
function buildSliders(){
 const s=$('sliders');s.innerHTML='';
 for(const [name,[lo,hi,step]] of Object.entries(state.ranges)){
  const lab=document.createElement('label');lab.textContent=name;
  const inp=document.createElement('input');inp.type='range';inp.min=lo;inp.max=hi;inp.step=step;inp.value=params[name];inp.dataset.name=name;
  const val=document.createElement('span');val.id='val-'+name;val.textContent=params[name];
  inp.oninput=()=>{params[name]=parseFloat(inp.value);val.textContent=inp.value;scheduleLive();};
  s.append(lab,inp,val);
 }
}
function syncSliders(){for(const inp of $('sliders').querySelectorAll('input')){inp.value=params[inp.dataset.name];$('val-'+inp.dataset.name).textContent=params[inp.dataset.name];}}
async function loadRows(){rows=await getJSON('/api/pairs?filter='+$('filter').value);pos=Math.min(pos,Math.max(0,rows.length-1));page=Math.floor(pos/PAGE);}
function cur(){return rows[pos];}
function render(){
 updateCounts();renderThumbs();
 const r=cur();if(!r){$('info').textContent='no pairs in this filter';return;}
 $('info').innerHTML='<b>'+(pos+1)+'/'+rows.length+'</b> &nbsp; '+r.key+' &nbsp; '+r.flags.map(f=>'<span class="tag '+f+'">'+f+'</span>').join('')+
  (r.stale?'<span class="tag stale">stale decision</span>':'')+' stored: raw '+fmtPct(r.raw_coverage)+' final '+fmtPct(r.final_coverage)+
  (r.size?' size '+r.size[0]+'x'+r.size[1]:'')+' &nbsp; decision: <b>'+(r.decision||'-')+'</b>';
 $('btn-accept').className=r.decision==='accept'?'on':'';$('btn-reject').className=r.decision==='reject'?'on':'';
 $('img-chosen').src='/preview/'+r.index+'/chosen.jpg';$('img-rejected').src='/preview/'+r.index+'/rejected.jpg';
 if(r.params){storedParams={...state.params,...r.params};}
 refreshLive();
 if(fsOn)updateFs();
}
// ---- full-screen viewer: native resolution, same index-only preview routes ----
function fsUrl(side,mode){
 const r=cur();if(!r)return '';
 if(mode==='heat')return '/preview/'+r.index+'/heat_full.jpg?'+qs();
 if(mode==='overlay')return '/preview/'+r.index+'/'+(side==='chosen'?'overlay_full':'overlay_rejected_full')+'.jpg?'+qs();
 return '/preview/'+r.index+'/'+side+'_full.jpg';
}
function updateFs(){
 const r=cur();if(!r){closeFs();return;}
 $('fs-img').src=fsUrl(fsSide,fsMode);
 const other=new Image();other.src=fsUrl(fsSide==='chosen'?'rejected':'chosen',fsMode);
 $('fs-hud').textContent=(pos+1)+'/'+rows.length+'  '+r.key+'  |  '+fsSide.toUpperCase()+(fsMode==='overlay'?' + mask':fsMode==='heat'?'  heatmap':'')+
  '  |  '+r.flags.join(',')+'  |  decision: '+(r.decision||'-');
}
function openFs(){
 fsOn=true;$('fs').classList.add('on');updateFs();
 const el=$('fs');if(el.requestFullscreen&&!document.fullscreenElement)el.requestFullscreen().catch(()=>{});
}
function closeFs(){
 fsOn=false;$('fs').classList.remove('on');
 if(document.fullscreenElement&&document.exitFullscreen)document.exitFullscreen().catch(()=>{});
}
document.addEventListener('fullscreenchange',()=>{if(!document.fullscreenElement&&fsOn)closeFs();});
for(const id of ['img-chosen','img-rejected','img-overlay','img-heat'])$(id).onclick=openFs;
$('fs-img').onclick=()=>{fsSide=fsSide==='chosen'?'rejected':'chosen';updateFs();};
function scheduleLive(){clearTimeout(liveTimer);liveTimer=setTimeout(refreshLive,150);}
async function refreshLive(){
 const r=cur();if(!r)return;
 const q=qs();
 $('img-overlay').src=overlayOn?'/preview/'+r.index+'/overlay.jpg?'+q:'/preview/'+r.index+'/chosen.jpg';
 $('img-heat').src='/preview/'+r.index+'/heat.jpg?'+q;
 const d=await getJSON('/api/pair/'+r.index+'?'+q);
 if(cur()!==r)return;
 $('live').innerHTML='live with sliders: '+d.live.flags.map(f=>'<span class="tag '+f+'">'+f+'</span>').join('')+' raw '+fmtPct(d.live.raw_coverage)+' final '+fmtPct(d.live.final_coverage)+
  (d.live_hash!==hashOf(storedParams)?' <i>(differs from stored; accepting stores these parameters for this pair)</i>':'');
}
function hashOf(p){return JSON.stringify(p);}
function renderThumbs(){
 const t=$('thumbs');t.innerHTML='';const start=page*PAGE;
 $('pg-label').textContent=rows.length?('page '+(page+1)+'/'+Math.max(1,Math.ceil(rows.length/PAGE))):'';
 rows.slice(start,start+PAGE).forEach((r,i)=>{
  const d=document.createElement('div');d.className=(start+i===pos?'cur ':'')+(r.decision==='accept'?'acc':r.decision==='reject'?'rej':'');
  const img=document.createElement('img');img.loading='lazy';img.src='/preview/'+r.index+'/thumb.jpg';
  const sp=document.createElement('span');sp.textContent=r.flags[0]+' '+fmtPct(r.final_coverage);
  d.append(img,sp);d.onclick=()=>{pos=start+i;render();};t.appendChild(d);
 });
}
function updateCounts(){const c=state.counts;$('counts').textContent=c.total+' pairs, '+c.accepted+' accepted, '+c.rejected+' rejected, '+(c.total-c.accepted-c.rejected)+' undecided';}
async function decide(decision){
 const r=cur();if(!r)return;
 const res=await getJSON('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:r.index,decision,params})});
 Object.assign(r,res);const s=await getJSON('/api/state');state.counts=s.counts;
 if(decision&&pos<rows.length-1){pos++;page=Math.floor(pos/PAGE);}
 render();
}
function move(d){if(!rows.length)return;pos=Math.max(0,Math.min(rows.length-1,pos+d));page=Math.floor(pos/PAGE);render();}
function nextUndecided(){for(let i=pos+1;i<rows.length;i++){if(!rows[i].decision){pos=i;page=Math.floor(pos/PAGE);render();return;}}}
async function rerun(){
 if(!confirm('Re-run proposals for ALL pairs with the current sliders? Stored masks are replaced; decisions are kept but marked stale where parameters changed.'))return;
 await getJSON('/api/rerun',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({params})});
}
async function pollRerun(){
 const s=await getJSON('/api/state');state.counts=s.counts;
 if(s.rerun.running){$('status').textContent='re-running '+s.rerun.done+'/'+s.rerun.total;}
 else if(s.rerun.stats&&$('status').textContent.startsWith('re-running')){$('status').textContent='re-run done';state.params=s.params;storedParams={...s.params};await loadRows();render();}
 else if(s.rerun.error){$('status').textContent='re-run failed: '+s.rerun.error;}
 setTimeout(pollRerun,1500);
}
document.addEventListener('keydown',e=>{
 if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
 const k=e.key.toLowerCase();
 if(fsOn){
  if(e.key==='Escape')closeFs();
  else if(e.key===' ')  {fsSide=fsSide==='chosen'?'rejected':'chosen';updateFs();}
  else if(k==='o'){fsMode=fsMode==='overlay'?'plain':'overlay';updateFs();}
  else if(k==='h'){fsMode=fsMode==='heat'?'plain':'heat';updateFs();}
  else if(k==='f')closeFs();
  else if(k==='a')decide('accept');else if(k==='r')decide('reject');else if(k==='u')decide(null);
  else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(k==='n')nextUndecided();
  else return;e.preventDefault();return;
 }
 if(k==='f'){openFs();e.preventDefault();return;}
 if(k==='a')decide('accept');else if(k==='r')decide('reject');else if(k==='u')decide(null);
 else if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(k==='n')nextUndecided();
 else if(k==='o'){overlayOn=!overlayOn;$('cap-overlay').textContent=overlayOn?'chosen + proposed mask':'chosen (overlay off)';refreshLive();}
 else return;e.preventDefault();
});
$('btn-accept').onclick=()=>decide('accept');$('btn-reject').onclick=()=>decide('reject');$('btn-clear').onclick=()=>decide(null);
$('btn-reset').onclick=()=>{params={...storedParams};syncSliders();refreshLive();};
$('btn-rerun').onclick=rerun;
$('filter').onchange=async()=>{pos=0;await loadRows();render();};
$('pg-prev').onclick=()=>{if(page>0){page--;renderThumbs();}};
$('pg-next').onclick=()=>{if((page+1)*PAGE<rows.length){page++;renderThumbs();}};
init();
</script></body></html>
"""


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("dataset")
        p.add_argument("--chosen-pattern", default=DEFAULT_CHOSEN_PATTERN)
        p.add_argument("--rejected-pattern", default=DEFAULT_REJECTED_PATTERN)

    p_propose = sub.add_parser("propose", help="compute proposal masks into <dataset>/.automask")
    add_common(p_propose)
    p_propose.add_argument(
        "--include-covered", type=float, default=None, help="also redo pairs whose existing mask covers > this fraction"
    )
    p_propose.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    add_param_arguments(p_propose)

    p_review = sub.add_parser("review", help="serve the review page on 127.0.0.1")
    p_review.add_argument("dataset")
    p_review.add_argument("--port", type=int, default=8766)
    p_review.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="for re-run all")

    p_apply = sub.add_parser("apply", help="write accepted masks to chosen/ and rejected/")
    p_apply.add_argument("dataset")
    p_apply.add_argument("--overwrite", action="store_true", help="replace existing masks (backed up first)")

    p_undo = sub.add_parser("undo", help="reverse an apply run")
    p_undo.add_argument("dataset")
    p_undo.add_argument("--run", type=int, default=None, help="run index in applied.json (default: latest)")

    args = parser.parse_args(argv)

    if args.command == "propose":
        params = params_from_args(args)
        print(f"proposing with {params.to_dict()}")

        def progress(done, total):
            if done % 100 == 0 or done == total:
                print(f"  {done}/{total}", flush=True)

        stats = run_propose(
            args.dataset,
            params,
            args.chosen_pattern,
            args.rejected_pattern,
            args.include_covered,
            args.workers,
            progress,
        )
        if args.include_covered is not None:
            proposals = load_proposals(args.dataset)
            proposals["include_covered"] = args.include_covered
            save_proposals(args.dataset, proposals)
        print_stats(stats)
    elif args.command == "review":
        proposals = load_proposals(args.dataset)
        if not proposals["pairs"]:
            print("no proposals yet: run propose first")
            return 1
        server = make_server(args.dataset, args.port, args.workers)
        print(
            f"review {len(proposals['pairs'])} pairs at http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)"
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "apply":
        print_stats(run_apply(args.dataset, args.overwrite))
    elif args.command == "undo":
        print_stats(run_undo(args.dataset, args.run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
