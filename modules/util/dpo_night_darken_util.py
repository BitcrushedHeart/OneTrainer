"""Night Darken: pull chosen DPO images whose caption says "night" / "dark" down
to a brightness that matches the caption, keeping the originals.

Launch
------
The tool is the "Night Darken" card on the web GUI's Tools page (backend
routes ``/api/dpo/night-darken/*``, ``DPONightDarkenService``). Start the
web backend as usual (``scripts/start_web.py`` / the Electron app); nothing
else is needed.

Flow
----
1. **Scan** a DPO concept folder (``<concept>/chosen/**``): every chosen image
   whose caption ``.txt`` matches :data:`NIGHT_PHRASES` (whole words, case
   insensitive) gets a luma statistic. Images that are already genuinely
   dark (:func:`is_already_dark`, thresholds in :class:`DarkThresholds`) are
   counted and skipped; the rest are the candidates. Minor Edits and
   Off-policy datasets are never sources — the user picks the source folder
   and the tool refuses a folder whose ``concepts.json`` / ``workspace.json``
   registration says it is one of those.
2. **Review**, one candidate at a time: a before/after wipe over the image,
   a strength slider that re-renders the darkened side, *Accept* (keeps the
   current strength), *Skip*, *Back*. Decisions live in the ledger at
   ``<concept>/.night_darken/ledger.json`` so a closed modal resumes.
3. **Commit**: every accepted image is darkened at its strength and replaced
   in place (same file name and format, PNG text chunks / EXIF / ICC kept).
   The original goes to ``<concept>/.night_darken/backup/<split>/<name>``,
   listed in ``<concept>/.night_darken/backup/manifest.csv`` (path, strength,
   stats). The most changed :data:`DEFAULT_MINOR_EDITS_FRACTION` of the
   accepted set (ranked by mean absolute luma change) is also written as new
   pairs into the Minor Edits folder: darkened = chosen, original = rejected,
   the same caption, and an all-white ``-masklabel.png`` beside both sides so
   the whole image counts under masked DPO.
4. **Restore** reverses a commit: backups move back, the concept manifest
   entries return to what they were, the Minor Edits pairs (files, masks,
   captions, manifest entries) are removed.

Darkening
---------
:func:`darken_rgb` works in linear light. Each pixel's linear luminance ``Y``
is pulled down by an exposure drop of ``EXPOSURE_STOPS * strength`` stops
and a gamma of ``1 + GAMMA_LIFT * strength`` (shadows fall faster than
highlights, so light sources keep reading as lights), and the RGB channels
are scaled by the resulting ``Y_out / Y`` so hue and saturation stay put.
Clipped highlights (``Y`` above :data:`HIGHLIGHT_KNEE`) keep
:data:`HIGHLIGHT_KEEP` of their original luminance so lamps, screens and
specular hits survive. Strength 0 is the identity.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass

from modules.util.dpo_curation_util import (
    _next_pair_id,
    _normalize_source_path,
    compute_file_sha256,
    load_manifest,
    resolve_aspect_ratio,
    save_manifest,
)
from modules.util.path_util import supported_image_extensions

import numpy as np

# --------------------------------------------------------------------------- #
# Constants (all tunable from the UI where it matters)
# --------------------------------------------------------------------------- #

#: Caption phrases that claim night or darkness. Whole words, case insensitive;
#: a space matches any run of spaces or hyphens ("low light" == "low-light").
NIGHT_PHRASES: tuple[str, ...] = (
    "night",
    "nighttime",
    "night time",
    "at night",
    "late night",
    "midnight",
    "after dark",
    "dark",
    "darkness",
    "darkened",
    "dimly lit",
    "dim lighting",
    "dim light",
    "low light",
    "poorly lit",
    "unlit",
    "moonlit",
    "moonlight",
    "candlelit",
    "candlelight",
    "nocturnal",
    "pitch black",
)

#: Nouns that make a bare "dark" describe an object, not the lighting
#: ("dark hair", "dark jeans"). Such a match does not make a candidate.
DARK_OBJECT_WORDS: tuple[str, ...] = (
    "hair",
    "haired",
    "skin",
    "skinned",
    "eyes",
    "eyed",
    "eyebrows",
    "brows",
    "lashes",
    "beard",
    "stubble",
    "brown",
    "blonde",
    "blond",
    "red",
    "blue",
    "green",
    "grey",
    "gray",
    "purple",
    "pink",
    "navy",
    "clothing",
    "clothes",
    "outfit",
    "jeans",
    "shirt",
    "t-shirt",
    "tshirt",
    "top",
    "dress",
    "skirt",
    "jacket",
    "coat",
    "hoodie",
    "sweater",
    "jumper",
    "pants",
    "trousers",
    "shorts",
    "suit",
    "tie",
    "shoes",
    "boots",
    "socks",
    "lingerie",
    "bra",
    "underwear",
    "bikini",
    "swimsuit",
    "leather",
    "wood",
    "wooden",
    "metal",
    "fabric",
    "curtains",
    "sofa",
    "couch",
    "background",
    "backdrop",
    "wall",
    "walls",
    "floor",
    "tiles",
    "stone",
    "marble",
    "nipples",
    "areolas",
    "areolae",
    "lipstick",
    "makeup",
    "nail",
    "nails",
    "tattoo",
    "tattoos",
    "circles",
    "roots",
    "bob",
    "pixie",
    "ponytail",
    "bun",
    "braid",
    "braids",
    "curls",
    "waves",
    "locks",
    "strands",
)

#: Luma (Rec.709 on 8-bit sRGB) at or below which an image already reads as
#: dark; both the median and the mean must agree.
DEFAULT_DARK_MEDIAN = 60.0
DEFAULT_DARK_MEAN = 80.0
#: A pixel counts as bright above this luma (for the "bright fraction" stat).
BRIGHT_PIXEL_LUMA = 128

#: Darkening curve at strength 1 (every term scales linearly with strength).
EXPOSURE_STOPS = 2.5
GAMMA_LIFT = 0.8
#: Linear luminance from which highlights start keeping part of their
#: original luminance, and how much of it a fully clipped pixel keeps.
HIGHLIGHT_KNEE = 0.8
HIGHLIGHT_KEEP = 0.6

#: Share of the accepted images (most changed first) that also become Minor
#: Edits pairs.
DEFAULT_MINOR_EDITS_FRACTION = 0.33
#: Default accept strength.
DEFAULT_STRENGTH = 0.5

#: Long side of the image the scan statistics are computed on.
SCAN_MAX_SIDE = 256
#: Long side of the review preview.
PREVIEW_MAX_SIDE = 1280

LEDGER_DIR = ".night_darken"
LEDGER_FILE = "ledger.json"
BACKUP_DIR = "backup"
BACKUP_MANIFEST = "manifest.csv"
HISTORY_ACTION = "night_darken"
MASK_SUFFIX = "-masklabel.png"
#: Optimise workspace dataset types this tool never reads from.
EXCLUDED_WORKSPACE_TYPES = ("minor_edits", "offpolicy")
#: Folder-name words that mark an excluded source when no workspace registry says so.
EXCLUDED_FOLDER_WORDS = ("minor edit", "minor_edit", "off policy", "offpolicy", "off-policy", "off_policy")

BACKUP_CSV_FIELDS = (
    "committed_at",
    "commit_id",
    "chosen",
    "backup",
    "strength",
    "median",
    "mean",
    "bright_fraction",
    "delta_luma",
    "minor_edits_dir",
    "minor_edits_pair_id",
)

_LEDGER_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
# Caption matching
# --------------------------------------------------------------------------- #


def _phrase_pattern(phrase: str) -> str:
    words = [re.escape(w) for w in re.split(r"[\s-]+", phrase.strip()) if w]
    return r"\b" + r"[\s-]+".join(words) + r"\b"


def compile_phrases(phrases: tuple[str, ...] | list[str] = NIGHT_PHRASES) -> list[tuple[str, re.Pattern]]:
    """One compiled whole-word pattern per phrase, longest phrase first."""
    ordered = sorted({p.strip() for p in phrases if p.strip()}, key=lambda p: (-len(p), p))
    return [(p, re.compile(_phrase_pattern(p), re.IGNORECASE)) for p in ordered]


# "dark hair", "dark brown hair", "dark wavy hair": at most one word between.
_DARK_OBJECT_RE = re.compile(
    r"\bdark(?:er|est)?\b(?:\s+[\w-]+)?\s+(?:" + "|".join(re.escape(w) for w in DARK_OBJECT_WORDS) + r")\b",
    re.IGNORECASE,
)
_BARE_DARK_RE = re.compile(r"\bdark(?:er|est)?\b", re.IGNORECASE)


def match_night_phrases(caption: str, patterns: list[tuple[str, re.Pattern]] | None = None) -> list[str]:
    """The phrases a caption claims, in list order. A bare ``dark`` whose
    every occurrence describes an object (``dark hair``) does not count."""
    if not caption:
        return []
    patterns = patterns if patterns is not None else compile_phrases()
    found: list[str] = []
    for phrase, pattern in patterns:
        if not pattern.search(caption):
            continue
        if phrase.lower() == "dark":
            spans = [m.span() for m in _BARE_DARK_RE.finditer(caption)]
            object_spans = [m.span() for m in _DARK_OBJECT_RE.finditer(caption)]
            lighting = [s for s in spans if not any(o[0] <= s[0] < o[1] for o in object_spans)]
            if not lighting:
                continue
        found.append(phrase)
    return found


# --------------------------------------------------------------------------- #
# Luma statistics
# --------------------------------------------------------------------------- #


@dataclass
class DarkThresholds:
    median: float = DEFAULT_DARK_MEDIAN
    mean: float = DEFAULT_DARK_MEAN

    @classmethod
    def from_dict(cls, raw: dict | None) -> DarkThresholds:
        raw = raw or {}
        out = cls()
        for key in ("median", "mean"):
            with suppress(TypeError, ValueError):
                if raw.get(key) is not None:
                    setattr(out, key, float(raw[key]))
        return out


@dataclass
class LumaStats:
    median: float
    mean: float
    p90: float
    bright_fraction: float
    dark_fraction: float

    def to_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in asdict(self).items()}


_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luma_u8(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an ``[H, W, 3]`` uint8 array, float32 in 0..255."""
    return rgb[..., :3].astype(np.float32) @ _LUMA_WEIGHTS


def luma_stats(rgb: np.ndarray) -> LumaStats:
    y = luma_u8(rgb).ravel()
    if y.size == 0:
        return LumaStats(0.0, 0.0, 0.0, 0.0, 0.0)
    return LumaStats(
        median=float(np.median(y)),
        mean=float(y.mean()),
        p90=float(np.percentile(y, 90)),
        bright_fraction=float((y > BRIGHT_PIXEL_LUMA).mean()),
        dark_fraction=float((y < 32).mean()),
    )


def is_already_dark(stats: LumaStats | dict, thresholds: DarkThresholds | None = None) -> bool:
    t = thresholds or DarkThresholds()
    median = stats["median"] if isinstance(stats, dict) else stats.median
    mean = stats["mean"] if isinstance(stats, dict) else stats.mean
    return float(median) <= t.median and float(mean) <= t.mean


def mean_abs_luma_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(luma_u8(a) - luma_u8(b)).mean())


# --------------------------------------------------------------------------- #
# Darkening
# --------------------------------------------------------------------------- #


def _srgb_to_linear_lut() -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


_SRGB_TO_LINEAR_LUT = _srgb_to_linear_lut()


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def darken_luminance(y: np.ndarray, strength: float) -> np.ndarray:
    """The curve on linear luminance in 0..1 (see the module docstring)."""
    s = float(min(max(strength, 0.0), 1.0))
    if s <= 0.0:
        return y
    gamma = 1.0 + GAMMA_LIFT * s
    gain = 2.0 ** (-EXPOSURE_STOPS * s)
    pulled = np.power(np.clip(y, 0.0, 1.0), gamma) * gain
    keep = HIGHLIGHT_KEEP * _smoothstep(HIGHLIGHT_KNEE, 1.0, y)
    return keep * y + (1.0 - keep) * pulled


def darken_rgb(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Darken an ``[H, W, 3]`` (or ``[H, W, 4]``, alpha untouched) uint8 sRGB
    array. Strength 0 returns an identical copy."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        raise TypeError("darken_rgb expects a uint8 array")
    s = float(min(max(strength, 0.0), 1.0))
    if s <= 0.0:
        return rgb.copy()
    alpha = rgb[..., 3:4] if rgb.shape[-1] == 4 else None
    lin = _SRGB_TO_LINEAR_LUT[rgb[..., :3]]
    y = lin @ _LUMA_WEIGHTS
    y_out = darken_luminance(y, s)
    ratio = np.where(y > 1e-6, y_out / np.maximum(y, 1e-6), 1.0).astype(np.float32)
    out_lin = lin * ratio[..., None]
    out = np.rint(_linear_to_srgb(out_lin) * 255.0).astype(np.uint8)
    if alpha is not None:
        out = np.concatenate([out, alpha], axis=-1)
    return out


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #


def load_rgb(path: str, max_side: int | None = None) -> np.ndarray:
    """Decode to ``[H, W, 3]`` uint8 (alpha dropped), optionally downscaled so
    the long side is at most ``max_side``."""
    return load_rgb_with_size(path, max_side)[0]


def load_rgb_with_size(path: str, max_side: int | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """:func:`load_rgb` plus the file's full pixel size."""
    from PIL import Image

    with Image.open(path) as img:
        size = img.size
        if max_side and max(img.size) > max_side:
            with suppress(Exception):
                img.draft("RGB", (max_side, max_side))
        img = img.convert("RGB")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        return np.asarray(img), size


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(rgb[..., :3]).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _png_info(src) -> object:
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    text = getattr(src, "text", None) or {}
    for key, value in text.items():
        if isinstance(key, str) and isinstance(value, str):
            with suppress(Exception):
                info.add_text(key, value)
    return info


def save_darkened_like(original_path: str, strength: float, target_path: str) -> dict:
    """Write the darkened original to ``target_path`` in the original's format,
    carrying its PNG text chunks (SwarmUI parameters, Triage & Fix provenance),
    EXIF, ICC profile and XMP across. Alpha is kept as is. Returns the luma
    change (``delta_luma``) and the original's stats.

    The file is written beside the target and renamed into place, so a crash
    leaves either the old file or the new one."""
    from PIL import Image

    with Image.open(original_path) as src:
        fmt = (src.format or os.path.splitext(original_path)[1].lstrip(".")).upper()
        if fmt == "JPG":
            fmt = "JPEG"
        info = dict(src.info or {})
        png_info = _png_info(src) if fmt == "PNG" else None
        has_alpha = src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in info)
        work = src.convert("RGBA" if has_alpha else "RGB")
        pixels = np.asarray(work)

    darkened = darken_rgb(pixels, strength)
    delta = mean_abs_luma_delta(pixels[..., :3], darkened[..., :3])
    stats = luma_stats(pixels[..., :3])

    kwargs: dict = {}
    for key in ("exif", "icc_profile", "xmp", "dpi"):
        if info.get(key):
            kwargs[key] = info[key]
    if fmt == "PNG":
        kwargs["pnginfo"] = png_info
        kwargs.pop("xmp", None)
    elif fmt == "JPEG":
        kwargs.update(quality=95, subsampling=0)
        kwargs.pop("xmp", None)
        darkened = darkened[..., :3]
    elif fmt == "WEBP":
        kwargs.update(quality=95, method=4)
        kwargs.pop("dpi", None)
    else:
        kwargs = {k: v for k, v in kwargs.items() if k in ("exif", "icc_profile")}
        darkened = darkened[..., :3]

    directory = os.path.dirname(target_path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(target_path)}.tmp")
    try:
        Image.fromarray(darkened).save(tmp, format=fmt, **kwargs)
        os.replace(tmp, target_path)
    finally:
        with suppress(OSError):
            if os.path.exists(tmp):
                os.remove(tmp)
    return {"delta_luma": round(delta, 4), "stats": stats.to_dict(), "format": fmt}


def write_full_mask(image_path: str, mask_path: str | None = None) -> str:
    """An all-white L-mode PNG at the image's exact size: the whole picture
    trains under masked DPO."""
    from PIL import Image

    with Image.open(image_path) as img:
        size = img.size
    target = mask_path or mask_path_for(image_path)
    Image.new("L", size, 255).save(target, format="PNG")
    return target


def mask_path_for(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + MASK_SUFFIX


# --------------------------------------------------------------------------- #
# Concept folder layout
# --------------------------------------------------------------------------- #


def rel(concept_dir: str, path: str) -> str:
    return os.path.relpath(path, concept_dir).replace("\\", "/")


def absolute(concept_dir: str, path: str | None) -> str | None:
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(concept_dir, path))


def ledger_dir(concept_dir: str) -> str:
    return os.path.join(concept_dir, LEDGER_DIR)


def ledger_path(concept_dir: str) -> str:
    return os.path.join(ledger_dir(concept_dir), LEDGER_FILE)


def backup_dir(concept_dir: str) -> str:
    return os.path.join(ledger_dir(concept_dir), BACKUP_DIR)


def is_mask_file(name: str) -> bool:
    return name.lower().endswith(MASK_SUFFIX)


def iter_chosen_images(concept_dir: str) -> list[dict]:
    """Every chosen image with its caption path, split and rejected twin.
    Dot-folders are skipped like mgds and Optimise skip them."""
    chosen_root = os.path.join(concept_dir, "chosen")
    rejected_root = os.path.join(concept_dir, "rejected")
    exts = supported_image_extensions()
    items: list[dict] = []
    if not os.path.isdir(chosen_root):
        return items
    for root, dirs, files in os.walk(chosen_root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith(".") or is_mask_file(name):
                continue
            stem, ext = os.path.splitext(name)
            if ext.lower() not in exts:
                continue
            path = os.path.join(root, name)
            split = os.path.relpath(root, chosen_root).replace("\\", "/")
            split = "" if split == "." else split
            caption_path = os.path.join(root, stem + ".txt")
            rejected = None
            rejected_dir = os.path.join(rejected_root, split) if split else rejected_root
            if os.path.isdir(rejected_dir):
                for candidate_ext in sorted(exts):
                    candidate = os.path.join(rejected_dir, stem + candidate_ext)
                    if os.path.isfile(candidate):
                        rejected = candidate
                        break
            items.append(
                {
                    "path": path,
                    "rel": rel(concept_dir, path),
                    "stem": stem,
                    "split": split,
                    "caption_path": caption_path if os.path.isfile(caption_path) else None,
                    "rejected_path": rejected,
                }
            )
    return items


def read_caption(path: str | None) -> str:
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def source_exclusion_reason(concept_dir: str) -> str | None:
    """Why a folder may not be a source: it is a Minor Edits or Off-policy
    dataset (by its Optimise workspace registration, else by its name)."""
    concept_dir = os.path.abspath(concept_dir)
    workspace = os.path.join(os.path.dirname(concept_dir), "workspace.json")
    if os.path.isfile(workspace):
        try:
            with open(workspace, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
        for dataset in data.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            folder = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(workspace), dataset.get("dir", ""))))
            if folder == os.path.normcase(concept_dir) and dataset.get("type") in EXCLUDED_WORKSPACE_TYPES:
                return f"{os.path.basename(concept_dir)} is registered as a {dataset.get('type')} dataset"
    lowered = os.path.basename(concept_dir).lower()
    for word in EXCLUDED_FOLDER_WORDS:
        if word in lowered:
            return f"{os.path.basename(concept_dir)} looks like a {word} dataset"
    return None


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def new_ledger(concept_dir: str) -> dict:
    return {
        "version": 1,
        "concept_dir": os.path.abspath(concept_dir),
        "scan": None,
        "candidates": [],
        "decisions": {},
        "commits": [],
    }


def load_ledger(concept_dir: str) -> dict | None:
    path = ledger_path(concept_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_ledger(concept_dir: str, ledger: dict) -> None:
    path = ledger_path(concept_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with _LEDGER_LOCK:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # noqa: PERF203 — Windows rename retry
                if attempt == 19:
                    raise
                time.sleep(0.05)


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #


def scan_candidates(
    concept_dir: str,
    *,
    phrases: tuple[str, ...] | list[str] = NIGHT_PHRASES,
    thresholds: DarkThresholds | None = None,
    progress=None,
    stop=None,
) -> dict:
    """Find the chosen images whose caption claims night/darkness and that are
    not dark. Returns ``{"candidates": [...], "counts": {...}}``; the
    candidates carry their stats, the already-dark ones are only counted
    (their paths are listed under ``already_dark``). ``progress(done, total)``
    is called per caption match; ``stop`` is a ``threading.Event``."""
    thresholds = thresholds or DarkThresholds()
    patterns = compile_phrases(phrases)
    images = iter_chosen_images(concept_dir)
    matched = []
    for item in images:
        found = match_night_phrases(read_caption(item["caption_path"]), patterns)
        if found:
            matched.append((item, found))
    candidates: list[dict] = []
    already_dark: list[str] = []
    unreadable: list[str] = []
    for done, (item, found) in enumerate(matched, start=1):
        if stop is not None and stop.is_set():
            break
        try:
            small, size = load_rgb_with_size(item["path"], SCAN_MAX_SIDE)
        except Exception:
            unreadable.append(item["rel"])
            continue
        stats = luma_stats(small)
        if is_already_dark(stats, thresholds):
            already_dark.append(item["rel"])
        else:
            candidates.append(
                {
                    "rel": item["rel"],
                    "stem": item["stem"],
                    "split": item["split"],
                    "phrases": found,
                    "stats": stats.to_dict(),
                    "size": list(size),
                    "has_caption": item["caption_path"] is not None,
                    "has_rejected": item["rejected_path"] is not None,
                }
            )
        if progress is not None:
            progress(done, len(matched))
    return {
        "candidates": candidates,
        "already_dark": already_dark,
        "unreadable": unreadable,
        "counts": {
            "images": len(images),
            "caption_matches": len(matched),
            "already_dark": len(already_dark),
            "unreadable": len(unreadable),
            "candidates": len(candidates),
        },
    }


# --------------------------------------------------------------------------- #
# Minor Edits pairs
# --------------------------------------------------------------------------- #


def _side_dir(minor_dir: str, side: str, split: str) -> str:
    base = os.path.join(minor_dir, side)
    return os.path.join(base, split) if split else base


def minor_edits_split(minor_dir: str) -> str:
    """``train`` when the destination uses ``chosen/train`` (the Optimise
    layout), else flat."""
    return "train" if os.path.isdir(os.path.join(minor_dir, "chosen", "train")) else ""


def _disk_pair_numbers(minor_dir: str) -> set[int]:
    numbers: set[int] = set()
    for side in ("chosen", "rejected"):
        base = os.path.join(minor_dir, side)
        if not os.path.isdir(base):
            continue
        for _root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                match = re.match(r"pair_(\d+)", name)
                if match:
                    numbers.add(int(match.group(1)))
    return numbers


def minor_edits_preflight(minor_dir: str, count: int) -> str | None:
    """Why ``count`` new pairs cannot be written: the folder is not a DPO pair
    folder, or the next ids its manifest would hand out already exist on disk
    (pairs the manifest does not know — Optimise's database would hand out
    the same numbers)."""
    if count <= 0:
        return None
    if not minor_dir:
        return "No Minor Edits folder set"
    if os.path.isdir(minor_dir) and not os.path.isfile(os.path.join(minor_dir, "manifest.json")):
        entries = [e for e in os.listdir(minor_dir) if not e.startswith(".")]
        if entries and not {"chosen", "rejected"} & set(entries):
            return "The Minor Edits folder is not empty and is not a DPO pair folder (no manifest.json)"
    manifest = load_manifest(minor_dir) if os.path.isdir(minor_dir) else {"pairs": []}
    first = _next_pair_id(manifest)
    taken = _disk_pair_numbers(minor_dir) & set(range(first, first + count))
    if taken:
        return f"pair numbers {sorted(taken)[:5]} already exist on disk but not in manifest.json"
    return None


def export_night_pair(
    minor_dir: str,
    manifest: dict,
    *,
    darkened_path: str,
    original_path: str,
    caption_path: str | None,
    lineage: dict,
) -> dict:
    """One Minor Edits pair: the darkened image as chosen, the original as
    rejected, the caption copied byte for byte, an all-white mask beside both
    sides. Follows ``export_single_pair``'s numbering and manifest entry and
    adds the destination's ``train`` split when it has one."""
    split = minor_edits_split(minor_dir)
    chosen_dir = _side_dir(minor_dir, "chosen", split)
    rejected_dir = _side_dir(minor_dir, "rejected", split)
    os.makedirs(chosen_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    pair_id = _next_pair_id(manifest)
    stem = f"pair_{pair_id:04d}"
    chosen_ext = os.path.splitext(darkened_path)[1].lower()
    rejected_ext = os.path.splitext(original_path)[1].lower()
    chosen_target = os.path.join(chosen_dir, stem + chosen_ext)
    rejected_target = os.path.join(rejected_dir, stem + rejected_ext)
    for target in (chosen_target, rejected_target):
        if os.path.exists(target):
            raise FileExistsError(target)

    written: list[str] = []
    try:
        shutil.copy2(darkened_path, chosen_target)
        written.append(chosen_target)
        shutil.copy2(original_path, rejected_target)
        written.append(rejected_target)
        caption_target = None
        if caption_path and os.path.isfile(caption_path):
            caption_target = os.path.join(chosen_dir, stem + ".txt")
            shutil.copyfile(caption_path, caption_target)
            written.append(caption_target)
        masks = [write_full_mask(chosen_target), write_full_mask(rejected_target)]
        written.extend(masks)
    except Exception:
        for path in written:
            with suppress(OSError):
                os.remove(path)
        raise

    prefix = f"{split}/" if split else ""
    entry = {
        "pair_id": pair_id,
        "prompt": read_caption(caption_path),
        "aspectratio": resolve_aspect_ratio("", chosen_target),
        "split": split or None,
        "chosen_file": prefix + os.path.basename(chosen_target),
        "rejected_file": prefix + os.path.basename(rejected_target),
        "caption_file": (prefix + stem + ".txt") if caption_target else None,
        "chosen_source": _normalize_source_path(darkened_path),
        "rejected_source": _normalize_source_path(original_path),
        "chosen_sha256": compute_file_sha256(chosen_target),
        "rejected_sha256": compute_file_sha256(rejected_target),
        "chosen_size": os.path.getsize(chosen_target),
        "rejected_size": os.path.getsize(rejected_target),
        "tool": HISTORY_ACTION,
        "lineage": lineage,
    }
    manifest.setdefault("pairs", []).append(entry)
    save_manifest(minor_dir, manifest)
    return {
        "dir": os.path.abspath(minor_dir),
        "pair_id": pair_id,
        "files": [rel(minor_dir, p) for p in written],
    }


def remove_night_pair(minor_dir: str, manifest: dict, record: dict) -> None:
    """Reverse :func:`export_night_pair`: its files and manifest entry go."""
    for name in record.get("files") or []:
        path = absolute(minor_dir, name)
        with suppress(OSError):
            if path:
                os.remove(path)
    pairs = manifest.get("pairs", [])
    manifest["pairs"] = [p for p in pairs if p.get("pair_id") != record.get("pair_id")]
    save_manifest(minor_dir, manifest)


# --------------------------------------------------------------------------- #
# Commit / restore
# --------------------------------------------------------------------------- #


def _manifest_entry_for(manifest: dict, stem: str) -> dict | None:
    for entry in manifest.get("pairs", []):
        name = str(entry.get("chosen_file") or "")
        if name and os.path.splitext(os.path.basename(name))[0] == stem:
            return entry
    return None


_ENTRY_KEYS = ("chosen_sha256", "chosen_size")


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def backup_original(concept_dir: str, chosen_path: str) -> str:
    """Copy the chosen image into the backup tree, mirroring its split."""
    chosen_root = os.path.join(concept_dir, "chosen")
    inner = os.path.relpath(chosen_path, chosen_root)
    target = os.path.join(backup_dir(concept_dir), inner)
    if os.path.exists(target):
        stem, ext = os.path.splitext(target)
        target = f"{stem}__{_timestamp()}{ext}"
        i = 1
        while os.path.exists(target):
            target = f"{stem}__{_timestamp()}_{i}{ext}"
            i += 1
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(chosen_path, target)
    return target


def append_backup_csv(concept_dir: str, rows: list[dict]) -> None:
    path = os.path.join(backup_dir(concept_dir), BACKUP_MANIFEST)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.isfile(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BACKUP_CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def minor_edits_count(total: int, fraction: float) -> int:
    """How many of ``total`` accepted images become pairs: the rounded share,
    at least one when the share is positive."""
    if total <= 0 or fraction <= 0:
        return 0
    return max(1, int(round(total * min(fraction, 1.0))))


def select_minor_edits(items: list[dict], fraction: float) -> list[dict]:
    """The most changed share of ``items`` by ``delta_luma``."""
    count = minor_edits_count(len(items), fraction)
    ranked = sorted(items, key=lambda it: (-float(it.get("delta_luma") or 0.0), it["rel"]))
    return ranked[:count]


def commit_accepted(
    concept_dir: str,
    ledger: dict,
    *,
    minor_edits_dir: str | None,
    fraction: float = DEFAULT_MINOR_EDITS_FRACTION,
    progress=None,
    stop=None,
) -> dict:
    """Darken every accepted, uncommitted candidate in place (backed up), then
    add the most changed share as Minor Edits pairs. Returns the commit
    record, which is also appended to ``ledger["commits"]``. The ledger is
    saved after every image so a crash mid-way leaves a restorable record."""
    concept_dir = os.path.abspath(concept_dir)
    by_rel = {c["rel"]: c for c in ledger.get("candidates", [])}
    pending = [
        (r, d)
        for r, d in ledger.get("decisions", {}).items()
        if d.get("state") == "accepted" and not d.get("committed") and r in by_rel
    ]
    pending.sort(key=lambda rd: rd[0])
    minor_count = minor_edits_count(len(pending), fraction)
    if minor_count and minor_edits_dir:
        problem = minor_edits_preflight(minor_edits_dir, minor_count)
        if problem:
            raise RuntimeError(f"Minor Edits: {problem}")
    if minor_count and not minor_edits_dir:
        raise RuntimeError("Minor Edits: no destination folder set (set the fraction to 0 to skip the pairs)")

    commit = {
        "id": f"{_timestamp()}_{len(ledger.get('commits', [])) + 1}",
        "at": time.time(),
        "minor_edits_dir": os.path.abspath(minor_edits_dir) if minor_edits_dir else None,
        "fraction": fraction,
        "items": [],
        "restored": False,
        "error": None,
    }
    ledger.setdefault("commits", []).append(commit)
    manifest = load_manifest(concept_dir)
    chosen_items = iter_chosen_images(concept_dir)
    by_path_rel = {it["rel"]: it for it in chosen_items}

    try:
        for done, (r, decision) in enumerate(pending, start=1):
            if stop is not None and stop.is_set():
                break
            item = by_path_rel.get(r)
            if item is None:
                continue
            strength = float(decision.get("strength") or 0.0)
            chosen_path = item["path"]
            backup = backup_original(concept_dir, chosen_path)
            try:
                result = save_darkened_like(backup, strength, chosen_path)
            except Exception:
                with suppress(OSError):
                    shutil.copy2(backup, chosen_path)
                with suppress(OSError):
                    os.remove(backup)
                raise
            entry = _manifest_entry_for(manifest, item["stem"])
            before = None
            history_len = None
            if entry is not None:
                before = {k: entry.get(k) for k in _ENTRY_KEYS if k in entry}
                history_len = len(entry.get("history") or [])
                entry["chosen_sha256"] = compute_file_sha256(chosen_path)
                entry["chosen_size"] = os.path.getsize(chosen_path)
                entry.setdefault("history", []).append(
                    {
                        "at": time.time(),
                        "action": HISTORY_ACTION,
                        "strength": strength,
                        "retired": [rel(concept_dir, backup)],
                    }
                )
            record = {
                "rel": r,
                "stem": item["stem"],
                "split": item["split"],
                "strength": strength,
                "backup": rel(concept_dir, backup),
                "delta_luma": result["delta_luma"],
                "stats": result["stats"],
                "format": result["format"],
                "caption": rel(concept_dir, item["caption_path"]) if item["caption_path"] else None,
                "manifest_before": before,
                "history_len": history_len,
                "minor_pair": None,
            }
            commit["items"].append(record)
            decision["committed"] = commit["id"]
            save_ledger(concept_dir, ledger)
            if progress is not None:
                progress(done, len(pending), "darken")
        if any(it["manifest_before"] is not None for it in commit["items"]):
            save_manifest(concept_dir, manifest)

        chosen = select_minor_edits(commit["items"], fraction) if commit["minor_edits_dir"] else []
        if chosen:
            minor_manifest = load_manifest(commit["minor_edits_dir"])
            for done, record in enumerate(chosen, start=1):
                if stop is not None and stop.is_set():
                    break
                item = by_path_rel[record["rel"]]
                record["minor_pair"] = export_night_pair(
                    commit["minor_edits_dir"],
                    minor_manifest,
                    darkened_path=item["path"],
                    original_path=absolute(concept_dir, record["backup"]),
                    caption_path=item["caption_path"],
                    lineage={
                        "tool": HISTORY_ACTION,
                        "source_concept": concept_dir,
                        "source_chosen": record["rel"],
                        "strength": record["strength"],
                        "delta_luma": record["delta_luma"],
                        "commit": commit["id"],
                    },
                )
                save_ledger(concept_dir, ledger)
                if progress is not None:
                    progress(done, len(chosen), "pairs")
    except Exception as ex:
        commit["error"] = str(ex)
        save_ledger(concept_dir, ledger)
        raise
    finally:
        rows = [
            {
                "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(commit["at"])),
                "commit_id": commit["id"],
                "chosen": it["rel"],
                "backup": it["backup"],
                "strength": it["strength"],
                "median": it["stats"]["median"],
                "mean": it["stats"]["mean"],
                "bright_fraction": it["stats"]["bright_fraction"],
                "delta_luma": it["delta_luma"],
                "minor_edits_dir": (it["minor_pair"] or {}).get("dir", ""),
                "minor_edits_pair_id": (it["minor_pair"] or {}).get("pair_id", ""),
            }
            for it in commit["items"]
        ]
        if rows:
            append_backup_csv(concept_dir, rows)
        save_ledger(concept_dir, ledger)
    return commit


def restore_commit(concept_dir: str, ledger: dict, commit_id: str) -> dict:
    """Undo one commit: backups return over the darkened files, manifest
    entries get their old fields and history back, Minor Edits pairs go.
    Items whose backup is missing are reported, not silently skipped."""
    concept_dir = os.path.abspath(concept_dir)
    commit = next((c for c in ledger.get("commits", []) if c.get("id") == commit_id), None)
    if commit is None:
        raise KeyError(commit_id)
    if commit.get("restored"):
        return {"restored": 0, "pairs_removed": 0, "missing": []}
    manifest = load_manifest(concept_dir)
    restored = 0
    missing: list[str] = []
    pairs_removed = 0
    minor_manifest = None
    for record in commit.get("items", []):
        backup = absolute(concept_dir, record.get("backup"))
        target = absolute(concept_dir, record.get("rel"))
        if not backup or not os.path.isfile(backup) or not target:
            missing.append(record.get("rel", "?"))
            continue
        if record.get("minor_pair"):
            minor_dir = record["minor_pair"]["dir"]
            if minor_manifest is None:
                minor_manifest = load_manifest(minor_dir)
            remove_night_pair(minor_dir, minor_manifest, record["minor_pair"])
            pairs_removed += 1
            record["minor_pair_removed"] = True
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(backup, target)
        entry = _manifest_entry_for(manifest, record["stem"])
        if entry is not None and record.get("manifest_before") is not None:
            for key in _ENTRY_KEYS:
                if key in record["manifest_before"]:
                    entry[key] = record["manifest_before"][key]
                else:
                    entry.pop(key, None)
            history = entry.get("history") or []
            entry["history"] = history[: int(record.get("history_len") or 0)]
            if not entry["history"]:
                entry.pop("history", None)
        decision = ledger.get("decisions", {}).get(record["rel"])
        if decision is not None:
            decision.pop("committed", None)
        restored += 1
    if restored:
        save_manifest(concept_dir, manifest)
    commit["restored"] = not missing
    commit["restored_at"] = time.time()
    if missing:
        commit["restore_missing"] = missing
    save_ledger(concept_dir, ledger)
    return {"restored": restored, "pairs_removed": pairs_removed, "missing": missing}
