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
   insensitive) AND says both "night" and "dark" (:data:`REQUIRED_GROUPS`:
   one of :data:`NIGHT_TERMS` and one of :data:`DARK_TERMS`; "dark hair",
   "dark brown skin" don't count as dark) gets a luma statistic. Images that are already genuinely
   dark (:func:`is_already_dark`, thresholds in :class:`DarkThresholds`) are
   counted and skipped; the rest are the candidates. Minor Edits and
   Off-policy datasets are never sources — the user picks the source folder
   and the tool refuses a folder whose ``concepts.json`` / ``workspace.json``
   registration says it is one of those.
2. **Review**, one candidate at a time: a before/after wipe over the image
   and four sliders that re-render the adjusted side — darkening strength,
   noise, hue shift and saturation — then *Accept* (keeps the current
   settings), *Skip*, *Back*. Decisions live in the ledger at
   ``<concept>/.night_darken/ledger.json`` so a closed modal resumes.
3. **Commit**: every accepted image is re-rendered at full size with its
   accepted settings and replaced in place (same file name and format, PNG
   text chunks / EXIF / ICC kept). The original goes to
   ``<concept>/.night_darken/backup/<split>/<name>``, listed in
   ``<concept>/.night_darken/backup/manifest.csv`` (path, strength, noise,
   hue, saturation, stats, change). The most changed
   :data:`DEFAULT_MINOR_EDITS_FRACTION` of the accepted set (ranked by the
   mean CIELAB ΔE between original and result, so every adjustment counts) is
   also written as new pairs into the **Night Edits** dataset (default
   ``<workspace>/DPO Night Edits`` beside the source concept, created on the
   first commit with the Minor Edits layout: ``chosen/train`` +
   ``rejected/train``, ``pair_NNNN`` from 0001, its own ``manifest.json`` via
   ``load_manifest`` / ``save_manifest`` and a ``concepts.json``): adjusted =
   chosen, original = rejected, the caption copied byte for byte, and an
   all-white ``-masklabel.png`` beside both sides so the whole image counts
   under masked DPO. The destination is editable but a folder the Optimise
   workspace already registers (Minor Edits, Off-policy, RLHF_8, ...) is
   refused, so the tool never collides with Optimise's pair numbering; the
   new dataset can later be imported into Optimise as its own dataset.
4. **Restore** reverses a commit: backups move back, the concept manifest
   entries return to what they were, the Night Edits pairs (files, masks,
   captions, manifest entries) are removed.

Adjustments (:func:`apply_adjustments`, in this order)
-------------------------------------------------------
1. **Darken** (:func:`darken_rgb`), in linear light. Each pixel's linear
   luminance ``Y`` is pulled down by an exposure drop of
   ``EXPOSURE_STOPS * strength`` stops and a gamma of
   ``1 + GAMMA_LIFT * strength`` (shadows fall faster than highlights, so
   light sources keep reading as lights), and the RGB channels are scaled by
   the resulting ``Y_out / Y`` so hue and saturation stay put. Clipped
   highlights (``Y`` above :data:`HIGHLIGHT_KNEE`) keep :data:`HIGHLIGHT_KEEP`
   of their original luminance so lamps, screens and specular hits survive.
   Strength 0 is the identity.
2. **Hue / saturation** (:func:`hue_saturation_rgb`), on gamma-encoded RGB
   as Rec.709 Y'CbCr: the chroma vector ``(Cb, Cr)`` is rotated by the hue
   angle and scaled by the saturation multiplier, luma is untouched, the
   result is clipped. Hue 0 and saturation 1 are the identity.
3. **Noise** (:func:`add_noise_rgb`), sensor-like, last. With ``a = noise *
   NOISE_MAX_SIGMA`` and the pixel's luma ``Y'`` (0..1), the per-pixel
   amplitude is ``a * (NOISE_FLOOR + (1 - NOISE_FLOOR) * (1 - Y'))``: full in
   the shadows, ``NOISE_FLOOR`` of it in the highlights. A luminance draw
   (one Gaussian shared by the three channels) plus a milder chroma draw
   (independent Gaussians per channel, ``NOISE_CHROMA`` of the amplitude) is
   added and clipped. The generator is seeded from the image's file stem, so
   a given image always gets the same noise field at a given size; the
   preview renders it at preview size, the commit at full size.

Change metric: :func:`mean_delta_e`, the mean CIE76 ΔE between the original
and the result in CIELAB (D65), computed at preview size when accepting and
at full size on commit.

Night presets (:data:`NIGHT_PRESETS`: Dusk, Urban night, Moonlit night)
-------------------------------------------------------------------------
Suggested looks given as a brightness multiplier, a saturation multiplier,
a hue and a noise sigma, mapped onto the sliders without new maths: the
darken strength is solved by bisection so a mid-grey sRGB 118 pixel ends at
``118 * brightness`` (:func:`solve_strength_for_brightness`), saturation and
hue are taken as they are, and the noise sigma becomes the slider value
whose per-channel sigma equals it at the darkened mid-grey level
(:func:`noise_slider_for_sigma`). Applying a preset to an image jitters
strength, saturation and noise by ±10 % with a generator seeded from the
file stem and the preset (:func:`jittered_preset`), so no two images get
exactly the same darkness; the preset name travels in the adjust dict, the
ledger, the manifest history and the backup CSV. "Darken more"
(:func:`next_preset`) steps none → Dusk → Urban → Moonlit.
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

#: A caption qualifies only when it matches at least one phrase of EVERY
#: group (the user's rule: it must say both "night" and "dark"; "dark brown
#: skin" alone, or "at night" alone, is not enough). Phrases outside the
#: groups are still reported but never qualify a caption on their own.
NIGHT_TERMS: tuple[str, ...] = ("night", "nighttime", "night time", "at night", "late night", "midnight")
DARK_TERMS: tuple[str, ...] = ("dark", "darkness", "darkened")
REQUIRED_GROUPS: tuple[tuple[str, ...], ...] = (NIGHT_TERMS, DARK_TERMS)

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

#: Share of the accepted images (most changed first) that also become Night
#: Edits pairs.
DEFAULT_MINOR_EDITS_FRACTION = 0.33
#: Default accept strength.
DEFAULT_STRENGTH = 0.5
#: Noise at slider 1: sigma in 0..1 units for a black pixel (about 31/255).
NOISE_MAX_SIGMA = 0.12
#: Share of the shadow amplitude that a white pixel still gets.
NOISE_FLOOR = 0.25
#: Chroma noise amplitude relative to the luminance noise.
NOISE_CHROMA = 0.35
#: Hue slider range in degrees.
HUE_RANGE = 180.0
#: Saturation slider range (multiplier).
SATURATION_MAX = 2.0

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
#: Default destination dataset for the new pairs, beside the source concept.
NIGHT_EDITS_DIR_NAME = "DPO Night Edits"
#: The destination's layout: ``chosen/train`` + ``rejected/train``, like Minor Edits.
NIGHT_EDITS_SPLIT = "train"
#: Pair numbering in a fresh destination starts here.
NIGHT_EDITS_FIRST_PAIR_ID = 1
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
    "noise",
    "hue",
    "saturation",
    "median",
    "mean",
    "bright_fraction",
    "delta_e",
    "delta_luma",
    "night_edits_dir",
    "night_edits_pair_id",
    "preset",
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


def qualifies(found: list[str], groups: tuple[tuple[str, ...], ...] = REQUIRED_GROUPS) -> bool:
    """Whether the matched phrases cover every required group."""
    lowered = {f.lower() for f in found}
    return all(lowered & {g.lower() for g in group} for group in groups)


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
# Hue / saturation, noise, the combined adjustment and its change metric
# --------------------------------------------------------------------------- #

# Rec.709 Y'CbCr on gamma-encoded values.
_CB_SCALE = 1.8556
_CR_SCALE = 1.5748


def hue_saturation_rgb(rgb: np.ndarray, hue_degrees: float, saturation: float) -> np.ndarray:
    """Rotate the chroma of an ``[H, W, 3|4]`` uint8 image by ``hue_degrees``
    and scale it by ``saturation``; luma is untouched. Neutral (0, 1) returns
    an identical copy."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        raise TypeError("hue_saturation_rgb expects a uint8 array")
    hue = float(hue_degrees)
    sat = float(max(saturation, 0.0))
    if abs(hue) < 1e-9 and abs(sat - 1.0) < 1e-9:
        return rgb.copy()
    alpha = rgb[..., 3:4] if rgb.shape[-1] == 4 else None
    x = rgb[..., :3].astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / _CB_SCALE
    cr = (r - y) / _CR_SCALE
    theta = np.deg2rad(hue)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    cb2 = sat * (cb * cos_t - cr * sin_t)
    cr2 = sat * (cb * sin_t + cr * cos_t)
    r2 = y + _CR_SCALE * cr2
    b2 = y + _CB_SCALE * cb2
    g2 = (y - 0.2126 * r2 - 0.0722 * b2) / 0.7152
    out = np.rint(np.clip(np.stack([r2, g2, b2], axis=-1), 0.0, 1.0) * 255.0).astype(np.uint8)
    if alpha is not None:
        out = np.concatenate([out, alpha], axis=-1)
    return out


def noise_seed(name: str) -> int:
    """A stable 63-bit seed from an image's file stem (any path form)."""
    import hashlib

    stem = os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0]
    return int.from_bytes(hashlib.sha256(stem.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def add_noise_rgb(rgb: np.ndarray, noise: float, seed: int) -> np.ndarray:
    """Sensor-style noise on an ``[H, W, 3|4]`` uint8 image: luminance noise
    shared by the channels plus milder per-channel chroma noise, both scaled
    by a shadow weight so dark pixels get the most. ``noise`` 0 returns an
    identical copy; the same ``seed`` and shape give the same field."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        raise TypeError("add_noise_rgb expects a uint8 array")
    n = float(min(max(noise, 0.0), 1.0))
    if n <= 0.0:
        return rgb.copy()
    alpha = rgb[..., 3:4] if rgb.shape[-1] == 4 else None
    x = rgb[..., :3].astype(np.float32) / 255.0
    luma = x @ _LUMA_WEIGHTS
    amplitude = (n * NOISE_MAX_SIGMA) * (NOISE_FLOOR + (1.0 - NOISE_FLOOR) * (1.0 - luma))
    rng = np.random.default_rng(int(seed))
    shape = x.shape[:2]
    luminance = rng.standard_normal(shape, dtype=np.float32)
    chroma = rng.standard_normal((*shape, 3), dtype=np.float32)
    delta = amplitude[..., None] * (luminance[..., None] + NOISE_CHROMA * chroma)
    out = np.rint(np.clip(x + delta, 0.0, 1.0) * 255.0).astype(np.uint8)
    if alpha is not None:
        out = np.concatenate([out, alpha], axis=-1)
    return out


@dataclass
class Adjust:
    """The four review sliders; the defaults are the identity. ``preset`` is
    the name of the night preset the values came from ("" when set by hand)."""

    strength: float = 0.0
    noise: float = 0.0
    hue: float = 0.0
    saturation: float = 1.0
    preset: str = ""

    @classmethod
    def from_dict(cls, raw: dict | None) -> Adjust:
        raw = raw or {}
        out = cls()
        for key in ("strength", "noise", "hue", "saturation"):
            value = raw.get(key)
            if value is None:
                continue
            with suppress(TypeError, ValueError):
                setattr(out, key, float(value))
        out.strength = min(max(out.strength, 0.0), 1.0)
        out.noise = min(max(out.noise, 0.0), 1.0)
        out.hue = min(max(out.hue, -HUE_RANGE), HUE_RANGE)
        out.saturation = min(max(out.saturation, 0.0), SATURATION_MAX)
        preset = raw.get("preset")
        out.preset = str(preset) if isinstance(preset, str) and preset in {p["id"] for p in NIGHT_PRESETS} else ""
        return out

    def to_dict(self) -> dict:
        out = {k: round(float(v), 4) for k, v in asdict(self).items() if k != "preset"}
        out["preset"] = self.preset
        return out

    def is_neutral(self) -> bool:
        return self.strength <= 0 and self.noise <= 0 and abs(self.hue) < 1e-9 and abs(self.saturation - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Night presets: suggested night looks mapped onto the sliders
# --------------------------------------------------------------------------- #

#: The suggested settings, in their own terms: ``brightness`` a multiplier on
#: the sRGB pixel value, ``saturation`` a multiplier, ``hue`` degrees,
#: ``noise_sigma`` a Gaussian sigma on the 0..1 pixel scale. Weakest first;
#: "Darken more" walks this order.
NIGHT_PRESETS: tuple[dict, ...] = (
    {"id": "dusk", "label": "Dusk", "brightness": 0.65, "saturation": 0.85, "hue": 0.0, "noise_sigma": 0.008},
    {"id": "urban", "label": "Urban night", "brightness": 0.45, "saturation": 0.90, "hue": 0.0, "noise_sigma": 0.025},
    {
        "id": "moonlit",
        "label": "Moonlit night",
        "brightness": 0.30,
        "saturation": 0.65,
        "hue": 0.0,
        "noise_sigma": 0.015,
    },
)
#: The sRGB level the brightness multiplier is matched at (mid-grey).
PRESET_REFERENCE_SRGB = 118
#: Per-image jitter on strength, saturation and noise when a preset is applied.
PRESET_JITTER = 0.10


def _srgb_value_to_linear(value: float) -> float:
    x = value / 255.0
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


def solve_strength_for_brightness(brightness: float, reference: int = PRESET_REFERENCE_SRGB) -> float:
    """The darken strength at which the sRGB ``reference`` grey ends at
    ``reference * brightness``, by bisection on :func:`darken_luminance`
    (monotone in the strength). The curve constants stay the only source of
    truth. 0 for a brightness of 1 or more; 1 when the target is below what
    full strength reaches."""
    if brightness >= 1.0:
        return 0.0
    y0 = _srgb_value_to_linear(float(reference))
    target = _srgb_value_to_linear(float(reference) * brightness)
    lo, hi = 0.0, 1.0
    if float(darken_luminance(np.array([y0], dtype=np.float64), 1.0)[0]) > target:
        return 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if float(darken_luminance(np.array([y0], dtype=np.float64), mid)[0]) > target:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def noise_slider_for_sigma(sigma: float, brightness: float, reference: int = PRESET_REFERENCE_SRGB) -> float:
    """The noise slider value whose per-channel sigma equals ``sigma`` (0..1
    scale) at the reference grey's post-darken level. The level is
    ``Y' = brightness * reference / 255``, where the amplitude is
    ``slider * NOISE_MAX_SIGMA * (NOISE_FLOOR + (1 - NOISE_FLOOR) * (1 - Y'))``,
    and a channel's sigma is that amplitude times ``sqrt(1 + NOISE_CHROMA^2)``
    (the shared luminance draw plus its own chroma draw). Matching at the
    darkened midtone rather than at black keeps the typical pixel at the
    suggested sigma; black pixels get a little more, highlights less."""
    if sigma <= 0:
        return 0.0
    level = min(max(brightness * reference / 255.0, 0.0), 1.0)
    weight = NOISE_FLOOR + (1.0 - NOISE_FLOOR) * (1.0 - level)
    amplitude = sigma / float(np.sqrt(1.0 + NOISE_CHROMA**2))
    return round(min(max(amplitude / (NOISE_MAX_SIGMA * weight), 0.0), 1.0), 4)


_PRESET_CACHE: dict[str, Adjust] = {}


def preset_adjust(preset_id: str) -> Adjust:
    """The un-jittered slider values of a preset (cached after the first solve)."""
    cached = _PRESET_CACHE.get(preset_id)
    if cached is not None:
        return Adjust(**asdict(cached))
    preset = next((p for p in NIGHT_PRESETS if p["id"] == preset_id), None)
    if preset is None:
        raise KeyError(preset_id)
    out = Adjust(
        strength=solve_strength_for_brightness(preset["brightness"]),
        noise=noise_slider_for_sigma(preset["noise_sigma"], preset["brightness"]),
        hue=float(preset["hue"]),
        saturation=float(preset["saturation"]),
        preset=preset_id,
    )
    _PRESET_CACHE[preset_id] = out
    return Adjust(**asdict(out))


def preset_table() -> list[dict]:
    """Every preset with its suggested values and the solved slider values."""
    return [{**p, "adjust": preset_adjust(p["id"]).to_dict()} for p in NIGHT_PRESETS]


def jittered_preset(preset_id: str, name: str, jitter: float = PRESET_JITTER) -> Adjust:
    """A preset for one image: strength, saturation and noise each scaled by
    ``1 + U(-jitter, +jitter)`` drawn from a generator seeded by the image's
    file stem and the preset id, so the same image always gets the same
    values and a dataset never learns one exact darkness. Clamped to the
    slider ranges; hue is left as the preset has it."""
    base = preset_adjust(preset_id)
    rng = np.random.default_rng((noise_seed(name) ^ noise_seed(f"preset:{preset_id}")) & 0x7FFFFFFFFFFFFFFF)
    factors = 1.0 + rng.uniform(-jitter, jitter, size=3)
    return Adjust.from_dict(
        {
            "strength": base.strength * factors[0],
            "saturation": base.saturation * factors[1],
            "noise": base.noise * factors[2],
            "hue": base.hue,
            "preset": preset_id,
        }
    )


def next_preset(current: Adjust | dict | None) -> str:
    """ "Darken more": the next stronger preset after the current settings —
    the one after the recorded preset, else the first preset darker than the
    current strength; the strongest stays the strongest."""
    current = current if isinstance(current, Adjust) else Adjust.from_dict(current)
    order = [p["id"] for p in NIGHT_PRESETS]
    if current.preset in order:
        return order[min(order.index(current.preset) + 1, len(order) - 1)]
    for preset_id in order:
        if preset_adjust(preset_id).strength > current.strength + 1e-3:
            return preset_id
    return order[-1]


def apply_adjustments(rgb: np.ndarray, adjust: Adjust | dict | None, seed: int | str) -> np.ndarray:
    """Darken, then hue/saturation, then noise (see the module docstring).
    ``seed`` is an int or the image name :func:`noise_seed` hashes."""
    adjust = adjust if isinstance(adjust, Adjust) else Adjust.from_dict(adjust)
    seed_value = noise_seed(seed) if isinstance(seed, str) else int(seed)
    out = darken_rgb(rgb, adjust.strength)
    out = hue_saturation_rgb(out, adjust.hue, adjust.saturation)
    return add_noise_rgb(out, adjust.noise, seed_value)


_RGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float32,
)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of an ``[..., 3]`` uint8 sRGB array, float32."""
    lin = _SRGB_TO_LINEAR_LUT[np.asarray(rgb)[..., :3]]
    xyz = (lin @ _RGB_TO_XYZ.T) / _D65
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def mean_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean CIE76 ΔE between two uint8 images of the same size."""
    return float(np.linalg.norm(rgb_to_lab(a) - rgb_to_lab(b), axis=-1).mean())


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


def save_adjusted_like(
    original_path: str, adjust: Adjust | dict | None, target_path: str, seed: int | str | None = None
) -> dict:
    """Write the adjusted original to ``target_path`` in the original's format,
    carrying its PNG text chunks (SwarmUI parameters, Triage & Fix provenance),
    EXIF, ICC profile and XMP across. Alpha is kept as is. Returns the change
    (``delta_e``, ``delta_luma``) and the original's stats. The noise seed
    defaults to the original's file stem.

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

    darkened = apply_adjustments(pixels, adjust, seed if seed is not None else original_path)
    delta_e = mean_delta_e(pixels[..., :3], darkened[..., :3])
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
    return {"delta_e": round(delta_e, 4), "delta_luma": round(delta, 4), "stats": stats.to_dict(), "format": fmt}


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


def workspace_registration(folder: str) -> dict | None:
    """The Optimise ``workspace.json`` entry registering ``folder`` as a
    dataset (looked up in the folder's parent), or ``None``."""
    folder = os.path.abspath(folder)
    workspace = os.path.join(os.path.dirname(folder), "workspace.json")
    if not os.path.isfile(workspace):
        return None
    try:
        with open(workspace, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    for dataset in data.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        registered = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(workspace), dataset.get("dir", ""))))
        if registered == os.path.normcase(folder):
            return dataset
    return None


def source_exclusion_reason(concept_dir: str) -> str | None:
    """Why a folder may not be a source: it is a Minor Edits or Off-policy
    dataset (by its Optimise workspace registration, else by its name)."""
    concept_dir = os.path.abspath(concept_dir)
    dataset = workspace_registration(concept_dir)
    if dataset is not None and dataset.get("type") in EXCLUDED_WORKSPACE_TYPES:
        return f"{os.path.basename(concept_dir)} is registered as a {dataset.get('type')} dataset"
    lowered = os.path.basename(concept_dir).lower()
    for word in EXCLUDED_FOLDER_WORDS:
        if word in lowered:
            return f"{os.path.basename(concept_dir)} looks like a {word} dataset"
    return None


def destination_exclusion_reason(night_dir: str) -> str | None:
    """Why a folder may not take the new pairs: it is a dataset Optimise
    already owns (any ``workspace.json`` registration: Minor Edits,
    Off-policy, RLHF_8, ...), or it is named like one. New pairs go into
    their own dataset so they never collide with Optimise's numbering."""
    night_dir = os.path.abspath(night_dir)
    dataset = workspace_registration(night_dir)
    if dataset is not None:
        return (
            f"{os.path.basename(night_dir)} is registered in the Optimise workspace as a {dataset.get('type')} dataset"
        )
    lowered = os.path.basename(night_dir).lower()
    for word in EXCLUDED_FOLDER_WORDS:
        if word in lowered:
            return f"{os.path.basename(night_dir)} looks like a {word} dataset"
    return None


def default_night_edits_dir(concept_dir: str) -> str:
    """``<workspace>/DPO Night Edits`` next to the source concept."""
    return os.path.join(os.path.dirname(os.path.abspath(concept_dir)), NIGHT_EDITS_DIR_NAME)


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
    groups: tuple[tuple[str, ...], ...] = REQUIRED_GROUPS,
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
    # the required words always count, whatever phrase list the UI sends
    patterns = compile_phrases(tuple(phrases) + tuple(t for g in groups for t in g))
    partial = 0
    images = iter_chosen_images(concept_dir)
    matched = []
    for item in images:
        found = match_night_phrases(read_caption(item["caption_path"]), patterns)
        if found and qualifies(found, groups):
            matched.append((item, found))
        elif found:
            partial += 1
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
            "partial_matches": partial,  # matched a phrase but not both "night" and "dark"
            "already_dark": len(already_dark),
            "unreadable": len(unreadable),
            "candidates": len(candidates),
        },
    }


# --------------------------------------------------------------------------- #
# Night Edits pairs (their own dataset, never an Optimise-owned one)
# --------------------------------------------------------------------------- #


def _side_dir(night_dir: str, side: str) -> str:
    return os.path.join(night_dir, side, NIGHT_EDITS_SPLIT)


def _disk_pair_numbers(night_dir: str) -> set[int]:
    numbers: set[int] = set()
    for side in ("chosen", "rejected"):
        base = os.path.join(night_dir, side)
        if not os.path.isdir(base):
            continue
        for _root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                match = re.match(r"pair_(\d+)", name)
                if match:
                    numbers.add(int(match.group(1)))
    return numbers


def next_night_pair_id(manifest: dict) -> int:
    return max(_next_pair_id(manifest), NIGHT_EDITS_FIRST_PAIR_ID)


def night_edits_preflight(night_dir: str, count: int) -> str | None:
    """Why ``count`` new pairs cannot be written: no folder, an Optimise-owned
    folder, a non-dataset folder, or the next ids the destination's manifest
    would hand out already exist on disk (pairs its manifest does not know)."""
    if count <= 0:
        return None
    if not night_dir:
        return "No Night Edits folder set"
    reason = destination_exclusion_reason(night_dir)
    if reason:
        return reason
    if os.path.isdir(night_dir) and not os.path.isfile(os.path.join(night_dir, "manifest.json")):
        entries = [e for e in os.listdir(night_dir) if not e.startswith(".")]
        if entries and not {"chosen", "rejected"} & set(entries):
            return "The Night Edits folder is not empty and is not a DPO pair folder (no manifest.json)"
    manifest = load_manifest(night_dir) if os.path.isdir(night_dir) else {"pairs": []}
    first = next_night_pair_id(manifest)
    taken = _disk_pair_numbers(night_dir) & set(range(first, first + count))
    if taken:
        return f"pair numbers {sorted(taken)[:5]} already exist on disk but not in manifest.json"
    return None


def ensure_night_edits_dataset(night_dir: str) -> None:
    """Create the destination layout (``chosen/train``, ``rejected/train``) and,
    for a brand-new folder, a ``concepts.json`` in the Minor Edits shape so
    the dataset drops straight into a concept list."""
    os.makedirs(_side_dir(night_dir, "chosen"), exist_ok=True)
    os.makedirs(_side_dir(night_dir, "rejected"), exist_ok=True)
    os.makedirs(os.path.join(night_dir, "chosen", "val"), exist_ok=True)
    os.makedirs(os.path.join(night_dir, "rejected", "val"), exist_ok=True)
    concepts = os.path.join(night_dir, "concepts.json")
    if not os.path.isfile(concepts):
        payload = [
            {
                "name": "DPO train",
                "path": os.path.abspath(night_dir),
                "type": "STANDARD",
                "enabled": True,
                "include_subdirectories": False,
                "dpo_chosen_pattern": "chosen/train/{}",
                "dpo_rejected_pattern": "rejected/train/{}",
            },
            {
                "name": "DPO val",
                "path": os.path.abspath(night_dir),
                "type": "VALIDATION",
                "enabled": True,
                "include_subdirectories": False,
                "dpo_chosen_pattern": "chosen/val/{}",
                "dpo_rejected_pattern": "rejected/val/{}",
            },
        ]
        with open(concepts, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def export_night_pair(
    night_dir: str,
    manifest: dict,
    *,
    adjusted_path: str,
    original_path: str,
    caption_path: str | None,
    lineage: dict,
) -> dict:
    """One Night Edits pair: the adjusted image as chosen, the original as
    rejected, the caption copied byte for byte, an all-white mask beside both
    sides. Numbering and the manifest entry follow ``export_single_pair``,
    starting at ``pair_0001`` in a fresh dataset; files live in ``<side>/train``."""
    ensure_night_edits_dataset(night_dir)
    chosen_dir = _side_dir(night_dir, "chosen")
    rejected_dir = _side_dir(night_dir, "rejected")

    pair_id = next_night_pair_id(manifest)
    stem = f"pair_{pair_id:04d}"
    chosen_ext = os.path.splitext(adjusted_path)[1].lower()
    rejected_ext = os.path.splitext(original_path)[1].lower()
    chosen_target = os.path.join(chosen_dir, stem + chosen_ext)
    rejected_target = os.path.join(rejected_dir, stem + rejected_ext)
    for target in (chosen_target, rejected_target):
        if os.path.exists(target):
            raise FileExistsError(target)

    written: list[str] = []
    try:
        shutil.copy2(adjusted_path, chosen_target)
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

    prefix = f"{NIGHT_EDITS_SPLIT}/"
    entry = {
        "pair_id": pair_id,
        "prompt": read_caption(caption_path),
        "aspectratio": resolve_aspect_ratio("", chosen_target),
        "split": NIGHT_EDITS_SPLIT,
        "chosen_file": prefix + os.path.basename(chosen_target),
        "rejected_file": prefix + os.path.basename(rejected_target),
        "caption_file": (prefix + stem + ".txt") if caption_target else None,
        "chosen_source": _normalize_source_path(adjusted_path),
        "rejected_source": _normalize_source_path(original_path),
        "chosen_sha256": compute_file_sha256(chosen_target),
        "rejected_sha256": compute_file_sha256(rejected_target),
        "chosen_size": os.path.getsize(chosen_target),
        "rejected_size": os.path.getsize(rejected_target),
        "tool": HISTORY_ACTION,
        "lineage": lineage,
    }
    manifest.setdefault("pairs", []).append(entry)
    save_manifest(night_dir, manifest)
    return {
        "dir": os.path.abspath(night_dir),
        "pair_id": pair_id,
        "files": [rel(night_dir, p) for p in written],
    }


def remove_night_pair(night_dir: str, manifest: dict, record: dict) -> None:
    """Reverse :func:`export_night_pair`: its files and manifest entry go."""
    for name in record.get("files") or []:
        path = absolute(night_dir, name)
        with suppress(OSError):
            if path:
                os.remove(path)
    pairs = manifest.get("pairs", [])
    manifest["pairs"] = [p for p in pairs if p.get("pair_id") != record.get("pair_id")]
    save_manifest(night_dir, manifest)


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


def night_edits_count(total: int, fraction: float) -> int:
    """How many of ``total`` accepted images become pairs: the rounded share,
    at least one when the share is positive."""
    if total <= 0 or fraction <= 0:
        return 0
    return max(1, int(round(total * min(fraction, 1.0))))


def select_night_edits(items: list[dict], fraction: float) -> list[dict]:
    """The most changed share of ``items`` by ``delta_e`` (mean CIELAB ΔE)."""
    count = night_edits_count(len(items), fraction)
    ranked = sorted(items, key=lambda it: (-float(it.get("delta_e") or 0.0), it["rel"]))
    return ranked[:count]


def decision_adjust(decision: dict) -> Adjust:
    """The accepted settings of a decision (older ledgers hold only ``strength``)."""
    raw = dict(decision.get("adjust") or {})
    if "strength" not in raw and decision.get("strength") is not None:
        raw["strength"] = decision["strength"]
    return Adjust.from_dict(raw)


def commit_accepted(
    concept_dir: str,
    ledger: dict,
    *,
    night_edits_dir: str | None,
    fraction: float = DEFAULT_MINOR_EDITS_FRACTION,
    progress=None,
    stop=None,
) -> dict:
    """Adjust every accepted, uncommitted candidate in place (backed up), then
    add the most changed share as Night Edits pairs. Returns the commit
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
    pair_count = night_edits_count(len(pending), fraction)
    if pair_count and not night_edits_dir:
        raise RuntimeError("Night Edits: no destination folder set (set the fraction to 0 to skip the pairs)")
    if pair_count:
        problem = night_edits_preflight(night_edits_dir, pair_count)
        if problem:
            raise RuntimeError(f"Night Edits: {problem}")

    commit = {
        "id": f"{_timestamp()}_{len(ledger.get('commits', [])) + 1}",
        "at": time.time(),
        "night_edits_dir": os.path.abspath(night_edits_dir) if night_edits_dir else None,
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
            adjust = decision_adjust(decision)
            chosen_path = item["path"]
            backup = backup_original(concept_dir, chosen_path)
            try:
                result = save_adjusted_like(backup, adjust, chosen_path, seed=item["stem"])
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
                        "adjust": adjust.to_dict(),
                        "retired": [rel(concept_dir, backup)],
                    }
                )
            record = {
                "rel": r,
                "stem": item["stem"],
                "split": item["split"],
                "adjust": adjust.to_dict(),
                "strength": adjust.strength,
                "backup": rel(concept_dir, backup),
                "delta_e": result["delta_e"],
                "delta_luma": result["delta_luma"],
                "stats": result["stats"],
                "format": result["format"],
                "caption": rel(concept_dir, item["caption_path"]) if item["caption_path"] else None,
                "manifest_before": before,
                "history_len": history_len,
                "night_pair": None,
            }
            commit["items"].append(record)
            decision["committed"] = commit["id"]
            save_ledger(concept_dir, ledger)
            if progress is not None:
                progress(done, len(pending), "adjust")
        if any(it["manifest_before"] is not None for it in commit["items"]):
            save_manifest(concept_dir, manifest)

        chosen = select_night_edits(commit["items"], fraction) if commit["night_edits_dir"] else []
        if chosen:
            ensure_night_edits_dataset(commit["night_edits_dir"])
            night_manifest = load_manifest(commit["night_edits_dir"])
            for done, record in enumerate(chosen, start=1):
                if stop is not None and stop.is_set():
                    break
                item = by_path_rel[record["rel"]]
                record["night_pair"] = export_night_pair(
                    commit["night_edits_dir"],
                    night_manifest,
                    adjusted_path=item["path"],
                    original_path=absolute(concept_dir, record["backup"]),
                    caption_path=item["caption_path"],
                    lineage={
                        "tool": HISTORY_ACTION,
                        "source_concept": concept_dir,
                        "source_chosen": record["rel"],
                        "adjust": record["adjust"],
                        "delta_e": record["delta_e"],
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
                "strength": it["adjust"]["strength"],
                "noise": it["adjust"]["noise"],
                "hue": it["adjust"]["hue"],
                "saturation": it["adjust"]["saturation"],
                "median": it["stats"]["median"],
                "mean": it["stats"]["mean"],
                "bright_fraction": it["stats"]["bright_fraction"],
                "delta_e": it["delta_e"],
                "delta_luma": it["delta_luma"],
                "night_edits_dir": (it["night_pair"] or {}).get("dir", ""),
                "night_edits_pair_id": (it["night_pair"] or {}).get("pair_id", ""),
                "preset": it["adjust"].get("preset", ""),
            }
            for it in commit["items"]
        ]
        if rows:
            append_backup_csv(concept_dir, rows)
        save_ledger(concept_dir, ledger)
    return commit


def restore_commit(concept_dir: str, ledger: dict, commit_id: str) -> dict:
    """Undo one commit: backups return over the adjusted files, manifest
    entries get their old fields and history back, Night Edits pairs go.
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
    night_manifest = None
    for record in commit.get("items", []):
        backup = absolute(concept_dir, record.get("backup"))
        target = absolute(concept_dir, record.get("rel"))
        if not backup or not os.path.isfile(backup) or not target:
            missing.append(record.get("rel", "?"))
            continue
        if record.get("night_pair"):
            night_dir = record["night_pair"]["dir"]
            if night_manifest is None:
                night_manifest = load_manifest(night_dir)
            remove_night_pair(night_dir, night_manifest, record["night_pair"])
            pairs_removed += 1
            record["night_pair_removed"] = True
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
