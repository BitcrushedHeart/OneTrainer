"""Prompt adherence for DPO curation: does this image show what its prompt says?

The "Bitcrush Score" experiment. Every image in a curation group shares one
prompt, so adherence never needs to be calibrated in absolute terms — it only
needs to *rank* the group and say *which part* of the prompt an image missed.
That is the same contract the realism score has (ordering is the product).

How it works, and why it is shaped this way:

1. **The prompt is decomposed into propositions**, DSG/VQAScore-style. The
   SoReal! caption preamble is already a list of atomic claims ("low angle",
   "side view", "medium full shot", "heavy grain", ...), and each of those maps
   onto a Bitcrush Engine group with a closed class list. Those become *axis
   propositions* and are scored against their siblings: "low angle" is scored
   against "high angle" and "eye level", not against nothing. The rest of the
   caption is chopped into short *free phrases* and scored on their own.

2. **A SigLIP 2 bi-encoder answers each proposition.** A bi-encoder is a poor
   judge of a whole 50-word caption (bag-of-words, 64-token truncation) but a
   good judge of a five-word phrase, which is exactly what the decomposition
   hands it. Axis propositions use a softmax over sibling logits (SigLIP's bias
   cancels); free phrases use SigLIP's own sigmoid, which its training
   calibrated as P(match).

3. **Image embeddings are cached** in the curation cache, keyed on the model,
   so scoring a group against its prompt is a handful of dot products after the
   first pass, and re-scoring with a different proposition set is free.

The model half is injected (``embedder``) so the pure half is testable without
weights; ``SiglipEmbedder`` is the real one.
"""

from __future__ import annotations

import base64
import contextlib
import gc
import math
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from modules.util.caption_style import is_tag_line, normalize_caption

import numpy as np

DEFAULT_MODEL_NAME = "google/siglip2-so400m-patch16-naflex"
#: NaFlex patch budget per image. 784 x 16px patches is ~448^2 of pixels — enough
#: for hair colour and framing, ~3x cheaper than the 1024 the model allows.
DEFAULT_MAX_PATCHES = 784
#: SigLIP 2 text towers are trained at 64 tokens; longer input is truncated.
TEXT_MAX_LENGTH = 64

#: Free-phrase length bounds, in words. Below the minimum a chunk is a leftover
#: ("of a"); above the maximum a bi-encoder starts to average the phrase away.
MIN_PHRASE_WORDS = 2
MAX_PHRASE_WORDS = 12

ADHERENCE_DEVICES = ("auto", "cpu", "cuda")
#: Free VRAM below which "auto" stays on the CPU. The SO400M model is ~0.8 GB in
#: fp16 plus activations; this leaves a generation job sharing the card alone.
_AUTO_CUDA_MIN_FREE_BYTES = 2_500_000_000


# --------------------------------------------------------------------------- #
# Vocabulary — mirrors the Bitcrush Engine group class lists
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AxisClass:
    label: str
    #: Case-insensitive regex that finds this class in a caption; ``None`` for a
    #: sibling that is only ever scored against, never matched (e.g. "a photo
    #: taken by someone else" on the selfie axis).
    pattern: str | None
    #: The zero-shot sentence scored for this class.
    text: str


@dataclass(frozen=True)
class Axis:
    id: str
    classes: tuple[AxisClass, ...]
    #: Whether the matched span is blanked out of the caption before free
    #: phrases are cut. Preamble axes are (they are not grammatical parts of
    #: the subject description); subject descriptors (gender, age, ethnicity,
    #: skin) are not, or "a smiling young woman" would lose its noun.
    mask_residue: bool = True


def _c(label: str, pattern: str | None, text: str) -> AxisClass:
    return AxisClass(label, pattern, text)


_SUBJECT_NOUN = (
    r"(?=\s+(?:woman|women|man|men|girl|boy|male|female|person|people|teen|teenager|lady|guy|bride|groom|couple|model))"
)

AXES: tuple[Axis, ...] = (
    Axis(
        "shot_type",
        (
            _c("Extreme Close-up", r"extreme close[- ]?up", "an extreme close-up photo showing only part of a face"),
            _c("Close-up", r"close[- ]?up", "a close-up photo of a face"),
            _c("Medium Close-up", r"medium close[- ]?up", "a medium close-up photo, head and shoulders"),
            _c("Medium shot", r"medium[- ]shot", "a medium shot photo, waist up"),
            _c("Cowboy shot", r"cowboy[- ]shot", "a cowboy shot photo, framed from mid-thigh up"),
            _c("Medium full shot", r"medium[- ]full[- ]shot", "a medium full shot photo, framed from the knees up"),
            _c("Full shot", r"full[- ]shot", "a full shot photo of the whole body, head to toe"),
            _c("Wide shot", r"wide[- ]shot", "a wide shot photo, a small figure in a large scene"),
            _c("Extreme wide shot", r"extreme wide[- ]shot", "an extreme wide shot photo of a vast scene"),
        ),
    ),
    Axis(
        "vertical_angle",
        (
            _c(
                "From Below",
                r"from (?:directly )?below|worm'?s[- ]eye",
                "a photo taken from directly below, looking straight up",
            ),
            _c("Very Low Angle", r"very low[- ]angle", "a photo taken from a very low angle, looking steeply up"),
            _c("Low Angle", r"low[- ]angle", "a photo taken from a low angle, looking slightly up"),
            _c(
                "Eye-Level", r"eye[- ]level|straight[- ]on", "a photo taken at eye level, camera level with the subject"
            ),
            _c("High Angle", r"high[- ]angle", "a photo taken from a high angle, looking slightly down"),
            _c("Very High Angle", r"very high[- ]angle", "a photo taken from a very high angle, looking steeply down"),
            _c(
                "From Above",
                r"from (?:directly )?above|bird'?s[- ]eye|top[- ]down",
                "a photo taken from directly above, looking straight down",
            ),
        ),
    ),
    Axis(
        "horizontal_angle",
        (
            _c(
                "Front View",
                r"front[- ]view|from the front|facing the camera",
                "a photo of a person seen from the front, facing the camera",
            ),
            _c(
                "Three-Quarter View", r"three[- ]quarter[- ]view", "a photo of a person seen from a three-quarter angle"
            ),
            _c(
                "Side View",
                r"side[- ]view|in profile|from the side",
                "a photo of a person seen from the side, in profile",
            ),
            _c(
                "Three-Quarter Rear",
                r"three[- ]quarter[- ]rear(?:[- ]view)?",
                "a photo of a person seen from a three-quarter rear angle",
            ),
            _c(
                "Rear View",
                r"rear[- ]view|from behind|back[- ]view",
                "a photo of a person seen from behind, back to the camera",
            ),
        ),
    ),
    Axis(
        "camera_type",
        (
            _c("Amateur Photo", r"\bamateur\b", "an amateur snapshot photo"),
            _c("Professional Photo", r"\bprofessional\b", "a professional studio photograph"),
            _c("Movie Scene", r"movie scene|film still|movie still", "a still frame from a movie"),
            _c("Video Screenshot", r"video screenshot|screenshot from a video|\bwebcam\b", "a screenshot from a video"),
        ),
    ),
    Axis(
        "grain",
        (
            _c(
                "Clean Image",
                r"\bclean\b(?=[ ,]+(?:image|amateur|professional|photo|shot|medium|close|wide|full|cowboy|extreme|selfie|portrait))",
                "a clean, noise-free photo",
            ),
            _c("Fine Grain", r"fine[- ]grain(?:ed)?", "a photo with fine film grain"),
            _c("Moderate Grain", r"moderate[- ]grain(?:ed)?", "a photo with moderate film grain"),
            _c("Heavy Grain", r"heavy[- ]grain(?:ed)?|very grainy", "a photo with heavy film grain"),
            _c("Noisy Image", r"\bnoisy\b", "a very noisy, low-quality photo"),
        ),
    ),
    Axis(
        "saturation",
        (
            _c("Black and White", r"black[- ]and[- ]white|monochrome|\bb&w\b|gr[ae]yscale", "a black and white photo"),
            _c("Muted", r"\bmuted\b", "a photo with muted, desaturated colours"),
            _c("Subdued", r"\bsubdued\b", "a photo with subdued colours"),
            _c(
                "Neutral",
                r"\bneutral\b(?=\s*(?:,|photo|shot|image|tones?|colou?rs?))",
                "a photo with neutral, natural colours",
            ),
            _c("Colourful", r"colou?rful", "a colourful photo with rich colours"),
            _c("Vivid", r"\bvivid\b", "a vivid photo with highly saturated colours"),
        ),
    ),
    Axis(
        "brightness",
        (
            _c(
                "Dark",
                r"dimly[- ]lit|low[- ]key|dark (?:lighting|scene|room|setting)|in (?:the )?dark(?:ness)?\b",
                "a dark, low-key photo",
            ),
            _c("Dim", r"\bdim (?:light|lighting)", "a dimly lit photo"),
            _c(
                "Bright",
                r"bright(?:ly)?[- ]lit|bright (?:light|lighting|daylight|sunlight)|well[- ]lit",
                "a brightly lit photo",
            ),
            _c("High-key", r"high[- ]key|overexposed", "a high-key, very bright photo"),
        ),
    ),
    Axis(
        "warmth",
        (
            _c(
                "Warm",
                r"warm (?:tones?|light(?:ing)?|colou?rs?|glow|hue)|golden hour",
                "a photo with warm, golden tones",
            ),
            _c("Cool", r"cool (?:tones?|light(?:ing)?|colou?rs?|hue)|\bbluish\b", "a photo with cool, bluish tones"),
        ),
    ),
    Axis(
        "contrast",
        (
            _c("Flat", r"flat (?:lighting|contrast|light)|low[- ]contrast", "a flat, low-contrast photo"),
            _c("Punchy", r"\bpunchy\b|high[- ]contrast|harsh (?:light|shadows)", "a punchy, high-contrast photo"),
        ),
    ),
    Axis(
        "selfie",
        (
            _c("Mirror Selfie", r"mirror selfie", "a mirror selfie photo"),
            _c("Selfie", r"\bselfie\b", "a selfie photo, arm outstretched holding the phone"),
            _c("Not a selfie", None, "a photo taken by another person"),
        ),
    ),
    Axis(
        "ethnicity",
        (
            _c("Caucasian", r"caucasian|\bwhite" + _SUBJECT_NOUN, "a photo of a Caucasian person"),
            _c("East Asian", r"east asian|\b(?:japanese|chinese|korean)\b", "a photo of an East Asian person"),
            _c(
                "Southeast Asian",
                r"south[- ]?east asian|\b(?:thai|filipina|filipino|vietnamese|indonesian)\b",
                "a photo of a Southeast Asian person",
            ),
            _c("Indian", r"\bindian\b|south asian", "a photo of an Indian person"),
            _c("African", r"\bafrican\b|\bblack" + _SUBJECT_NOUN, "a photo of a Black person"),
            _c("Mixed Race", r"mixed[- ]race|biracial", "a photo of a mixed race person"),
            _c("Native American", r"native american|indigenous", "a photo of a Native American person"),
            _c("Middle Eastern", r"middle eastern|\barab\b|persian", "a photo of a Middle Eastern person"),
            _c("Hispanic", r"hispanic|latina|latino", "a photo of a Hispanic person"),
        ),
        mask_residue=False,
    ),
    Axis(
        "skin_tone",
        (
            _c("Pale Skin", r"pale skin", "a photo of a person with pale skin"),
            _c("Fair Skin", r"fair skin", "a photo of a person with fair skin"),
            _c("Light Skin", r"light skin", "a photo of a person with light skin"),
            _c("Tanned Skin", r"tanned skin|tan skin", "a photo of a person with tanned skin"),
            _c("Olive Skin", r"olive skin", "a photo of a person with olive skin"),
            _c("Light Brown Skin", r"light[- ]brown skin", "a photo of a person with light brown skin"),
            _c("Medium Brown Skin", r"medium[- ]brown skin|\bbrown skin", "a photo of a person with medium brown skin"),
            _c("Dark Brown Skin", r"dark[- ]brown skin|dark skin", "a photo of a person with dark brown skin"),
            _c("Black Skin", r"black skin|very dark skin", "a photo of a person with very dark skin"),
        ),
        mask_residue=False,
    ),
    Axis(
        "gender",
        (
            _c(
                "Female",
                r"\b(?:woman|women|girl|girls|female|lady|ladies|bride|she|her|hers|herself)\b",
                "a photo of a woman",
            ),
            _c("Male", r"\b(?:man|men|boy|boys|male|guy|guys|groom|he|him|his|himself)\b", "a photo of a man"),
        ),
        mask_residue=False,
    ),
)

# Age is ordinal and numeric in SoReal! captions ("34-year-old"), so it gets its
# own extractor rather than a per-class regex. Bins are coarse on purpose: a
# bi-encoder cannot tell 31 from 34, but it can tell a child from an adult.
_AGE_BINS: tuple[tuple[str, int, str], ...] = (
    ("child", 13, "a photo of a child"),
    ("teenager", 20, "a photo of a teenager"),
    ("twenties", 30, "a photo of a young adult in their twenties"),
    ("thirties", 40, "a photo of an adult in their thirties"),
    ("forties", 50, "a photo of an adult in their forties"),
    ("middle-aged", 65, "a photo of a middle-aged person in their fifties or sixties"),
    ("elderly", 10**9, "a photo of an elderly person"),
)
_AGE_TEXTS: tuple[str, ...] = tuple(text for _, _, text in _AGE_BINS)
_AGE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[- ]year[- ]old\b", re.IGNORECASE)
_AGE_WORD_RE: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(?:child|kid|toddler|little (?:girl|boy))\b", re.IGNORECASE), "child"),
    (re.compile(r"\bteen(?:age|ager|aged)?\b", re.IGNORECASE), "teenager"),
    (re.compile(r"\bmiddle[- ]aged\b", re.IGNORECASE), "middle-aged"),
    (re.compile(r"\b(?:elderly|old (?:woman|man|lady)|senior)\b", re.IGNORECASE), "elderly"),
)


def _age_bin(years: int) -> str:
    for label, upper, _ in _AGE_BINS:
        if years < upper:
            return label
    return _AGE_BINS[-1][0]


#: Width of the age-model agreement curve, in years. Roughly the head's error
#: scale: an estimate one sigma off the claim still scores ~0.6, three sigmas
#: off scores ~0.01. Ordering within a group is what matters, so the exact
#: value is a tuning knob, not a threshold.
AGE_SIGMA_YEARS = 6.0
#: Age estimates with more ``__none__`` mass than this are not trusted — the
#: model is saying "no readable subject", and its expected value is noise.
AGE_NONE_GATE = 0.5
AGE_MODEL_AXIS = "age_model"


def claimed_age_range(prompt: str) -> tuple[float, float] | None:
    """The age a caption claims, as an inclusive ``(low, high)`` range in years
    — a point for "34-year-old", a band for "teenage" — or ``None`` when the
    caption says nothing about age (so the age model is never consulted)."""
    text = normalize_caption(prompt)
    if not text:
        return None
    numeric = {int(m.group(1)) for m in _AGE_NUMERIC_RE.finditer(text)}
    if len(numeric) == 1:
        years = float(numeric.pop())
        return years, years
    if numeric:
        return None  # several ages: a couple or a group, no single claim
    label, _spans = _extract_age(text)
    if label is None:
        return None
    lower = 0.0
    for lbl, upper, _ in _AGE_BINS:
        if lbl == label:
            return lower, float(min(upper, 100) - 1)
        lower = float(upper)
    return None


def age_agreement(estimated: float, claimed: tuple[float, float], *, sigma: float = AGE_SIGMA_YEARS) -> float:
    """P-like agreement in (0, 1]: 1 inside the claimed range, a Gaussian
    fall-off in the distance to its nearest edge outside it."""
    low, high = claimed
    distance = max(low - estimated, estimated - high, 0.0)
    return math.exp(-0.5 * (distance / sigma) ** 2)


def _age_text(label: str) -> str:
    return next(text for lbl, _, text in _AGE_BINS if lbl == label)


@dataclass(frozen=True)
class Proposition:
    """One scoreable claim about the image.

    ``texts[0]`` is always ``text``; for an axis proposition the rest are the
    sibling classes it is scored against, for a free phrase there are none.
    """

    text: str
    axis: str | None
    label: str | None
    texts: Sequence[str]


_COMPILED: dict[str, list[tuple[AxisClass, re.Pattern]]] = {
    axis.id: [(cls, re.compile(cls.pattern, re.IGNORECASE)) for cls in axis.classes if cls.pattern] for axis in AXES
}


def _match_axis(axis: Axis, text: str) -> tuple[AxisClass | None, list[tuple[int, int]]]:
    """The class this caption claims on ``axis``, plus every matched span.

    Longest span wins, so "medium full shot" beats the "full shot" inside it.
    Two *different* classes surviving overlap removal is a contradiction (a
    couple's "bride and groom" on the gender axis) and yields ``None``: the
    axis is skipped rather than scored against the wrong sibling.
    """
    hits: list[tuple[int, int, AxisClass]] = []
    for cls, pattern in _COMPILED[axis.id]:
        hits.extend((m.start(), m.end(), cls) for m in pattern.finditer(text))
    if not hits:
        return None, []
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    kept: list[tuple[int, int, AxisClass]] = []
    for start, end, cls in hits:
        if all(end <= ks or start >= ke for ks, ke, _ in kept):
            kept.append((start, end, cls))
    spans = [(s, e) for s, e, _ in kept]
    labels = {cls.label for _, _, cls in kept}
    if len(labels) > 1:
        return None, spans
    return kept[0][2], spans


def _extract_age(text: str) -> tuple[str | None, list[tuple[int, int]]]:
    labels: set[str] = set()
    spans: list[tuple[int, int]] = []
    for m in _AGE_NUMERIC_RE.finditer(text):
        labels.add(_age_bin(int(m.group(1))))
        spans.append((m.start(), m.end()))
    if not labels:
        for pattern, label in _AGE_WORD_RE:
            for m in pattern.finditer(text):
                labels.add(label)
                spans.append((m.start(), m.end()))
    if len(labels) != 1:
        return None, spans
    return labels.pop(), spans


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "photo",
        "photograph",
        "shot",
        "image",
        "picture",
        "showing",
        "shows",
        "taken",
        "in",
        "on",
        "at",
    }
)
_PHRASE_SPLIT_RE = re.compile(
    r"[.,;:!?]|\b(?:with|wearing|while|and|against|holding|beside|next to|behind|near)\b", re.IGNORECASE
)
_LEADING_JUNK_RE = re.compile(
    r"^(?:(?:of|a|an|the|shot|photo|photograph|image|picture|showing|shows|taken|is|are)\s+)+", re.IGNORECASE
)
_WS_RE = re.compile(r"\s+")
_ARTICLE_RE = re.compile(r"^(?:a|an|the)\b", re.IGNORECASE)


def _phrase_text(chunk: str) -> str:
    words = chunk.split()
    if len(words) > MAX_PHRASE_WORDS:
        chunk = " ".join(words[:MAX_PHRASE_WORDS])
    if _ARTICLE_RE.match(chunk):
        return "a photo of " + chunk if not chunk.lower().startswith(("a photo", "an image", "a picture")) else chunk
    if words and words[0].lower().endswith("ing"):
        return "a photo of a person " + chunk
    return "a photo of " + chunk


def _free_phrases(residue: str, min_words: int) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for raw in _PHRASE_SPLIT_RE.split(residue):
        chunk = _WS_RE.sub(" ", raw).strip(" -'\"")
        chunk = _LEADING_JUNK_RE.sub("", chunk).strip()
        if not chunk:
            continue
        content = [w for w in chunk.lower().split() if w not in _STOP_WORDS]
        if len(content) < min_words:
            continue
        text = _phrase_text(chunk)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(text)
    return phrases


def extract_propositions(prompt: str) -> list[Proposition]:
    """Decompose a caption line into axis propositions and free phrases.

    Axis propositions come first, in vocabulary order; free phrases follow in
    caption order. ``UNCONDITIONAL`` and empty prompts yield nothing, which the
    scorer treats as "nothing to score" rather than an error.
    """
    text = normalize_caption(prompt)
    if not text or text == "UNCONDITIONAL":
        return []

    props: list[Proposition] = []
    masked = list(text)

    def _blank(spans: list[tuple[int, int]]) -> None:
        for start, end in spans:
            for i in range(start, end):
                masked[i] = " "

    for axis in AXES:
        cls, spans = _match_axis(axis, text)
        if axis.mask_residue:
            _blank(spans)
        if cls is None:
            continue
        siblings = [c.text for c in axis.classes if c is not cls]
        props.append(Proposition(text=cls.text, axis=axis.id, label=cls.label, texts=[cls.text, *siblings]))

    age_label, _age_spans = _extract_age(text)
    if age_label is not None:
        own = _age_text(age_label)
        props.append(
            Proposition(text=own, axis="age", label=age_label, texts=[own, *[t for t in _AGE_TEXTS if t != own]])
        )

    min_words = 1 if is_tag_line(text) else MIN_PHRASE_WORDS
    props.extend(
        Proposition(text=phrase, axis=None, label=None, texts=[phrase])
        for phrase in _free_phrases("".join(masked), min_words)
    )
    return props


# --------------------------------------------------------------------------- #
# Scoring maths
# --------------------------------------------------------------------------- #


def axis_probability(sims: np.ndarray, *, scale: float, bias: float) -> float:
    """P(own class) as a softmax over sibling logits. ``sims[0]`` is the own
    class. The SigLIP bias is a constant across siblings and cancels; it is
    accepted so both probability functions share one signature."""
    logits = np.asarray(sims, dtype=np.float64) * scale + bias
    logits -= logits.max()
    exp = np.exp(logits)
    return float(exp[0] / exp.sum())


def phrase_probability(sim: float, *, scale: float, bias: float) -> float:
    """SigLIP's own P(match): the sigmoid its loss calibrated."""
    z = float(sim) * scale + bias
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def aggregate(props: Sequence[Proposition], probs: Sequence[float]) -> dict:
    """Per-image summary of proposition probabilities.

    ``adherence`` weights the axis family and the phrase family equally rather
    than averaging every proposition, so a caption with ten free phrases and
    two axis claims does not drown the axes — the axes are the propositions
    the bi-encoder is most reliable on.
    """
    axis_probs = [p for prop, p in zip(props, probs, strict=True) if prop.axis is not None]
    phrase_probs = [p for prop, p in zip(props, probs, strict=True) if prop.axis is None]
    axis_mean = float(np.mean(axis_probs)) if axis_probs else None
    axis_min = float(min(axis_probs)) if axis_probs else None
    phrase_mean = float(np.mean(phrase_probs)) if phrase_probs else None
    families = [v for v in (axis_mean, phrase_mean) if v is not None]
    adherence = float(np.mean(families)) if families else None
    return {
        "adherence": adherence,
        "axis_mean": axis_mean,
        "axis_min": axis_min,
        "phrase_mean": phrase_mean,
        "n_axis": len(axis_probs),
        "n_phrase": len(phrase_probs),
    }


# --------------------------------------------------------------------------- #
# The model half
# --------------------------------------------------------------------------- #


def _pick_device(device: str) -> str:
    device = str(device).lower()
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return "cpu"
    return "cuda" if free >= _AUTO_CUDA_MIN_FREE_BYTES else "cpu"


class SiglipEmbedder:
    """SigLIP 2 image/text embedder with the interface the scorer expects:
    ``embed_images(paths) -> (n, d)``, ``embed_texts(texts) -> (n, d)``, both
    L2-normalised float32, plus ``scale``/``bias``/``model_version``.

    Loads lazily: constructing one costs nothing, and ``model_version`` is
    known without the weights, so a group whose image vectors, prompt vectors
    and calibration are all cached never loads them. The first ``embed_*`` (or
    a ``scale``/``bias`` read with no cached calibration) loads; a failed load
    raises ``RuntimeError`` there. ``unload_model`` frees the weights and the
    next use loads them again.

    A row of NaN marks an image that could not be decoded; the scorer drops it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_patches: int = DEFAULT_MAX_PATCHES,
        local_files_only: bool = False,
    ):
        self.requested_device = device
        self.device: str | None = None
        self.max_patches = int(max_patches)
        self.model_name = model_name
        self.model_version = f"{model_name}@{self.max_patches}"
        self.local_files_only = local_files_only
        self._torch = None
        self._processor = None
        self._model = None
        self._dtype = None
        self._scale: float | None = None
        self._bias: float | None = None
        self._load_lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def set_calibration(self, scale: float, bias: float) -> None:
        """Supply the model's logit scale/bias from a cache so scoring cached
        vectors needs no weights."""
        self._scale = float(scale)
        self._bias = float(bias)

    @property
    def scale(self) -> float:
        if self._scale is None:
            self.load_model()
        return float(self._scale)

    @property
    def bias(self) -> float:
        if self._bias is None:
            self.load_model()
        return float(self._bias)

    def load_model(self) -> None:
        """Load the weights (idempotent). Local files first, the network only
        on a miss unless ``local_files_only``. Raises ``RuntimeError``."""
        with self._load_lock:
            if self.loaded:
                return
            try:
                import torch

                from transformers import AutoModel, AutoProcessor
            except ImportError as ex:
                raise RuntimeError("Adherence scoring needs torch and transformers.") from ex
            device = _pick_device(self.requested_device)
            dtype = torch.float16 if device == "cuda" else torch.float32
            attempts = (True,) if self.local_files_only else (True, False)
            last: Exception | None = None
            for local in attempts:
                try:
                    processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=local)
                    model = AutoModel.from_pretrained(self.model_name, local_files_only=local, dtype=dtype)
                    model = model.to(device).eval()
                    break
                except Exception as ex:  # noqa: BLE001 — the last failure is the message
                    last = ex
            else:
                raise RuntimeError(f"Adherence scoring could not load {self.model_name}: {last}")
            self._torch = torch
            self._processor = processor
            self._model = model
            self._dtype = dtype
            self.device = device
            self._scale = float(model.logit_scale.exp().item())
            self._bias = float(model.logit_bias.item())

    def unload_model(self) -> None:
        """Free the weights; ``model_version`` and any calibration survive."""
        with self._load_lock:
            model, self._model = self._model, None
            self._processor = None
            if model is None:
                return
            with contextlib.suppress(Exception):
                model.to("cpu")
            del model
            gc.collect()
            torch = self._torch
            with contextlib.suppress(Exception):
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        from PIL import Image

        self.load_model()

        def _open_rgb(path: str):
            try:
                with Image.open(path) as handle:
                    return handle.convert("RGB")
            except Exception:
                return None

        opened = [_open_rgb(path) for path in paths]
        images = [img for img in opened if img is not None]
        ok = [img is not None for img in opened]
        dim = (
            self._model.config.text_config.projection_size
            if hasattr(self._model.config.text_config, "projection_size")
            else None
        )
        if not images:
            return np.full((len(paths), dim or 1), np.nan, dtype=np.float32)
        inputs = self._processor.image_processor(images=images, max_num_patches=self.max_patches, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self._dtype)
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        feats = self._torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float32)
        out = np.full((len(paths), feats.shape[1]), np.nan, dtype=np.float32)
        out[np.asarray(ok, dtype=bool)] = feats
        return out

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.load_model()
        tokens = self._processor.tokenizer(
            list(texts), padding="max_length", max_length=TEXT_MAX_LENGTH, truncation=True, return_tensors="pt"
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with self._torch.no_grad():
            feats = self._model.get_text_features(**tokens)
        return self._torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float32)


def load_siglip_embedder(
    device: str = "auto", max_patches: int = DEFAULT_MAX_PATCHES, model_name: str = DEFAULT_MODEL_NAME
) -> SiglipEmbedder:
    """The real embedder handle, or a ``RuntimeError`` the UI can show.

    Nothing is loaded here: the weights come in on the first embedding call
    that the cache cannot answer (local files preferred; a first run
    downloads them), and a failed load raises ``RuntimeError`` from that call.
    """
    try:
        import torch  # noqa: F401

        import transformers  # noqa: F401
    except ImportError as ex:
        raise RuntimeError("Adherence scoring needs torch and transformers.") from ex
    return SiglipEmbedder(model_name, device=device, max_patches=max_patches, local_files_only=False)


# --------------------------------------------------------------------------- #
# Embedding cache encoding
# --------------------------------------------------------------------------- #


def encode_embedding(vector) -> str:
    """Compact JSON-safe form: float16 bytes, base64. ~3 KB per SO400M vector
    instead of ~20 KB of decimal text; float16 keeps cosine error < 1e-3."""
    arr = np.asarray(vector, dtype=np.float32).astype(np.float16)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_embedding(blob: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(blob.encode("ascii")), dtype=np.float16).astype(np.float32)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


#: Scan-cache fields for the model-side values the scorer needs beyond the
#: image vectors: one text vector per proposition sentence, and the model's
#: logit calibration. Both are keyed on the model version, not on a file.
TEXT_CACHE_FIELD = "adherence_text"
CALIBRATION_CACHE_FIELD = "adherence_calibration"


def _text_vectors(embedder, texts: list[str], cache=None) -> np.ndarray:
    """One vector per text: the embedder's memo first, then the scan cache,
    the model only for what neither has (and written back to both)."""
    memo: dict[str, np.ndarray] | None = getattr(embedder, "_adherence_text_memo", None)
    if memo is None:
        memo = {}
        with contextlib.suppress(Exception):
            embedder._adherence_text_memo = memo
    missing = [t for t in dict.fromkeys(texts) if t not in memo]
    version = getattr(embedder, "model_version", None)
    if missing and cache is not None:
        still_missing: list[str] = []
        for text in missing:
            blob = cache.peek_global(TEXT_CACHE_FIELD, text, version)
            if isinstance(blob, str):
                try:
                    memo[text] = decode_embedding(blob)
                    continue
                except Exception:  # noqa: BLE001 — a corrupt blob is a miss
                    pass
            still_missing.append(text)
        missing = still_missing
    if missing:
        vecs = embedder.embed_texts(missing)
        for text, vec in zip(missing, vecs, strict=True):
            memo[text] = np.asarray(vec, dtype=np.float32)
            if cache is not None:
                cache.store_global(TEXT_CACHE_FIELD, text, encode_embedding(memo[text]), version)
    return np.stack([memo[t] for t in texts])


def _calibration(embedder, cache=None) -> tuple[float, float]:
    """``(scale, bias)`` — from the scan cache when it has them for this model
    version, otherwise from the embedder (which loads) and written back."""
    version = getattr(embedder, "model_version", None)
    if cache is not None:
        cached = cache.peek_global(CALIBRATION_CACHE_FIELD, "logit", version)
        if isinstance(cached, dict) and "scale" in cached and "bias" in cached:
            scale, bias = float(cached["scale"]), float(cached["bias"])
            setter = getattr(embedder, "set_calibration", None)
            if callable(setter):
                setter(scale, bias)
            return scale, bias
    scale, bias = float(embedder.scale), float(embedder.bias)
    if cache is not None:
        cache.store_global(CALIBRATION_CACHE_FIELD, "logit", {"scale": scale, "bias": bias}, version)
    return scale, bias


def prompt_texts(prompt: str) -> list[str]:
    """Every sentence the scorer will embed for ``prompt`` (axis siblings
    included), or ``[]`` when the prompt has nothing to score."""
    texts: list[str] = []
    for prop in extract_propositions(prompt):
        texts.extend(prop.texts)
    return list(dict.fromkeys(texts))


def warm_prompt(prompt: str, *, embedder, cache) -> bool:
    """Put everything a later ``score_images_adherence`` for ``prompt`` needs
    beyond the image vectors into ``cache``: its text vectors and the model's
    calibration. Returns False when the prompt has nothing to score."""
    texts = prompt_texts(prompt)
    if not texts:
        return False
    _text_vectors(embedder, texts, cache=cache)
    _calibration(embedder, cache=cache)
    return True


def image_embeddings(
    paths: Sequence[str],
    *,
    embedder,
    cache=None,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Image vectors for every readable path, cache first, model for the rest."""
    vectors: dict[str, np.ndarray] = {}
    misses: list[str] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        cached = cache.peek_adherence_embedding(path, embedder.model_version) if cache is not None else None
        if cached is not None:
            vectors[path] = np.asarray(cached, dtype=np.float32)
        else:
            misses.append(path)
    done = len(vectors)
    total = len(paths)
    if progress is not None:
        progress(done, total)
    for start in range(0, len(misses), batch_size):
        chunk = misses[start : start + batch_size]
        vecs = np.asarray(embedder.embed_images(chunk), dtype=np.float32)
        for path, vec in zip(chunk, vecs, strict=True):
            if np.isnan(vec).any():
                continue
            vectors[path] = vec
            if cache is not None:
                cache.store_adherence_embedding(path, vec, embedder.model_version)
        done += len(chunk)
        if progress is not None:
            progress(done, total)
    return vectors


def score_images_adherence(
    paths: Sequence[str],
    prompt: str,
    *,
    embedder=None,
    cache=None,
    device: str = "auto",
    batch_size: int = 8,
    max_patches: int = DEFAULT_MAX_PATCHES,
    progress: Callable[[int, int], None] | None = None,
    age_estimates: Mapping[str, Mapping] | None = None,
) -> dict[str, dict]:
    """Adherence of each readable path to ``prompt``.

    Returns ``{path: {adherence, axis_mean, axis_min, phrase_mean, n_axis,
    n_phrase, propositions: [{text, axis, label, prob}]}}``. Unreadable files
    are skipped; a prompt with nothing to score returns ``{}`` without touching
    the model. Raises ``RuntimeError`` when the model cannot be loaded.

    ``age_estimates`` (``{path: {"age", "p_none"}}`` from
    ``age_estimate_util.estimate_ages``) adds one more axis proposition —
    agreement between the caption's claimed age and the Engine age head's
    estimate — **only when the caption claims an age**. Captions without one
    never consult the age model; estimates whose ``p_none`` exceeds
    :data:`AGE_NONE_GATE` are treated as unavailable for that image.
    """
    props = extract_propositions(prompt)
    if not props or not paths:
        return {}
    claimed = claimed_age_range(prompt) if age_estimates else None
    if embedder is None:
        embedder = load_siglip_embedder(device=device, max_patches=max_patches)

    vectors = image_embeddings(list(paths), embedder=embedder, cache=cache, batch_size=batch_size, progress=progress)
    if not vectors:
        return {}

    all_texts: list[str] = []
    for prop in props:
        all_texts.extend(prop.texts)
    text_vecs = _text_vectors(embedder, all_texts, cache=cache)
    text_index = {t: i for i, t in enumerate(all_texts)}

    scale, bias = _calibration(embedder, cache=cache)
    results: dict[str, dict] = {}
    for path, vec in vectors.items():
        sims = text_vecs @ vec
        probs: list[float] = []
        detail: list[dict] = []
        for prop in props:
            if prop.axis is not None:
                own_sims = np.asarray([sims[text_index[t]] for t in prop.texts])
                prob = axis_probability(own_sims, scale=scale, bias=bias)
            else:
                prob = phrase_probability(float(sims[text_index[prop.text]]), scale=scale, bias=bias)
            probs.append(prob)
            detail.append({"text": prop.text, "axis": prop.axis, "label": prop.label, "prob": prob})
        # The SigLIP propositions make the adherence summary. The age head is
        # reported beside it, not folded in: it is a different model with a
        # different error profile, and the combiner weights it on its own.
        summary = aggregate(props, probs)
        age_summary = _age_summary(claimed, age_estimates.get(path) if age_estimates else None)
        summary["age_agreement"] = age_summary["age_agreement"]
        summary["estimated_age"] = age_summary["estimated_age"]
        summary["propositions"] = [*detail, *age_summary["propositions"]]
        results[path] = summary
    return results


def _age_summary(claimed: tuple[float, float] | None, estimate: Mapping | None) -> dict:
    """The age head's verdict for one image against the caption's claim:
    ``{age_agreement, estimated_age, propositions}`` (the proposition list is
    empty or the single age-model line)."""
    summary: dict = {"age_agreement": None, "estimated_age": None, "propositions": []}
    if claimed is None or estimate is None or estimate.get("age") is None:
        return summary
    estimated = float(estimate["age"])
    p_none = float(estimate.get("p_none", 0.0))
    summary["estimated_age"] = estimated
    if p_none > AGE_NONE_GATE:
        return summary
    prob = age_agreement(estimated, claimed)
    low, high = claimed
    label = f"{low:g}-year-old" if low == high else f"{low:g}-{high:g} years old"
    summary["age_agreement"] = prob
    summary["propositions"].append(
        {
            "text": f"age model: claimed {label}, estimated {estimated:.0f}",
            "axis": AGE_MODEL_AXIS,
            "label": label,
            "prob": prob,
            "estimated_age": estimated,
            "p_none": p_none,
        }
    )
    return summary


def score_images_age_agreement(
    paths: Sequence[str],
    prompt: str,
    *,
    age_estimates: Mapping[str, Mapping] | None,
) -> dict[str, dict]:
    """The age axis on its own, without the SigLIP embedder.

    Same per-path shape as :func:`score_images_adherence` — ``adherence``,
    ``axis_mean`` and ``phrase_mean`` are ``None`` — so a caller that has
    dropped the SigLIP propositions keeps its output contract. Returns ``{}``
    when the caption claims no age or there are no estimates.
    """
    claimed = claimed_age_range(prompt) if age_estimates else None
    if claimed is None:
        return {}
    results: dict[str, dict] = {}
    for path in paths:
        estimate = age_estimates.get(path) if age_estimates else None
        if estimate is None:
            continue
        results[path] = {
            "adherence": None,
            "axis_mean": None,
            "phrase_mean": None,
            **_age_summary(claimed, estimate),
        }
    return results
