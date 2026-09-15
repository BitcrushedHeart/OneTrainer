"""Pure logic for the on-image text reward: what to ask the model to write,
and how to pair what a detector found with what was asked for.

The reward term (see :mod:`modules.util.text_reward_model`) is DRaFT-1 /
ReFL shaped: a no-grad rollout to a late timestep, one grad-bearing x0
prediction, a VAE decode of the text region, and a frozen CTC recognizer whose
loss flows back into the pixels. Everything in *this* module is torch-free so
it can be tested without weights:

* :func:`generate_text_spec` draws a prompt of the form
  ``"<Surface> text reading \\"<string>\\""`` where the string comes from one of
  several *format families* — dictionary words in mixed case, prices, dates,
  times, URLs, abbreviations, numbers, non-words — so the model is rewarded for
  composing glyph sequences from sub-word tokens rather than recalling one
  memorised word. Every character is filtered against the recognizer's
  vocabulary: the reward can only teach what the recognizer can read, so a
  target it cannot represent must never reach the loss.

* :func:`align_words` pairs detector boxes (each read once, no grad) with the
  target words by best edit-similarity in reading order, so the CTC loss is
  computed only on boxes that plausibly *are* the target word. A target with no
  box is reported as missing (it carries no gradient — there is nothing to push
  on — but it counts against the metric); a spurious box is ignored.

* :func:`text_reward_metrics` turns readings into the scalars that go to
  TensorBoard: mean per-word edit ratio and the fraction of prompts rendered
  verbatim. These come from the reward recognizer itself, so they say whether
  the *reward* is satisfied — the independent verdict stays with the curation
  OCR (PaddleOCR-VL edit distance), never the network being optimised against.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from modules.util.caption_text_extract import levenshtein_ratio

#: docTR's ``english`` vocabulary: ASCII letters, digits, punctuation, ° and
#: the common currency symbols. The recognizer's real vocab is read off the
#: loaded model and passed in; this is the default for tests and the prompt
#: generator when no model is loaded.
DEFAULT_VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿"

#: Surfaces the prompt template draws from. Style is assumed learnt already;
#: the surface only keeps the reward from collapsing onto one rendering mode.
DEFAULT_SURFACES = (
    "Handwritten",
    "Sign",
    "Neon",
    "Cursive",
    "Typeset",
    "Digital",
    "Painted",
    "Embossed",
    "Engraved",
    "Chalkboard",
    "Stencilled",
    "Stamped",
    "Printed",
    "Carved",
)

#: Families and their default draw weights. ``words`` is the plain
#: "1-8 dictionary words" case; the structured ones force composition.
DEFAULT_FAMILY_WEIGHTS = {
    "words": 6.0,
    "price": 2.0,
    "date": 1.5,
    "time": 1.0,
    "url": 1.0,
    "abbrev": 1.0,
    "number": 1.0,
    "nonword": 1.0,
}

#: Built-in dictionary: everyday words of mixed length so a prompt with eight
#: of them still fits on a sign. A user word list (one word per line) replaces
#: it via ``word_list``.
BUILTIN_WORDS = (
    "apple",
    "bread",
    "cheese",
    "coffee",
    "dinner",
    "eggs",
    "fish",
    "garden",
    "honey",
    "island",
    "juice",
    "kettle",
    "lemon",
    "market",
    "night",
    "orange",
    "pepper",
    "quiet",
    "river",
    "summer",
    "table",
    "umbrella",
    "village",
    "water",
    "yellow",
    "winter",
    "bakery",
    "butcher",
    "florist",
    "grocer",
    "tailor",
    "barber",
    "chemist",
    "library",
    "station",
    "harbour",
    "theatre",
    "museum",
    "castle",
    "bridge",
    "tower",
    "church",
    "school",
    "hotel",
    "cafe",
    "diner",
    "pub",
    "inn",
    "lodge",
    "cottage",
    "manor",
    "farm",
    "mill",
    "forge",
    "yard",
    "dock",
    "quay",
    "open",
    "closed",
    "welcome",
    "exit",
    "entrance",
    "push",
    "pull",
    "private",
    "parking",
    "reserved",
    "sale",
    "fresh",
    "local",
    "organic",
    "vintage",
    "classic",
    "premium",
    "deluxe",
    "daily",
    "weekly",
    "monthly",
    "north",
    "south",
    "east",
    "west",
    "high",
    "low",
    "old",
    "new",
    "big",
    "small",
    "long",
    "short",
    "fast",
    "slow",
    "red",
    "blue",
    "green",
    "black",
    "white",
    "gold",
    "silver",
    "copper",
    "bronze",
    "ivory",
    "amber",
    "coral",
    "morning",
    "evening",
    "midnight",
    "sunrise",
    "sunset",
    "autumn",
    "spring",
    "monday",
    "friday",
    "sunday",
    "tea",
    "milk",
    "sugar",
    "butter",
    "flour",
    "salt",
    "oil",
    "wine",
    "beer",
    "cider",
    "ale",
    "stout",
    "lager",
    "boots",
    "shoes",
    "hats",
    "coats",
    "gloves",
    "scarves",
    "rings",
    "watches",
    "jewels",
    "pearls",
    "books",
    "maps",
    "prints",
    "frames",
    "candles",
    "soaps",
    "lamps",
    "clocks",
    "mirrors",
    "thank",
    "you",
    "please",
    "sorry",
    "hello",
    "goodbye",
    "cheers",
    "love",
    "hope",
    "joy",
    "peace",
    "luck",
    "street",
    "road",
    "lane",
    "avenue",
    "square",
    "court",
    "close",
    "crescent",
    "drive",
    "walk",
    "way",
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "for",
    "with",
    "by",
    "at",
    "to",
    "in",
    "on",
    "from",
    "woollen",
    "linen",
    "cotton",
    "leather",
    "timber",
    "stone",
    "brick",
    "glass",
    "iron",
    "steel",
    "baker",
    "miller",
    "smith",
    "taylor",
    "cooper",
    "fletcher",
    "turner",
    "mason",
    "potter",
    "wright",
)

_ABBREVIATIONS = (
    "Ltd.",
    "Co.",
    "Inc.",
    "St.",
    "Rd.",
    "Ave.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "No.",
    "Est.",
    "Dept.",
    "Bros.",
    "etc.",
    "approx.",
    "Tel:",
    "Ref:",
    "Vol.",
    "pp.",
    "Ch.",
)

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

_TLDS = ("com", "co.uk", "org", "net", "io", "shop", "cafe")


@dataclass(frozen=True)
class TextSpec:
    """One reward prompt: the surface, the exact string the image must show,
    the string split into the words the detector is expected to box, and the
    family it was drawn from (for per-family metrics)."""

    surface: str
    text: str
    words: tuple[str, ...]
    family: str

    @property
    def prompt(self) -> str:
        return f'{self.surface} text reading "{self.text}"'


def filter_to_vocab(text: str, vocab: str) -> str:
    """Drop every character the recognizer cannot emit. Spaces survive because
    they are word boundaries, not glyphs."""
    return "".join(ch for ch in text if ch == " " or ch in vocab)


def _case(word: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.45:
        return word.lower()
    if r < 0.85:
        return word[:1].upper() + word[1:].lower()
    return word.upper()


def _nonword(rng: random.Random, length: int) -> str:
    consonants = "bcdfghjklmnprstvwz"
    vowels = "aeiou"
    return "".join(rng.choice(consonants if i % 2 == 0 else vowels) for i in range(length))


def _price(rng: random.Random, vocab: str) -> str:
    symbols = [s for s in ("£", "$", "€") if s in vocab] or [""]
    symbol = rng.choice(symbols)
    if rng.random() < 0.5:
        return f"{symbol}{rng.randint(1, 999)}.{rng.randint(0, 99):02d}"
    if rng.random() < 0.5:
        return f"{symbol}{rng.randint(1, 99)}"
    return f"{rng.randint(1, 99)}p" if symbol == "£" else f"{symbol}{rng.randint(1, 99)}.99"


def _date(rng: random.Random) -> str:
    day = rng.randint(1, 28)
    month = rng.randint(1, 12)
    year = rng.randint(1890, 2030)
    r = rng.random()
    if r < 0.3:
        return f"{day:02d}/{month:02d}/{year}"
    if r < 0.55:
        return f"{_MONTHS[month - 1]} {day}, {year}"
    if r < 0.8:
        return f"{day} {_MONTHS[month - 1][:3]} {year}"
    return f"Est. {year}"


def _time(rng: random.Random) -> str:
    h = rng.randint(0, 23)
    m = rng.choice((0, 15, 30, 45, rng.randint(0, 59)))
    if rng.random() < 0.5:
        return f"{h:02d}:{m:02d}"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {'am' if h < 12 else 'pm'}"


def _url(rng: random.Random, words: Sequence[str]) -> str:
    host = rng.choice(words).lower() + rng.choice(("", rng.choice(words).lower()))
    url = f"{host}.{rng.choice(_TLDS)}"
    if rng.random() < 0.4:
        url = "www." + url
    if rng.random() < 0.3:
        url += "/" + rng.choice(words).lower()
    return url


def _number(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.4:
        return str(rng.randint(0, 9999))
    if r < 0.6:
        return f"{rng.randint(1, 99)}%"
    if r < 0.8:
        return f"No. {rng.randint(1, 999)}"
    return f"{rng.randint(1, 500)}{rng.choice(('kg', 'm', 'km', 'ml', 'cm', 'g'))}"


def generate_text_spec(
    rng: random.Random,
    vocab: str = DEFAULT_VOCAB,
    *,
    words: Sequence[str] = BUILTIN_WORDS,
    surfaces: Sequence[str] = DEFAULT_SURFACES,
    family_weights: dict[str, float] | None = None,
    min_words: int = 1,
    max_words: int = 8,
) -> TextSpec:
    """Draw one prompt. Deterministic for a given ``rng`` state."""
    weights = dict(DEFAULT_FAMILY_WEIGHTS)
    if family_weights:
        weights.update(family_weights)
    families = [f for f, w in weights.items() if w > 0]
    family = rng.choices(families, weights=[weights[f] for f in families], k=1)[0]
    n = rng.randint(min_words, max_words)

    if family == "words":
        tokens = [_case(rng.choice(words), rng) for _ in range(n)]
    elif family == "nonword":
        tokens = [_case(_nonword(rng, rng.randint(3, 8)), rng) for _ in range(max(1, n // 2))]
    elif family == "price":
        head = [_case(rng.choice(words), rng) for _ in range(max(1, n - 2))]
        tokens = [*head, "-", _price(rng, vocab)] if rng.random() < 0.5 else [*head, _price(rng, vocab)]
    elif family == "date":
        head = [_case(rng.choice(words), rng) for _ in range(max(0, n - 3))]
        tokens = [*head, *_date(rng).split(" ")]
    elif family == "time":
        head = [_case(rng.choice(words), rng) for _ in range(max(0, n - 2))]
        tokens = [*head, *_time(rng).split(" ")]
    elif family == "url":
        tokens = [_url(rng, words)]
    elif family == "abbrev":
        head = [_case(rng.choice(words), rng) for _ in range(max(1, n - 1))]
        abbrev = rng.choice(_ABBREVIATIONS)
        tokens = [abbrev, *head] if abbrev.endswith(":") else [*head, abbrev]
    elif family == "number":
        head = [_case(rng.choice(words), rng) for _ in range(max(0, n - 2))]
        tokens = [*head, *_number(rng).split(" ")]
    else:  # pragma: no cover - guarded by the weights table
        raise ValueError(family)

    cleaned = [filter_to_vocab(t, vocab) for t in tokens]
    cleaned = [t for t in cleaned if t]
    if not cleaned:
        cleaned = [_case(rng.choice(words), rng)]
    return TextSpec(
        surface=rng.choice(list(surfaces)),
        text=" ".join(cleaned),
        words=tuple(cleaned),
        family=family,
    )


def load_word_list(path: str | None) -> tuple[str, ...]:
    """One word per line; blank lines and ``#`` comments ignored. Falls back to
    the built-in list when ``path`` is empty or unreadable."""
    if not path:
        return BUILTIN_WORDS
    try:
        with open(path, encoding="utf-8") as f:
            words = tuple(w.strip() for w in f if w.strip() and not w.lstrip().startswith("#"))
    except OSError:
        return BUILTIN_WORDS
    return words or BUILTIN_WORDS


@dataclass
class WordAlignment:
    """Result of :func:`align_words`. ``pairs`` are ``(box_index, target_index)``
    in target order; ``missing`` are target indices no box was assigned to."""

    pairs: list[tuple[int, int]] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)


def reading_order(boxes: Sequence[Sequence[float]], line_tolerance: float = 0.5) -> list[int]:
    """Indices of ``boxes`` (``xmin, ymin, xmax, ymax`` in any consistent unit)
    sorted top-to-bottom, then left-to-right within a line. Two boxes share a
    line when their vertical centres differ by less than ``line_tolerance``
    times the smaller box height."""
    idx = sorted(range(len(boxes)), key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
    lines: list[list[int]] = []
    for i in idx:
        b = boxes[i]
        cy = (b[1] + b[3]) / 2
        h = max(b[3] - b[1], 1e-6)
        placed = False
        if lines:
            last = lines[-1]
            ref = boxes[last[0]]
            ref_cy = (ref[1] + ref[3]) / 2
            ref_h = max(ref[3] - ref[1], 1e-6)
            if abs(cy - ref_cy) < line_tolerance * min(h, ref_h):
                last.append(i)
                placed = True
        if not placed:
            lines.append([i])
    out: list[int] = []
    for line in lines:
        out.extend(sorted(line, key=lambda i: boxes[i][0]))
    return out


def align_words(
    readings: Sequence[str],
    targets: Sequence[str],
    *,
    min_ratio: float = 0.34,
) -> WordAlignment:
    """Monotone alignment of ``readings`` (one no-grad decode per box, already
    in reading order) to ``targets``. Standard edit-distance DP over words
    where the match cost is ``1 - levenshtein_ratio``; a pair is kept only if
    its ratio clears ``min_ratio`` so a box that reads nothing like the target
    (a spurious detection, or a different word) never receives its gradient.
    Case-insensitive for the *pairing* only: the CTC target keeps its case."""
    n, m = len(readings), len(targets)
    if m == 0:
        return WordAlignment()
    if n == 0:
        return WordAlignment(missing=list(range(m)))

    sim = [[levenshtein_ratio(r.lower(), t.lower()) for t in targets] for r in readings]
    # dp[i][j]: min cost aligning readings[:i] with targets[:j]
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    skip_read, skip_target = 0.6, 1.0
    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == inf:
                continue
            if i < n and j < m:
                c = dp[i][j] + (1.0 - sim[i][j])
                if c < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = c
                    back[i + 1][j + 1] = (i, j)
            if i < n:
                c = dp[i][j] + skip_read
                if c < dp[i + 1][j]:
                    dp[i + 1][j] = c
                    back[i + 1][j] = (i, j)
            if j < m:
                c = dp[i][j] + skip_target
                if c < dp[i][j + 1]:
                    dp[i][j + 1] = c
                    back[i][j + 1] = (i, j)

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = back[i][j]
        if pi == i - 1 and pj == j - 1 and sim[pi][pj] >= min_ratio:
            pairs.append((pi, pj))
        i, j = pi, pj
    pairs.reverse()
    matched = {t for _, t in pairs}
    return WordAlignment(pairs=pairs, missing=[t for t in range(m) if t not in matched])


@dataclass(frozen=True)
class TextRewardMetrics:
    """Scalars for TensorBoard, from the reward recognizer's own decode."""

    word_ratio: float  #: mean edit ratio over target words (missing = 0)
    exact: float  #: fraction of prompts where every word read verbatim
    missing: float  #: fraction of target words with no box
    count: int


def text_reward_metrics(
    readings_per_image: Sequence[Sequence[str]],
    targets_per_image: Sequence[Sequence[str]],
    alignments: Sequence[WordAlignment],
) -> TextRewardMetrics:
    ratios: list[float] = []
    exact_hits = 0
    missing = 0
    total_words = 0
    for readings, targets, al in zip(readings_per_image, targets_per_image, alignments, strict=True):
        per_word = [0.0] * len(targets)
        for bi, ti in al.pairs:
            per_word[ti] = levenshtein_ratio(readings[bi], targets[ti])
        ratios.extend(per_word)
        missing += len(al.missing)
        total_words += len(targets)
        if targets and all(r >= 1.0 - 1e-9 for r in per_word):
            exact_hits += 1
    count = len(targets_per_image)
    return TextRewardMetrics(
        word_ratio=sum(ratios) / len(ratios) if ratios else 0.0,
        exact=exact_hits / count if count else 0.0,
        missing=missing / total_words if total_words else 0.0,
        count=count,
    )


def policy_read_metrics(
    policy_readings: dict[int, str],
    targets: Sequence[str],
    alignment: WordAlignment,
) -> TextRewardMetrics:
    """Metrics for one offline-source sample, where the boxes and the
    alignment come from the clean source image and only the words inside the
    grad-decoded crop get a policy reading (``policy_readings``: box index ->
    reading).

    * ``missing``: fraction of *all* target words the source alignment left
      unboxed, as in :func:`text_reward_metrics`.
    * ``word_ratio`` / ``exact``: over the target words that were either read
      from the policy crop or are missing (scored as 0). Words aligned on the
      source but lying outside the decoded crop have no policy reading and
      are left out rather than counted as 0."""
    read_pairs = [(bi, ti) for bi, ti in alignment.pairs if bi in policy_readings]
    counted = sorted({ti for _, ti in read_pairs} | set(alignment.missing))
    remap = {ti: k for k, ti in enumerate(counted)}
    sub = WordAlignment(
        pairs=[(k, remap[ti]) for k, (_, ti) in enumerate(read_pairs)],
        missing=[remap[t] for t in alignment.missing],
    )
    inner = text_reward_metrics([[policy_readings[bi] for bi, _ in read_pairs]], [[targets[t] for t in counted]], [sub])
    return TextRewardMetrics(
        word_ratio=inner.word_ratio,
        exact=inner.exact,
        missing=len(alignment.missing) / len(targets) if targets else 0.0,
        count=1,
    )


# -- scheduling and the per-epoch prompt bank ---------------------------------


def derive_text_reward_interval(primary_batches: int, samples_per_epoch: int) -> int:
    """Standard micro-batches per reward sample so ``samples_per_epoch`` reward
    steps land inside one epoch of the primary loader (mirrors the DPO add-on
    interval). ``samples_per_epoch >= primary_batches`` means every step."""
    if samples_per_epoch <= 0 or primary_batches <= 0:
        return 0
    return max(1, round(primary_batches / samples_per_epoch))


class TextRewardScheduler:
    """Decides on which micro-batches the reward term runs.

    The term fires every ``interval`` micro-batches, but never on a step that
    already carries the DPO add-on pair: the two would otherwise share one GPU
    step and peak memory. A blocked firing is carried over to the next free
    step so the per-epoch budget is still met."""

    def __init__(self, interval: int):
        self.interval = max(0, int(interval))
        self.pending = 0

    def due(self, epoch_step: int, blocked: bool) -> bool:
        if self.interval <= 0:
            return False
        if epoch_step % self.interval == 0:
            self.pending += 1
        if blocked or self.pending <= 0:
            return False
        self.pending -= 1
        return True

    def start_epoch(self, interval: int) -> None:
        self.interval = max(0, int(interval))
        self.pending = 0


@dataclass
class TextRewardSample:
    spec: TextSpec
    embedding: object  #: per-prompt conditioning, as the model's encode_text returns it
    negative_embedding: object | None
    image_path: str | None = None  #: offline source image (OFFLINE_IMAGES), else None
    image_sha256: str | None = None  #: dataset tool's content hash of ``image_path``, when known


class TextRewardBank:
    """One epoch's worth of reward prompts with their conditioning encoded
    once up front (the text encoder is offloaded during training; encoding
    per step would page it in every time). Cycles when exhausted."""

    def __init__(
        self,
        specs: Sequence[TextSpec],
        embeddings: Sequence[object],
        negative_embedding: object | None,
        image_paths: Sequence[str] | None = None,
        image_sha256s: Sequence[str | None] | None = None,
    ):
        if len(specs) != len(embeddings):
            raise ValueError("one embedding per spec")
        if image_paths is not None and len(image_paths) != len(specs):
            raise ValueError("one image per spec")
        if image_sha256s is not None and (image_paths is None or len(image_sha256s) != len(image_paths)):
            raise ValueError("one sha256 (or None) per image")
        self.specs = list(specs)
        self.embeddings = list(embeddings)
        self.negative_embedding = negative_embedding
        self.image_paths = list(image_paths) if image_paths is not None else None
        self.image_sha256s = list(image_sha256s) if image_sha256s is not None else None
        self.cursor = 0

    def __len__(self) -> int:
        return len(self.specs)

    def next(self) -> TextRewardSample:
        if not self.specs:
            raise RuntimeError("empty text reward bank")
        i = self.cursor % len(self.specs)
        self.cursor += 1
        return TextRewardSample(
            self.specs[i],
            self.embeddings[i],
            self.negative_embedding,
            self.image_paths[i] if self.image_paths is not None else None,
            self.image_sha256s[i] if self.image_sha256s is not None else None,
        )


def generate_text_specs(
    count: int,
    seed: int,
    vocab: str,
    *,
    words: Sequence[str] = BUILTIN_WORDS,
    min_words: int = 1,
    max_words: int = 8,
) -> list[TextSpec]:
    rng = random.Random(seed)
    return [generate_text_spec(rng, vocab, words=words, min_words=min_words, max_words=max_words) for _ in range(count)]


# -- offline dataset (Text Alignment Dataset Tool output) and wildcard export --

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_QUOTED = re.compile(r'"([^"]+)"')


def target_text_from_prompt(prompt: str) -> str | None:
    """The string a reward prompt asks the image to show: the first quoted
    span. ``Neon text reading "Open 24h"`` -> ``Open 24h``."""
    m = _QUOTED.search(prompt)
    return m.group(1).strip() if m else None


def spec_from_text(text: str, vocab: str, *, surface: str = "", family: str = "offline") -> TextSpec | None:
    words = [w for w in (filter_to_vocab(t, vocab) for t in text.split(" ")) if w]
    if not words:
        return None
    return TextSpec(surface=surface, text=" ".join(words), words=tuple(words), family=family)


def spec_from_prompt(prompt: str, vocab: str, *, family: str = "offline") -> TextSpec | None:
    """Build a :class:`TextSpec` from a full prompt (offline samples carry the
    prompt SwarmUI stamped into the image). Characters outside the
    recognizer's vocabulary are dropped from the target, as in generation."""
    text = target_text_from_prompt(prompt)
    if not text:
        return None
    words = [w for w in (filter_to_vocab(t, vocab) for t in text.split(" ")) if w]
    if not words:
        return None
    surface = prompt.split(" text reading", 1)[0].strip() if " text reading" in prompt else ""
    return TextSpec(surface=surface, text=" ".join(words), words=tuple(words), family=family)


class TextRewardDatasetEntry(tuple):
    """One offline sample: unpacks, indexes and compares exactly like the
    ``(spec, prompt, image_path)`` 3-tuple, and additionally carries
    ``sha256`` — the dataset tool's content hash of the image from
    ``<stem>.json`` (``None`` when there is no JSON or it has no hash)."""

    def __new__(cls, spec: TextSpec, prompt: str, image_path: str, sha256: str | None = None):
        entry = super().__new__(cls, (spec, prompt, image_path))
        entry.sha256 = sha256
        return entry

    def __getnewargs__(self):
        return (*self, self.sha256)

    @property
    def spec(self) -> TextSpec:
        return self[0]

    @property
    def prompt(self) -> str:
        return self[1]

    @property
    def image_path(self) -> str:
        return self[2]


def load_text_reward_dataset(folder: str, vocab: str) -> list[TextRewardDatasetEntry]:
    """``[(spec, prompt, image_path), ...]`` for every image in ``folder``
    (recursive) with a ``<stem>.txt`` prompt beside it. A ``<stem>.json`` from
    the dataset tool supplies ``prompt``/``text`` when present, and its
    ``sha256`` rides along as ``entry.sha256``. Sorted by path so the order is
    stable across epochs."""
    out: list[TextRewardDatasetEntry] = []
    if not folder or not os.path.isdir(folder):
        return out
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            image_path = os.path.join(root, name)
            prompt = None
            explicit_text = None
            sha256 = None
            json_path = os.path.join(root, stem + ".json")
            if os.path.isfile(json_path):
                try:
                    with open(json_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    prompt = meta.get("prompt") or None
                    explicit_text = meta.get("text") or None
                    digest = meta.get("sha256")
                    if isinstance(digest, str) and digest.strip():
                        sha256 = digest.strip().lower()
                except (OSError, ValueError):
                    prompt = None
            if prompt is None:
                txt_path = os.path.join(root, stem + ".txt")
                if not os.path.isfile(txt_path):
                    continue
                with open(txt_path, encoding="utf-8") as f:
                    prompt = f.readline().strip()
            if explicit_text:
                # The dataset tool's target text is authoritative: a long real
                # caption may hold spoken quotes before the sign text.
                spec = spec_from_text(explicit_text, vocab, surface=prompt.split(" text reading", 1)[0].strip())
            else:
                spec = spec_from_prompt(prompt, vocab)
            if spec is None:
                continue
            out.append(TextRewardDatasetEntry(spec, prompt, image_path, sha256))
    return out


def write_wildcard_file(
    path: str,
    count: int,
    seed: int,
    vocab: str = DEFAULT_VOCAB,
    *,
    words: Sequence[str] = BUILTIN_WORDS,
    min_words: int = 1,
    max_words: int = 8,
) -> int:
    """Write ``count`` reward prompts, one per line, as a SwarmUI wildcard
    (``<wildcard:name>`` picks a line). Returns the number written."""
    specs = generate_text_specs(count, seed, vocab, words=words, min_words=min_words, max_words=max_words)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for spec in specs:
            f.write(spec.prompt + "\n")
    return len(specs)
