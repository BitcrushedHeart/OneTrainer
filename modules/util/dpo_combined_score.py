"""Combining the DPO curation axes — on-image text, adherence, age, realism,
scalar quality — into one ordering for a group.

Two combiners, because they answer different questions:

* :func:`gate_then_rank` is the curation rule. On-image text is the first
  *gate* when the caption asks for text, adherence the next, and age
  agreement the third when the caption claims an age: an image that fails its
  prompt can never outrank one that passes, however photographic it looks,
  because a chosen image that ignores the prompt teaches the model to ignore
  prompts — and a chosen image with the wrong words on its sign teaches it to
  misspell. Inside the gates realism ranks (lower P(AI) first) and scalar
  quality breaks ties. Nothing is summed, so the UI can still say *which*
  axis put an image where it is.

* :func:`weighted_z` is a single scalar for a badge and for the experiment's
  comparison against the gate: within-group z-scores, sign-corrected, weighted.
  A scalar throws away the "which axis" information, which is exactly why it
  is not the curation rule.

Inputs are ``{path: {"text", "adherence", "age", "realism", "quality"}}`` with
any value ``None`` for unscored. ``realism`` is P(AI-generated) — lower is
better; the others are higher-is-better (``text`` is the OCR match against the
text the caption asks for, absent when it asks for none; ``age`` is the
age-head agreement with the caption's claim, absent when the caption claims no
age).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import cmp_to_key

#: ``text`` carries the heaviest default weight: a wrong or missing word on a
#: chosen image is poison for a text-rendering model, so when the caption asks
#: for text that axis should dominate the scalar the way it dominates the gate.
#: It exceeds the other defaults combined (3.5), so in a pair where every other
#: axis prefers the garbled sign, the verbatim one still comes out ahead.
DEFAULT_WEIGHTS = {"text": 4.0, "adherence": 1.0, "age": 1.0, "realism": 1.0, "quality": 0.5}
#: Signs that turn each axis into higher-is-better.
_SIGN = {"text": 1.0, "adherence": 1.0, "age": 1.0, "realism": -1.0, "quality": 1.0}

#: Text similarity is an edit-distance ratio on a few words; one wrong letter
#: in a short sign is ~0.1, so only rounding-level differences are a tie.
DEFAULT_TEXT_MARGIN = 0.02
#: Adherence differences smaller than this are "the same tier" and fall
#: through to realism. Adherence is a mean of probabilities from a zero-shot
#: model; a few hundredths is noise, not a verdict.
DEFAULT_ADHERENCE_MARGIN = 0.05
#: Age agreement is a Gaussian in years; a tenth is about two years of
#: estimate difference near the claim, which is inside the head's error.
DEFAULT_AGE_MARGIN = 0.1
DEFAULT_REALISM_MARGIN = 0.0


def _cmp_axis(a: float | None, b: float | None, *, sign: float, margin: float) -> int:
    """+1 if ``a`` beats ``b`` on this axis, -1 if it loses, 0 if it is a tie
    (within ``margin``) or both are unscored. A scored value beats ``None``."""
    if a is None and b is None:
        return 0
    if a is None:
        return -1
    if b is None:
        return 1
    delta = (float(a) - float(b)) * sign
    if abs(delta) <= margin:
        return 0
    return 1 if delta > 0 else -1


def prefer(
    a: Mapping,
    b: Mapping,
    *,
    adherence_margin: float = DEFAULT_ADHERENCE_MARGIN,
    realism_margin: float = DEFAULT_REALISM_MARGIN,
    age_margin: float = DEFAULT_AGE_MARGIN,
    text_margin: float = DEFAULT_TEXT_MARGIN,
) -> int:
    """Lexicographic preference: text, then adherence, then age, then realism,
    then quality. The three prompt axes gate before realism ranks: an image
    that garbles the requested text, misses the prompt, or misses the claimed
    age cannot be the chosen side. Returns +1 when ``a`` should rank above
    ``b``, -1 below, 0 indifferent."""
    for axis, margin in (
        ("text", text_margin),
        ("adherence", adherence_margin),
        ("age", age_margin),
        ("realism", realism_margin),
        ("quality", 0.0),
    ):
        verdict = _cmp_axis(a.get(axis), b.get(axis), sign=_SIGN[axis], margin=margin)
        if verdict:
            return verdict
    return 0


def gate_then_rank(
    images: Mapping[str, Mapping],
    *,
    adherence_margin: float = DEFAULT_ADHERENCE_MARGIN,
    realism_margin: float = DEFAULT_REALISM_MARGIN,
    text_margin: float = DEFAULT_TEXT_MARGIN,
) -> list[str]:
    """Paths best-first under :func:`prefer`.

    The margin makes the comparison non-transitive in principle (0.80 ~ 0.85 ~
    0.90 but 0.80 < 0.90); with the small default margin and a stable sort this
    is a rounding-level effect, and the alternative — hard tiers — puts 0.849
    and 0.851 in different buckets, which is worse.
    """

    def _cmp(x: str, y: str) -> int:
        # sorted() puts "smaller" first; we want the preferred one first.
        return -prefer(
            images[x],
            images[y],
            adherence_margin=adherence_margin,
            realism_margin=realism_margin,
            text_margin=text_margin,
        )

    return sorted(images.keys(), key=cmp_to_key(_cmp))


def weighted_z(images: Mapping[str, Mapping], weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """One scalar per path: the weighted mean of its within-group z-scores.

    Axes an image lacks are dropped from *its* mean (the weights renormalise
    over what it has); an axis constant across the group contributes 0. An
    image with nothing scored sits at 0, the group mean.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    stats: dict[str, tuple[float, float]] = {}
    for axis in weights:
        values = [float(v[axis]) for v in images.values() if v.get(axis) is not None]
        if len(values) < 2:
            stats[axis] = (float(values[0]) if values else 0.0, 0.0)
            continue
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        stats[axis] = (mean, math.sqrt(var))

    out: dict[str, float] = {}
    for path, scores in images.items():
        numerator = 0.0
        denominator = 0.0
        for axis, weight in weights.items():
            value = scores.get(axis)
            if value is None or weight == 0:
                continue
            mean, std = stats[axis]
            z = (float(value) - mean) / std if std > 0 else 0.0
            numerator += weight * _SIGN[axis] * z
            denominator += weight
        out[path] = numerator / denominator if denominator > 0 else 0.0
    return out
