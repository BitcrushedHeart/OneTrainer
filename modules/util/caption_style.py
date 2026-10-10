"""Tag-list vs natural-language caption classification.

Ported verbatim (logic-for-logic) from the Bitcrush Suite's caption SSoT,
``packages/pyshared/src/bitcrush_pyshared/caption/utils.py`` plus the
``CaptionType`` classifier from ``caption/autosort.py`` and the on-image-text
token shapes from ``caption/image_text.py``. That implementation
is battle-tested against a real 96k-caption dataset and is deliberately
stdlib-only, which is why it is copied rather than imported: importing it
would pull in the whole ``bitcrush_pyshared`` package (structlog, pillow,
sqlalchemy, ...) for two functions.

Keep this file in sync with the upstream SSoT by porting changes, not by
re-deriving behaviour — the tests carry its ground-truth corpora. The one
intentional difference is marked ``LOCAL DEVIATION`` in :func:`is_tag_line`.

Used here to keep tag-style captions out of off-policy DPO generation: a tag
list is a poor prompt for a realism negative (it under-specifies the scene,
so the generation shares little with the photo beyond keywords).
"""

import re
import unicodedata
from enum import Enum

_WHITESPACE_RE = re.compile(r"\s+")

#: Prefix of an on-image-text tag (``text: "OPEN 24 HOURS"`` or the legacy
#: ``text:<reading>``).
TEXT_TAG_PREFIX = "text:"

#: The quoted tag-line token, e.g. ``text: "OPEN 24 HOURS"``.
IMAGE_TEXT_TOKEN_RE = re.compile(r'text:\s*"[^"\r\n]*"', re.IGNORECASE)


def normalize_caption(text: str | None) -> str:
    """NFC-normalize, strip, and collapse internal whitespace runs."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text).strip())


def split_caption_lines(text: str | None) -> list[str]:
    """A caption blob split into individually-classifiable lines."""
    if not text:
        return []
    return [normalized for raw in text.splitlines() if (normalized := normalize_caption(raw))]


NL_STARTERS: frozenset[str] = frozenset({
    "her", "his", "their", "its", "my", "your", "our",
    "the", "a", "an",
    "and", "or", "but", "yet", "so", "nor",
    "she", "he", "they", "it", "we",
    "who", "which", "that", "this", "these", "those",
    "some", "any", "each", "every",
    "with", "from", "to", "by", "for", "into", "onto",
    "about", "between", "through", "during", "after", "before",
    "over", "under", "behind", "beside", "near", "next",
    "against", "along", "across",
    "while", "as", "if", "when", "where", "then", "also", "not", "no",
})  # fmt: skip

NL_TAG_ING_ALLOWLIST: frozenset[str] = frozenset({
    "ring", "string", "piercing", "king", "bling", "sling",
    "licking", "looking", "kissing", "sitting", "standing",
    "kneeling", "lying", "smiling", "riding", "squatting",
    "straddling", "fingering",
})  # fmt: skip

NL_TAG_ED_ALLOWLIST: frozenset[str] = frozenset({"red", "bed"})

NL_GRAMMAR_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "her", "his", "their", "its",
    "my", "your", "our", "this", "that", "these", "those",
    "some", "any", "each", "every",
})  # fmt: skip

NL_SENTENCE_MARKERS: tuple[str, ...] = (
    " is ", " are ", " was ", " were ", " has ", " have ", " had ",
    " the ", " a ", " an ", " with ", " on ", " in ", " from ", " at ",
    " and ", " or ", " than ", " then ",
)  # fmt: skip

# " at " is deliberately absent here: booru-style tags like "looking at
# viewer" must stay tag-shaped. is_nl_fragment still catches genuine NL
# uses of "at" because its marker scan is gated on a grammar word.
TAG_SEGMENT_SENTENCE_INDICATORS: tuple[str, ...] = (
    " is ", " are ", " was ", " were ", " has ", " have ", " had ",
    " the ", " a ", " an ", " with ", " on ", " in ", " from ",
)  # fmt: skip

TAG_VERB_STARTS: tuple[str, ...] = (
    "photo of", "image of", "picture of", "high-quality", "professional",
)  # fmt: skip

TAG_KNOWN_PATTERNS: tuple[str, ...] = (
    "cm", "year-old", "years old",
    "skin", "hair", "build", "height",
    "caucasian", "european", "asian", "african", "american", "hispanic",
    "slim", "athletic", "average", "petite", "tall", "short",
    "shot", "angle", "view", "selfie", "photo",
    "breasts", "areolae", "areolas", "penis",
    "lighting", "expression",
)  # fmt: skip

# Performance guard only — genuine booru tag dumps routinely exceed 300
# characters, so prose protection comes from the clause checks below, not
# from line length.
_MAX_TAG_LINE_LENGTH = 2000
#: A comma segment of this many words is prose, never a tag token.
_MAX_TAG_SEGMENT_WORDS = 10
_PIECE_SPLIT_RE = re.compile(r"\.\s+")

# Verbs that mark a comma/period piece as a genuine clause rather than a
# tag idiom — "A young woman stands in the studio" is prose; "screenshot
# from a video" is a tag despite its article.
NL_CLAUSE_VERBS: frozenset[str] = frozenset({
    "is", "are", "was", "were", "has", "have", "had",
    "stands", "sits", "poses", "wears", "holds", "looks", "takes",
    "walks", "smiles", "shows", "reveals", "features", "includes",
    "contains", "appears", "stand", "sit", "pose", "wear", "hold",
    "walk", "smile", "show",
})  # fmt: skip

# Exact class tags that TAG_VERB_STARTS would otherwise reject.
TAG_EXACT_ALLOWLIST: frozenset[str] = frozenset({
    "professional photo", "amateur photo", "professional", "amateur",
})  # fmt: skip

# Proper-name metadata that resembles a clause. Case-sensitive, and only
# accepted inside a list with at least three independently tag-shaped pieces.
_TAG_LIST_CLAUSE_LABELS: frozenset[str] = frozenset({"She Is New"})
_TAG_LIST_PREPOSITIONS: frozenset[str] = frozenset({"on", "in", "from"})
_TAG_NAME_QUESTION_STARTERS = NL_STARTERS | {"what", "why", "how", "really"}


def is_nl_fragment(segment: str) -> bool:
    """Whether a comma/period ``segment`` reads as a natural-language fragment."""
    lower = segment.lower().strip()
    if not lower:
        return False
    words = lower.split()
    if not words:
        return False
    first = words[0]
    if first in NL_STARTERS:
        return True
    if first.endswith("ing") and len(first) > 4 and first not in NL_TAG_ING_ALLOWLIST:
        return True
    # A past-participle opener only reads as a clause when it has an object or
    # complement behind it ("seated on a bench", "captured at a low angle"). A
    # one- or two-word segment is a descriptor tag ("subdued", "tanned skin",
    # "mixed race"); reading those as prose tipped whole tag lines under the
    # tag ratio.
    if first.endswith("ed") and len(first) > 3 and first not in NL_TAG_ED_ALLOWLIST and len(words) >= 3:
        return True
    if "'s " in lower or "’s " in lower or "?s " in lower:
        return True
    if len(words) >= 2 and any(word in NL_GRAMMAR_WORDS for word in words):
        padded = f" {lower} "
        for marker in NL_SENTENCE_MARKERS:
            if marker in padded:
                return True
    return False


def is_tag_like_segment(segment: str) -> bool:
    """Whether a single comma ``segment`` looks like a tag rather than prose."""
    segment = segment.strip()
    if not segment:
        return False
    words = segment.split()
    if len(words) > 5:
        return False
    lower = segment.lower()
    if lower in TAG_EXACT_ALLOWLIST:
        return True
    padded = f" {lower} "
    for indicator in TAG_SEGMENT_SENTENCE_INDICATORS:
        if indicator in padded:
            return False
    for start in TAG_VERB_STARTS:
        if lower.startswith(start):
            return False
    for pattern in TAG_KNOWN_PATTERNS:
        if pattern in lower:
            return True
    return len(words) <= 3


def _neutralise_image_text(line: str) -> str:
    """Collapse on-image-text tokens to the bare tag ``text``.

    A transcription is whatever the image says — often a full sentence — so it
    must never vote on whether its line is prose. Both the quoted tag-line token
    (``text: "OPEN 24 HOURS"``) and a legacy ``text:<reading>`` segment count as
    one ordinary tag.
    """
    line = IMAGE_TEXT_TOKEN_RE.sub("text", line)
    return ",".join(
        " text" if segment.strip().casefold().startswith(TEXT_TAG_PREFIX) else segment for segment in line.split(",")
    ).strip()


def _is_tag_list_piece(piece: str, nl_segments: set[str] | None) -> bool:
    """Vote on list metadata without relaxing standalone fragment detection."""
    lower = piece.lower()
    words = lower.split()
    if (nl_segments and lower in nl_segments) or is_nl_fragment(piece) or len(words) >= 6:
        return False
    if is_tag_like_segment(piece.rstrip(".")):
        return True
    # Short descriptors such as "watermark in center", "motion blur on hand"
    # and "Shot on Canon" have prepositions but no sentence grammar. Keep
    # article-bearing fragments ("screenshot from a video") as negative votes.
    return bool(_TAG_LIST_PREPOSITIONS.intersection(words)) and not (
        NL_GRAMMAR_WORDS.intersection(words) or NL_CLAUSE_VERBS.intersection(words)
    )


def _is_damaged_tag_name(segment: str) -> bool:
    """Recognize replacement '?' characters in short proper-name metadata."""
    words = segment.split()
    return (
        2 <= len(words) <= 4
        and not segment.endswith("?")
        and words[0].rstrip("?").lower() not in _TAG_NAME_QUESTION_STARTERS
        and not is_nl_fragment(segment)
        and not NL_CLAUSE_VERBS.intersection(segment.lower().split())
        and all(word.replace("?", "a").istitle() for word in words)
        and all(index > 0 and segment[index - 1].isalpha() for index, char in enumerate(segment) if char == "?")
    )


def is_tag_line(line: str, nl_segments: set[str] | None = None) -> bool:
    """Classify a caption line as a tag list ("blonde hair, blue eyes,
    outdoors") versus natural language ("A woman stands in a garden.").

    ``nl_segments``: optional lowercased comma-split segments from the
    caption's prose portion; a line or segment matching one is treated as
    natural language, not a tag.
    """
    line = line.strip()
    if not line or len(line) > _MAX_TAG_LINE_LENGTH:
        return False
    if TEXT_TAG_PREFIX in line.casefold():
        line = _neutralise_image_text(line)
    if "—" in line or "–" in line:  # em/en dashes signal prose
        return False
    if nl_segments and line.lower() in nl_segments:
        return False

    had_period = line.endswith(".")
    line = line.removesuffix(".").rstrip()
    if not line:
        return False
    segments = [segment.strip() for segment in line.split(",") if segment.strip()]
    if not segments:
        return False
    # Scraped name tags can contain lost accents ("Alexander Skarsg?rd",
    # "Chlo? Grace Moretz"). Only tolerate them within multi-token lists;
    # question marks at the end of a segment remain a decisive prose signal.
    if "?" in line and (
        len(segments) < 3 or any("?" in segment and not _is_damaged_tag_name(segment) for segment in segments)
    ):
        return False
    # A genuine clause anywhere in the line means prose is present and the
    # line must never be classified as a tag list. A clause needs a verb —
    # article-bearing tag idioms ("screenshot from a video") and short tag
    # pieces around a stray period ("fair skin. nipples") are tolerated.
    pieces = [
        (index, piece.strip())
        for index, segment in enumerate(segments)
        for piece in _PIECE_SPLIT_RE.split(segment)
        if piece.strip()
    ]
    tag_count = sum(_is_tag_list_piece(piece, nl_segments) for _, piece in pieces) if len(segments) >= 3 else 0
    clause_label_indices: set[int] = set()
    if len(segments) >= 4 and tag_count >= 3 and not had_period:
        # A named label or viewer-perspective tag may resemble a sentence.
        # It must follow other metadata, occupy a whole comma segment, and
        # stay short and unpunctuated. It does not itself earn a tag vote.
        clause_label_indices = {
            index
            for index, segment in enumerate(segments)
            if index > 0
            and not (nl_segments and segment.lower() in nl_segments)
            and not any(char in segment for char in ".?!;:")
            and len(segment.split()) < _MAX_TAG_SEGMENT_WORDS
            and (segment in _TAG_LIST_CLAUSE_LABELS or segment.lower().startswith("the viewer is "))
        }
    if any(
        len(words := [word.strip(".") for word in piece.lower().split()]) >= 3
        and is_nl_fragment(piece)
        and NL_CLAUSE_VERBS.intersection(words)
        and index not in clause_label_indices
        for index, piece in pieces
    ):
        return False

    # No tag token is ten words long. When one comma-free run of that length
    # carries more than half the line's words, the line IS that sentence and
    # the short list pieces in front of it are its preamble ("Neutral, front
    # view, high-angle, fine grain amateur close-up of a woman ... into her
    # hair"); the piece ratio must not outvote it. A mixed paragraph whose
    # long piece is a minority of the words keeps its list classification.
    longest = max(len(segment.split()) for segment in segments)
    if longest >= _MAX_TAG_SEGMENT_WORDS and longest * 2 > len(line.split()):
        return False

    if len(segments) <= 2:
        if had_period or any(
            len(segment.split()) >= 6 or bool(nl_segments and segment.lower() in nl_segments) or is_nl_fragment(segment)
            for segment in segments
        ):
            return False
        return len(segments) == 2 or len(segments[0].split()) < 3 or is_tag_like_segment(segments[0])

    # A trailing full stop on a short comma list signals a prose sentence
    # with appended tags; only list-shaped lines (4+ segments) tolerate one.
    if had_period and len(segments) < 4:
        return False
    # LOCAL DEVIATION from the upstream SSoT (remove once upstream handles it):
    # a line that opens with an article and closes with a full stop is a
    # sentence with a comma-separated modifier preamble ("A subdued, eye-level,
    # straight on, three-quarter view, clean photo of a person with dark
    # hair."). Upstream's preposition-tag vote counts "straight on" as a tag
    # and tips this shape over the ratio.
    if had_period and segments[0].split()[0].lower() in {"a", "an", "the"}:
        return False
    return tag_count / len(pieces) >= 0.6


class CaptionType(str, Enum):
    DESCRIPTIVE = "Descriptive"
    SHORT = "Short"
    TAG = "Tag"


DEFAULT_TYPE_THRESHOLD = 250
DEFAULT_TYPE_THRESHOLD_UNIT = "char"
#: A natural-language caption needs at least this many words; anything
#: shorter is a tag-style line. 8 keeps clipped clauses and bare fragments
#: ("a woman") out of the descriptive bucket.
DEFAULT_MIN_NL_WORDS = 8


def _measure(text: str, unit: str) -> int:
    if unit == "word":
        return len(text.split())
    return len(text)


def classify_caption_type(
    text: str,
    *,
    threshold: int = DEFAULT_TYPE_THRESHOLD,
    unit: str = DEFAULT_TYPE_THRESHOLD_UNIT,
    min_nl_words: int = DEFAULT_MIN_NL_WORDS,
) -> CaptionType:
    """``CaptionType`` for one caption. Tag detection wins over length; a
    non-tag line shorter than ``min_nl_words`` words is also TAG (set 0 to
    disable). Otherwise DESCRIPTIVE at or above ``threshold``, else SHORT."""
    if is_tag_line(text):
        return CaptionType.TAG
    words = text.split()
    if words and min_nl_words > 0 and len(words) < min_nl_words:
        return CaptionType.TAG
    if _measure(text, unit) >= threshold:
        return CaptionType.DESCRIPTIVE
    return CaptionType.SHORT
