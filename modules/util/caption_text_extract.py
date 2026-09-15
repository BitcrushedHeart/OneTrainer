"""Which parts of a caption are on-image text — Bitcrush Studio's Text Tool rule.

Vendored from ``bitcrush_pyshared.caption.text_extract`` (Bitcrush Suite) so the
DPO curation tool and Studio agree on what "the caption mentions text" means.
Kept as close to verbatim as possible so fixes can be carried across; the only
deliberate divergences are the pure-Python Levenshtein (Studio uses rapidfuzz,
which OneTrainer does not ship) and the dropped caption-editing helpers
(``replace_identified_text`` / ``clear_sentence_containing``), which a scorer
never needs.

The two-tier identification heuristic:

    Tier 1 — paired delimiters (straight/curly, single/double quotes).
    Tier 2 — verb cues followed by an unquoted clause ("a sign reads OPEN").

Both tiers run on every caption; results dedupe against each other before being
returned. Tier 1 wins on overlap (it has the stronger evidence — the writer
chose to delimit the string).

The matching primitives normalise both sides (lowercase, strip punctuation incl.
all quote variants, collapse whitespace, alias ``&``/number words) and compare
with a Levenshtein ratio and a token-set ratio.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Tier = Literal[1, 2]


@dataclass(frozen=True)
class IdentifiedText:
    """One VLM-identified text reference lifted out of a caption."""

    start: int  # caption[start:end] is the substring to replace
    end: int
    text: str  # the inner text (without delimiters / verb cue)
    tier: Tier  # 1 = quoted, 2 = verb-cued
    delimiter: str = ""  # for Tier 1: opening delimiter char; "" for Tier 2


# ── Tier 1: quoted strings ────────────────────────────────────────────────────

# (open, close) pairs. Curly singles are slightly tricky because U+2019 is also
# the typographic apostrophe ("don't" -> "don\u2019t"); we still try to match
# them as a pair but discard matches whose contents look like a contraction.
_QUOTE_PAIRS = [
    ('"', '"'),
    ("“", "”"),  # curly double
    ("'", "'"),
    ("\u2018", "\u2019"),  # curly single
    ("«", "»"),  # guillemets
    ("「", "」"),  # CJK corner brackets
    ("`", "`"),  # transcription models sometimes emit Markdown quotes
]

# Common transcription mismatch: a model opens with a straight quote and
# closes with a curly quote (or vice versa). Keep the canonical pair above for
# replacement, but accept either close while identifying the source span.
_QUOTE_CLOSES = {
    '"': ('"', "”"),
    "“": ("”", '"'),
    "'": ("'", "\u2019"),
    "\u2018": ("\u2019", "'"),
}
_QUOTE_OPENERS = {open_c for open_c, _ in _QUOTE_PAIRS}
_QUOTE_EDGE_CHARS = _QUOTE_OPENERS | {close_c for _, close_c in _QUOTE_PAIRS}
_SPEECH_CUES = {"say", "says", "said"}
_HUMAN_SPEECH_SUBJECTS = {
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "man",
    "woman",
    "boy",
    "girl",
    "person",
    "people",
    "child",
}
_BARE_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _word_before(caption: str, idx: int) -> str:
    """The bare word ending just before `idx` (an opening-quote position)."""
    j = idx
    while j > 0 and caption[j - 1].isspace():
        j -= 1
    k = j
    while k > 0 and not caption[k - 1].isspace():
        k -= 1
    return caption[k:j].strip(_EDGE_PUNCT).lower()


def _word_after(caption: str, idx: int) -> str:
    """The bare word starting at/after `idx` (just past a closing quote)."""
    n = len(caption)
    j = idx
    while j < n and caption[j].isspace():
        j += 1
    k = j
    while k < n and not caption[k].isspace():
        k += 1
    return caption[j:k].strip(_EDGE_PUNCT).lower()


def _human_speech_subject_before(caption: str, cue_start: int) -> bool:
    """Whether the word immediately before a speech cue is a human subject."""
    words = _BARE_WORD_RE.findall(caption[:cue_start].lower())
    return bool(words and words[-1] in _HUMAN_SPEECH_SUBJECTS)


_QUOTE_CLOSERS = {close_c for _, close_c in _QUOTE_PAIRS} | {'"', "'", "’", "”"}
_AUXILIARY_VERBS = {"is", "are", "was", "were", "be", "been", "being"}


def _quoted_object_before_cue(caption: str, cue_start: int) -> bool:
    """Whether a verb cue's object is the quoted string right before it.

    OneTrainer addition (carry back to Studio): in ``'Full Time' written
    underneath`` or ``"OPEN" is printed above the door`` the cue's text is
    the Tier 1 quote and the words *after* the cue say where it sits, not
    what it says. Forward capture from such a cue needs the same caps/digit
    evidence as a participial cue, so ``underneath`` transcribes nothing while
    ``'A' reads HAPPY BIRTHDAY`` still does. An auxiliary between quote and
    cue (``'X' is written``) is looked through.
    """
    j = cue_start
    while j > 0 and caption[j - 1].isspace():
        j -= 1
    k = j
    while k > 0 and not caption[k - 1].isspace():
        k -= 1
    if caption[k:j].lower() in _AUXILIARY_VERBS:
        j = k
        while j > 0 and caption[j - 1].isspace():
            j -= 1
    return j > 0 and caption[j - 1] in _QUOTE_CLOSERS


def _is_spoken_quote(caption: str, quote_start: int) -> bool:
    """Reject direct speech while retaining object text (``a note says 'hi'``)."""
    words = _BARE_WORD_RE.findall(caption[:quote_start].lower())
    return len(words) >= 2 and words[-1] in _SPEECH_CUES and words[-2] in _HUMAN_SPEECH_SUBJECTS


def _find_quoted(caption: str) -> list[IdentifiedText]:
    out: list[IdentifiedText] = []
    for open_c, close_c in _QUOTE_PAIRS:
        i = 0
        while True:
            start = caption.find(open_c, i)
            if start < 0:
                break
            # For straight quotes, only treat as opener if preceded by start-of-string,
            # whitespace, or sentence punctuation — avoids treating apostrophes as
            # quote openers (e.g. "don't").
            if open_c == close_c:
                prev = caption[start - 1] if start > 0 else ""
                if prev and not (prev.isspace() or prev in "([{,.;:!?\u2014\u2013-"):
                    i = start + 1
                    continue
            end = min(
                (
                    candidate
                    for candidate in (
                        caption.find(candidate_close, start + 1)
                        for candidate_close in _QUOTE_CLOSES.get(open_c, (close_c,))
                    )
                    if candidate >= 0
                ),
                default=-1,
            )
            if end < 0:
                # Unbalanced — skip this opener and move on.
                i = start + 1
                continue
            inner = caption[start + 1 : end]
            # Reject empty quotes and obvious contractions for curly singles.
            if not inner.strip():
                i = end + 1
                continue
            if (
                open_c == "\u2018"
                and close_c == "\u2019"
                and start > 0
                and caption[start - 1].isalpha()
                and " " not in inner
            ):
                # Curly-single often means contraction (\u2019 is the apostrophe).
                # If the matched span has no whitespace and is wedged inside a word,
                # skip it.
                i = start + 1
                continue
            # ISSUE-0486 F6b: a straight-single-quote span whose inner text is a
            # single lowercase letter is a spoken abbreviation ("rock 'n' roll"),
            # not on-image text.
            # Sentence punctuation immediately inside a closing quote belongs
            # to the caption sentence, not to the OCR candidate.
            inner_stripped = inner.strip().rstrip(".!?").rstrip()
            if not inner_stripped:
                i = end + 1
                continue
            if open_c == "'" and len(inner_stripped) == 1 and inner_stripped.isalpha() and inner_stripped.islower():
                i = end + 1
                continue
            # ISSUE-0486 F4: scare-quote context \u2014 reject a quoted token that is
            # named/nicknamed or trailed by a shape word ('V' shape, nicknamed 'Joe').
            if _word_before(caption, start) in _SCARE_BEFORE:
                i = end + 1
                continue
            if _word_after(caption, end + 1) in _SCARE_AFTER:
                i = end + 1
                continue
            if _is_spoken_quote(caption, start):
                i = end + 1
                continue
            out.append(
                IdentifiedText(
                    start=start,
                    end=end + 1,
                    text=inner_stripped,
                    tier=1,
                    delimiter=open_c,
                )
            )
            i = end + 1
    return out


# ── Tier 2: verb cues ─────────────────────────────────────────────────────────

# Trigger phrases. Each cue finds its own matches; we walk forward token-by-token
# from the cue position to extract the captured clause. Bounded by:
#   - sentence-end punctuation (. ! ?)
#   - comma / semicolon / colon / em-dash
#   - conjunction (and / or / but / while)
#   - state-verb / preposition (is, are, was, were, on, in, at, above, below, ...)
#   - another verb cue (lets nested cues each surface their own row)
#   - max 6 tokens (bounds runaway capture against narrative continuations
#     the spec didn't anticipate, e.g. "the sign reads HELLO shines brightly")
_VERB_CUES = [
    "reading",  # listed before "read" so longest-match-first picks "reading"
    "reads",
    "read",
    "says",
    "say",
    "said",
    "written",
    "writes",
    "write",
    "labelled",
    "labeled",
    "titled",
    "captioned",
    "spelled out",
    "spelling out",
    "spells out",
    "spell out",
    "spelling",
    "printed",
    "stamped",
    "inscribed",
    "engraved",
    "emblazoned",
    "embroidered",
    "the words",
    "the text",
]
_DISPLAY_CUES = ["displays text", "displays words", "displaying text", "displaying words"]
_ALL_CUES = _VERB_CUES + _DISPLAY_CUES

_ARTICLES = {"a", "an", "the", "some", "that", "this"}
_CONJ_WORDS = {"and", "or", "but", "while"}
_STATE_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "on",
    "in",
    "at",
    "above",
    "below",
    "near",
    "inside",
    "outside",
    "with",  # ISSUE-0486 F2a: prose boundary ("titled WAR with a poster …")
    "across",  # ISSUE-0486 F5: prose boundary ("emblazoned across the banner")
}
_CUE_FIRST_WORDS = {c.split()[0] for c in _ALL_CUES}

# ISSUE-0486 F5: participial cue verbs double as clothing/prop adjectives
# ("a printed dress", "embroidered boots"). Re-audit R1: for THESE cues only,
# always demand caps/digit evidence in the capture — a bare lowercase noun after
# them is a described material/pattern, not transcribed text.
_EVIDENCE_ONLY_CUES = {
    "labelled",
    "labeled",
    "printed",
    "stamped",
    "inscribed",
    "engraved",
    "emblazoned",
    "embroidered",
}

# ISSUE-0486 F3: figurative reads/says — a capture that opens with one of these
# is an idiom ("says nothing", "reads like", "says it all"), not transcription.
# Re-audit R3: only when the source token is lowercase — genuine ALL-CAPS text
# ("says LIKE US ON FACEBOOK") must survive.
_FIGURATIVE_FIRST = {"nothing", "like", "it", "north", "south", "east", "west"}

# ISSUE-0486 F4: Tier-1 scare-quote context. A quoted token is a description,
# not on-image text, when it is introduced as a name or trailed by a shape word.
# Re-audit R4: "called"/"named" dropped — they routinely precede genuine on-image
# names ('a boat named "Serenity"'); the stronger nicknamed/dubbed/so-called stay.
_SCARE_BEFORE = {"nicknamed", "dubbed", "so-called", "credit", "credited"}
_SCARE_AFTER = {"shape", "shaped", "pattern", "formation", "gesture"}
_EDGE_PUNCT = "\"'.,;:!?()[]{}"


def _is_caps_run(token: str) -> bool:
    """True when `token` is on-image evidence: digit, all-caps, or uncased script.

    Re-audit R2: Title-Case prose ("Bob", "Maria") is not a caps run — it has
    lowercase letters — so it never qualifies as a comma-list continuation item.
    """
    if any(c.isdigit() for c in token):
        return True
    if not any(c.isalpha() for c in token):
        return False
    # Scripts without case (for example CJK) have neither lowercase nor
    # uppercase letters; absence of lowercase is the useful evidence gate.
    return not any(c.islower() for c in token if c.isalpha())


_MAX_TIER2_TOKENS = 6
_SENTENCE_TERMINATORS = ".!?,;:\u2014\u2013\n"
_TOKEN_RE = re.compile(r"\S+")
_PASSIVE_CUE_RE = re.compile(
    r"\b(?:is|are|was|were)\s+"
    r"(?:written|printed|stamped|inscribed|engraved|emblazoned|embroidered)\b",
    re.IGNORECASE,
)


def _find_verb_cued(caption: str) -> list[IdentifiedText]:
    out: list[IdentifiedText] = []
    lower = caption.lower()
    seen_spans: set[tuple[int, int]] = set()
    # Regions already claimed by a (longer) cue match: "spelling out X" must
    # not ALSO fire the bare "spelling" cue and capture "out X".
    cue_regions: list[tuple[int, int]] = []

    # Sort cues longest-first so "the text" fires before "the" (not a cue) and
    # "displays text" fires before "displays" (which isn't a standalone cue).
    cues_sorted = sorted(_ALL_CUES, key=len, reverse=True)

    for cue in cues_sorted:
        i = 0
        while True:
            idx = lower.find(cue, i)
            if idx < 0:
                break
            # Word boundary check: previous char must be non-word, next char after cue must be space
            # or punct.
            prev_ok = idx == 0 or not lower[idx - 1].isalnum()
            after_idx = idx + len(cue)
            next_ok = after_idx >= len(lower) or not lower[after_idx].isalnum()
            if not (prev_ok and next_ok):
                i = idx + 1
                continue
            if any(idx >= rs and after_idx <= re_ for rs, re_ in cue_regions):
                i = idx + 1
                continue
            if cue in _SPEECH_CUES and _human_speech_subject_before(caption, idx):
                i = after_idx + 1
                continue
            cue_regions.append((idx, after_idx))

            text_span = _walk_capture(
                caption, after_idx, cue in _EVIDENCE_ONLY_CUES or _quoted_object_before_cue(caption, idx)
            )
            i = after_idx + 1
            if text_span is None:
                continue
            ts, te = text_span
            if (ts, te) not in seen_spans:
                seen_spans.add((ts, te))
                out.append(IdentifiedText(start=ts, end=te, text=caption[ts:te], tier=2, delimiter=""))
            # ISSUE-0486 F7: a caps/digit capture that stopped at a comma opens a
            # list — emit each following comma/and/or-separated caps run as its
            # own span ("reading STOP, YIELD, and MERGE").
            captured = caption[ts:te]
            if any(ch.isupper() or ch.isdigit() for ch in captured):
                p = te
                while p < len(caption) and caption[p].isspace():
                    p += 1
                if p < len(caption) and caption[p] == ",":
                    for a, b in _scan_comma_list(caption, p):
                        if (a, b) not in seen_spans:
                            seen_spans.add((a, b))
                            out.append(IdentifiedText(start=a, end=b, text=caption[a:b], tier=2, delimiter=""))
    return out


_PASSIVE_LINK_TOKENS = {"&", "+", "/"}


def _find_passive_cued(caption: str) -> list[IdentifiedText]:
    """Capture a conservative caps/digit run before a passive text cue.

    Forward walking cannot see ``WELCOME is written on the wall``. Walking
    backward without a strong gate would be destructive, so passive extraction
    accepts only a trailing all-caps/digit (or uncased-script) run.
    """
    out: list[IdentifiedText] = []
    seen_spans: set[tuple[int, int]] = set()
    for cue_match in _PASSIVE_CUE_RE.finditer(caption):
        text_span = _walk_passive_capture(caption, cue_match.start())
        if text_span is None or text_span in seen_spans:
            continue
        seen_spans.add(text_span)
        start, end = text_span
        out.append(
            IdentifiedText(
                start=start,
                end=end,
                text=caption[start:end],
                tier=2,
                delimiter="",
            )
        )
    return out


def _walk_passive_capture(caption: str, end: int) -> tuple[int, int] | None:
    """Return the trailing transcription-evidence run immediately before `end`."""
    candidates = list(_TOKEN_RE.finditer(caption, 0, end))
    captured: list[tuple[int, int, str]] = []
    for match in reversed(candidates):
        if len(captured) >= _MAX_TIER2_TOKENS:
            break
        start, token_end = match.span()

        while start < token_end and caption[start] in _QUOTE_EDGE_CHARS:
            start += 1
        while token_end > start and (
            caption[token_end - 1] in _QUOTE_EDGE_CHARS or caption[token_end - 1] in _SENTENCE_TERMINATORS
        ):
            token_end -= 1
        clean = caption[start:token_end]
        if not clean:
            continue
        if _is_caps_run(clean):
            captured.append((start, token_end, clean))
            continue
        if clean in _PASSIVE_LINK_TOKENS and captured:
            captured.append((start, token_end, clean))
            continue
        break

    while captured and captured[-1][2] in _PASSIVE_LINK_TOKENS:
        captured.pop()
    if not captured:
        return None
    return captured[-1][0], captured[0][1]


_LIST_CONJ = {"and", "or"}
_LIST_LEADERS = {"another"}


def _scan_comma_list(caption: str, start: int) -> list[tuple[int, int]]:
    """Emit comma/and/or-separated caps-run items following a Tier-2 capture.

    `start` points at the comma that ended the base capture. Each item must open
    with a caps/digit token; the scan stops at the first non-qualifying item, at
    sentence-end, or after 5 items.
    """
    out: list[tuple[int, int]] = []
    n = len(caption)
    pos = start
    while len(out) < 5:
        # Require a comma separator.
        while pos < n and caption[pos].isspace():
            pos += 1
        if pos >= n or caption[pos] != ",":
            break
        pos += 1
        while pos < n and caption[pos].isspace():
            pos += 1
        # Skip a single leading list conjunction ("and"/"or") or the explicit
        # cueless-item marker "another" ("OPEN, another CLOSED").
        m = _TOKEN_RE.match(caption, pos)
        if m and caption[m.start() : m.end()].strip(_EDGE_PUNCT).lower() in _LIST_CONJ | _LIST_LEADERS:
            pos = m.end()
            while pos < n and caption[pos].isspace():
                pos += 1

        item_begin: int | None = None
        item_end: int | None = None
        first = True
        stop_after = False
        while pos < n:
            while pos < n and caption[pos].isspace():
                pos += 1
            if pos >= n:
                break
            ch = caption[pos]
            if ch == ",":
                break
            if ch in _SENTENCE_TERMINATORS:
                stop_after = True
                break
            m = _TOKEN_RE.match(caption, pos)
            if not m:
                break
            wstart, wend = m.span()
            word_raw = caption[wstart:wend]
            trim = 0
            while trim < len(word_raw) and word_raw[-(trim + 1)] in _SENTENCE_TERMINATORS:
                trim += 1
            word_end = wend - trim
            word_text = caption[wstart:word_end]
            if not word_text:
                pos = wend
                continue
            if first:
                if not _is_caps_run(word_text):
                    return out  # non-qualifying item (prose/Title-Case) ends the list
                first = False
                item_begin = wstart
                item_end = word_end
            elif word_text.isalpha() and word_text.islower():
                break  # prose word — item ends here
            else:
                item_end = word_end
            pos = min(wend, word_end)
        if item_begin is not None and item_end is not None:
            out.append((item_begin, item_end))
        if stop_after:
            break
    return out


def _walk_capture(caption: str, start: int, cue_requires_evidence: bool = False) -> tuple[int, int] | None:
    """Walk forward from `start` collecting up to _MAX_TIER2_TOKENS words.

    Skips a single leading article. Stops when a token is a terminator class
    (state-verb, conjunction, another cue) or when a punctuation terminator
    appears. Returns the absolute (begin, end) span of the captured text.

    `cue_requires_evidence` (participial cues, R1) forces the caps/digit gate on
    regardless of article/"with" skipping — "a printed dress" transcribes nothing.
    """
    pos = start
    # Skip whitespace.
    while pos < len(caption) and caption[pos].isspace():
        pos += 1
    # ISSUE-0486 F1: a colon immediately after the cue ("a sign reads: OPEN") is
    # a lead-in, not a terminator. Skip a single leading colon + following space.
    if pos < len(caption) and caption[pos] == ":":
        pos += 1
        while pos < len(caption) and caption[pos].isspace():
            pos += 1
    # A paired quote will be emitted by Tier 1 and win on overlap. For an
    # unclosed quote, start after the opener so Tier 2 never treats punctuation
    # as part of the editable OCR text.
    if pos < len(caption) and caption[pos] in _QUOTE_OPENERS:
        pos += 1
        while pos < len(caption) and caption[pos].isspace():
            pos += 1
    if pos >= len(caption):
        return None

    tokens: list[tuple[int, int]] = []  # absolute (begin, end) spans
    skipped_article = False
    skipped_with = False
    require_evidence = False  # a leading article/"with" skip demands caps/digit text

    while pos < len(caption) and len(tokens) < _MAX_TIER2_TOKENS:
        # Skip whitespace.
        while pos < len(caption) and caption[pos].isspace():
            pos += 1
        if pos >= len(caption):
            break
        # Punctuation terminator stops the walk before any further tokens.
        if caption[pos] in _SENTENCE_TERMINATORS:
            break

        # Pull next "word" — non-whitespace run, but trim trailing terminators.
        m = _TOKEN_RE.match(caption, pos)
        if not m:
            break
        wstart, wend = m.span()
        # Trim trailing punctuation/quote marks so "STOP." and an unclosed
        # ``reads "STOP`` both identify just the editable text.
        word_end = wend
        had_terminator_after_word = False
        while word_end > wstart and (
            caption[word_end - 1] in _SENTENCE_TERMINATORS or caption[word_end - 1] in _QUOTE_EDGE_CHARS
        ):
            if caption[word_end - 1] in _SENTENCE_TERMINATORS:
                had_terminator_after_word = True
            word_end -= 1
        word_text = caption[wstart:word_end].lower()

        # Skip a single leading article (only once, only as the very first token).
        # Only skip when the article is *lowercase* in the source text; an
        # uppercase article (e.g. "THE END") is part of the on-image text.
        word_raw_clean = caption[wstart:word_end]
        if not tokens and not skipped_article and word_text in _ARTICLES and word_raw_clean.islower():
            skipped_article = True
            require_evidence = True
            pos = wend
            continue
        # ISSUE-0486 F2a: skip a single leading lowercase "with" the same way,
        # so "printed with JUST DO IT" captures the text, not the preposition.
        if not tokens and not skipped_with and word_text == "with" and word_raw_clean.islower():
            skipped_with = True
            require_evidence = True
            pos = wend
            continue

        # Terminator-class words stop the walk before adding this token.
        if (word_text in _CONJ_WORDS or word_text in _STATE_WORDS) and not _is_caps_run(word_raw_clean):
            break
        # Another verb cue: stop so the nested cue gets its own row.
        if word_text in _CUE_FIRST_WORDS and tokens and not _is_caps_run(word_raw_clean):
            break

        tokens.append((wstart, word_end))
        pos = wend

        if had_terminator_after_word:
            break

    if not tokens:
        return None

    # ISSUE-0486 F3 / R3: figurative reads/says — "says nothing", "reads like",
    # "says it all" open on an idiom word and transcribe nothing. Only when the
    # source token is lowercase; ALL-CAPS "says IT WORKS" is genuine text.
    first_source = caption[tokens[0][0] : tokens[0][1]]
    if first_source.islower() and first_source in _FIGURATIVE_FIRST:
        return None

    # ISSUE-0486 F2b: caps-run truncation. When the first captured token carries
    # transcription evidence (a capital or digit), the on-image text ends at the
    # first entirely-lowercase word — everything from there is narrative prose.
    # Skipped only for article/"with" captures ("reading a WARNING to visitors"),
    # where the trailing lowercase words are part of the described message.
    first_tok = caption[tokens[0][0] : tokens[0][1]]
    if not require_evidence and any(ch.isupper() or ch.isdigit() for ch in first_tok):
        for k in range(len(tokens)):
            tok = caption[tokens[k][0] : tokens[k][1]]
            if tok.isalpha() and tok.islower():
                tokens = tokens[:k]
                break
    if not tokens:
        return None

    captured = caption[tokens[0][0] : tokens[-1][1]]
    # An article/"with" after the cue signals an object noun-phrase ("reading a
    # menu", "says a lot", "printed with cotton"), not transcribed text — keep
    # the capture only when the text itself shows transcription evidence (caps or
    # digits; all-lowercase on-image text is written quoted, i.e. Tier 1). Mined
    # from confirmed review dismissals (ISSUE-0067).
    if (require_evidence or cue_requires_evidence) and not any(ch.isupper() or ch.isdigit() for ch in captured):
        return None
    return tokens[0][0], tokens[-1][1]


def _dedupe_nested_tier2(rows: list[IdentifiedText]) -> list[IdentifiedText]:
    """Drop Tier 2 spans that strictly contain another Tier 2 span.

    "the text spelling VICTORY" yields a "the text" hit covering the longer
    clause and a "spelling" hit covering just "VICTORY"; the inner cue is more
    specific so we keep it and discard the outer.
    """
    out: list[IdentifiedText] = []
    for r in rows:
        contains_other = any(
            other is not r
            and other.tier == 2
            and other.start >= r.start
            and other.end <= r.end
            and (other.start, other.end) != (r.start, r.end)
            for other in rows
        )
        if not contains_other:
            out.append(r)
    return out


def _dedupe_contained_tier1(rows: list[IdentifiedText]) -> list[IdentifiedText]:
    """Drop Tier-1 spans strictly contained within another Tier-1 span.

    A quote nested inside a larger quote — 'says "He said \'hi\' loudly"' yields
    the outer double-quoted span plus the inner 'hi' single-quoted one — is part
    of the outer transcription, not a separate reference (ISSUE-0486 F6a).
    """
    out: list[IdentifiedText] = []
    for r in rows:
        contained = any(
            other is not r
            and other.start <= r.start
            and other.end >= r.end
            and (other.start, other.end) != (r.start, r.end)
            for other in rows
        )
        if not contained:
            out.append(r)
    return out


# ── Public: identification ────────────────────────────────────────────────────


def extract_identified_text(caption: str) -> list[IdentifiedText]:
    """Return all identified text references in the caption.

    Tier 1 (quoted) results take precedence over overlapping Tier 2 (verb-cued)
    results. Returned in caption-order (ascending start offset).
    """
    if not caption:
        return []

    t1 = _dedupe_contained_tier1(_find_quoted(caption))
    t2 = _dedupe_nested_tier2(_find_verb_cued(caption) + _find_passive_cued(caption))

    # Tier 1 spans win on overlap.
    def overlaps(a: IdentifiedText, b: IdentifiedText) -> bool:
        return not (a.end <= b.start or b.end <= a.start)

    kept_t2 = [m for m in t2 if not any(overlaps(m, q) for q in t1)]

    # Dedupe by (normalised text); Tier 1 wins ties.
    seen_text: dict[str, IdentifiedText] = {}
    for m in t1 + kept_t2:
        key = normalize_for_match(m.text)
        if key not in seen_text or (seen_text[key].tier == 2 and m.tier == 1):
            seen_text[key] = m

    return sorted(seen_text.values(), key=lambda m: m.start)


# ── Public: normalisation + safeguard match ───────────────────────────────────

_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)

# ISSUE-0486 F8: symbol + number-word aliases so a caption and its OCR reading
# normalise identically ("Rock & Roll" ≡ "Rock and Roll", "two for one" ≡ "2 for 1").
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
}  # fmt: skip


def normalize_for_match(text: str) -> str:
    """Lowercase, strip all punctuation (incl. all quote variants), collapse whitespace, trim."""
    if not text:
        return ""
    # Decompose Unicode so curly quotes / accented chars normalise predictably,
    # then drop combining marks.
    nfkd = unicodedata.normalize("NFKD", text)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = no_marks.lower()
    # Alias symbols to their spoken form *before* punctuation stripping erases them.
    lowered = lowered.replace("&", " and ").replace("%", " percent ")
    # Replace any non-alphanumeric with a single space; collapse whitespace.
    spaced = _PUNCT_RE.sub(" ", lowered)
    # Map standalone English number words to digits so both sides agree.
    tokens = [_NUMBER_WORDS.get(t, t) for t in spaced.split()]
    return " ".join(tokens).strip()


def levenshtein_distance(a: str, b: str) -> int:
    """Plain edit distance (insert / delete / substitute, unit costs). Pure
    Python: the strings here are a few words of on-image text, so the O(len a
    x len b) table is trivial and a compiled dependency is not worth it."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """Similarity in [0, 1]: 1 - distance / max(len) (the normalised
    Levenshtein similarity Studio gets from its compiled dependency)."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / longest


def token_set_ratio(a: str, b: str) -> float:
    """Jaccard-style ratio over token sets — order-invariant, dup-invariant.

    Defined as |intersection| / |union| over set(words in a) and set(words in b).
    Approximates fuzzywuzzy's token_set_ratio behaviour for the cases the
    safeguard cares about (rearranged / partially-overlapping phrases).
    """
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union
