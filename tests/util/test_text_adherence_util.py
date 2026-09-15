"""Tests for modules.util.text_adherence_util — the on-image text axis of the
DPO curation tool.

The pure half is pinned without any model: which captions claim on-image
text (Studio's Text Tool rule, vendored), how a span is aligned to the best
OCR window, how several spans share one reading, and what the score is when
the reader saw the text verbatim / with a typo / not at all. The model half
is exercised through a fake reader and the versioned cache.
"""

from modules.util import text_adherence_util as tau
from modules.util.dpo_curation_util import DpoScanCache

import pytest

# --------------------------------------------------------------------------- #
# Which captions mention text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "caption, spans",
    [
        ('A neon sign reading "OPEN 24 HOURS" above the door.', ["OPEN 24 HOURS"]),
        ("A man in a t-shirt printed with JUST DO IT.", ["JUST DO IT"]),
        ("A storefront whose sign reads: BAKERY, with a chalkboard that says FRESH BREAD.", ["BAKERY", "FRESH BREAD"]),
        ("A low angle medium shot of a 34-year-old woman, heavy grain.", []),
        ("UNCONDITIONAL", []),
        ("", []),
        # Direct speech is not on-image text.
        ('A woman says "hello there" to her friend.', []),
        # A described pattern is not on-image text either.
        ("A woman in a printed dress.", []),
        # Several strings in one caption: quoted names, a verb-cued score and
        # a quoted phrase whose cue comes *after* it. "underneath" says where
        # 'Full Time' sits, not what it says, so it is not a fifth span.
        (
            "A Scoreline between 'Hull City' and 'Real Madrid' reads 7-1, with 'Full Time' written underneath.",
            ["Hull City", "Real Madrid", "7-1", "Full Time"],
        ),
        ("'OPEN' is written above the door.", ["OPEN"]),
        ("the words 'FULL TIME' written underneath", ["FULL TIME"]),
        # A cue after a quote still transcribes forward when the forward text
        # carries caps/digit evidence of its own.
        ("A card labelled 'A' reads HAPPY BIRTHDAY.", ["A", "HAPPY BIRTHDAY"]),
    ],
)
def test_caption_text_spans_follow_the_text_tool_rule(caption, spans):
    assert tau.caption_text_spans(caption) == spans


def test_caption_text_spans_dedupe_on_normalised_text():
    assert tau.caption_text_spans('A sign reading "OPEN" and a door marked "open!"') == ["OPEN"]


# --------------------------------------------------------------------------- #
# Alignment and scoring
# --------------------------------------------------------------------------- #


def test_verbatim_reading_scores_one():
    summary = tau.score_text(["OPEN 24 HOURS"], "WELCOME\nOPEN 24 HOURS\nno parking")
    assert summary["text"] == pytest.approx(1.0)
    assert summary["text_min"] == pytest.approx(1.0)
    assert summary["matches"][0].read == "OPEN 24 HOURS"
    assert summary["matches"][0].verdict == "verbatim"


def test_case_punctuation_and_symbol_aliases_do_not_count_against_the_image():
    summary = tau.score_text(["Rock & Roll"], "rock and roll!")
    assert summary["text"] == pytest.approx(1.0)


def test_a_typo_scores_below_absent_and_reports_what_was_read():
    # "Kama" for "Karma" is a wrong sign, and a chosen image with a wrong sign
    # teaches the misspelling; it must rank below an image with no sign.
    summary = tau.score_text(["Karma"], "Kama")
    match = summary["matches"][0]
    assert match.read == "Kama" and match.verdict == "wrong"
    assert match.ratio == pytest.approx(0.8)
    assert 0.0 < summary["text"] < tau.ABSENT_SCORE


def test_one_wrong_letter_is_wrong_however_long_the_word():
    short = tau.score_text(["OPEN"], "OPEM")["matches"][0]
    long = tau.score_text(["HAPPY BIRTHDAY"], "HAPPY BIRTHDAV")["matches"][0]
    assert short.verdict == long.verdict == "wrong"
    assert long.score < tau.ABSENT_SCORE


def test_missing_text_scores_the_absent_value():
    # An empty reading, no reading, and a reading of unrelated text all mean
    # the requested text is not on the image — none of them earns the small
    # credit any two strings share under an edit distance.
    for reading in ("", None, "a cat on a mat"):
        summary = tau.score_text(["BAKERY"], reading)
        assert summary["text"] == tau.ABSENT_SCORE
        assert summary["matches"][0].read is None
        assert summary["matches"][0].verdict == "absent"


def test_several_spans_claim_disjoint_windows_and_average():
    summary = tau.score_text(["BAKERY", "FRESH BREAD"], "BAKERY\nFRESH BREAD daily")
    assert summary["text"] == pytest.approx(1.0)
    assert [m.read for m in summary["matches"]] == ["BAKERY", "FRESH BREAD"]
    half = tau.score_text(["BAKERY", "FRESH BREAD"], "BAKERY only")
    assert half["text"] == pytest.approx((1.0 + tau.ABSENT_SCORE) / 2)
    assert half["text_min"] == tau.ABSENT_SCORE
    assert half["n_text"] == 2


SCORELINE = "A Scoreline between 'Hull City' and 'Real Madrid' reads 7-1, with 'Full Time' written underneath."


def test_a_scoreboard_with_every_string_scores_one():
    spans = tau.caption_text_spans(SCORELINE)
    for reading in (
        "HULL CITY 7 - 1 REAL MADRID\nFULL TIME",
        "Hull City 7-1 Real Madrid Full Time",
        "FULL TIME\nHULL CITY   7-1   REAL MADRID",
    ):
        summary = tau.score_text(spans, reading)
        assert summary["text"] == pytest.approx(1.0), reading
        assert summary["n_text"] == 4


def test_a_scoreboard_missing_one_string_is_ranked_below_a_complete_one():
    spans = tau.caption_text_spans(SCORELINE)
    complete = tau.score_text(spans, "HULL CITY 7-1 REAL MADRID FULL TIME")
    no_full_time = tau.score_text(spans, "HULL CITY 7-1 REAL MADRID")
    wrong_score = tau.score_text(spans, "HULL CITY 1-7 REAL MADRID FULL TIME")
    # A missing "Full Time" is a lesser sin than a reversed scoreline.
    assert complete["text"] > no_full_time["text"] > wrong_score["text"]
    assert [m.read for m in no_full_time["matches"]][-1] is None
    assert no_full_time["text_min"] == tau.ABSENT_SCORE
    assert [m.verdict for m in wrong_score["matches"]] == ["verbatim", "verbatim", "wrong", "verbatim"]


def test_numbers_are_order_sensitive_but_words_keep_studios_layout_tolerance():
    # A reversed score is a different score...
    assert tau.similarity("7 1", "1 7") < 0.7
    assert tau.similarity("10 30", "30 10") < 0.7
    # ...while a two-line sign read in column order is still that sign.
    assert tau.similarity("open 24 hours", "open 24 hours") == 1.0
    assert tau.similarity("rock and roll", "roll and rock") == 1.0


def test_a_repeated_sign_cannot_serve_two_different_spans():
    # Only one "OPEN" is on the image; the second span must not reuse it.
    summary = tau.score_text(["OPEN", "OPEN NOW"], "OPEN")
    reads = sorted(m.read or "" for m in summary["matches"])
    assert reads.count("OPEN") == 1


def test_the_ordering_is_verbatim_then_absent_then_wrong():
    good = tau.score_text(["HAPPY BIRTHDAY"], "HAPPY BIRTHDAY")["text"]
    typo = tau.score_text(["HAPPY BIRTHDAY"], "HAPPY BIRTHDAV")["text"]
    garbled = tau.score_text(["HAPPY BIRTHDAY"], "HAPY BRTHDY")["text"]
    absent = tau.score_text(["HAPPY BIRTHDAY"], "")["text"]
    # No sign beats a wrong sign; among wrong signs the closer miss ranks higher.
    assert good > absent > typo > garbled > 0.0


def test_no_spans_scores_nothing():
    assert tau.score_text([], "anything")["text"] is None


# --------------------------------------------------------------------------- #
# The model half through a fake reader
# --------------------------------------------------------------------------- #


class FakeReader:
    model_version = "fake-ocr/1"

    def __init__(self, readings: dict[str, str]):
        self.readings = readings
        self.calls = 0

    def read(self, pil):
        self.calls += 1
        return self.readings[pil.size]


@pytest.fixture
def images(tmp_path):
    from PIL import Image

    paths = []
    for i in range(3):
        p = tmp_path / f"t{i}.png"
        Image.new("RGB", (8 + i, 8), (0, 0, 0)).save(p)
        paths.append(str(p))
    return paths


def _reader_for(images):
    from PIL import Image

    sizes = [Image.open(p).size for p in images]
    return FakeReader({sizes[0]: "OPEN", sizes[1]: "OPEM", sizes[2]: ""})


def test_score_images_text_returns_every_readable_path(images):
    res = tau.score_images_text(images, 'A sign reading "OPEN".', backend=_reader_for(images))
    assert list(res) == images
    scores = [res[p]["text"] for p in images]
    assert scores[0] == pytest.approx(1.0) and scores[2] == tau.ABSENT_SCORE and 0 < scores[1] < scores[2]
    prop = res[images[1]]["propositions"][0]
    assert prop["axis"] == tau.TEXT_OCR_AXIS and prop["label"] == "OPEN" and prop["read"] == "OPEM"
    assert prop["verdict"] == "wrong" and prop["prob"] == scores[1] and prop["ratio"] == pytest.approx(0.75)
    assert "misrendered" in prop["text"]
    assert res[images[2]]["propositions"][0]["read"] is None


def test_a_caption_without_text_never_touches_the_reader(images):
    reader = _reader_for(images)
    assert tau.score_images_text(images, "A woman in a red dress.", backend=reader) == {}
    assert reader.calls == 0


def test_readings_are_cached_per_model_version(images, tmp_path):
    cache = DpoScanCache(str(tmp_path / "cache.json"))
    reader = _reader_for(images)
    first = tau.read_image_text(images, backend=reader, cache=cache)
    assert reader.calls == 3
    again = tau.read_image_text(images, backend=reader, cache=cache)
    assert again == first and reader.calls == 3
    reader.model_version = "fake-ocr/2"
    tau.read_image_text(images, backend=reader, cache=cache)
    assert reader.calls == 6


def test_unreadable_and_missing_files_are_skipped(images, tmp_path):
    bad = str(tmp_path / "bad.png")
    with open(bad, "wb") as handle:
        handle.write(b"not an image")
    reader = _reader_for(images)
    res = tau.read_image_text([*images, bad, str(tmp_path / "missing.png")], backend=reader)
    assert set(res) == set(images)


def test_a_reader_that_cannot_load_raises_a_message_not_a_crash(images):
    class Broken:
        model_version = "x"

        def read(self, pil):
            raise RuntimeError("PaddleOCR-VL could not load")

    with pytest.raises(RuntimeError, match="could not load"):
        tau.read_image_text(images, backend=Broken())


def test_load_ocr_backend_falls_back_to_the_other_reader(monkeypatch):
    built = []

    class Fails(tau.OCRBackendBase):
        display_name = "Fails"

        def _load_impl(self):
            built.append("fails")
            self.last_error = "remote code broke"
            return False

    class Works(tau.OCRBackendBase):
        display_name = "Works"

        def _load_impl(self):
            built.append("works")
            self.model = self.processor = object()
            return True

    monkeypatch.setattr(tau, "_BACKEND_CLASSES", {"paddleocr-vl": Fails, "got-ocr2": Works})
    reader = tau.load_ocr_backend(device="cpu")
    # A handle, not a model: nothing loads until a reading is needed.
    assert built == [] and not reader.loaded
    assert reader.load_model() and built == ["fails", "works"]
    assert reader.display_name == "Works" and reader.loaded

    monkeypatch.setattr(tau, "_BACKEND_CLASSES", {"paddleocr-vl": Fails, "got-ocr2": Fails})
    with pytest.raises(RuntimeError, match="remote code broke"):
        tau.load_ocr_backend(device="cpu").read(None)


def test_a_cached_reading_never_loads_the_reader(images, tmp_path):
    """The backfill's promise: once every reading is cached, Selection asks
    the cache and the weights stay on disk."""
    cache = DpoScanCache(str(tmp_path / "cache.json"))
    first = _reader_for(images)
    first.model_version = "fake-ocr@1"
    tau.read_image_text(images, backend=first, cache=cache)

    class NeverLoads(tau.OCRBackendBase):
        display_name = "NeverLoads"
        repo = "fake-ocr"
        revision = "1"

        def _load_impl(self):
            raise AssertionError("every reading is cached — the model must not load")

    class Handle(tau.OCRReader):
        pass

    lazy = Handle(device="cpu")
    lazy._order = ["never"]
    tau._BACKEND_CLASSES["never"] = NeverLoads
    try:
        assert lazy.candidate_versions == ["fake-ocr@1"]
        again = tau.read_image_text(images, backend=lazy, cache=cache)
    finally:
        del tau._BACKEND_CLASSES["never"]
    assert set(again) == set(images) and not lazy.loaded


def test_readings_from_the_fallback_reader_still_count(images, tmp_path):
    cache = DpoScanCache(str(tmp_path / "cache.json"))
    fallback = _reader_for(images)
    fallback.model_version = "fallback/1"
    tau.read_image_text(images, backend=fallback, cache=cache)

    class Preferred:
        model_version = "preferred/1"
        candidate_versions = ["preferred/1", "fallback/1"]
        calls = 0

        def read(self, pil):
            type(self).calls += 1
            return "x"

    assert tau.read_image_text(images, backend=Preferred(), cache=cache)[images[0]] == "OPEN"
    assert Preferred.calls == 0
