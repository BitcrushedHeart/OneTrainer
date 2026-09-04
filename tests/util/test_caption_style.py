"""The ported tag/natural-language caption classifier.

Ground truth comes from the Bitcrush Suite's own corpora (its
``test_caption_line_classify.py`` TAG_LINES/NL_LINES and the boundary matrix
in ``test_caption.py``), so a divergence between the port and the upstream
SSoT shows up here rather than as a silently different filter.
"""

from modules.util.caption_style import (
    CaptionType,
    classify_caption_type,
    is_nl_fragment,
    is_tag_like_segment,
    is_tag_line,
    normalize_caption,
    split_caption_lines,
)

import pytest

TAG_LINES = [
    "blonde hair, blue eyes, 1girl, outdoors",
    "car interior, car",
    "Woman, car",
    "sunset",
    "desert",
    "Underwater",
    "London Underground",
    "Kingston-Upon-Hull, Shot on Canon, f/5.6, 1/30s, ISO 100, 27mm",
    "front view, high angle, professional photo",
    "looking at viewer",
    "1girl, solo, long hair, looking at viewer, smile",
    "medium shot, eye-level",
    "video screenshot",
    "Adult, Adults Only, Anticipation, Back Seat, Business, Businesswoman",
    "red dress, standing",
    "f/1.8, bokeh, 85mm",
    "professional photo, studio lighting, high angle",
    "Shot on Panasonic Lumix, eye-level, side view, professional photo",
    "blonde hair, blue eyes, outdoors, smiling.",
    "fair skin, 14-year-old, slim build",
    "amateur photo, mirror selfie, bathroom",
    # Big booru dump with a stray internal period and article-bearing tag
    # idioms — still a tag list.
    "Looking at camera, Three-Quarter Rear, arm tattoos, asian, ass, "
    "bare back, black hair, breasts, bright socks, fair skin, fair skin., "
    "indoor setting, nipples, nude, nude body, pink stockings, "
    "questionable, round buttocks, short dark hair, side view, solo",
    "blonde hair, breasts, necklace, large areolae, nude, open mouth, "
    "outdoors, pool, questionable, screenshot from a video, solo, "
    "video screenshot, Caucasian woman",
]

NL_LINES = [
    "A woman stands in a garden with flowers.",
    "A woman sits in a car, smiling",
    "She has long blonde hair and a warm smile",
    "The quick brown fox jumps over the lazy dog, beautiful sunset",
    "A photograph of a woman in a red dress, taken at golden hour",
    "video screenshot, screenshot from a video",
    "Sunset.",
    "reddit.",
    "A woman stands.",
    "Woman, car.",
    "tag1, tag2, tag3, professional photo. She has a warm smile.",
    "A young woman stands in the studio., caucasian, amateur photo",
    "A woman poses., Caucasian, Light Skin",
    "She stands in the studio, Caucasian, Light Skin.",
    "Is this a photo of a woman?",
    "The light — sharp and cold — cuts across the platform",
    "wearing a red dress",
    "clad in a black shirt",
    "her hands on the wheel",
    "with red hair",
    "A large orange sun sits low above distant hills",
    "photo of a woman",
    "tag one, tag two, " + "x" * 2000,
    "",
    "   ",
]


@pytest.mark.parametrize("line", TAG_LINES)
def test_tag_lines(line: str):
    assert is_tag_line(line) is True, line


@pytest.mark.parametrize("line", NL_LINES)
def test_natural_language_lines(line: str):
    assert is_tag_line(line) is False, line


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", False),
        ("blue eyes", True),
        ("long blonde hair", True),
        ("screenshot from a video", False),
        ("red dress, blue eyes", True),
        ("red dress, blue eyes.", False),
        ("solo, red dress, blue eyes.", False),
        ("1girl, solo, red dress, blue eyes.", True),
        ("fair skin. nipples, blue eyes, solo", True),
        ("A young woman stands in studio, blonde hair, blue eyes, solo", False),
        ("tag one, tag two, tag three, screenshot from a video, a woman", True),
        ("tag one, tag two, screenshot from a video, a woman", False),
    ],
)
def test_tag_line_boundary_matrix(line: str, expected: bool):
    assert is_tag_line(line) is expected


def test_questions_and_dashes_are_never_tags():
    assert not is_tag_line("what is this?")
    assert not is_tag_line("close-up — soft light, blue eyes")


def test_nl_segments_override_the_vote():
    line = "blue eyes, red dress, a woman"
    assert is_tag_line(line) is True
    assert is_tag_line(line, nl_segments={"blue eyes", "red dress"}) is False


def test_segment_and_fragment_helpers():
    assert is_tag_like_segment("blue eyes")
    assert is_tag_like_segment("professional photo")  # exact allowlist
    assert not is_tag_like_segment("she is in the garden")
    assert not is_tag_like_segment("a woman with a hat on a bench somewhere")  # >5 words
    assert is_nl_fragment("her hands on the wheel")
    assert is_nl_fragment("wearing a red dress")
    assert not is_nl_fragment("looking at viewer")  # allowlisted booru gerund
    assert not is_nl_fragment("blue eyes")


def test_the_users_own_caption_style_is_natural_language():
    """SoReal captions open with comma-separated modifiers before the prose;
    they must not be filtered out as tag lists."""
    for caption in (
        "A subdued, eye-level, straight on, three-quarter view, clean photo of a person with dark hair.",
        "A front view, clean shows a professional photo of a woman standing in the surf, "
        "the background shows the surface of the water, blurred out of focus.",
        "A colourful, clean professional photo of a person in a kitchen, the light is dim.",
    ):
        assert is_tag_line(caption) is False, caption


def test_classify_caption_type():
    assert classify_caption_type("blonde hair, blue eyes, outdoors") is CaptionType.TAG
    # Non-tag but too short to be natural language.
    assert classify_caption_type("She is smiling at the camera.") is CaptionType.TAG
    assert classify_caption_type("She is smiling at the camera.", min_nl_words=0) is CaptionType.SHORT
    assert classify_caption_type("A woman stands in a sunlit garden holding a watering can.") is CaptionType.SHORT
    assert classify_caption_type("A woman stands in a sunlit garden. " + "x" * 250) is CaptionType.DESCRIPTIVE
    assert classify_caption_type("A woman stands in a sunlit garden holding a can.", threshold=6, unit="word") is (
        CaptionType.DESCRIPTIVE
    )
    assert classify_caption_type("") is CaptionType.SHORT


def test_normalize_and_split():
    assert normalize_caption("  a b\n c  ") == "a b c"
    assert normalize_caption(None) == ""
    assert split_caption_lines("one\n\n  two  \nthree") == ["one", "two", "three"]
    assert split_caption_lines(None) == []
