"""Tests for the SigLIP-based prompt-adherence scorer in
modules.util.adherence_score_util.

No model is loaded anywhere here. The scorer is split into a pure half
(turning a SoReal!-style caption into scoreable propositions, and turning
cosine similarities into per-proposition probabilities) and a model half that
is injected as an ``embedder`` object. The tests pin the pure half, because
that is where a silent mistake costs the most: a proposition that is never
extracted is never scored, and nothing downstream can tell.
"""

import math

from modules.util import adherence_score_util as asu
from modules.util.dpo_curation_util import DpoScanCache

import numpy as np
import pytest
from PIL import Image

# --------------------------------------------------------------------------- #
# Proposition extraction
# --------------------------------------------------------------------------- #

SOREAL_PROMPT = (
    "A muted, side view, low angle, heavy grain medium full shot of a 14-year-old "
    "young teenage girl with long blonde hair standing sideways against a white wall."
)


def _axis_props(props):
    return {p.axis: p for p in props if p.axis is not None}


def test_axis_terms_in_a_soreal_preamble_are_all_found():
    props = asu.extract_propositions(SOREAL_PROMPT)
    by_axis = _axis_props(props)
    assert by_axis["saturation"].label == "Muted"
    assert by_axis["horizontal_angle"].label == "Side View"
    assert by_axis["vertical_angle"].label == "Low Angle"
    assert by_axis["grain"].label == "Heavy Grain"
    assert by_axis["shot_type"].label == "Medium full shot"
    assert by_axis["gender"].label == "Female"
    assert by_axis["age"].label == "teenager"


def test_axis_proposition_carries_its_siblings_with_itself_first():
    props = asu.extract_propositions("A high-angle medium shot of a smiling woman.")
    angle = _axis_props(props)["vertical_angle"]
    assert angle.texts[0] == angle.text
    # Every sibling is a distinct full sentence, not a bare label.
    assert len(set(angle.texts)) == len(angle.texts) > 3
    assert all(" " in t for t in angle.texts)


def test_longest_match_wins_within_an_axis():
    # "medium full shot" contains "full shot"; "very low angle" contains "low angle".
    props = asu.extract_propositions("A very low angle medium full shot of a man.")
    by_axis = _axis_props(props)
    assert by_axis["shot_type"].label == "Medium full shot"
    assert by_axis["vertical_angle"].label == "Very Low Angle"


def test_hyphenated_and_spaced_forms_are_the_same_class():
    a = _axis_props(asu.extract_propositions("An eye-level shot of a woman"))["vertical_angle"]
    b = _axis_props(asu.extract_propositions("An eye level shot of a woman"))["vertical_angle"]
    assert a.label == b.label == "Eye-Level"


def test_neutral_expression_is_not_the_saturation_class():
    props = asu.extract_propositions("A close-up of a woman with a neutral expression.")
    assert "saturation" not in _axis_props(props)


def test_mixed_gender_prompt_skips_the_gender_axis():
    props = asu.extract_propositions("A medium shot of a Black bride and groom cutting a cake.")
    by_axis = _axis_props(props)
    assert "gender" not in by_axis
    assert by_axis["ethnicity"].label == "African"


def test_skin_tone_prefers_the_longer_phrase():
    props = asu.extract_propositions("A woman with light brown skin and dark hair.")
    assert _axis_props(props)["skin_tone"].label == "Light Brown Skin"


def test_free_phrases_come_from_the_residue_after_axis_terms():
    props = asu.extract_propositions(SOREAL_PROMPT)
    free = [p.text for p in props if p.axis is None]
    joined = " | ".join(free)
    assert "blonde hair" in joined
    assert "white wall" in joined
    # Preamble leftovers ("A", "of a") never become propositions.
    assert all(len(t.split()) >= asu.MIN_PHRASE_WORDS for t in free)
    assert not any("of of" in t.lower() for t in free)


def test_free_phrases_are_capped_in_length():
    long_prompt = "A photo of " + " ".join(["word"] * 40) + "."
    props = asu.extract_propositions(long_prompt)
    assert all(len(p.text.split()) <= asu.MAX_PHRASE_WORDS + 3 for p in props)


def test_tag_line_prompt_yields_one_proposition_per_tag():
    props = asu.extract_propositions("1girl, long hair, red dress, looking at viewer, outdoors")
    free = [p for p in props if p.axis is None]
    assert len(free) >= 4
    assert any("red dress" in p.text for p in free)


def test_unconditional_and_empty_prompts_yield_nothing():
    assert asu.extract_propositions("") == []
    assert asu.extract_propositions("UNCONDITIONAL") == []


def test_age_bins_are_ordinal_neighbours():
    child = _axis_props(asu.extract_propositions("A photo of a 7-year-old girl"))["age"]
    adult = _axis_props(asu.extract_propositions("A photo of a 34-year-old woman"))["age"]
    assert child.label == "child"
    assert adult.label == "thirties"
    assert child.text != adult.text
    assert set(child.texts) == set(adult.texts)


# --------------------------------------------------------------------------- #
# Scoring maths
# --------------------------------------------------------------------------- #


def test_axis_probability_is_a_softmax_over_siblings_and_ignores_bias():
    # cos: own class 0.30, siblings 0.10 and 0.05. Bias must not change a softmax.
    sims = np.array([0.30, 0.10, 0.05])
    p1 = asu.axis_probability(sims, scale=10.0, bias=-10.0)
    p2 = asu.axis_probability(sims, scale=10.0, bias=0.0)
    expected = math.exp(3.0) / (math.exp(3.0) + math.exp(1.0) + math.exp(0.5))
    assert p1 == pytest.approx(expected)
    assert p2 == pytest.approx(expected)


def test_phrase_probability_is_the_siglip_sigmoid():
    assert asu.phrase_probability(0.2, scale=10.0, bias=-10.0) == pytest.approx(1 / (1 + math.exp(8.0)))
    assert asu.phrase_probability(1.2, scale=10.0, bias=-10.0) == pytest.approx(1 / (1 + math.exp(-2.0)))


def test_aggregate_reports_axis_phrase_and_overall_means():
    props = [
        asu.Proposition(text="a low angle photo", axis="vertical_angle", label="Low Angle", texts=["a", "b"]),
        asu.Proposition(text="a close-up photo", axis="shot_type", label="Close-up", texts=["a", "b"]),
        asu.Proposition(text="a photo of a red dress", axis=None, label=None, texts=["a photo of a red dress"]),
    ]
    probs = [0.8, 0.6, 0.2]
    agg = asu.aggregate(props, probs)
    assert agg["axis_mean"] == pytest.approx(0.7)
    assert agg["axis_min"] == pytest.approx(0.6)
    assert agg["phrase_mean"] == pytest.approx(0.2)
    assert agg["n_axis"] == 2 and agg["n_phrase"] == 1
    # The headline adherence weights the two families equally, not per-prop.
    assert agg["adherence"] == pytest.approx((0.7 + 0.2) / 2)


def test_aggregate_with_one_family_missing_uses_the_other():
    props = [asu.Proposition(text="x", axis=None, label=None, texts=["x"])]
    agg = asu.aggregate(props, [0.4])
    assert agg["axis_mean"] is None
    assert agg["adherence"] == pytest.approx(0.4)
    assert asu.aggregate([], [])["adherence"] is None


# --------------------------------------------------------------------------- #
# End-to-end with a fake embedder + cache
# --------------------------------------------------------------------------- #


class FakeEmbedder:
    """Deterministic unit vectors: images and texts hash to directions so the
    scorer's plumbing (batching, caching, per-proposition mapping) can be
    checked without a model."""

    model_version = "fake-siglip/1"
    scale = 10.0
    bias = -10.0

    def __init__(self):
        self.image_calls: list[list[str]] = []
        self.text_calls: list[list[str]] = []

    @staticmethod
    def _vec(key: str) -> np.ndarray:
        seed = sum((i + 1) * ord(ch) for i, ch in enumerate(key)) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(8)
        return v / np.linalg.norm(v)

    def embed_images(self, paths):
        self.image_calls.append(list(paths))
        return np.stack([self._vec(p) for p in paths])

    def embed_texts(self, texts):
        self.text_calls.append(list(texts))
        return np.stack([self._vec(t) for t in texts])


@pytest.fixture
def images(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), (i * 40, 0, 0)).save(p)
        paths.append(str(p))
    return paths


def test_score_images_returns_one_result_per_readable_path(images):
    emb = FakeEmbedder()
    results = asu.score_images_adherence(images, "A low angle medium shot of a woman in a red dress.", embedder=emb)
    assert set(results) == set(images)
    for r in results.values():
        assert 0.0 <= r["adherence"] <= 1.0
        assert r["n_axis"] >= 2
        assert all(0.0 <= p["prob"] <= 1.0 for p in r["propositions"])


def test_image_embeddings_are_cached_by_model_version(images, tmp_path):
    cache = DpoScanCache(str(tmp_path / "cache.json"))
    emb = FakeEmbedder()
    prompt = "A close-up of a man."
    asu.score_images_adherence(images, prompt, embedder=emb, cache=cache)
    assert sum(len(c) for c in emb.image_calls) == 3
    asu.score_images_adherence(images, prompt, embedder=emb, cache=cache)
    assert sum(len(c) for c in emb.image_calls) == 3, "second pass must hit the cache"
    # A different model version invalidates.
    emb.model_version = "fake-siglip/2"
    asu.score_images_adherence(images, prompt, embedder=emb, cache=cache)
    assert sum(len(c) for c in emb.image_calls) == 6


def test_cached_embeddings_survive_a_save_and_reload(images, tmp_path):
    path = str(tmp_path / "cache.json")
    cache = DpoScanCache(path)
    emb = FakeEmbedder()
    asu.score_images_adherence(images, "A close-up of a man.", embedder=emb, cache=cache)
    cache.save()
    reloaded = DpoScanCache(path)
    vec = reloaded.peek_adherence_embedding(images[0], emb.model_version)
    assert vec is not None and len(vec) == 8
    assert reloaded.peek_adherence_embedding(images[0], "other") is None


def test_text_embeddings_are_memoised_per_embedder(images):
    emb = FakeEmbedder()
    prompt = "A close-up of a man."
    asu.score_images_adherence(images[:1], prompt, embedder=emb)
    asu.score_images_adherence(images[1:], prompt, embedder=emb)
    texts = [t for call in emb.text_calls for t in call]
    assert len(texts) == len(set(texts)), "no text should be embedded twice"


def test_unreadable_paths_are_skipped_not_raised(images, tmp_path):
    emb = FakeEmbedder()
    missing = str(tmp_path / "nope.png")
    results = asu.score_images_adherence([*images, missing], "A close-up of a man.", embedder=emb)
    assert missing not in results and len(results) == 3


def test_prompt_with_no_propositions_scores_nothing(images):
    emb = FakeEmbedder()
    assert asu.score_images_adherence(images, "UNCONDITIONAL", embedder=emb) == {}
    assert emb.image_calls == []


# --------------------------------------------------------------------------- #
# Age model proposition — only when the caption claims an age
# --------------------------------------------------------------------------- #


def test_claimed_age_range_is_a_point_for_numeric_and_a_band_for_words():
    assert asu.claimed_age_range("A photo of a 34-year-old woman") == (34.0, 34.0)
    low, high = asu.claimed_age_range("A photo of a teenage girl")
    assert low == 13.0 and high == 19.0
    assert asu.claimed_age_range("A photo of a woman in a red dress") is None
    assert asu.claimed_age_range("A 30-year-old man and a 5-year-old boy") is None


def test_age_agreement_is_one_inside_the_range_and_decays_outside():
    assert asu.age_agreement(34.0, (34.0, 34.0)) == pytest.approx(1.0)
    assert asu.age_agreement(15.0, (13.0, 19.0)) == pytest.approx(1.0)
    near = asu.age_agreement(40.0, (34.0, 34.0))
    far = asu.age_agreement(60.0, (34.0, 34.0))
    assert 0.0 < far < near < 1.0


def test_age_proposition_is_added_only_when_the_caption_claims_an_age(images):
    emb = FakeEmbedder()
    estimates = {p: {"age": 34.0, "p_none": 0.1} for p in images}
    with_age = asu.score_images_adherence(
        images, "A close-up of a 34-year-old woman.", embedder=emb, age_estimates=estimates
    )
    for r in with_age.values():
        age_props = [p for p in r["propositions"] if p["axis"] == asu.AGE_MODEL_AXIS]
        assert len(age_props) == 1
        assert age_props[0]["prob"] == pytest.approx(1.0)
        assert age_props[0]["estimated_age"] == 34.0
        assert r["age_agreement"] == pytest.approx(1.0)
        # Adherence itself is the SigLIP verdict only - age is its own axis.
        assert r["n_axis"] == len([p for p in r["propositions"] if p["axis"] and p["axis"] != asu.AGE_MODEL_AXIS])
    without = asu.score_images_adherence(images, "A close-up of a woman.", embedder=emb, age_estimates=estimates)
    for r in without.values():
        assert not any(p["axis"] == asu.AGE_MODEL_AXIS for p in r["propositions"])
        assert r["age_agreement"] is None


def test_age_proposition_ranks_the_closer_estimate_higher(images):
    emb = FakeEmbedder()
    estimates = {images[0]: {"age": 35.0, "p_none": 0.0}, images[1]: {"age": 55.0, "p_none": 0.0}}
    res = asu.score_images_adherence(
        images[:2], "A close-up of a 34-year-old woman.", embedder=emb, age_estimates=estimates
    )
    prob = {p: next(x["prob"] for x in r["propositions"] if x["axis"] == asu.AGE_MODEL_AXIS) for p, r in res.items()}
    assert prob[images[0]] > prob[images[1]]


def test_age_proposition_is_skipped_when_the_model_saw_no_subject(images):
    emb = FakeEmbedder()
    estimates = {images[0]: {"age": 34.0, "p_none": 0.9}}
    res = asu.score_images_adherence(
        images[:1], "A close-up of a 34-year-old woman.", embedder=emb, age_estimates=estimates
    )
    assert not any(p["axis"] == asu.AGE_MODEL_AXIS for p in res[images[0]]["propositions"])
    assert res[images[0]]["age_agreement"] is None
    assert res[images[0]]["estimated_age"] == 34.0  # still reported, just not trusted


def test_age_agreement_alone_matches_the_siglip_scorer_without_an_embedder(images):
    """The DPO tool scores age without SigLIP: the standalone scorer must
    return the same age verdicts as the full scorer, with the adherence keys
    present but ``None``, and never touch an embedder."""
    prompt = "A close-up of a 34-year-old woman."
    estimates = {
        images[0]: {"age": 35.0, "p_none": 0.0},
        images[1]: {"age": 55.0, "p_none": 0.0},
        images[2]: {"age": 34.0, "p_none": 0.9},
    }
    alone = asu.score_images_age_agreement(images, prompt, age_estimates=estimates)
    full = asu.score_images_adherence(images, prompt, embedder=FakeEmbedder(), age_estimates=estimates)
    assert set(alone) == set(images)
    for p in images:
        assert alone[p]["adherence"] is None and alone[p]["axis_mean"] is None and alone[p]["phrase_mean"] is None
        assert alone[p]["age_agreement"] == full[p]["age_agreement"]
        assert alone[p]["estimated_age"] == full[p]["estimated_age"]
        assert alone[p]["propositions"] == [x for x in full[p]["propositions"] if x["axis"] == asu.AGE_MODEL_AXIS]
    assert alone[images[0]]["age_agreement"] > alone[images[1]]["age_agreement"]
    assert alone[images[2]]["age_agreement"] is None  # p_none over the gate
    assert asu.score_images_age_agreement(images, "A close-up of a woman.", age_estimates=estimates) == {}
    assert asu.score_images_age_agreement(images, prompt, age_estimates=None) == {}
