"""Tests for modules.util.age_estimate_util — Engine's Age head via ONNX.

The session is faked, so these pin what matters and would otherwise fail
silently: the class list decodes to the right ages, the expected value is
taken over the real classes only, ``__none__`` mass is reported rather than
folded into the age, and cache hits keyed on the model version skip the model.
"""

from modules.util import age_estimate_util as aeu
from modules.util.dpo_curation_util import DpoScanCache

import numpy as np
import pytest
from PIL import Image

SPEC = {
    "model_version": "age-test/1",
    "aspect_ratio_buckets": [[512, 512], [576, 448], [448, 576]],
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
    "class_names": ["__none__", "0-year-old", "1-year-old", "2-year-old", "3-year-old"],
    "none_index": 0,
    "min_age": 0,
}


class FakeSession:
    def __init__(self, logits_fn):
        self._fn = logits_fn
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        n = feeds["pixels"].shape[0]
        return [np.stack([self._fn(i) for i in range(n)])]


def test_class_ages_parse_the_engine_class_names():
    ages, none_index = aeu.class_ages(SPEC)
    assert none_index == 0
    assert np.isnan(ages[0])
    assert ages[1:].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_decode_takes_the_expected_value_over_real_classes_only():
    # Equal mass on 1 and 3 -> EV 2, whatever __none__ holds.
    logits = np.log(np.array([[0.5, 0.0, 0.25, 0.0, 0.25]]) + 1e-12)
    out = aeu.decode_ages(logits, SPEC)[0]
    assert out["age"] == pytest.approx(2.0, abs=1e-6)
    assert out["p_none"] == pytest.approx(0.5, abs=1e-6)
    assert out["argmax_age"] is None  # the mode is __none__


def test_decode_reports_the_mode_age_when_it_is_a_real_class():
    logits = np.log(np.array([[0.1, 0.0, 0.7, 0.2, 0.0]]) + 1e-12)
    out = aeu.decode_ages(logits, SPEC)[0]
    assert out["argmax_age"] == 1.0
    assert out["p_none"] == pytest.approx(0.1, abs=1e-6)


@pytest.fixture
def images(tmp_path):
    paths = []
    for i, size in enumerate([(64, 64), (96, 64), (64, 96)]):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", size, (10 * i, 20, 30)).save(p)
        paths.append(str(p))
    return paths


def test_estimate_ages_batches_by_bucket_shape_and_returns_every_path(images):
    session = FakeSession(lambda i: np.array([0.0, 0.0, 30.0, 0.0, 0.0]))
    out = aeu.estimate_ages(images, loader=lambda: (session, SPEC))
    assert set(out) == set(images)
    # Three distinct aspects -> three shapes -> three batches.
    assert session.calls == 3
    for r in out.values():
        assert r["age"] == pytest.approx(1.0, abs=1e-3)


def test_cache_hits_skip_the_model_and_are_keyed_on_the_model_version(images, tmp_path):
    cache = DpoScanCache(str(tmp_path / "c.json"))
    session = FakeSession(lambda i: np.array([0.0, 0.0, 30.0, 0.0, 0.0]))
    aeu.estimate_ages(images, loader=lambda: (session, SPEC), cache=cache)
    first = session.calls
    aeu.estimate_ages(images, loader=lambda: (session, SPEC), cache=cache)
    assert session.calls == first
    other = dict(SPEC, model_version="age-test/2")
    aeu.estimate_ages(images, loader=lambda: (session, other), cache=cache)
    assert session.calls == 2 * first


def test_unreadable_files_are_skipped(images, tmp_path):
    session = FakeSession(lambda i: np.zeros(5))
    out = aeu.estimate_ages([*images, str(tmp_path / "missing.png")], loader=lambda: (session, SPEC))
    assert len(out) == 3


def test_missing_model_raises_a_message_not_a_crash(images):
    with pytest.raises(RuntimeError, match="Age estimation"):
        aeu.estimate_ages(images, loader=lambda: (None, None))
