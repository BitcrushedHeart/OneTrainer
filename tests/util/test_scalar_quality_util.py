"""Tests for modules.util.scalar_quality_util — the port of Bitcrush Studio's
"Simple" quality score (sharpness / contrast / exposure / compression).

The formula is pinned by construction rather than by magic numbers: a sharper
image must outscore its blurred copy, a flat or black frame must collapse to
the floor, and the compression term must be *off* by default because the DPO
tool compares images across codecs (a PNG generation against a JPEG photo),
where bits-per-pixel says nothing about quality.
"""

from modules.util import scalar_quality_util as sq

import numpy as np
import pytest
from PIL import Image, ImageFilter


def _noise_image(size=(128, 128), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sharp_and_blurred(tmp_path):
    sharp = _noise_image()
    blurred = sharp.filter(ImageFilter.GaussianBlur(3))
    sp, bp = tmp_path / "sharp.png", tmp_path / "blur.png"
    sharp.save(sp)
    blurred.save(bp)
    return str(sp), str(bp)


def test_sharper_image_scores_higher(sharp_and_blurred):
    sharp, blurred = sharp_and_blurred
    a = sq.scalar_quality(sharp)
    b = sq.scalar_quality(blurred)
    assert a is not None and b is not None
    assert a["sharpness"] > b["sharpness"]
    assert a["total"] > b["total"]


def test_components_are_bounded_0_100(sharp_and_blurred):
    for path in sharp_and_blurred:
        r = sq.scalar_quality(path)
        for key in ("total", "sharpness", "contrast", "exposure", "compression"):
            assert 0.0 <= r[key] <= 100.0


def test_black_frame_scores_zero(tmp_path):
    p = tmp_path / "black.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(p)
    assert sq.scalar_quality(str(p))["total"] == 0.0


def test_flat_grey_frame_scores_zero(tmp_path):
    p = tmp_path / "grey.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(p)
    assert sq.scalar_quality(str(p))["total"] == 0.0


def test_compression_is_excluded_by_default_and_weights_renormalise(sharp_and_blurred):
    sharp, _ = sharp_and_blurred
    without = sq.scalar_quality(sharp)
    with_ = sq.scalar_quality(sharp, include_compression=True)
    # Same components either way; only the blend differs.
    assert without["sharpness"] == with_["sharpness"]
    w = sq.DEFAULT_WEIGHTS
    expected = (
        without["sharpness"] * w["sharpness"]
        + without["contrast"] * w["contrast"]
        + without["exposure"] * w["exposure"]
    ) / (w["sharpness"] + w["contrast"] + w["exposure"])
    assert without["total"] == pytest.approx(min(100.0, expected), abs=1e-6)
    assert with_["total"] != pytest.approx(without["total"])


def test_unreadable_path_returns_none(tmp_path):
    assert sq.scalar_quality(str(tmp_path / "missing.png")) is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert sq.scalar_quality(str(bad)) is None


def test_score_many_skips_unreadable(sharp_and_blurred, tmp_path):
    sharp, blurred = sharp_and_blurred
    out = sq.score_images_quality([sharp, blurred, str(tmp_path / "nope.png")])
    assert set(out) == {sharp, blurred}
    assert out[sharp] > out[blurred]
