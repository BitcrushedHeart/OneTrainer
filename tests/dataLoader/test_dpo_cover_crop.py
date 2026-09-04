"""The rejected half of a DPO pair is cover-scaled and cropped, never stretched."""

from modules.dataLoader.dpo.CoverScaleCropImage import (
    CoverScaleCropImage,
    cover_scale_size,
    relative_crop_offset,
)

import torch
from torchvision import transforms
from torchvision.transforms import functional

import pytest


def _gradient(h: int, w: int) -> torch.Tensor:
    """3xHxW image whose channels encode x and y position — a stretch or a
    wrong crop shows up as a wrong gradient slope."""
    ys = torch.linspace(0, 1, h).view(h, 1).expand(h, w)
    xs = torch.linspace(0, 1, w).view(1, w).expand(h, w)
    return torch.stack([xs, ys, torch.zeros(h, w)])


def _module(image, scale_hw, crop_hw, offset):
    module = CoverScaleCropImage(
        names=["image_rejected"],
        scale_resolution_in_name="scale_resolution",
        crop_resolution_in_name="crop_resolution",
        crop_offset_in_name="crop_offset",
    )
    values = {
        "image_rejected": image,
        "scale_resolution": list(scale_hw),
        "crop_resolution": list(crop_hw),
        "crop_offset": tuple(offset),
    }
    module._get_previous_length = lambda name: 1
    module._get_previous_item = lambda variation, name, index: values[name]
    return module


def test_cover_scale_size_never_undershoots_the_crop_box():
    assert cover_scale_size((100, 200), (64, 64)) == (64, 128)
    assert cover_scale_size((200, 100), (64, 64)) == (128, 64)
    assert cover_scale_size((64, 64), (64, 64)) == (64, 64)
    # Rounding can't leave a side short.
    h, w = cover_scale_size((333, 500), (96, 128))
    assert h >= 96 and w >= 128


def test_relative_offset_keeps_the_chosen_framing_fraction():
    # Chosen: 40 px of vertical slack, jittered to the very bottom.
    assert relative_crop_offset((40, 0), (104, 64), (64, 64), (200, 64)) == (136, 0)
    # Chosen exactly on-bucket (no slack) -> centre whatever slack we have.
    assert relative_crop_offset((0, 0), (64, 64), (64, 64), (64, 100)) == (0, 18)
    # No slack here either -> zero.
    assert relative_crop_offset((7, 7), (80, 80), (64, 64), (64, 64)) == (0, 0)


def test_same_bucket_matches_scale_crop_image_exactly():
    """A rejected image with the chosen image's aspect must come out identical
    to what ScaleCropImage produced before — this change is a no-op there."""
    image = _gradient(300, 450)  # 3:2
    scale_hw, crop_hw, offset = (96, 144), (96, 128), (0, 9)
    expected = functional.crop(
        transforms.Resize(scale_hw, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(image),
        offset[0],
        offset[1],
        *crop_hw,
    )
    out = _module(image, scale_hw, crop_hw, offset).get_item(0, 0)["image_rejected"]
    assert out.shape == expected.shape
    assert torch.allclose(out, expected)


def test_mismatched_aspect_is_cropped_not_stretched():
    """Chosen is 3:2 landscape; rejected is 7:4. The x-gradient slope of the
    output must equal that of an aspect-preserving resize (i.e. the visible
    field narrows), not the slope of a full-width squash."""
    rejected = _gradient(400, 700)  # 7:4
    scale_hw, crop_hw, offset = (96, 144), (96, 144), (0, 0)  # chosen fills its bucket
    out = _module(rejected, scale_hw, crop_hw, offset).get_item(0, 0)["image_rejected"]
    assert out.shape == (3, 96, 144)

    # Aspect-preserving cover scale: height 96 -> width 168; a 144-wide centre
    # crop keeps x in [12/168, 156/168].
    x_left, x_right = out[0, 48, 0].item(), out[0, 48, -1].item()
    assert x_left == pytest.approx(12 / 167, abs=0.02)
    assert x_right == pytest.approx(155 / 167, abs=0.02)
    # A stretch would have spanned the full [0, 1] range.
    assert x_left > 0.03 and x_right < 0.97
    # y is unaffected: the full height is visible.
    assert out[1, 0, 72].item() == pytest.approx(0.0, abs=0.02)
    assert out[1, -1, 72].item() == pytest.approx(1.0, abs=0.02)


def test_portrait_rejected_against_landscape_chosen_crops_vertically():
    rejected = _gradient(600, 400)  # 2:3 portrait
    scale_hw, crop_hw, offset = (96, 144), (96, 144), (0, 0)
    out = _module(rejected, scale_hw, crop_hw, offset).get_item(0, 0)["image_rejected"]
    assert out.shape == (3, 96, 144)
    # Width drives the scale (144 wide -> 216 tall); centre crop keeps the
    # middle 96 rows, so the y-gradient no longer spans [0, 1].
    assert out[1, 0, 72].item() > 0.2
    assert out[1, -1, 72].item() < 0.8
    # x spans the full width.
    assert out[0, 48, 0].item() == pytest.approx(0.0, abs=0.02)
    assert out[0, 48, -1].item() == pytest.approx(1.0, abs=0.02)


def test_video_layout_uses_trailing_spatial_dims():
    rejected = _gradient(400, 700).unsqueeze(0).expand(2, 3, 400, 700).clone()  # F,C,H,W
    out = _module(rejected, (96, 144), (96, 144), (0, 0)).get_item(0, 0)["image_rejected"]
    assert out.shape == (2, 3, 96, 144)


def test_none_image_passes_through():
    assert _module(None, (96, 144), (96, 144), (0, 0)).get_item(0, 0)["image_rejected"] is None
