"""Torch-side helpers of the text reward: latent window selection and the
frozen OCR pipeline. The OCR tests need the docTR weights and are skipped
when the package is missing."""

import importlib.util

from modules.util.text_reward_model import boxes_inside_region, latent_crop_region

import torch

import pytest

HAS_DOCTR = importlib.util.find_spec("doctr") is not None


def test_latent_crop_region_covers_boxes_with_margin_and_respects_cap():
    lh = lw = 128  # 1024 px
    boxes = [(0.10, 0.40, 0.45, 0.48), (0.50, 0.40, 0.80, 0.48)]
    y0, y1, x0, x1 = latent_crop_region(boxes, lh, lw, margin=0.05)
    assert x0 <= 0.05 * lw and x1 >= 0.85 * lw
    assert y0 <= 0.35 * lh and y1 >= 0.53 * lh

    capped = latent_crop_region(boxes, lh, lw, margin=0.05, max_area=32 * 32)
    cy0, cy1, cx0, cx1 = capped
    assert (cy1 - cy0) * (cx1 - cx0) <= 32 * 32
    assert 0 <= cy0 < cy1 <= lh and 0 <= cx0 < cx1 <= lw

    assert latent_crop_region([], lh, lw) == (0, lh, 0, lw)


def test_boxes_inside_region_filters_words_outside_the_decoded_window():
    lh = lw = 128
    boxes = [(0.10, 0.40, 0.30, 0.48), (0.70, 0.40, 0.90, 0.48)]
    region = (int(0.35 * lh), int(0.55 * lh), 0, int(0.5 * lw))
    assert boxes_inside_region(boxes, region, lh, lw) == [0]


@pytest.mark.skipif(not HAS_DOCTR, reason="python-doctr not installed")
def test_frozen_ocr_reads_rendered_text_and_ctc_gradient_reaches_pixels():
    from modules.util.text_reward_model import TextRewardModel

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (768, 512), (200, 190, 170))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 56)
    except OSError:
        pytest.skip("no TrueType font available for the render")
    draw.text((60, 180), "The Woollen Mill - £11.50", fill=(30, 30, 40), font=font)
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()[None] / 255.0

    reward = TextRewardModel(torch.device("cpu"), augment=False)
    assert "£" in reward.vocab
    detected = reward.detect(x)[0]
    targets = ["The", "Woollen", "Mill", "-", "£11.50"]
    alignment = reward.align(detected, targets)
    # a lone dash is below the detector's minimum box height and may go unboxed
    assert alignment.missing in ([], [3]), (detected.readings, alignment)
    for bi, ti in alignment.pairs:
        assert detected.readings[bi] == targets[ti]

    x.requires_grad_(True)
    boxes = [detected.boxes[bi] for bi, _ in alignment.pairs]
    words = [targets[ti] for _, ti in alignment.pairs]
    loss = reward.ctc_loss(x, boxes, words)
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and float(x.grad.abs().sum()) > 0

    # a wrong target costs more than the right one
    misspelt = {"The": "Tne", "Woollen": "Wollen", "Mill": "MiII", "-": "_", "£11.50": "£17.50"}
    wrong = reward.ctc_loss(x.detach(), boxes, [misspelt[w] for w in words])
    assert float(wrong) > float(loss)
