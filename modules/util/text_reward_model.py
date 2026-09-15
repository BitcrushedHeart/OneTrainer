"""Frozen OCR as a differentiable reward for on-image text.

docTR supplies both halves as plain torch modules: a DBNet text *detector*
(``db_mobilenet_v3_large``) that runs under ``no_grad`` on a decoded image and
returns word boxes, and a CRNN *recognizer* (``crnn_vgg16_bn``, ~15M params,
126-character vocabulary with £ € ¥ and full ASCII punctuation) whose CTC loss
is a differentiable function of its input crop. Frozen means the OCR never
learns; it does not mean the gradient stops — ``loss.backward()`` reaches
whatever produced the pixels.

Only the recognizer is in the gradient path. Boxes are constants (like the
reward in DRaFT), the crop is ``roi_align`` (differentiable in the image, not
in the box), and a light differentiable augmentation — box jitter, blur,
noise — sits between the crop and the recognizer so the policy cannot satisfy
the CTC head with a pixel pattern that a small resample would break.

Memory: the detector runs at 1024² in bf16 under ``no_grad`` (tens of MB of
activations); the recognizer sees ``(words, 3, 32, 128)`` crops. Neither is
what costs VRAM in the reward step — the VAE decode with grad is, which is
why :func:`latent_crop_region` exists: decode only the text region.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

from modules.util.text_reward_util import (
    WordAlignment,
    align_words,
    reading_order,
)

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

DETECTOR_ARCH = "db_mobilenet_v3_large"
RECOGNIZER_ARCH = "parseq"
#: Recognizers with a differentiable loss under ``forward(x, target=...)``.
#: PARSeq (ViT, permuted autoregressive, teacher-forced CE) reads stylised and
#: curved text far better than the CRNN (CTC); both use the 126-char vocab.
RECOGNIZER_ARCHS = ("parseq", "crnn_vgg16_bn", "vitstr_base", "sar_resnet31", "master")
#: TrOCR (transformers ``VisionEncoderDecoderModel``): ViT encoder + text decoder,
#: teacher-forced cross-entropy loss, byte-pair vocabulary (any character).
#: ``trocr-base-str`` is the scene-text fine-tune, the right one for signs.
TROCR_ARCHS = ("trocr-base-str", "trocr-base-printed", "trocr-large-str", "trocr-large-printed")
TROCR_MAX_TOKENS = 32


@dataclass
class DetectedWords:
    """Boxes for one image, relative ``xmin, ymin, xmax, ymax`` in [0, 1],
    already in reading order, and the recognizer's no-grad reading of each."""

    boxes: list[tuple[float, float, float, float]]
    readings: list[str]


def latent_crop_region(
    boxes: Sequence[Sequence[float]],
    latent_h: int,
    latent_w: int,
    *,
    margin: float = 0.08,
    max_area: int | None = None,
) -> tuple[int, int, int, int]:
    """Latent-space window ``(y0, y1, x0, x1)`` covering ``boxes`` (relative
    coords) with a relative ``margin``, optionally shrunk toward the boxes'
    centre so ``(y1-y0)*(x1-x0) <= max_area``. Falls back to the whole latent
    when there are no boxes."""
    if not boxes:
        y0, y1, x0, x1 = 0, latent_h, 0, latent_w
    else:
        xmin = min(b[0] for b in boxes) - margin
        ymin = min(b[1] for b in boxes) - margin
        xmax = max(b[2] for b in boxes) + margin
        ymax = max(b[3] for b in boxes) + margin
        x0 = max(0, math.floor(xmin * latent_w))
        y0 = max(0, math.floor(ymin * latent_h))
        x1 = min(latent_w, math.ceil(xmax * latent_w))
        y1 = min(latent_h, math.ceil(ymax * latent_h))
    # keep at least 2x2 latents
    if x1 - x0 < 2:
        x1 = min(latent_w, x0 + 2)
        x0 = max(0, x1 - 2)
    if y1 - y0 < 2:
        y1 = min(latent_h, y0 + 2)
        y0 = max(0, y1 - 2)
    if max_area is not None and (y1 - y0) * (x1 - x0) > max_area:
        scale = math.sqrt(max_area / ((y1 - y0) * (x1 - x0)))
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        h = max(2, int((y1 - y0) * scale))
        w = max(2, int((x1 - x0) * scale))
        y0 = int(max(0, min(latent_h - h, round(cy - h / 2))))
        x0 = int(max(0, min(latent_w - w, round(cx - w / 2))))
        y1, x1 = y0 + h, x0 + w
    return int(y0), int(y1), int(x0), int(x1)


def boxes_inside_region(
    boxes: Sequence[Sequence[float]],
    region: tuple[int, int, int, int],
    latent_h: int,
    latent_w: int,
    *,
    min_overlap: float = 0.9,
) -> list[int]:
    """Indices of boxes that lie (at least ``min_overlap`` of their area)
    inside the latent ``region``. Words outside the decoded crop cannot be
    scored and must not be counted as missing."""
    y0, y1, x0, x1 = region
    rx0, ry0, rx1, ry1 = x0 / latent_w, y0 / latent_h, x1 / latent_w, y1 / latent_h
    out = []
    for i, b in enumerate(boxes):
        area = max(b[2] - b[0], 1e-6) * max(b[3] - b[1], 1e-6)
        ix = max(0.0, min(b[2], rx1) - max(b[0], rx0))
        iy = max(0.0, min(b[3], ry1) - max(b[1], ry0))
        if ix * iy / area >= min_overlap:
            out.append(i)
    return out


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-9)


def _suppress_overlaps(
    candidates: Sequence[tuple[float, float, float, float, float]], iou_threshold: float = 0.4
) -> list[tuple[float, float, float, float]]:
    """Greedy NMS across detection scales: highest confidence first; a box
    that mostly overlaps a kept one, or lies almost entirely inside it, is
    dropped (a fragment of a giant word found at native scale vs the whole
    word found shrunk)."""
    kept: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1, _conf in sorted(candidates, key=lambda c: -c[4]):
        box = (x0, y0, x1, y1)
        area = max((x1 - x0) * (y1 - y0), 1e-9)
        drop = False
        for k in kept:
            ix = max(0.0, min(x1, k[2]) - max(x0, k[0]))
            iy = max(0.0, min(y1, k[3]) - max(y0, k[1]))
            if _iou(box, k) >= iou_threshold or (ix * iy) / area >= 0.7:
                drop = True
                break
        if not drop:
            kept.append(box)
    return kept


class TextRewardModel:
    """Frozen docTR detector + recognizer. Construct once, keep on the training
    device; ``detect`` and ``ctc_loss`` are the two entry points."""

    def __init__(
        self,
        device: torch.device,
        *,
        detector_arch: str = DETECTOR_ARCH,
        recognizer_arch: str = RECOGNIZER_ARCH,
        augment: bool = True,
        min_box_confidence: float = 0.15,
        min_box_height: float = 0.012,
    ):
        from doctr.models import detection, recognition

        self.device = device
        self.detector_arch = detector_arch
        self.recognizer_arch = recognizer_arch
        self.is_trocr = recognizer_arch.startswith("trocr")
        self.detector = getattr(detection, detector_arch)(pretrained=True).eval().to(device)
        for p in self.detector.parameters():
            p.requires_grad_(False)
        self._det_shape = tuple(self.detector.cfg["input_shape"][1:])
        self._det_mean = torch.tensor(self.detector.cfg["mean"], device=device).view(1, 3, 1, 1)
        self._det_std = torch.tensor(self.detector.cfg["std"], device=device).view(1, 3, 1, 1)

        if self.is_trocr:
            from modules.util.text_reward_util import DEFAULT_VOCAB

            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            repo = f"microsoft/{recognizer_arch}"
            self.processor = TrOCRProcessor.from_pretrained(repo)
            self.recognizer = VisionEncoderDecoderModel.from_pretrained(repo).eval().to(device)
            self.tokenizer = self.processor.tokenizer
            # The published checkpoints omit these; teacher forcing needs them.
            cfg = self.recognizer.config
            if getattr(cfg, "decoder_start_token_id", None) is None:
                cfg.decoder_start_token_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
            if getattr(cfg, "pad_token_id", None) is None:
                cfg.pad_token_id = self.tokenizer.pad_token_id
            # BPE vocabulary: every printable character is representable.
            self.vocab = DEFAULT_VOCAB
            size = self.processor.image_processor.size
            self._rec_shape = (int(size["height"]), int(size["width"]))
            mean = self.processor.image_processor.image_mean
            std = self.processor.image_processor.image_std
        else:
            self.recognizer = getattr(recognition, recognizer_arch)(pretrained=True).eval().to(device)
            self.vocab = self.recognizer.vocab
            self._rec_shape = tuple(self.recognizer.cfg["input_shape"][1:])
            mean, std = self.recognizer.cfg["mean"], self.recognizer.cfg["std"]
        for p in self.recognizer.parameters():
            p.requires_grad_(False)
        self.augment = augment
        self.min_box_confidence = min_box_confidence
        self.min_box_height = min_box_height
        self._rec_mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self._rec_std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def to(self, device: torch.device) -> TextRewardModel:
        self.device = device
        self.detector.to(device)
        self.recognizer.to(device)
        for name in ("_det_mean", "_det_std", "_rec_mean", "_rec_std"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # -- detection (no grad) -------------------------------------------------

    @torch.no_grad()
    def detect(self, images01: torch.Tensor, scales: Sequence[float] = (1.0, 0.5, 0.3)) -> list[DetectedWords]:
        """``images01``: ``(B, 3, H, W)`` in [0, 1]. Returns per-image boxes
        in reading order with their no-grad readings.

        Scene-text detectors are tuned for small text; a sign filling a third
        of the frame goes unboxed at native scale. So the image is also
        detected shrunk into the corner of the detector canvas (``scales`` <
        1), which turns giant glyphs into ordinary ones, and the boxes are
        merged with a greedy IoU suppression favouring the higher confidence."""
        det_h, det_w = self._det_shape
        per_image: list[list[tuple[float, float, float, float, float]]] = [[] for _ in range(images01.shape[0])]
        for scale in scales:
            h = max(32, int(det_h * scale) // 32 * 32)
            w = max(32, int(det_w * scale) // 32 * 32)
            x = F.interpolate(images01.float(), size=(h, w), mode="bilinear", align_corners=False)
            if scale < 1.0:
                canvas = torch.ones((x.shape[0], 3, det_h, det_w), device=x.device, dtype=x.dtype)
                canvas *= x.mean(dim=(2, 3), keepdim=True)
                canvas[:, :, :h, :w] = x
                x = canvas
            x = (x - self._det_mean) / self._det_std
            preds = self.detector(x, return_preds=True)["preds"]
            sx, sy = det_w / w, det_h / h
            for i, pred in enumerate(preds):
                for b in pred["words"]:
                    conf = float(b[4])
                    x0, y0, x1, y1 = float(b[0]) * sx, float(b[1]) * sy, float(b[2]) * sx, float(b[3]) * sy
                    if x0 >= 1.0 or y0 >= 1.0:
                        continue  # canvas padding
                    x1, y1 = min(x1, 1.0), min(y1, 1.0)
                    if conf >= self.min_box_confidence and (y1 - y0) >= self.min_box_height:
                        per_image[i].append((x0, y0, x1, y1, conf))

        out: list[DetectedWords] = []
        for i, cands in enumerate(per_image):
            boxes = _suppress_overlaps(cands)
            order = reading_order(boxes)
            boxes = [boxes[k] for k in order]
            readings = self.read(images01[i : i + 1], boxes) if boxes else []
            out.append(DetectedWords(boxes=boxes, readings=readings))
        return out

    @torch.no_grad()
    def read(self, image01: torch.Tensor, boxes: Sequence[Sequence[float]]) -> list[str]:
        crops = self.crop(image01, boxes, augment=False)
        if self.is_trocr:
            ids = self.recognizer.generate(
                pixel_values=self._rec_norm(crops), max_new_tokens=TROCR_MAX_TOKENS, num_beams=1
            )
            return [t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]
        preds = self.recognizer(self._rec_norm(crops), return_preds=True)["preds"]
        return [p[0] for p in preds]

    # -- crops and loss (grad) -----------------------------------------------

    def crop(self, image01: torch.Tensor, boxes: Sequence[Sequence[float]], *, augment: bool) -> torch.Tensor:
        """``roi_align`` word crops of a single ``(1, 3, H, W)`` image at the
        recognizer's input size. Differentiable in the image."""
        _, _, h, w = image01.shape
        rois = []
        for b in boxes:
            x0, y0, x1, y1 = b
            if augment:
                jw, jh = (x1 - x0) * 0.06, (y1 - y0) * 0.12
                x0 += (torch.rand(()).item() * 2 - 1) * jw
                x1 += (torch.rand(()).item() * 2 - 1) * jw
                y0 += (torch.rand(()).item() * 2 - 1) * jh
                y1 += (torch.rand(()).item() * 2 - 1) * jh
            rois.append([0.0, x0 * w, y0 * h, x1 * w, y1 * h])
        rois_t = torch.tensor(rois, dtype=torch.float32, device=image01.device)
        crops = roi_align(
            image01.float(),
            rois_t,
            output_size=self._rec_shape,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        if augment:
            crops = self._augment(crops)
        return crops

    @staticmethod
    def _augment(crops: torch.Tensor) -> torch.Tensor:
        """Differentiable, cheap, per-batch: gaussian blur with a random small
        sigma, then gaussian noise. Enough to make single-pixel adversarial
        structure unstable; not enough to blur real glyphs at 32 px."""
        sigma = 0.3 + torch.rand(()).item() * 0.6
        k = 5
        ax = torch.arange(k, dtype=crops.dtype, device=crops.device) - (k - 1) / 2
        g = torch.exp(-(ax**2) / (2 * sigma**2))
        g = (g / g.sum()).view(1, 1, 1, k)
        c = crops.shape[1]
        crops = F.conv2d(F.pad(crops, (2, 2, 0, 0), mode="replicate"), g.expand(c, 1, 1, k), groups=c)
        crops = F.conv2d(F.pad(crops, (0, 0, 2, 2), mode="replicate"), g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)
        crops = crops + torch.randn_like(crops) * 0.01
        return crops.clamp(0, 1)

    def _rec_norm(self, crops: torch.Tensor) -> torch.Tensor:
        return (crops - self._rec_mean) / self._rec_std

    def ctc_loss(
        self,
        image01: torch.Tensor,
        boxes: Sequence[Sequence[float]],
        targets: Sequence[str],
    ) -> torch.Tensor:
        """Mean CTC loss of the recognizer reading ``targets[k]`` from
        ``boxes[k]`` of the (grad-bearing) image. Empty ``boxes`` gives a zero
        that still depends on the image so callers can backward unconditionally."""
        if not boxes:
            return image01.sum() * 0.0
        crops = self.crop(image01, boxes, augment=self.augment)
        if self.is_trocr:
            # Teacher-forced cross-entropy of the target tokens given the crop;
            # padding is masked with -100 as transformers expects.
            enc = self.tokenizer(
                list(targets), padding=True, truncation=True, max_length=TROCR_MAX_TOKENS, return_tensors="pt"
            )
            labels = enc.input_ids.to(self.device)
            labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
            out = self.recognizer(pixel_values=self._rec_norm(crops), labels=labels)
            return out.loss
        # The CRNN's LSTM sits in eval mode (frozen); cuDNN only implements the
        # RNN backward in training mode, so route this tiny model through the
        # native kernels instead.
        with torch.backends.cudnn.flags(enabled=False):
            out = self.recognizer(self._rec_norm(crops), target=list(targets))
        return out["loss"]

    def align(self, detected: DetectedWords, target_words: Sequence[str]) -> WordAlignment:
        return align_words(detected.readings, target_words)


@contextlib.contextmanager
def autocast_off():
    """The recognizer's CTC in fp32; docTR models are fp32 and small."""
    with torch.autocast(device_type="cuda", enabled=False):
        yield
