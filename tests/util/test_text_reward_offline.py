"""Offline (pre-generated image) source of the text reward: dataset loading,
prompt parsing, wildcard export, and the re-noise reward step on a fake model."""

import glob
import importlib.util
import json
import os
import pickle
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupTextRewardMixin import ModelSetupTextRewardMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TextRewardSource import TextRewardSource
from modules.util.text_reward_util import (
    DEFAULT_VOCAB,
    TextRewardBank,
    TextRewardSample,
    TextSpec,
    WordAlignment,
    align_words,
    load_text_reward_dataset,
    policy_read_metrics,
    spec_from_prompt,
    target_text_from_prompt,
    write_wildcard_file,
)

import torch
import torch.nn.functional as F

import pytest

HAS_DOCTR = importlib.util.find_spec("doctr") is not None


def test_prompt_parsing_and_vocab_filter():
    assert target_text_from_prompt('Neon text reading "Open Daily - £4.50"') == "Open Daily - £4.50"
    assert target_text_from_prompt("no quotes here") is None
    spec = spec_from_prompt('Neon text reading "Open Daily - £4.50"', DEFAULT_VOCAB)
    assert spec.words == ("Open", "Daily", "-", "£4.50") and spec.surface == "Neon"
    ascii_only = DEFAULT_VOCAB.replace("£", "")
    assert spec_from_prompt('Sign text reading "£"', ascii_only) is None
    assert spec_from_prompt('Sign text reading "£4"', ascii_only).words == ("4",)


def test_dataset_loader_reads_txt_and_json(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text('Sign text reading "Bakery"\nsecond line ignored\n', encoding="utf-8")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "b.json").write_text(
        json.dumps({"prompt": 'Neon text reading "Est. 1920"', "text": "Est. 1920", "sha256": " ABCdef01 "})
    )
    (tmp_path / "c.png").write_bytes(b"x")  # no prompt -> skipped
    (tmp_path / "d.png").write_bytes(b"x")
    (tmp_path / "d.txt").write_text("no quoted text\n")  # unparsable -> skipped
    rows = load_text_reward_dataset(str(tmp_path), DEFAULT_VOCAB)
    assert [(r[0].text, os.path.basename(r[2])) for r in rows] == [("Bakery", "a.png"), ("Est. 1920", "b.jpg")]
    assert load_text_reward_dataset(str(tmp_path / "missing"), DEFAULT_VOCAB) == []
    # the JSON's sha256 rides along without changing the 3-tuple shape
    assert [r.sha256 for r in rows] == [None, "abcdef01"]
    spec, prompt, path = rows[1]
    assert rows[1] == (spec, prompt, path) and len(rows[1]) == 3
    assert (rows[1].spec, rows[1].prompt, rows[1].image_path) == (spec, prompt, path)
    clone = pickle.loads(pickle.dumps(rows[1]))
    assert clone == rows[1] and clone.sha256 == "abcdef01"


def test_wildcard_file_is_one_prompt_per_line(tmp_path):
    path = tmp_path / "wild" / "text_alignment.txt"
    n = write_wildcard_file(str(path), 25, seed=3)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert n == 25 and len(lines) == 25
    assert all(' text reading "' in line and line.endswith('"') for line in lines)
    assert write_wildcard_file(str(path), 25, seed=3) == 25
    assert path.read_text(encoding="utf-8").splitlines() == lines  # deterministic


def test_bank_carries_image_paths():
    specs = [TextSpec("Sign", "a", ("a",), "offline"), TextSpec("Sign", "b", ("b",), "offline")]
    bank = TextRewardBank(specs, [1, 2], None, ["x.png", "y.png"])
    assert [bank.next().image_path for _ in range(3)] == ["x.png", "y.png", "x.png"]
    with pytest.raises(ValueError):
        TextRewardBank(specs, [1, 2], None, ["x.png"])
    hashed = TextRewardBank(specs, [1, 2], None, ["x.png", "y.png"], ["ab", None])
    assert [hashed.next().image_sha256 for _ in range(3)] == ["ab", None, "ab"]
    with pytest.raises(ValueError):
        TextRewardBank(specs, [1, 2], None, ["x.png", "y.png"], ["ab"])


def test_policy_read_metrics_skips_unread_boxes_but_counts_missing():
    targets = ["Open", "Daily", "Now", "Here"]
    # boxes 0,1,2 aligned on the source to targets 0,1,2; target 3 missing
    alignment = WordAlignment(pairs=[(0, 0), (1, 1), (2, 2)], missing=[3])
    # box 2 lay outside the decoded crop: no policy reading, left out of the ratio
    metrics = policy_read_metrics({0: "Open", 1: "Daily"}, targets, alignment)
    assert metrics.missing == pytest.approx(0.25)
    assert metrics.word_ratio == pytest.approx(2 / 3)  # Open 1, Daily 1, Here (missing) 0
    assert metrics.exact == 0.0
    full = policy_read_metrics({0: "Open", 1: "Daily", 2: "Now"}, targets[:3], WordAlignment(pairs=alignment.pairs))
    assert full.word_ratio == pytest.approx(1.0) and full.exact == 1.0 and full.missing == 0.0


def test_offline_validation_needs_a_populated_folder(tmp_path):
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.text_reward_enabled = True
    config.text_reward_source = TextRewardSource.OFFLINE_IMAGES
    config.text_reward_dataset_dir = ""
    assert "dataset folder" in config.validate_text_reward_startup()
    config.text_reward_dataset_dir = str(tmp_path)
    assert "no image" in config.validate_text_reward_startup()
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text('Sign text reading "Bakery"\n')
    assert config.validate_text_reward_startup() is None


class _FakeVae(torch.nn.Module):
    """Latent = image downsampled 8x (3 channels); encode/decode are exact inverses up to resampling."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=3, scaling_factor=1.0, shift_factor=0.0)
        self.gain = torch.nn.Parameter(torch.ones(()))

    def encode(self, pixels, return_dict=False):
        latent = F.interpolate((pixels + 1.0) * 0.5, scale_factor=1 / 8, mode="area")
        return (SimpleNamespace(mode=lambda: latent),)

    def decode(self, latent, return_dict=False):
        image = F.interpolate(latent * self.gain, scale_factor=8, mode="bilinear", align_corners=False)
        return (image * 2.0 - 1.0,)


class _FakeSetup(ModelSetupTextRewardMixin):
    """Velocity = eps - x0 for the exact x0 the image encodes to: the flow
    target, so one step lands back on the source image."""

    def __init__(self):
        super().__init__()
        self.train_device = torch.device("cpu")
        self.temp_device = torch.device("cpu")
        self.forwards = 0
        self.grad_forwards = 0
        self.source = None

    def predict_distill_velocity(
        self, model, batch, config, train_progress, latent, sigma, *, conditioning="conditional"
    ):
        self.forwards += 1
        if torch.is_grad_enabled():
            self.grad_forwards += 1
        s = sigma.view(-1, 1, 1, 1)
        return (latent - self.source) / s


def _offline_model(vae):
    return SimpleNamespace(
        vae=vae,
        train_dtype=SimpleNamespace(torch_dtype=lambda: torch.float32),
        unscale_latents=lambda x: x,
        scale_latents=lambda x: x,
        vae_to=lambda device: None,
    )


def _log_calls(obj, name, log, label=None):
    original = getattr(obj, name)

    def wrapper(*args, **kwargs):
        log.append(label or name)
        return original(*args, **kwargs)

    setattr(obj, name, wrapper)


@pytest.mark.skipif(not HAS_DOCTR, reason="python-doctr not installed")
def test_offline_reward_step_is_one_forward_and_backpropagates(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    words = ("Open", "Daily", "£4.50")
    image = Image.new("RGB", (768, 512), (215, 205, 190))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 96)
    except OSError:
        pytest.skip("no TrueType font available for the render")
    draw.text((60, 200), " ".join(words), fill=(25, 25, 35), font=font)
    image.save(tmp_path / "s.png")
    (tmp_path / "s.txt").write_text(f'Sign text reading "{" ".join(words)}"\n', encoding="utf-8")

    setup = _FakeSetup()
    reward = setup.text_reward_model()
    reward.augment = False
    vae = _FakeVae()
    model = _offline_model(vae)
    config = TrainConfig.default_values()
    config.text_reward_source = TextRewardSource.OFFLINE_IMAGES
    config.text_reward_dataset_dir = str(tmp_path)
    config.text_reward_resolution = 768
    config.text_reward_rollout_steps = 25
    config.text_reward_random_grad_step = True
    config.text_reward_max_decode_pixels = 512
    config.text_reward_save_interval = 0
    config.workspace_dir = str(tmp_path)
    config.cache_dir = str(tmp_path / "cache")

    rows = load_text_reward_dataset(str(tmp_path), reward.vocab)
    spec, _prompt, path = rows[0]

    # the fake velocity needs the true x0 = what the image encodes to
    with torch.no_grad():
        setup.source = setup._text_reward_encode_image(model, path, config)
    assert setup.source.shape == (1, 3, 64, 96)  # 512x768 -> /8, non-square preserved

    calls = []
    _log_calls(vae, "encode", calls)
    _log_calls(reward, "detect", calls)

    sample = TextRewardSample(spec, torch.zeros(4, 8), None, path)
    progress = SimpleNamespace(global_step=5, epoch=0, epoch_step=5)
    backwards = []
    value = setup.calculate_text_reward_loss(model, sample, config, progress, backwards.append)

    assert setup.forwards == 1 and setup.grad_forwards == 1  # the grad step only: no prefix, no probe
    assert calls.count("encode") == 1 and calls.count("detect") == 1  # cache miss
    assert len(glob.glob(os.path.join(config.cache_dir, "text_reward", "*.pt"))) == 1
    assert len(backwards) == 1 and torch.isfinite(value)
    backwards[0].backward()
    assert float(vae.gain.grad.abs()) > 0
    metrics = setup.get_last_text_reward_metrics()
    assert metrics["scored_words"] >= 2 and metrics["word_ratio"] > 0.5

    # same image, same config: served from the cache, bit-identical step
    calls.clear()
    setup.forwards = setup.grad_forwards = 0
    backwards_hit = []
    value_hit = setup.calculate_text_reward_loss(model, sample, config, progress, backwards_hit.append)
    assert calls == []  # no VAE encode, no detect
    assert setup.forwards == 1 and setup.grad_forwards == 1
    assert float(value_hit) == float(value)
    assert setup.get_last_text_reward_metrics() == metrics


# -- the offline step with a scripted OCR (no docTR needed) --------------------


class _ScriptedReward:
    """Stands in for TextRewardModel: fixed detections, a differentiable
    "loss" and readings that echo the scripted words. Logs every call."""

    detector_arch = "scripted_det"
    recognizer_arch = "scripted_rec"
    min_box_confidence = 0.15
    min_box_height = 0.012
    device = torch.device("cpu")
    vocab = DEFAULT_VOCAB

    def __init__(self, words, boxes, log):
        self.words = list(words)
        self.boxes = list(boxes)
        self.log = log

    def detect(self, images01):
        from modules.util.text_reward_model import DetectedWords

        self.log.append("detect")
        return [DetectedWords(boxes=list(self.boxes), readings=list(self.words))]

    def align(self, detected, targets):
        return align_words(detected.readings, targets)

    def ctc_loss(self, image01, boxes, targets):
        self.log.append("ctc_loss")
        return image01.mean() * len(boxes)

    def read(self, image01, boxes):
        self.log.append("read")
        return self.words[: len(boxes)]


_WORDS = ("Open", "Daily")
_BOXES = [(0.10, 0.40, 0.35, 0.60), (0.40, 0.40, 0.70, 0.60)]


def _scripted_setup(tmp_path, *, save_interval=0):
    from PIL import Image

    rng = torch.Generator().manual_seed(0)
    pixels = (torch.rand((128, 256, 3), generator=rng) * 255).to(torch.uint8).numpy()
    path = tmp_path / "img.png"
    Image.fromarray(pixels).save(path)

    log = []
    setup = _FakeSetup()
    setup.source = torch.zeros(())
    setup._text_reward_model = _ScriptedReward(_WORDS, _BOXES, log)
    vae = _FakeVae()
    _log_calls(vae, "encode", log)
    model = _offline_model(vae)
    config = TrainConfig.default_values()
    config.text_reward_source = TextRewardSource.OFFLINE_IMAGES
    config.text_reward_resolution = 256
    config.text_reward_rollout_steps = 25
    config.text_reward_random_grad_step = True
    config.text_reward_max_decode_pixels = 512
    config.text_reward_save_interval = save_interval
    config.workspace_dir = str(tmp_path / "ws")
    config.cache_dir = str(tmp_path / "cache")
    sample = TextRewardSample(TextSpec("Sign", " ".join(_WORDS), _WORDS, "words"), torch.zeros(4, 8), None, str(path))
    return setup, model, vae, config, sample, log


def _spy_decode(setup, log):
    original = setup._text_reward_decode

    def spy(model_, latent, *, tiled=False):
        log.append("decode_tiled" if tiled else "decode_crop")
        return original(model_, latent, tiled=tiled)

    setup._text_reward_decode = spy


def test_offline_preview_decode_and_save_run_after_backward(tmp_path):
    setup, model, vae, config, sample, log = _scripted_setup(tmp_path, save_interval=2)
    _spy_decode(setup, log)
    _log_calls(setup, "_save_text_reward_sample", log, "save")
    progress = SimpleNamespace(global_step=7, epoch=0, epoch_step=7)

    def backward(loss):
        log.append("backward")
        loss.backward()

    setup.calculate_text_reward_loss(model, sample, config, progress, backward)  # sample 1: a save step

    assert setup.forwards == 1 and setup.grad_forwards == 1
    assert log == ["encode", "detect", "decode_crop", "ctc_loss", "backward", "read", "decode_tiled", "save"]
    assert float(vae.gain.grad.abs()) > 0
    assert os.path.exists(os.path.join(config.workspace_dir, "text_reward", "00000007.png"))
    with open(os.path.join(config.workspace_dir, "text_reward", "00000007.txt"), encoding="utf-8") as f:
        assert f.read().splitlines()[1] == "read: Open Daily"
    metrics = setup.get_last_text_reward_metrics()
    assert metrics["scored_words"] == 2 and metrics["word_ratio"] == 1.0 and metrics["missing"] == 0.0

    # sample 2 is not a save step: cache hit and no full-frame decode at all
    log.clear()
    setup.calculate_text_reward_loss(model, sample, config, progress, backward)
    assert log == ["decode_crop", "ctc_loss", "backward", "read"]


def test_online_rollout_still_runs_the_probe_before_the_grad_forward(tmp_path):
    setup, model, _vae, config, _sample, log = _scripted_setup(tmp_path)
    config.text_reward_source = TextRewardSource.ONLINE_ROLLOUT
    config.text_reward_rollout_steps = 4
    config.text_reward_random_grad_step = False  # grad step 3 -> 3 prefix forwards
    sample = TextRewardSample(TextSpec("Sign", " ".join(_WORDS), _WORDS, "words"), torch.zeros(4, 8), None)
    grad_flags = []
    original_predict = setup.predict_distill_velocity

    def predict_spy(*args, **kwargs):
        grad_flags.append(torch.is_grad_enabled())
        log.append("forward")
        return original_predict(*args, **kwargs)

    setup.predict_distill_velocity = predict_spy
    _spy_decode(setup, log)
    progress = SimpleNamespace(global_step=3, epoch=0, epoch_step=3)
    setup.calculate_text_reward_loss(model, sample, config, progress, lambda loss: log.append("backward"))

    # 3 prefix forwards + the no-grad probe, its full decode and detect, then the grad forward
    assert grad_flags == [False, False, False, False, True]
    assert log == ["forward"] * 4 + ["decode_tiled", "detect", "forward", "decode_crop", "ctc_loss", "backward"]
    assert not os.path.isdir(os.path.join(config.cache_dir, "text_reward"))
