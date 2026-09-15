"""Offline (pre-generated image) source of the text reward: dataset loading,
prompt parsing, wildcard export, and the re-noise reward step on a fake model."""

import importlib.util
import json
import os
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupTextRewardMixin import ModelSetupTextRewardMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TextRewardSource import TextRewardSource
from modules.util.text_reward_util import (
    DEFAULT_VOCAB,
    TextRewardBank,
    TextSpec,
    load_text_reward_dataset,
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
    (tmp_path / "b.json").write_text(json.dumps({"prompt": 'Neon text reading "Est. 1920"', "text": "Est. 1920"}))
    (tmp_path / "c.png").write_bytes(b"x")  # no prompt -> skipped
    (tmp_path / "d.png").write_bytes(b"x")
    (tmp_path / "d.txt").write_text("no quoted text\n")  # unparsable -> skipped
    rows = load_text_reward_dataset(str(tmp_path), DEFAULT_VOCAB)
    assert [(r[0].text, os.path.basename(r[2])) for r in rows] == [("Bakery", "a.png"), ("Est. 1920", "b.jpg")]
    assert load_text_reward_dataset(str(tmp_path / "missing"), DEFAULT_VOCAB) == []


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


@pytest.mark.skipif(not HAS_DOCTR, reason="python-doctr not installed")
def test_offline_reward_step_skips_the_prefix_and_backpropagates(tmp_path):
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
    setup.text_reward_model().augment = False
    vae = _FakeVae()
    model = SimpleNamespace(
        vae=vae,
        train_dtype=SimpleNamespace(torch_dtype=lambda: torch.float32),
        unscale_latents=lambda x: x,
        scale_latents=lambda x: x,
        vae_to=lambda device: None,
    )
    config = TrainConfig.default_values()
    config.text_reward_source = TextRewardSource.OFFLINE_IMAGES
    config.text_reward_dataset_dir = str(tmp_path)
    config.text_reward_resolution = 768
    config.text_reward_rollout_steps = 25
    config.text_reward_random_grad_step = True
    config.text_reward_max_decode_pixels = 512
    config.text_reward_save_interval = 0
    config.workspace_dir = str(tmp_path)

    rows = load_text_reward_dataset(str(tmp_path), setup.text_reward_model().vocab)
    spec, _prompt, path = rows[0]
    from modules.util.text_reward_util import TextRewardSample

    # the fake velocity needs the true x0 = what the image encodes to
    with torch.no_grad():
        setup.source = setup._text_reward_encode_image(model, path, config)
    assert setup.source.shape == (1, 3, 64, 96)  # 512x768 -> /8, non-square preserved

    sample = TextRewardSample(spec, torch.zeros(4, 8), None, path)
    progress = SimpleNamespace(global_step=5, epoch=0, epoch_step=5)
    backwards = []
    value = setup.calculate_text_reward_loss(model, sample, config, progress, backwards.append)

    assert setup.forwards == 2 and setup.grad_forwards == 1  # probe + grad step, no prefix
    assert len(backwards) == 1 and torch.isfinite(value)
    backwards[0].backward()
    assert float(vae.gain.grad.abs()) > 0
    metrics = setup.get_last_text_reward_metrics()
    assert metrics["scored_words"] >= 2 and metrics["word_ratio"] > 0.5
