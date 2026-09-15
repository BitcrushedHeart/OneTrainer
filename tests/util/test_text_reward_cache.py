"""On-disk cache of the offline text reward's source latent and detection:
key/identity rules, weights-only round trip, invalidation, corrupt entries and
the VAE paging of a miss. No docTR needed: the OCR is scripted."""

import glob
import os
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupTextRewardMixin import ModelSetupTextRewardMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.text_reward_cache import (
    TEXT_REWARD_CACHE_SCHEMA,
    TextRewardSourceCache,
    text_reward_cache_key,
    text_reward_image_identity,
)
from modules.util.text_reward_model import DetectedWords
from modules.util.text_reward_util import TextRewardSample, TextSpec

import torch
import torch.nn.functional as F

import pytest


def _key(**overrides):
    fields = {
        "image_identity": "sha256:ab:10",
        "resolution": 1024,
        "vae_identity": "vae|FLOAT_32",
        "detector_arch": "det",
        "recognizer_arch": "parseq",
        "min_box_confidence": 0.15,
        "min_box_height": 0.012,
    }
    fields.update(overrides)
    return text_reward_cache_key(**fields)


def test_entry_round_trips_through_weights_only_load(tmp_path):
    cache = TextRewardSourceCache(str(tmp_path / "tr"))
    latent = torch.randn(1, 16, 12, 20)
    boxes = [(0.1, 0.2, 0.30000000000000004, 0.4), (0.5, 0.55, 0.9, 0.7)]
    assert cache.save(_key(), latent, boxes, ["Open", "Daily"])
    assert glob.glob(str(tmp_path / "tr" / "*.tmp")) == []
    hit = cache.load(_key())
    assert torch.equal(hit.latent, latent) and hit.latent.dtype == torch.float32
    assert hit.boxes == boxes  # float64 storage keeps the python floats exact
    assert hit.readings == ["Open", "Daily"]
    raw = torch.load(cache.path_for(_key()), map_location="cpu", weights_only=True)
    assert raw["key"]["schema"] == TEXT_REWARD_CACHE_SCHEMA

    # an entry with no boxes is valid too
    assert cache.save(_key(image_identity="other"), latent, [], [])
    empty = cache.load(_key(image_identity="other"))
    assert empty.boxes == [] and empty.readings == []


@pytest.mark.parametrize(
    "change",
    [
        {"image_identity": "sha256:cd:10"},
        {"resolution": 768},
        {"vae_identity": "vae|BFLOAT_16"},
        {"detector_arch": "det2"},
        {"recognizer_arch": "crnn_vgg16_bn"},
        {"min_box_confidence": 0.2},
        {"min_box_height": 0.02},
    ],
)
def test_every_key_field_separates_entries(tmp_path, change):
    cache = TextRewardSourceCache(str(tmp_path))
    cache.save(_key(), torch.zeros(1, 1, 2, 2), [], [])
    assert cache.path_for(_key(**change)) != cache.path_for(_key())
    assert cache.load(_key(**change)) is None


def test_stored_key_is_verified_not_just_the_filename(tmp_path):
    cache = TextRewardSourceCache(str(tmp_path))
    cache.save(_key(), torch.zeros(1, 1, 2, 2), [], [])
    # simulate a filename collision: another key's entry sitting at this path
    os.replace(cache.path_for(_key()), cache.path_for(_key(resolution=512)))
    assert cache.load(_key(resolution=512)) is None


def test_image_identity_prefers_sha256_and_falls_back_to_path_size_mtime(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"12345")
    by_hash = text_reward_image_identity(str(image), "ABC")
    by_file = text_reward_image_identity(str(image))
    assert by_hash == "sha256:abc:5"
    assert by_file.startswith("file:") and by_file.endswith(f":5:{os.stat(image).st_mtime_ns}")
    st = os.stat(image)
    os.utime(image, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert text_reward_image_identity(str(image), "abc") == by_hash  # a touch keeps the hash identity
    assert text_reward_image_identity(str(image)) != by_file


# -- through the mixin ------------------------------------------------------------


class _Vae(torch.nn.Module):
    def __init__(self, log):
        super().__init__()
        self.config = SimpleNamespace(latent_channels=3, scaling_factor=1.0, shift_factor=0.0)
        self.gain = torch.nn.Parameter(torch.ones(()))
        self.log = log
        self.where = None  # when set, parameters() reports this device (paging test)

    def parameters(self, recurse=True):
        if self.where is None:
            yield from super().parameters(recurse)
        else:
            yield SimpleNamespace(device=self.where, dtype=torch.float32)

    def encode(self, pixels, return_dict=False):
        self.log.append("encode")
        latent = F.interpolate((pixels + 1.0) * 0.5, scale_factor=1 / 8, mode="area")
        return (SimpleNamespace(mode=lambda: latent),)


class _Reward:
    detector_arch = "scripted_det"
    recognizer_arch = "scripted_rec"
    min_box_confidence = 0.15
    min_box_height = 0.012
    device = torch.device("cpu")

    def __init__(self, log):
        self.log = log

    def detect(self, images01):
        self.log.append("detect")
        assert images01.shape[1] == 3 and float(images01.min()) >= 0.0 and float(images01.max()) <= 1.0
        return [DetectedWords(boxes=[(0.1, 0.2, 0.5, 0.4)], readings=["Open"])]


class _Setup(ModelSetupTextRewardMixin):
    def __init__(self):
        super().__init__()
        self.train_device = torch.device("cpu")
        self.temp_device = torch.device("cpu")


@pytest.fixture
def rig(tmp_path):
    from PIL import Image

    image_path = tmp_path / "img.png"
    pixels = (torch.rand((96, 160, 3), generator=torch.Generator().manual_seed(1)) * 255).to(torch.uint8)
    Image.fromarray(pixels.numpy()).save(image_path)
    log = []
    setup = _Setup()
    reward = _Reward(log)
    vae = _Vae(log)
    moves = []

    def vae_to(device):
        moves.append(device)
        if vae.where is not None:
            vae.where = device

    model = SimpleNamespace(vae=vae, scale_latents=lambda x: x, vae_to=vae_to)
    config = TrainConfig.default_values()
    config.text_reward_resolution = 160
    config.cache_dir = str(tmp_path / "cache")
    spec = TextSpec("Sign", "Open", ("Open",), "offline")

    def run(sha256=None):
        log.clear()
        sample = TextRewardSample(spec, None, None, str(image_path), sha256)
        return setup._text_reward_offline_source(model, sample, config, reward)

    return SimpleNamespace(
        run=run, log=log, config=config, image_path=image_path, setup=setup, vae=vae, moves=moves, model=model
    )


def _entries(rig):
    return glob.glob(os.path.join(rig.config.cache_dir, "text_reward", "*.pt"))


def test_miss_then_hit_returns_identical_latent_and_detection(rig):
    latent, detected = rig.run()
    assert rig.log == ["encode", "detect"]
    assert latent.device.type == "cpu" and latent.dtype == torch.float32 and latent.shape == (1, 3, 12, 20)
    with torch.no_grad():
        direct = rig.setup._text_reward_encode_image(rig.model, str(rig.image_path), rig.config)
    assert torch.equal(latent, direct)
    assert len(_entries(rig)) == 1

    latent_hit, detected_hit = rig.run()
    assert rig.log == []  # no encode, no detect
    assert torch.equal(latent_hit, latent)
    assert detected_hit == detected


def test_resolution_change_invalidates(rig):
    rig.run()
    rig.config.text_reward_resolution = 240
    latent, _ = rig.run()
    assert rig.log == ["encode", "detect"] and latent.shape == (1, 3, 18, 30)
    rig.config.text_reward_resolution = 160  # the first entry is still there
    rig.run()
    assert rig.log == []
    # 160 -> 164 rounds to the same effective 160: still a hit
    rig.config.text_reward_resolution = 164
    rig.run()
    assert rig.log == []


def test_touching_or_rewriting_the_image_invalidates_without_sha256(rig):
    rig.run()
    st = os.stat(rig.image_path)
    os.utime(rig.image_path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    rig.run()
    assert rig.log == ["encode", "detect"]
    rig.run()
    assert rig.log == []
    with open(rig.image_path, "ab") as f:  # size change
        f.write(b"\0")
    rig.run()
    assert rig.log == ["encode", "detect"]


def test_sha256_identity_survives_a_touch_and_a_different_hash_invalidates(rig):
    rig.run("aa" * 32)
    assert rig.log == ["encode", "detect"]
    st = os.stat(rig.image_path)
    os.utime(rig.image_path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    rig.run("aa" * 32)
    assert rig.log == []
    rig.run("bb" * 32)
    assert rig.log == ["encode", "detect"]


def test_corrupt_entry_is_a_miss_reported_once_and_overwritten(rig, capsys):
    latent, detected = rig.run()
    (path,) = _entries(rig)
    with open(path, "wb") as f:
        f.write(b"not a torch file")
    latent2, detected2 = rig.run()
    assert rig.log == ["encode", "detect"]
    assert torch.equal(latent2, latent) and detected2 == detected
    rig.run()
    assert rig.log == []  # rewritten entry loads again
    with open(path, "wb") as f:
        f.write(b"garbage again")
    rig.run()
    assert rig.log == ["encode", "detect"]
    assert capsys.readouterr().out.count("unreadable entry") == 1


def test_no_cache_dir_means_no_cache(rig):
    rig.config.cache_dir = ""
    rig.run()
    rig.run()
    assert rig.log == ["encode", "detect"]


def test_a_miss_pages_the_vae_in_and_back_out_a_hit_never_touches_it(rig):
    rig.vae.where = torch.device("meta")  # "not on the train device"
    rig.setup.temp_device = torch.device("meta")
    rig.run()
    assert rig.moves == [torch.device("cpu"), torch.device("meta")]
    assert rig.vae.where == torch.device("meta")
    rig.moves.clear()
    rig.run()
    assert rig.log == [] and rig.moves == []

    # already on the train device: a miss leaves it there
    rig.vae.where = torch.device("cpu")
    rig.config.text_reward_resolution = 240
    rig.run()
    assert rig.log == ["encode", "detect"] and rig.moves == []
