"""Per-epoch text reward bank with the per-process prompt-embedding cache.

A fake model stands in for Z-Image: its ``encode_text`` derives each
embedding from the raw prompt (so a mix-up would show), templates the list in
place like the real one, and records every call; ``text_encoder_to`` records
every move. The pre-change builder is reproduced below as the reference for
order and content."""

import json
import os
import random
from types import SimpleNamespace

from modules.modelSetup.mixin.ModelSetupTextRewardMixin import TEXT_REWARD_ENCODE_CHUNK, ModelSetupTextRewardMixin
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TextRewardSource import TextRewardSource
from modules.util.text_reward_util import (
    DEFAULT_VOCAB,
    TextRewardBank,
    generate_text_specs,
    load_text_reward_dataset,
    load_word_list,
)

import torch

import pytest

TRAIN = torch.device("meta")  # distinct from the temp device, never really used
TEMP = torch.device("cpu")
NEGATIVE = "blurry, low quality"


def _embedding_for(prompt: str) -> torch.Tensor:
    rows = [[float(ord(ch)), float(len(prompt)), float(i)] for i, ch in enumerate(prompt)]
    return torch.tensor(rows or [[0.0, 0.0, 0.0]], dtype=torch.bfloat16)


class _FakeModel:
    def __init__(self):
        self.text_encoder = torch.nn.Linear(2, 2)
        self.encode_calls: list[dict] = []
        self.moves: list[torch.device] = []

    def text_encoder_to(self, device):
        self.moves.append(device)

    def encode_text(self, train_device, batch_size=1, text=None, pad_to_longest=False, **kwargs):
        raw = list(text)
        self.encode_calls.append({"prompts": raw, "batch_size": batch_size, "pad_to_longest": pad_to_longest})
        for i, prompt in enumerate(text):
            text[i] = f"<tmpl>{prompt}</tmpl>"  # the real encode_text templates in place
        return [_embedding_for(p) for p in raw]

    @property
    def encoded_prompts(self) -> list[str]:
        return [p for call in self.encode_calls for p in call["prompts"]]

    def reset(self):
        self.encode_calls.clear()
        self.moves.clear()


class _FakeSetup(ModelSetupTextRewardMixin):
    def __init__(self):
        super().__init__()
        self.train_device = TRAIN
        self.temp_device = TEMP

    def text_reward_model(self):
        return SimpleNamespace(vocab=DEFAULT_VOCAB)


def _legacy_build(setup, model, config, epoch) -> TextRewardBank:
    """The builder before the prompt cache (commit 8a10a403), verbatim apart
    from the reward-model lookup."""
    vocab = setup.text_reward_model().vocab
    image_paths = None
    image_sha256s = None
    if config.text_reward_source == TextRewardSource.OFFLINE_IMAGES:
        rows = load_text_reward_dataset(config.text_reward_dataset_dir, vocab)
        rng = random.Random(0x7E57 + epoch)
        rng.shuffle(rows)
        specs = [r[0] for r in rows]
        prompts = [r[1] for r in rows]
        image_paths = [r[2] for r in rows]
        image_sha256s = [getattr(r, "sha256", None) for r in rows]
    else:
        specs = generate_text_specs(
            max(1, config.text_reward_samples_per_epoch),
            seed=0x7E57 + epoch,
            vocab=vocab,
            words=load_word_list(config.text_reward_word_list),
            min_words=config.text_reward_min_words,
            max_words=config.text_reward_max_words,
        )
        prompts = [s.prompt for s in specs]
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(specs), 16):
            chunk_prompts = list(prompts[start : start + 16])
            encoded = model.encode_text(train_device=TRAIN, batch_size=len(chunk_prompts), text=chunk_prompts)
            embeddings.extend(e.detach().to("cpu") for e in encoded)
        negative = None
        if config.text_reward_cfg_scale > 1.0:
            negative = model.encode_text(
                train_device=TRAIN, batch_size=1, text=[config.text_reward_negative_prompt or ""]
            )[0]
    return TextRewardBank(specs, embeddings, negative, image_paths, image_sha256s)


def _write_sample(folder, stem: str, prompt: str, sha256: str | None = None):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, stem + ".png"), "wb") as f:
        f.write(b"")  # the loader never opens the image
    with open(os.path.join(folder, stem + ".txt"), "w", encoding="utf-8") as f:
        f.write(prompt + "\n")
    if sha256 is not None:
        with open(os.path.join(folder, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "sha256": sha256}, f)


def _offline_dataset(root, count: int = 21, first: int = 0):
    """``count`` samples split over two folders (the loader walks them), one
    prompt repeated on two images, every third sample with a JSON hash."""
    for i in range(first, first + count):
        folder = os.path.join(root, "a" if i % 2 == 0 else os.path.join("b", "nested"))
        prompt = f'Shop sign text reading "Word{i}"' if i != first + 5 else f'Shop sign text reading "Word{first + 4}"'
        _write_sample(folder, f"img_{i:03d}", prompt, sha256=f"{i:064x}" if i % 3 == 0 else None)


def _config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.latent_caching = True
    config.text_reward_cfg_scale = 1.0
    config.text_reward_negative_prompt = NEGATIVE
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _offline_config(root, **overrides) -> TrainConfig:
    return _config(text_reward_source=TextRewardSource.OFFLINE_IMAGES, text_reward_dataset_dir=str(root), **overrides)


def _online_config(**overrides) -> TrainConfig:
    return _config(text_reward_source=TextRewardSource.ONLINE_ROLLOUT, text_reward_samples_per_epoch=37, **overrides)


def _assert_same_bank(bank: TextRewardBank, reference: TextRewardBank, *, negative: bool = True):
    assert bank.specs == reference.specs
    assert bank.image_paths == reference.image_paths
    assert bank.image_sha256s == reference.image_sha256s
    assert len(bank.embeddings) == len(reference.embeddings)
    for got, want in zip(bank.embeddings, reference.embeddings, strict=True):
        assert got.device.type == "cpu"
        assert got.dtype == want.dtype
        assert torch.equal(got, want)
    if negative:
        if reference.negative_embedding is None:
            assert bank.negative_embedding is None
        else:
            assert torch.equal(bank.negative_embedding, reference.negative_embedding)


def _unique(prompts):
    return list(dict.fromkeys(prompts))


@pytest.mark.parametrize("epoch", [0, 1, 7])
def test_offline_bank_matches_pre_change_builder(tmp_path, epoch):
    _offline_dataset(tmp_path)
    config = _offline_config(tmp_path)
    reference = _legacy_build(_FakeSetup(), _FakeModel(), config, epoch)
    bank = _FakeSetup().build_text_reward_bank(_FakeModel(), config, epoch)

    rows = load_text_reward_dataset(str(tmp_path), DEFAULT_VOCAB)
    random.Random(0x7E57 + epoch).shuffle(rows)
    assert bank.image_paths == [r[2] for r in rows]
    assert bank.image_sha256s == [r.sha256 for r in rows]
    assert any(h is not None for h in bank.image_sha256s) and any(h is None for h in bank.image_sha256s)
    _assert_same_bank(bank, reference)


@pytest.mark.parametrize("epoch", [0, 3])
@pytest.mark.parametrize("cfg", [1.0, 4.5])
def test_online_bank_matches_pre_change_builder(epoch, cfg):
    config = _online_config(text_reward_cfg_scale=cfg)
    reference = _legacy_build(_FakeSetup(), _FakeModel(), config, epoch)
    bank = _FakeSetup().build_text_reward_bank(_FakeModel(), config, epoch)
    expected = generate_text_specs(
        37,
        seed=0x7E57 + epoch,
        vocab=DEFAULT_VOCAB,
        words=load_word_list(config.text_reward_word_list),
        min_words=config.text_reward_min_words,
        max_words=config.text_reward_max_words,
    )
    assert bank.specs == expected
    _assert_same_bank(bank, reference)
    assert (bank.negative_embedding is not None) == (cfg > 1.0)


def test_offline_second_epoch_hits_the_cache_without_moving_the_text_encoder(tmp_path):
    _offline_dataset(tmp_path)
    config = _offline_config(tmp_path)
    setup, model = _FakeSetup(), _FakeModel()

    first = setup.build_text_reward_bank(model, config, 1)
    prompts_1 = [r[1] for r in load_text_reward_dataset(str(tmp_path), DEFAULT_VOCAB)]
    assert sorted(model.encoded_prompts) == sorted(_unique(prompts_1))  # duplicate prompt encoded once
    assert len(model.encoded_prompts) == len(prompts_1) - 1
    assert all(call["pad_to_longest"] is True for call in model.encode_calls)
    assert all(
        1 <= call["batch_size"] == len(call["prompts"]) <= TEXT_REWARD_ENCODE_CHUNK for call in model.encode_calls
    )
    assert len(model.encode_calls) == 2
    assert model.moves == [TRAIN, TEMP]

    model.reset()
    second = setup.build_text_reward_bank(model, config, 2)
    assert model.encode_calls == []
    assert model.moves == []
    assert second.image_paths != first.image_paths  # a new shuffle ...
    assert sorted(second.image_paths) == sorted(first.image_paths)  # ... of the same rows
    _assert_same_bank(second, _legacy_build(_FakeSetup(), _FakeModel(), config, 2))


def test_changed_prompt_set_encodes_only_misses_and_evicts_stale_entries(tmp_path):
    _offline_dataset(tmp_path, count=21)
    config = _offline_config(tmp_path)
    setup, model = _FakeSetup(), _FakeModel()
    setup.build_text_reward_bank(model, config, 0)
    old_prompts = {r[1] for r in load_text_reward_dataset(str(tmp_path), DEFAULT_VOCAB)}

    # drop four samples, add three new ones
    for i in (0, 1, 2, 3):
        folder = os.path.join(tmp_path, "a" if i % 2 == 0 else os.path.join("b", "nested"))
        for ext in (".png", ".txt", ".json"):
            path = os.path.join(folder, f"img_{i:03d}{ext}")
            if os.path.exists(path):
                os.remove(path)
    for i in range(100, 103):
        _write_sample(os.path.join(tmp_path, "c"), f"img_{i:03d}", f'Poster text reading "Fresh{i}"')
    new_prompts = {r[1] for r in load_text_reward_dataset(str(tmp_path), DEFAULT_VOCAB)}

    model.reset()
    bank = setup.build_text_reward_bank(model, config, 1)
    assert sorted(model.encoded_prompts) == sorted(new_prompts - old_prompts)
    assert len(new_prompts - old_prompts) == 3
    assert model.moves == [TRAIN, TEMP]
    # the cache holds exactly this epoch's raw prompts (not the templated text)
    assert set(setup._text_reward_prompt_embeddings) == new_prompts
    assert not any(key.startswith("<tmpl>") for key in setup._text_reward_prompt_embeddings)
    _assert_same_bank(bank, _legacy_build(_FakeSetup(), _FakeModel(), config, 1))


def test_plain_offline_never_encodes_the_negative_prompt(tmp_path):
    _offline_dataset(tmp_path)
    config = _offline_config(tmp_path, text_reward_cfg_scale=5.0, text_reward_init_sigma=0.0)
    setup, model = _FakeSetup(), _FakeModel()
    bank = setup.build_text_reward_bank(model, config, 0)
    assert NEGATIVE not in model.encoded_prompts
    assert bank.negative_embedding is None
    assert NEGATIVE not in setup._text_reward_prompt_embeddings
    _assert_same_bank(bank, _legacy_build(_FakeSetup(), _FakeModel(), config, 0), negative=False)


def test_hybrid_offline_encodes_the_negative_prompt_for_its_prefix(tmp_path):
    _offline_dataset(tmp_path)
    config = _offline_config(tmp_path, text_reward_cfg_scale=5.0, text_reward_init_sigma=0.8)
    setup, model = _FakeSetup(), _FakeModel()
    bank = setup.build_text_reward_bank(model, config, 0)
    assert model.encoded_prompts.count(NEGATIVE) == 1
    assert torch.equal(bank.negative_embedding, _embedding_for(NEGATIVE))
    assert bank.image_paths is not None and len(bank) == 21


def test_online_encodes_and_caches_the_negative_prompt():
    config = _online_config(text_reward_cfg_scale=3.0)
    setup, model = _FakeSetup(), _FakeModel()
    bank = setup.build_text_reward_bank(model, config, 0)
    assert model.encoded_prompts.count(NEGATIVE) == 1
    assert torch.equal(bank.negative_embedding, _embedding_for(NEGATIVE))

    # a rebuild of the same epoch is all hits, the negative included
    model.reset()
    again = setup.build_text_reward_bank(model, config, 0)
    assert model.encode_calls == [] and model.moves == []
    _assert_same_bank(again, bank)

    # the next epoch draws new prompts; the negative survives eviction
    model.reset()
    nxt = setup.build_text_reward_bank(model, config, 1)
    assert NEGATIVE not in model.encoded_prompts
    assert model.moves == [TRAIN, TEMP]
    assert set(setup._text_reward_prompt_embeddings) == {s.prompt for s in nxt.specs} | {NEGATIVE}
    _assert_same_bank(nxt, _legacy_build(_FakeSetup(), _FakeModel(), config, 1))


def test_empty_negative_prompt_is_encoded_as_empty_string():
    config = _online_config(text_reward_cfg_scale=2.0, text_reward_negative_prompt="")
    setup, model = _FakeSetup(), _FakeModel()
    bank = setup.build_text_reward_bank(model, config, 0)
    assert "" in model.encoded_prompts
    assert torch.equal(bank.negative_embedding, _embedding_for(""))


def test_text_encoder_identity_change_invalidates_the_cache(tmp_path):
    _offline_dataset(tmp_path)
    config = _offline_config(tmp_path)
    setup, model = _FakeSetup(), _FakeModel()
    setup.build_text_reward_bank(model, config, 0)
    unique_count = len(model.encoded_prompts)

    model.reset()
    model.text_encoder = torch.nn.Linear(2, 2)  # a different encoder object
    setup.build_text_reward_bank(model, config, 1)
    assert len(model.encoded_prompts) == unique_count

    model.reset()
    model.text_encoder = model.text_encoder.to(torch.float16)  # same object, new runtime dtype
    setup.build_text_reward_bank(model, config, 2)
    assert len(model.encoded_prompts) == unique_count

    model.reset()
    config.base_model_name = "some/other-model"
    setup.build_text_reward_bank(model, config, 3)
    assert len(model.encoded_prompts) == unique_count

    model.reset()
    setup.build_text_reward_bank(model, config, 4)
    assert model.encode_calls == []


def test_text_encoder_stays_on_train_device_without_latent_caching(tmp_path):
    _offline_dataset(tmp_path)
    config = _offline_config(tmp_path, latent_caching=False)
    setup, model = _FakeSetup(), _FakeModel()
    setup.build_text_reward_bank(model, config, 0)
    assert model.moves == [TRAIN]


def test_failed_encode_still_restores_the_text_encoder(tmp_path):
    _offline_dataset(tmp_path)
    config = _offline_config(tmp_path)
    setup, model = _FakeSetup(), _FakeModel()

    def boom(**kwargs):
        raise RuntimeError("encoder exploded")

    model.encode_text = boom
    with pytest.raises(RuntimeError, match="exploded"):
        setup.build_text_reward_bank(model, config, 0)
    assert model.moves == [TRAIN, TEMP]
