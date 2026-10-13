"""cache_scrub_captions wiring: text caches get the mgds caption-privacy knobs."""

from random import Random

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.ZImageModel import ZImageModel
from modules.util.caption_privacy_util import token_tensor_names
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.enum.ModelType import ModelType

from mgds.pipelineModules.ConceptCacheRouter import ConceptCacheRouter
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

import torch

ZIMAGE_TEXT_NAMES = ["tokens", "tokens_mask", "text_encoder_hidden_state"]


class DataLoaderHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


def _caches(config, text_split_names=ZIMAGE_TEXT_NAMES, text_caching=True, concepts=None):
    loader = DataLoaderHarness()
    loader._concepts_for_cache = concepts or []
    config.latent_caching = True
    modules = loader._cache_modules_from_names(
        model=None,
        model_setup=None,
        image_split_names=["latent_image"],
        image_aggregate_names=["crop_resolution", "image_path"],
        text_split_names=text_split_names,
        sort_names=["latent_image", "crop_resolution", "image_path", *text_split_names, "prompt", "concept"],
        config=config,
        text_caching=text_caching,
    )
    return [m for m in modules if isinstance(m, SmartDiskCache | ConceptCacheRouter)]


def _config(tmp_path, **overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.cache_dir = str(tmp_path / "cache")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_scrub_is_on_by_default_and_migrated_on(tmp_path):
    assert TrainConfig.default_values().cache_scrub_captions is True
    # A config saved before the field existed (version 34) loads with it on.
    legacy = TrainConfig.default_values().to_dict()
    legacy.pop("cache_scrub_captions")
    legacy["__version"] = 34
    assert TrainConfig.default_values().from_dict(legacy).cache_scrub_captions is True
    # An explicit opt-out survives a save/load round trip.
    opted_out = TrainConfig.default_values()
    opted_out.cache_scrub_captions = False
    assert TrainConfig.default_values().from_dict(opted_out.to_dict()).cache_scrub_captions is False


def test_text_cache_scrubs_tokens_and_prompts(tmp_path):
    image_cache, text_cache = _caches(_config(tmp_path))
    assert text_cache.store_sourceless_prompts is False
    assert text_cache.scrub_split_names == ["tokens"]
    # The image cache carries no captions and is left alone.
    assert image_cache.store_sourceless_prompts is True
    assert list(image_cache.scrub_split_names) == []


def test_multi_encoder_token_names():
    names = ["tokens_1", "tokens_mask_1", "text_encoder_1_pooled_state", "tokens_2", "tokens_mask_2", "tokens"]
    assert token_tensor_names(names) == ["tokens_1", "tokens_2", "tokens"]


def test_disabled_leaves_text_cache_unchanged(tmp_path):
    _, text_cache = _caches(_config(tmp_path, cache_scrub_captions=False))
    assert text_cache.store_sourceless_prompts is True
    assert list(text_cache.scrub_split_names) == []


def test_uncached_text_builds_no_text_cache(tmp_path):
    # Text encoder trained / text not cached: nothing to scrub, no text cache.
    caches = _caches(_config(tmp_path), text_caching=False)
    assert [c.cache_dir.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for c in caches] == ["image"]
    assert caches[0].store_sourceless_prompts is True


def test_output_embeddings_keep_token_ids(tmp_path):
    config = _config(tmp_path)
    embedding = TrainEmbeddingConfig.default_values()
    embedding.is_output_embedding = True
    config.additional_embeddings = [embedding]
    _, text_cache = _caches(config)
    assert text_cache.store_sourceless_prompts is False
    assert text_cache.scrub_split_names == []


def test_routed_text_caches_get_the_knobs(tmp_path):
    concept = ConceptConfig.default_values()
    concept.name = "routed"
    concept.path = "/data/routed"
    concept.cache_dir = str(tmp_path / "shared")
    routers = _caches(_config(tmp_path), concepts=[concept])
    assert [type(r) for r in routers] == [ConceptCacheRouter, ConceptCacheRouter]
    assert routers[1].cache_kwargs["store_sourceless_prompts"] is False
    assert routers[1].cache_kwargs["scrub_split_names"] == ["tokens"]
    assert "scrub_split_names" not in routers[0].cache_kwargs


def test_zimage_cached_text_ignores_token_values():
    # Training with cached text passes the cached hidden state; token VALUES
    # must not matter (only the mask), so zeroed cached tokens train the same.
    model = ZImageModel(ModelType.Z_IMAGE)
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    real = torch.randint(1, 1000, (2, 6))

    def encode(tokens):
        return model.encode_text(
            train_device=torch.device("cpu"),
            batch_size=2,
            rand=Random(0),
            tokens=tokens,
            tokens_mask=mask,
            text_encoder_output=hidden,
        )

    with_real = encode(real)
    with_zeros = encode(torch.zeros_like(real))
    assert len(with_real) == len(with_zeros) == 2
    for a, b in zip(with_real, with_zeros, strict=True):
        assert torch.equal(a, b)
