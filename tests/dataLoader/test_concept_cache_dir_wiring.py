import os

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.concept_cache_util import concept_cache_roots, uses_concept_cache_roots
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig

from mgds.pipelineModules.ConceptCacheRouter import ConceptCacheRouter
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache


class DataLoaderHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


def _concept(name: str, cache_dir: str = "", enabled: bool = True) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.name = name
    concept.path = f"/data/{name}"
    concept.cache_dir = cache_dir
    concept.enabled = enabled
    return concept


def _cache_modules(concepts, cache_dir: str):
    loader = DataLoaderHarness()
    loader._concepts_for_cache = concepts
    config = TrainConfig.default_values()
    config.cache_dir = cache_dir
    config.latent_caching = True
    return loader._cache_modules_from_names(
        model=None,
        model_setup=None,
        image_split_names=["latent_image"],
        image_aggregate_names=["crop_resolution", "image_path"],
        text_split_names=["text_encoder_hidden_state"],
        sort_names=["latent_image", "crop_resolution", "image_path", "text_encoder_hidden_state", "prompt", "concept"],
        config=config,
        text_caching=True,
    )


def test_concept_cache_dir_defaults_to_the_run_cache():
    assert ConceptConfig.default_values().cache_dir == ""
    # Concept files written before the field existed load as "use the run cache".
    assert ConceptConfig.default_values().from_dict({"name": "legacy"}).cache_dir == ""


def test_runs_without_overrides_keep_plain_caches(tmp_path):
    run_cache = str(tmp_path / "run")
    modules = _cache_modules([_concept("a"), _concept("b", cache_dir=run_cache + os.sep)], run_cache)
    caches = [m for m in modules if isinstance(m, SmartDiskCache | ConceptCacheRouter)]

    assert [type(m) for m in caches] == [SmartDiskCache, SmartDiskCache]
    assert [m.cache_dir for m in caches] == [os.path.join(run_cache, "image"), os.path.join(run_cache, "text")]


def test_overrides_route_image_and_text_caches(tmp_path):
    run_cache = str(tmp_path / "run")
    shared = str(tmp_path / "shared")
    concepts = [
        _concept("a"),
        _concept("b", cache_dir=shared),
        _concept("off", cache_dir=str(tmp_path / "x"), enabled=False),
    ]
    modules = _cache_modules(concepts, run_cache)
    routers = [m for m in modules if isinstance(m, ConceptCacheRouter)]

    assert [r.subdir for r in routers] == ["image", "text"]
    for router in routers:
        assert router.default_root == run_cache
        assert router.sourceless_roots == [run_cache, shared]
    # Same kwargs as the plain path, so the routed children behave identically.
    assert routers[0].cache_kwargs["source_path_in_name"] == "image_path"
    assert routers[1].cache_kwargs["content_key_in_name"] == "prompt"


def test_cache_roots_skip_disabled_and_merge_spellings(tmp_path):
    run_cache = str(tmp_path / "run")
    shared = str(tmp_path / "shared")
    concepts = [
        _concept("b", cache_dir=shared),
        _concept("a"),
        _concept("c", cache_dir=shared + os.sep),
        _concept("off", cache_dir=str(tmp_path / "off"), enabled=False),
    ]

    assert concept_cache_roots(concepts, run_cache) == [shared, run_cache]
    assert uses_concept_cache_roots(concepts, run_cache)
    assert not uses_concept_cache_roots([_concept("a"), concepts[3]], run_cache)
    # Every concept overriding: the run cache itself is not used at all.
    assert concept_cache_roots([concepts[0]], run_cache) == [shared]
