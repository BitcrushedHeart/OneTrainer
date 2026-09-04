"""Prompt-only distill data source: local pipeline modules + mixin stage selection."""

import json

from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.dataLoader.promptOnly.CollectPromptPaths import CollectPromptPaths
from modules.dataLoader.promptOnly.GenerateZeroLatent import GenerateZeroLatent
from modules.dataLoader.promptOnly.SynthesizeResolution import SynthesizeResolution
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

import torch


def _fake_previous(module, concept, extra=None):
    """Wire a pipeline module to a single fake concept row without a pipeline."""
    values = {"concept": concept, "concept.include_subdirectories": concept.get("include_subdirectories", False)}
    values.update(extra or {})
    module._get_previous_length = lambda name: 1
    module._get_previous_item = lambda variation, name, index: values[name]


def _collect(tmp_path):
    module = CollectPromptPaths(
        concept_in_name="concept",
        path_in_name="path",
        include_subdirectories_in_name="concept.include_subdirectories",
        enabled_in_name="enabled",
        path_out_name="image_path",
        concept_out_name="concept",
        exclude_postfix=["-masklabel", "-condlabel"],
    )
    _fake_previous(module, {"path": str(tmp_path), "enabled": True})
    module.start(0)
    return module


def test_collect_prompt_paths_enumerates_txt_and_emits_synthetic_png(tmp_path):
    (tmp_path / "a.txt").write_text("a prompt")
    (tmp_path / "b.txt").write_text("b prompt")
    (tmp_path / "b.png").write_bytes(b"not really a png")  # image present: still enumerated once, via the txt
    (tmp_path / "c.webp").write_bytes(b"image without caption")  # no caption: not a row
    (tmp_path / "a-masklabel.txt").write_text("excluded")

    module = _collect(tmp_path)

    assert module.length() == 2
    paths = sorted(module.get_item(0, i)["image_path"] for i in range(module.length()))
    assert paths == [str(tmp_path / "a.png"), str(tmp_path / "b.png")]


def test_collect_prompt_paths_works_without_any_images(tmp_path):
    for i in range(3):
        (tmp_path / f"p{i}.txt").write_text(f"prompt {i}")
    module = _collect(tmp_path)
    assert module.length() == 3
    assert all(module.get_item(0, i)["image_path"].endswith(".png") for i in range(3))


def _synthesize(tmp_path, image_path, *, override_enabled=False, override="", target="1024"):
    module = SynthesizeResolution(
        path_in_name="image_path",
        resolution_out_name="original_resolution",
        target_resolution_in_name="settings.target_resolution",
        enable_resolution_override_in_name="concept.image.enable_resolution_override",
        resolution_override_in_name="concept.image.resolution_override",
    )
    _fake_previous(
        module,
        {"path": str(tmp_path)},
        extra={
            "image_path": image_path,
            "settings.target_resolution": target,
            "concept.image.enable_resolution_override": override_enabled,
            "concept.image.resolution_override": override,
        },
    )
    return module.get_item(0, 0)["original_resolution"]


def test_synthesize_resolution_prefers_sidecar(tmp_path):
    image_path = str(tmp_path / "x.png")
    (tmp_path / "x.json").write_text(json.dumps({"prompt": "p", "width": 832, "height": 1216}))
    assert _synthesize(tmp_path, image_path) == (1216, 832)


def test_synthesize_resolution_override_then_target(tmp_path):
    image_path = str(tmp_path / "y.png")  # no sidecar
    assert _synthesize(tmp_path, image_path, override_enabled=True, override="768x512") == (512, 768)
    assert _synthesize(tmp_path, image_path, override_enabled=False, override="768x512") == (1024, 1024)
    assert _synthesize(tmp_path, image_path, target="512,768") == (768, 768)


def test_generate_zero_latent_shape_dtype_and_offset():
    module = GenerateZeroLatent(
        crop_resolution_in_name="crop_resolution",
        latent_out_name="latent_image",
        crop_offset_out_name="crop_offset",
        channels=16,
        downscale_factor=8,
        dtype=torch.bfloat16,
    )
    module._get_previous_length = lambda name: 1
    module._get_previous_item = lambda variation, name, index: (1216, 832)
    item = module.get_item(0, 0)
    latent = item["latent_image"]
    assert latent.shape == (16, 152, 104)
    assert latent.dtype == torch.bfloat16
    assert float(latent.abs().sum()) == 0.0
    assert item["crop_offset"] == (0, 0)


class _Loader(DataLoaderText2ImageMixin):
    def _cache_modules(self, *args, **kwargs):
        raise NotImplementedError

    def _debug_modules(self, *args, **kwargs):
        raise NotImplementedError

    def _output_modules(self, *args, **kwargs):
        raise NotImplementedError

    def _preparation_modules(self, *args, **kwargs):
        raise NotImplementedError


def _prompt_only_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.prompt_only_data = True
    return config


def test_mixin_prompt_only_stage_selection():
    loader = _Loader()
    config = _prompt_only_config()

    enumerate_modules = loader._enumerate_input_modules(config)
    assert any(isinstance(m, CollectPromptPaths) for m in enumerate_modules)

    load_modules = loader._load_input_modules(config, DataType.FLOAT_32)
    names = [type(m).__name__ for m in load_modules]
    assert "LoadImage" not in names and "LoadVideo" not in names
    assert "SelectRandomText" in names

    assert loader._mask_augmentation_modules(config) == []
    assert loader._crop_modules(config) == []
    assert loader._inpainting_modules(config) == []

    aug_names = [type(m).__name__ for m in loader._augmentation_modules(config)]
    assert aug_names == ["DropTags", "CapitalizeTags", "ShuffleTags"]

    bucketing = loader._aspect_bucketing_in(config, 64)
    assert any(isinstance(m, SynthesizeResolution) for m in bucketing)
    assert all(type(m).__name__ != "CalcAspect" for m in bucketing)


def test_mixin_prompt_only_skips_image_cache():
    loader = _Loader()
    config = _prompt_only_config()
    config.latent_caching = True
    loader._aspect_bucketing_for_cache = None
    loader.stop_check_fun = lambda: False

    image_split = ["latent_image", "original_resolution", "crop_offset"]
    image_aggregate = ["crop_resolution", "image_path"]
    text_split = ["tokens", "tokens_mask", "text_encoder_hidden_state"]
    sort_names = image_aggregate + image_split + ["prompt", *text_split, "concept"]

    modules = loader._cache_modules_from_names(
        model=None,
        model_setup=None,
        image_split_names=image_split,
        image_aggregate_names=image_aggregate,
        text_split_names=text_split,
        sort_names=sort_names,
        config=config,
        text_caching=True,
    )

    names = [type(m).__name__ for m in modules]
    assert names.count("SmartDiskCache") == 1  # text cache only
    variation_sorting = [m for m in modules if type(m).__name__ == "VariationSorting"]
    assert len(variation_sorting) == 1
    # The live zero-latent names must survive into the sorting stage.
    kept = variation_sorting[0].names
    for name in ("latent_image", "crop_resolution", "crop_offset", "image_path", "original_resolution"):
        assert name in kept, name
    for name in text_split:
        assert name not in kept, name


def test_validate_prompt_only_requires_distill_mode(tmp_path):
    config = _prompt_only_config()
    assert "distill" in config.validate_prompt_only_startup()
    config.distill_enabled = True
    assert config.validate_prompt_only_startup() is None
    config.masked_training = True
    assert "masked_training" in config.validate_prompt_only_startup()
