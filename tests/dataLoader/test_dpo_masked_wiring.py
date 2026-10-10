"""Masked DPO in the dataloader: the rejected half gets its own mask (falling
back to the chosen one), rides through crop/augmentation with the pair, is
cached and watched, and the per-concept masked-training overrides travel."""

import inspect
from random import Random

from modules.dataLoader.dpo.ConceptLatentMaskRemove import ConceptLatentMaskRemove
from modules.dataLoader.dpo.CoverScaleCropImage import CoverScaleCropImage
from modules.dataLoader.dpo.PairedCircularMaskShrink import PairedCircularMaskShrink
from modules.dataLoader.dpo.ScaleMaskToLatent import ScaleMaskToLatent
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelType import ModelType

from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.SelectFirstInput import SelectFirstInput

import torch


class DataLoaderHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


def _config(*, rlhf: bool = True, masked: bool = True) -> TrainConfig:
    config = TrainConfig.default_values()
    config.rlhf_enabled = rlhf
    config.masked_training = masked
    return config


def _of(modules, cls):
    return [m for m in modules if isinstance(m, cls)]


# --- modules ---------------------------------------------------------------


def _wire(module, values: dict, rand_seed: int | None = None):
    module._get_previous_length = lambda name: 1
    module._get_previous_item = lambda variation, name, index: values.get(name)
    if rand_seed is not None:
        module._get_rand = lambda variation, index=-1: Random(rand_seed)
    return module


def test_scale_mask_to_latent_matches_the_rejected_latent_resolution():
    module = _wire(
        ScaleMaskToLatent("mask_rejected", "latent_image_rejected", "latent_mask_rejected"),
        {"mask_rejected": torch.ones(1, 64, 96), "latent_image_rejected": torch.zeros(4, 8, 12)},
    )
    out = module.get_item(0, 0)["latent_mask_rejected"]
    assert out.shape == (1, 8, 12)
    assert torch.allclose(out, torch.ones(1, 8, 12))


def test_scale_mask_to_latent_passes_an_already_sized_mask_through():
    mask = torch.rand(1, 8, 12)
    module = _wire(
        ScaleMaskToLatent("mask_rejected", "latent_image_rejected", "latent_mask_rejected"),
        {"mask_rejected": mask, "latent_image_rejected": torch.zeros(4, 8, 12)},
    )
    assert module.get_item(0, 0)["latent_mask_rejected"] is mask


def test_concept_mask_remove_unmasks_both_halves_together():
    masks = {"latent_mask": torch.zeros(1, 4, 4), "latent_mask_rejected": torch.zeros(1, 4, 4)}
    names = ["latent_mask", "latent_mask_rejected"]
    probability = "concept.unmasked_probability_override"

    # Random(3).random() is ~0.238: below 0.5 fires, above 0.1 does not.
    fired = _wire(ConceptLatentMaskRemove(names, probability, 0.0), {**masks, probability: 0.5}, rand_seed=3).get_item(
        0, 0
    )
    assert all(torch.equal(fired[name], torch.ones(1, 4, 4)) for name in names)

    kept = _wire(ConceptLatentMaskRemove(names, probability, 0.0), {**masks, probability: 0.1}, rand_seed=3).get_item(
        0, 0
    )
    assert all(kept[name] is masks[name] for name in names)


def test_concept_mask_remove_inherits_the_run_probability():
    masks = {"latent_mask": torch.zeros(1, 4, 4)}
    probability = "concept.unmasked_probability_override"
    # Random(3).random() is ~0.238. Inheriting (negative or absent) uses the
    # run's 0.5 and fires; an explicit 0 never unmasks.
    for inherited in (-1.0, None):
        module = _wire(
            ConceptLatentMaskRemove(["latent_mask"], probability, 0.5), {**masks, probability: inherited}, rand_seed=3
        )
        assert torch.equal(module.get_item(0, 0)["latent_mask"], torch.ones(1, 4, 4))
    module = _wire(ConceptLatentMaskRemove(["latent_mask"], probability, 0.5), {**masks, probability: 0.0}, rand_seed=3)
    assert module.get_item(0, 0)["latent_mask"] is masks["latent_mask"]
    module = _wire(
        ConceptLatentMaskRemove(["latent_mask"], probability, 0.0), {**masks, probability: -1.0}, rand_seed=3
    )
    assert module.get_item(0, 0)["latent_mask"] is masks["latent_mask"]


def test_paired_shrink_draws_the_chosen_modules_random_numbers():
    chosen = RandomCircularMaskShrink(
        mask_name="mask", enabled_in_name="e", shrink_probability=1.0, shrink_factor_min=0.2
    )
    chosen._get_rand = lambda variation, index=-1: Random(1000 + variation * 10 + index)
    paired = PairedCircularMaskShrink(
        paired_with=chosen,
        mask_name="mask_rejected",
        enabled_in_name="e",
        shrink_probability=1.0,
        shrink_factor_min=0.2,
    )
    assert paired._get_rand(2, 5).random() == chosen._get_rand(2, 5).random()

    # Identical masks therefore shrink to the identical disc.
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1.0
    chosen_out = _wire(chosen, {"mask": mask.clone(), "e": True}).get_item(0, 0)["mask"]
    paired_out = _wire(paired, {"mask_rejected": mask.clone(), "e": True}).get_item(0, 0)["mask_rejected"]
    assert torch.equal(chosen_out, paired_out)
    assert chosen_out.sum() < mask.sum()


# --- wiring ----------------------------------------------------------------


def test_rejected_mask_path_is_derived_only_for_masked_dpo_runs():
    loader = DataLoaderHarness()
    paths = [m for m in loader._enumerate_input_modules(_config()) if isinstance(m, ModifyPath)]
    rejected = [m for m in paths if m.out_name == "mask_path_rejected"]
    assert len(rejected) == 1
    assert rejected[0].in_name == "image_path_rejected"

    for config in (_config(masked=False), _config(rlhf=False)):
        assert not any(
            m.out_name == "mask_path_rejected"
            for m in loader._enumerate_input_modules(config)
            if isinstance(m, ModifyPath)
        )


def test_rejected_mask_loads_from_its_own_file_and_falls_back_to_the_chosen_mask():
    modules = DataLoaderHarness()._load_input_modules(_config(), DataType.FLOAT_32)
    load = [m for m in _of(modules, LoadImage) if m.path_in_name == "mask_path_rejected"]
    assert len(load) == 1
    assert load[0].image_out_name == "mask_rejected_file"
    select = _of(modules, SelectFirstInput)
    assert len(select) == 1
    assert select[0].in_names == ["mask_rejected_file", "mask"]
    assert select[0].out_name == "mask_rejected"
    # The fallback reads the loaded chosen mask, so it must come after it.
    chosen_load = next(m for m in _of(modules, LoadImage) if m.path_in_name == "mask_path")
    assert modules.index(chosen_load) < modules.index(load[0]) < modules.index(select[0])

    unmasked = DataLoaderHarness()._load_input_modules(_config(masked=False), DataType.FLOAT_32)
    assert not _of(unmasked, SelectFirstInput)


def test_rejected_mask_is_cover_cropped_with_the_rejected_image():
    crop = _of(DataLoaderHarness()._crop_modules(_config()), CoverScaleCropImage)
    assert crop[0].names == ["image_rejected", "mask_rejected"]
    crop = _of(DataLoaderHarness()._crop_modules(_config(masked=False)), CoverScaleCropImage)
    assert crop[0].names == ["image_rejected"]


def test_rejected_mask_follows_the_pair_through_mask_augmentations():
    loader = DataLoaderHarness()
    modules = loader._mask_augmentation_modules(_config())
    shrink = _of(modules, RandomCircularMaskShrink)
    assert [m.mask_name for m in shrink] == ["mask", "mask_rejected"]
    assert isinstance(shrink[1], PairedCircularMaskShrink)
    assert shrink[1].paired_with is shrink[0]
    rotate = _of(modules, RandomMaskRotateCrop)[0]
    assert rotate.mask_name == "mask"
    assert "mask_rejected" in rotate.additional_names

    assert not _of(loader._mask_augmentation_modules(_config(rlhf=False)), PairedCircularMaskShrink)


def test_rejected_latent_mask_is_sized_from_the_rejected_latent():
    modules = DataLoaderHarness()._dpo_rejected_mask_modules(_config())
    assert len(modules) == 1
    assert isinstance(modules[0], ScaleMaskToLatent)
    assert modules[0].get_inputs() == ["mask_rejected", "latent_image_rejected"]
    assert modules[0].get_outputs() == ["latent_mask_rejected"]
    assert DataLoaderHarness()._dpo_rejected_mask_modules(_config(masked=False)) == []


def _output_modules(config: TrainConfig) -> list:
    return DataLoaderHarness()._output_modules_from_out_names(
        model=None,
        model_setup=None,
        output_names=["latent_image", "latent_mask"] if config.masked_training else ["latent_image"],
        config=config,
        train_dtype=DataType.FLOAT_32,
    )


def test_rejected_latent_mask_and_unmasked_weight_reach_the_batch():
    output = next(m for m in _output_modules(_config()) if isinstance(m, OutputPipelineModule))
    assert "latent_mask_rejected" in output.output_names
    assert ("concept.unmasked_weight_override", "unmasked_weight_override") in list(
        zip(output.input_names, output.output_names, strict=True)
    )

    sft = next(m for m in _output_modules(_config(rlhf=False)) if isinstance(m, OutputPipelineModule))
    assert "latent_mask_rejected" not in sft.output_names
    assert "unmasked_weight_override" in sft.output_names

    unmasked = next(m for m in _output_modules(_config(masked=False)) if isinstance(m, OutputPipelineModule))
    assert "unmasked_weight_override" not in unmasked.output_names


def test_concept_unmasked_probability_applies_to_non_inpainting_masked_runs():
    remove = _of(_output_modules(_config()), ConceptLatentMaskRemove)
    assert len(remove) == 1
    assert remove[0].latent_mask_names == ["latent_mask", "latent_mask_rejected"]
    assert remove[0].probability_in_name == "concept.unmasked_probability_override"
    assert remove[0].default_probability == _config().unmasked_probability

    assert _of(_output_modules(_config(rlhf=False)), ConceptLatentMaskRemove)[0].latent_mask_names == ["latent_mask"]
    assert not _of(_output_modules(_config(masked=False)), ConceptLatentMaskRemove)

    inpainting = _config(rlhf=False)
    inpainting.model_type = ModelType.STABLE_DIFFUSION_15_INPAINTING
    modules = _output_modules(inpainting)
    assert not _of(modules, ConceptLatentMaskRemove)
    assert _of(modules, RandomLatentMaskRemove)


def test_rejected_mask_is_cached_as_a_split_tensor_and_watched():
    # _cache_modules_from_names builds SmartDiskCache in place (it needs a
    # model and a cache dir), so check the wiring at source level.
    source = inspect.getsource(DataLoaderText2ImageMixin._cache_modules_from_names)
    assert 'image_split_names + ["latent_mask_rejected"]' in source
    assert 'image_extra_watched.append("mask_path_rejected")' in source


def test_concept_overrides_default_to_inheriting():
    concept = ConceptConfig.default_values()
    assert concept.unmasked_weight_override == -1.0
    assert concept.unmasked_probability_override == -1.0
    legacy = ConceptConfig.default_values().from_dict({"name": "legacy", "dpo_beta_override": 0.0})
    assert legacy.unmasked_weight_override == -1.0
    assert legacy.unmasked_probability_override == -1.0
    assert ConceptConfig.default_values().to_dict()["unmasked_weight_override"] == -1.0
