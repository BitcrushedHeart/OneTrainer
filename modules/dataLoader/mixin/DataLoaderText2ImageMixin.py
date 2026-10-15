import os
import re
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

import modules.util.multi_gpu_util as multi
from modules.dataLoader.dpo.ConceptLatentMaskRemove import ConceptLatentMaskRemove
from modules.dataLoader.dpo.CoverScaleCropImage import CoverScaleCropImage
from modules.dataLoader.dpo.DeriveDPORejectedPath import DeriveDPORejectedPath
from modules.dataLoader.dpo.FilterDPOChosenPaths import FilterDPOChosenPaths
from modules.dataLoader.dpo.PairedCircularMaskShrink import PairedCircularMaskShrink
from modules.dataLoader.dpo.ScaleMaskToLatent import ScaleMaskToLatent
from modules.dataLoader.promptOnly.CollectPromptPaths import CollectPromptPaths
from modules.dataLoader.promptOnly.SynthesizeResolution import SynthesizeResolution
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util import path_util
from modules.util.caption_privacy_util import text_cache_scrub_kwargs
from modules.util.concept_cache_util import concept_cache_roots, uses_concept_cache_roots
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.sourceless_cache_util import sourceless_cache_problems
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

from mgds.OutputPipelineModule import OutputPipelineModule
from mgds.pipelineModules.AspectBatchSorting import AspectBatchSorting
from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect
from mgds.pipelineModules.CapitalizeTags import CapitalizeTags
from mgds.pipelineModules.CollectPaths import CollectPaths
from mgds.pipelineModules.ConceptCacheRouter import ConceptCacheRouter
from mgds.pipelineModules.DistributedSampler import DistributedSampler
from mgds.pipelineModules.DownloadHuggingfaceDatasets import DownloadHuggingfaceDatasets
from mgds.pipelineModules.DropTags import DropTags
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.GenerateImageLike import GenerateImageLike
from mgds.pipelineModules.GenerateMaskedConditioningImage import GenerateMaskedConditioningImage
from mgds.pipelineModules.GetFilename import GetFilename
from mgds.pipelineModules.ImageToVideo import ImageToVideo
from mgds.pipelineModules.InlineAspectBatchSorting import InlineAspectBatchSorting
from mgds.pipelineModules.InlineDistributedSampler import InlineDistributedSampler
from mgds.pipelineModules.LoadImage import LoadImage
from mgds.pipelineModules.LoadMultipleTexts import LoadMultipleTexts
from mgds.pipelineModules.LoadVideo import LoadVideo
from mgds.pipelineModules.ModifyPath import ModifyPath
from mgds.pipelineModules.RandomBrightness import RandomBrightness
from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink
from mgds.pipelineModules.RandomContrast import RandomContrast
from mgds.pipelineModules.RandomFlip import RandomFlip
from mgds.pipelineModules.RandomHue import RandomHue
from mgds.pipelineModules.RandomLatentMaskRemove import RandomLatentMaskRemove
from mgds.pipelineModules.RandomMaskRotateCrop import RandomMaskRotateCrop
from mgds.pipelineModules.RandomRotate import RandomRotate
from mgds.pipelineModules.RandomSaturation import RandomSaturation
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.ResolutionPolicy import target_resolutions_for_spec
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.ScaleCropImage import ScaleCropImage
from mgds.pipelineModules.SelectFirstInput import SelectFirstInput
from mgds.pipelineModules.SelectInput import SelectInput
from mgds.pipelineModules.SelectRandomText import SelectRandomText
from mgds.pipelineModules.ShuffleTags import ShuffleTags
from mgds.pipelineModules.SingleAspectCalculation import SingleAspectCalculation
from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
from mgds.pipelineModules.VariationSorting import VariationSorting

import torch

from diffusers import AutoencoderKL


def text_encode_batch_size() -> int:
    """Encode batch size and build worker count for text caches.

    Batching several captions per forward amortizes the per-forward fixed
    cost (weight streaming under layer offload, dequant, kernel launches)
    that dominates a bs=1 encode through a multi-billion-parameter text
    encoder. Set OT_TEXT_CACHE_BATCH=1 to restore strictly serial bs=1
    encoding.
    """
    try:
        return max(1, int(os.environ.get("OT_TEXT_CACHE_BATCH", "8")))
    except ValueError:
        return 8


class DataLoaderText2ImageMixin(metaclass=ABCMeta):
    @staticmethod
    def __as_output_mapping(name: str | tuple[str, str]) -> tuple[str, str]:
        return name if isinstance(name, tuple) else (name, name)

    def _enumerate_input_modules(self, config: TrainConfig, allow_videos: bool = False) -> list:
        supported_extensions = set()
        supported_extensions |= path_util.supported_image_extensions()

        if allow_videos:
            supported_extensions |= path_util.supported_video_extensions()

        download_datasets = DownloadHuggingfaceDatasets(
            concept_in_name="concept",
            path_in_name="path",
            enabled_in_name="enabled",
            concept_out_name="concept",
        )

        collect_paths = CollectPaths(
            concept_in_name="concept",
            path_in_name="path",
            include_subdirectories_in_name="concept.include_subdirectories",
            enabled_in_name="enabled",
            path_out_name="image_path",
            concept_out_name="concept",
            extensions=supported_extensions,
            include_postfix=None,
            exclude_postfix=["-masklabel", "-condlabel"],
        )

        mask_path = ModifyPath(in_name="image_path", out_name="mask_path", postfix="-masklabel", extension=".png")
        cond_path = ModifyPath(in_name="image_path", out_name="cond_path", postfix="-condlabel", extension=".png")
        sample_prompt_path = ModifyPath(
            in_name="image_path", out_name="sample_prompt_path", postfix="", extension=".txt"
        )

        if config.prompt_only_data:
            # Enumerate captions instead of images; image_path becomes the
            # synthetic (never opened) .png sibling. Mask/cond/DPO paths are
            # excluded by startup validation.
            collect_prompt_paths = CollectPromptPaths(
                concept_in_name="concept",
                path_in_name="path",
                include_subdirectories_in_name="concept.include_subdirectories",
                enabled_in_name="enabled",
                path_out_name="image_path",
                concept_out_name="concept",
                exclude_postfix=["-masklabel", "-condlabel"],
            )
            return [download_datasets, collect_prompt_paths, sample_prompt_path]

        modules = [download_datasets, collect_paths, FilterDPOChosenPaths(), sample_prompt_path]

        if config.rlhf_enabled:
            modules.append(DeriveDPORejectedPath())
        if config.masked_training:
            modules.append(mask_path)
            if config.rlhf_enabled:
                modules.append(
                    ModifyPath(
                        in_name="image_path_rejected",
                        out_name="mask_path_rejected",
                        postfix="-masklabel",
                        extension=".png",
                    )
                )
        if config.custom_conditioning_image:
            modules.append(cond_path)

        return modules

    def _load_input_modules(
        self,
        config: TrainConfig,
        train_dtype: DataType,
        vae_frame_dim: bool = False,
    ) -> list:
        load_image = LoadImage(
            path_in_name="image_path",
            image_out_name="image",
            range_min=0,
            range_max=1,
            supported_extensions=path_util.supported_image_extensions(),
            dtype=train_dtype.torch_dtype(),
        )
        load_video = LoadVideo(
            path_in_name="image_path",
            target_frame_count_in_name="settings.target_frames",
            video_out_name="image",
            range_min=0,
            range_max=1,
            target_frame_rate=24,
            supported_extensions=path_util.supported_video_extensions(),
            dtype=train_dtype.torch_dtype(),
        )
        image_to_video = ImageToVideo(in_name="image", out_name="image")

        generate_mask = GenerateImageLike(
            image_in_name="image", image_out_name="mask", color=255, range_min=0, range_max=1
        )
        load_mask = LoadImage(
            path_in_name="mask_path",
            image_out_name="mask",
            range_min=0,
            range_max=1,
            channels=1,
            supported_extensions={".png"},
            dtype=train_dtype.torch_dtype(),
        )
        mask_to_video = ImageToVideo(in_name="mask", out_name="mask")

        load_cond_image = LoadImage(
            path_in_name="cond_path",
            image_out_name="custom_conditioning_image",
            range_min=0,
            range_max=1,
            supported_extensions=path_util.supported_image_extensions(),
            dtype=train_dtype.torch_dtype(),
        )

        load_sample_prompts = LoadMultipleTexts(path_in_name="sample_prompt_path", texts_out_name="sample_prompts")
        load_concept_prompts = LoadMultipleTexts(
            path_in_name="concept.text.prompt_path", texts_out_name="concept_prompts"
        )
        filename_prompt = GetFilename(
            path_in_name="image_path", filename_out_name="filename_prompt", include_extension=False
        )
        select_prompt_input = SelectInput(
            setting_name="concept.text.prompt_source",
            out_name="prompts",
            setting_to_in_name_map={
                "sample": "sample_prompts",
                "concept": "concept_prompts",
                "filename": "filename_prompt",
            },
            default_in_name="sample_prompts",
        )
        select_random_text = SelectRandomText(texts_in_name="prompts", text_out_name="prompt")

        if config.prompt_only_data:
            return [load_sample_prompts, load_concept_prompts, filename_prompt, select_prompt_input, select_random_text]

        modules = [load_image, load_video]

        if config.rlhf_enabled:
            modules.append(
                LoadImage(
                    path_in_name="image_path_rejected",
                    image_out_name="image_rejected",
                    range_min=0,
                    range_max=1,
                    supported_extensions=path_util.supported_image_extensions(),
                    dtype=train_dtype.torch_dtype(),
                )
            )

        if vae_frame_dim:
            modules.append(image_to_video)
            if config.rlhf_enabled:
                modules.append(ImageToVideo(in_name="image_rejected", out_name="image_rejected"))

        modules.extend(
            [load_sample_prompts, load_concept_prompts, filename_prompt, select_prompt_input, select_random_text]
        )

        if config.masked_training:
            modules.append(generate_mask)
            modules.append(load_mask)
            if config.rlhf_enabled:
                # The rejected image's own -masklabel.png when it has one.
                # LoadImage yields None for a missing file, so an aligned pair
                # (an in-place fix of the chosen image, the common case) needs
                # no second mask and shares the chosen one.
                modules.append(
                    LoadImage(
                        path_in_name="mask_path_rejected",
                        image_out_name="mask_rejected_file",
                        range_min=0,
                        range_max=1,
                        channels=1,
                        supported_extensions={".png"},
                        dtype=train_dtype.torch_dtype(),
                    )
                )
                modules.append(SelectFirstInput(in_names=["mask_rejected_file", "mask"], out_name="mask_rejected"))
        elif config.model_type.has_mask_input():
            modules.append(generate_mask)

        if config.custom_conditioning_image:
            modules.append(load_cond_image)

        if vae_frame_dim:
            modules.append(mask_to_video)
            if config.masked_training and config.rlhf_enabled:
                modules.append(ImageToVideo(in_name="mask_rejected", out_name="mask_rejected"))

        return modules

    def _mask_augmentation_modules(self, config: TrainConfig) -> list:
        if config.prompt_only_data:
            return []

        inputs = ["image"]

        if config.rlhf_enabled:
            inputs.append("image_rejected")
            if config.masked_training:
                # Rotate/crop is driven by the chosen mask; the rejected mask
                # takes the same transform so the pair keeps one framing.
                inputs.append("mask_rejected")

        lowest_resolution = min([int(x.strip()) for x in re.split(r"\D", config.resolution) if x.strip() != ""])
        circular_mask_shrink = RandomCircularMaskShrink(
            mask_name="mask",
            shrink_probability=1.0,
            shrink_factor_min=0.2,
            shrink_factor_max=1.0,
            enabled_in_name="concept.image.enable_random_circular_mask_shrink",
        )
        random_mask_rotate_crop = RandomMaskRotateCrop(
            mask_name="mask",
            additional_names=inputs,
            min_size=lowest_resolution,
            min_padding_percent=10,
            max_padding_percent=30,
            max_rotate_angle=20,
            enabled_in_name="concept.image.enable_random_mask_rotate_crop",
        )

        rejected_mask_shrink = PairedCircularMaskShrink(
            paired_with=circular_mask_shrink,
            mask_name="mask_rejected",
            shrink_probability=1.0,
            shrink_factor_min=0.2,
            shrink_factor_max=1.0,
            enabled_in_name="concept.image.enable_random_circular_mask_shrink",
        )

        modules = []

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(circular_mask_shrink)
            if config.masked_training and config.rlhf_enabled:
                modules.append(rejected_mask_shrink)

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(random_mask_rotate_crop)

        return modules

    def _aspect_bucketing_in(
        self, config: TrainConfig, aspect_bucketing_quantization: int, frame_dim_enabled: bool = False
    ):
        if config.prompt_only_data:
            calc_aspect = SynthesizeResolution(
                path_in_name="image_path",
                resolution_out_name="original_resolution",
                target_resolution_in_name="settings.target_resolution",
                enable_resolution_override_in_name="concept.image.enable_resolution_override",
                resolution_override_in_name="concept.image.resolution_override",
            )
        else:
            calc_aspect = CalcAspect(image_in_name="image", resolution_out_name="original_resolution")
        additional_resolution_names = []
        if config.rlhf_enabled and config.adaptive_resolution:
            additional_resolution_names.append("original_resolution_rejected")
            calc_rejected_aspect = CalcAspect(
                image_in_name="image_rejected",
                resolution_out_name="original_resolution_rejected",
            )

        # DPO pairs no longer need a bucketing subclass: the rejected image is
        # a second tensor inside the chosen sample's row, so it inherits the
        # chosen image's bucket and crop target by construction.
        aspect_bucketing_quantization = AspectBucketing(
            quantization=aspect_bucketing_quantization,
            resolution_in_name="original_resolution",
            target_resolution_in_name="settings.target_resolution",
            enable_target_resolutions_override_in_name="concept.image.enable_resolution_override",
            target_resolutions_override_in_name="concept.image.resolution_override",
            target_frames_in_name="settings.target_frames",
            frame_dim_enabled=frame_dim_enabled,
            scale_resolution_out_name="scale_resolution",
            crop_resolution_out_name="crop_resolution",
            possible_resolutions_out_name="possible_resolutions",
            adaptive_resolution=config.adaptive_resolution,
            additional_resolution_in_names=additional_resolution_names,
            adaptive_resolution_ladder=config.adaptive_resolution_ladder,
        )

        single_aspect_calculation = SingleAspectCalculation(
            resolution_in_name="original_resolution",
            target_resolution_in_name="settings.target_resolution",
            enable_target_resolutions_override_in_name="concept.image.enable_resolution_override",
            target_resolutions_override_in_name="concept.image.resolution_override",
            scale_resolution_out_name="scale_resolution",
            crop_resolution_out_name="crop_resolution",
            possible_resolutions_out_name="possible_resolutions",
            adaptive_resolution=config.adaptive_resolution,
            additional_resolution_in_names=additional_resolution_names,
            adaptive_resolution_ladder=config.adaptive_resolution_ladder,
        )

        modules = [calc_aspect]
        if config.rlhf_enabled and config.adaptive_resolution:
            modules.append(calc_rejected_aspect)

        if config.aspect_ratio_bucketing:
            modules.append(aspect_bucketing_quantization)
            # Stash the instance so _cache_modules_from_names can wire it
            # into SmartDiskCache for bucket-method drift detection and
            # aspect-derived re-bucketing without an image decode.
            self._aspect_bucketing_for_cache = aspect_bucketing_quantization
        else:
            modules.append(single_aspect_calculation)
            self._aspect_bucketing_for_cache = single_aspect_calculation

        return modules

    def _crop_modules(self, config: TrainConfig):
        if config.prompt_only_data:
            # Nothing to crop; crop_offset is emitted by GenerateZeroLatent.
            return []

        inputs = ["image"]

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append("mask")

        if config.model_type.has_depth_input():
            inputs.append("depth")

        if config.custom_conditioning_image:
            inputs.append("custom_conditioning_image")

        scale_crop = ScaleCropImage(
            names=inputs,
            scale_resolution_in_name="scale_resolution",
            crop_resolution_in_name="crop_resolution",
            enable_crop_jitter_in_name="concept.image.enable_crop_jitter",
            crop_offset_out_name="crop_offset",
        )

        modules = [scale_crop]

        if config.rlhf_enabled:
            # The rejected image is NOT run through ScaleCropImage: that module
            # resizes every name to the chosen image's scale_resolution, which
            # stretches a rejected image whose aspect differs from the chosen
            # image's bucket — and a stretched generation is trivially
            # discriminable, so the pair teaches "stretched vs not". Instead it
            # is cover-scaled (aspect kept, one factor, covers the crop box)
            # and cropped with the chosen image's *relative* crop offset, so a
            # jittered pair keeps the same framing. Same-bucket pairs come out
            # identical to the old path.
            modules.append(
                CoverScaleCropImage(
                    names=["image_rejected"] + (["mask_rejected"] if config.masked_training else []),
                    scale_resolution_in_name="scale_resolution",
                    crop_resolution_in_name="crop_resolution",
                    crop_offset_in_name="crop_offset",
                )
            )

        return modules

    def _augmentation_modules(self, config: TrainConfig):
        inputs = ["image"]
        image_inputs = ["image"]

        if config.rlhf_enabled:
            # Each augmentation draws once per sample and applies the identical
            # transform to every listed name, keeping DPO pairs comparable.
            inputs.append("image_rejected")
            image_inputs.append("image_rejected")

        if config.masked_training or config.model_type.has_mask_input():
            inputs.append("mask")
            if config.masked_training and config.rlhf_enabled:
                inputs.append("mask_rejected")

        if config.model_type.has_depth_input():
            inputs.append("depth")

        if config.custom_conditioning_image:
            inputs.append("custom_conditioning_image")
            image_inputs.append("custom_conditioning_image")

        # image augmentations
        random_flip = RandomFlip(
            names=inputs,
            enabled_in_name="concept.image.enable_random_flip",
            fixed_enabled_in_name="concept.image.enable_fixed_flip",
        )
        random_rotate = RandomRotate(
            names=inputs,
            enabled_in_name="concept.image.enable_random_rotate",
            fixed_enabled_in_name="concept.image.enable_fixed_rotate",
            max_angle_in_name="concept.image.random_rotate_max_angle",
        )
        random_brightness = RandomBrightness(
            names=image_inputs,
            enabled_in_name="concept.image.enable_random_brightness",
            fixed_enabled_in_name="concept.image.enable_fixed_brightness",
            max_strength_in_name="concept.image.random_brightness_max_strength",
        )
        random_contrast = RandomContrast(
            names=image_inputs,
            enabled_in_name="concept.image.enable_random_contrast",
            fixed_enabled_in_name="concept.image.enable_fixed_contrast",
            max_strength_in_name="concept.image.random_contrast_max_strength",
        )
        random_saturation = RandomSaturation(
            names=image_inputs,
            enabled_in_name="concept.image.enable_random_saturation",
            fixed_enabled_in_name="concept.image.enable_fixed_saturation",
            max_strength_in_name="concept.image.random_saturation_max_strength",
        )
        random_hue = RandomHue(
            names=image_inputs,
            enabled_in_name="concept.image.enable_random_hue",
            fixed_enabled_in_name="concept.image.enable_fixed_hue",
            max_strength_in_name="concept.image.random_hue_max_strength",
        )

        # text augmentations
        drop_tags = DropTags(
            text_in_name="prompt",
            enabled_in_name="concept.text.tag_dropout_enable",
            probability_in_name="concept.text.tag_dropout_probability",
            dropout_mode_in_name="concept.text.tag_dropout_mode",
            special_tags_in_name="concept.text.tag_dropout_special_tags",
            special_tag_mode_in_name="concept.text.tag_dropout_special_tags_mode",
            delimiter_in_name="concept.text.tag_delimiter",
            keep_tags_count_in_name="concept.text.keep_tags_count",
            text_out_name="prompt",
            regex_enabled_in_name="concept.text.tag_dropout_special_tags_regex",
        )
        caps_randomize = CapitalizeTags(
            text_in_name="prompt",
            enabled_in_name="concept.text.caps_randomize_enable",
            probability_in_name="concept.text.caps_randomize_probability",
            capitalize_mode_in_name="concept.text.caps_randomize_mode",
            delimiter_in_name="concept.text.tag_delimiter",
            convert_lowercase_in_name="concept.text.caps_randomize_lowercase",
            text_out_name="prompt",
        )
        shuffle_tags = ShuffleTags(
            text_in_name="prompt",
            enabled_in_name="concept.text.enable_tag_shuffling",
            delimiter_in_name="concept.text.tag_delimiter",
            keep_tags_count_in_name="concept.text.keep_tags_count",
            text_out_name="prompt",
        )

        if config.prompt_only_data:
            return [drop_tags, caps_randomize, shuffle_tags]

        modules = [
            random_flip,
            random_rotate,
            random_brightness,
            random_contrast,
            random_saturation,
            random_hue,
            drop_tags,
            caps_randomize,
            shuffle_tags,
        ]

        return modules

    def _inpainting_modules(self, config: TrainConfig):
        if config.prompt_only_data:
            return []

        conditioning_image = GenerateMaskedConditioningImage(
            image_in_name="image",
            mask_in_name="mask",
            image_out_name="conditioning_image",
            image_range_min=0,
            image_range_max=1,
        )
        select_conditioning_image = SelectFirstInput(
            in_names=["custom_conditioning_image", "conditioning_image"], out_name="conditioning_image"
        )

        modules = []

        if config.model_type.has_conditioning_image_input():
            modules.append(conditioning_image)
            modules.append(select_conditioning_image)

        return modules

    def _output_modules_from_out_names(
        self,
        model: BaseModel,
        model_setup: ModelSetupText2ImageMixin,
        output_names: list[str | tuple[str, str]],
        config: TrainConfig,
        before_cache_image_fun: Callable[[], None] | None = None,
        use_conditioning_image: bool = False,
        vae: AutoencoderKL | None = None,
        autocast_context: list[torch.autocast | None] = None,
        train_dtype: DataType | None = None,
        is_validation: bool = False,
    ):
        if before_cache_image_fun is None:

            def prepare_vae():
                model.to(self.temp_device)
                model.vae_to(self.train_device)
                model.eval()
                torch_gc()

            before_cache_image_fun = prepare_vae

        if config.rlhf_enabled:
            output_names = output_names + ["latent_image_rejected"]
            if config.masked_training:
                output_names = output_names + ["latent_mask_rejected"]

        resolved_output_names = [self.__as_output_mapping(name) for name in output_names]

        output_names = resolved_output_names + [
            ("concept.loss_weight", "loss_weight"),
            ("concept.type", "concept_type"),
        ]

        if config.rlhf_enabled:
            # Per-concept DPO beta (0 = inherit the global beta). A batch mixes
            # concepts - AspectBatchSorting buckets by resolution across the whole
            # dataset - so beta has to travel per sample, exactly like loss_weight.
            # Concept fields never key the disk cache, so betas can be retuned
            # between runs without re-caching latents.
            output_names.append(("concept.dpo_beta_override", "dpo_beta_override"))
            # Per-concept supervised mix on the chosen half (negative = inherit
            # rlhf_supervised_mix), per sample for the same reason.
            output_names.append(("concept.supervised_mix_override", "supervised_mix_override"))

        if getattr(self, "is_kto", False):
            # The KTO loader runs the standard pipeline (config.rlhf_enabled is
            # forced off, so no rejected half) but feeds a preference loss: the
            # concept's Liked / Disliked label and its beta override travel per
            # sample like loss_weight, because a batch mixes concepts.
            output_names.append(("concept.kto_label", "kto_label"))
            output_names.append(("concept.dpo_beta_override", "dpo_beta_override"))

        if config.masked_training:
            # Per-concept unmasked weight (negative = inherit the global value).
            # Travels per sample for the same reason as the beta override.
            output_names.append(("concept.unmasked_weight_override", "unmasked_weight_override"))

        if config.validation:
            output_names.append(("concept.name", "concept_name"))
            output_names.append(("concept.path", "concept_path"))
            output_names.append(("concept.seed", "concept_seed"))

        final_output_names = [out_name for _, out_name in output_names]
        sort_names = final_output_names + ["concept"]

        mask_remove = RandomLatentMaskRemove(
            latent_mask_name="latent_mask",
            latent_conditioning_image_name="latent_conditioning_image" if use_conditioning_image else None,
            replace_probability=config.unmasked_probability,
            vae=vae,
            possible_resolutions_in_name="possible_resolutions",
            autocast_contexts=autocast_context,
            dtype=train_dtype.torch_dtype(),
            before_cache_fun=before_cache_image_fun,
        )

        concept_mask_remove = ConceptLatentMaskRemove(
            latent_mask_names=["latent_mask"] + (["latent_mask_rejected"] if config.rlhf_enabled else []),
            probability_in_name="concept.unmasked_probability_override",
            default_probability=config.unmasked_probability,
        )

        modules = []

        if config.model_type.has_mask_input():
            # Inpainting: mgds' module blanks the conditioning image too, but
            # knows only the run-wide probability.
            modules.append(mask_remove)
        elif config.masked_training:
            modules.append(concept_mask_remove)

        world_size = (
            multi.world_size() if config.multi_gpu else 1
        )  # world_size can be 1 for validation dataloader, even if multi.world_size() returns > 1
        if config.latent_caching:
            batch_sorting = AspectBatchSorting(
                resolution_in_name="crop_resolution", names=sort_names, batch_size=config.batch_size * world_size
            )
            distributed_sampler = DistributedSampler(names=sort_names, world_size=world_size, rank=multi.rank())
        else:
            batch_sorting = InlineAspectBatchSorting(
                resolution_in_name="crop_resolution", names=sort_names, batch_size=config.batch_size * world_size
            )
            distributed_sampler = InlineDistributedSampler(names=sort_names, world_size=world_size, rank=multi.rank())

        output = OutputPipelineModule(names=output_names)

        modules.append(batch_sorting)
        if world_size > 1:
            modules.append(distributed_sampler)

        modules.append(output)

        return modules

    def _cache_modules_from_names(
        self,
        model: BaseModel,
        model_setup: ModelSetupText2ImageMixin,
        image_split_names: list[str],
        image_aggregate_names: list[str],
        text_split_names: list[str],
        sort_names: list[str],
        config: TrainConfig,
        text_caching: bool,
        before_cache_image_fun: Callable[[], None] | None = None,
        text_cache_build_workers: int | None = None,
    ):
        image_cache_dir = os.path.join(config.cache_dir, "image")
        text_cache_dir = os.path.join(config.cache_dir, "text")

        if before_cache_image_fun is None:

            def prepare_vae():
                model.to(self.temp_device)
                model.vae_to(self.train_device)
                model.eval()
                torch_gc()

            before_cache_image_fun = prepare_vae

        def before_cache_text_fun():
            model_setup.prepare_text_caching(model, config)

        sourceless = config.sourceless_training and config.latent_caching

        def stop_check():
            return self.stop_check_fun()

        # If _aspect_bucketing_in ran first and stashed an AspectBucketing
        # instance, wire bucket-method drift detection into the image cache.
        # Sourceless training and non-bucketing flows leave this None — the
        # cache then behaves as a single-variant store with drift detection
        # disabled.
        aspect_bucketing = getattr(self, "_aspect_bucketing_for_cache", None)
        if aspect_bucketing is not None:
            bucket_method_provider = aspect_bucketing.compute_bucket_method_hash

            def rebucket_provider(aspect, _ab=aspect_bucketing):
                keys = []
                for tr in _ab.bucket_resolutions:
                    h, w = _ab.bucket_for_aspect(aspect, tr)
                    keys.append(f"{h}x{w}")
                return keys
        else:
            bucket_method_provider = None
            rebucket_provider = None

        # Sidecars whose edits should invalidate the bundled image entry
        # (latent_image + latent_mask + custom_conditioning_image are all
        # baked into the same .pt). Must match the ModifyPath stages
        # registered in _enumerate_input_modules — names not present in the
        # upstream pipeline would raise on resolution.
        image_extra_watched: list[str] = []
        if config.masked_training:
            image_extra_watched.append("mask_path")
        if config.custom_conditioning_image:
            image_extra_watched.append("cond_path")

        if config.rlhf_enabled:
            # The rejected latent is a new split tensor inside the chosen
            # row's .pt: SmartDiskCache's schema-drift augmentation encodes it
            # in place without invalidating the existing chosen latents.
            # The rejected file is watched as a sidecar of the chosen row: the
            # cache entry is keyed on the chosen path, so a rejected image
            # swapped in place (Replace Pairs, re-pairing) would otherwise
            # train on the stale cached latent forever. In adaptive mode it
            # also constrains tier eligibility.
            image_split_names = image_split_names + ["latent_image_rejected"]
            image_extra_watched.append("image_path_rejected")
            if config.masked_training:
                image_split_names = image_split_names + ["latent_mask_rejected"]
                image_extra_watched.append("mask_path_rejected")

        # The DPO patterns change which rows a concept produces, so they join
        # the variation group keys. Group keys only organize balancing and
        # sampling — they never invalidate cached .pt files. Gate on rlhf_enabled
        # so non-DPO runs keep the byte-identical upstream group key rather than a
        # 6-element key carrying two always-empty pattern fields.
        variations_group_extra = (
            ["concept.dpo_chosen_pattern", "concept.dpo_rejected_pattern"] if config.rlhf_enabled else []
        )

        image_cache_kwargs = {
            "split_names": image_split_names,
            "aggregate_names": image_aggregate_names,
            "variations_in_name": "concept.image_variations",
            "balancing_in_name": "concept.balancing",
            "balancing_strategy_in_name": "concept.balancing_strategy",
            "variations_group_in_name": [
                "concept.path",
                "concept.seed",
                "concept.include_subdirectories",
                "concept.image",
            ]
            + variations_group_extra,
            "group_enabled_in_name": "concept.enabled",
            "before_cache_fun": before_cache_image_fun,
            "stop_check_fun": stop_check,
            "modeltype": config.model_type.value,
            "source_path_in_name": "image_path",
            "sourceless": sourceless,
            "bucket_method_provider": bucket_method_provider,
            "rebucket_provider": rebucket_provider,
            "aspect_bucketing": aspect_bucketing,
            "extra_watched_paths_in_names": image_extra_watched or None,
            "trust_cache": config.skip_cache_validation,
            "adaptive_resolution": config.adaptive_resolution,
            "adaptive_resolution_ladder": config.adaptive_resolution_ladder,
            "additional_resolution_in_names": (
                ["original_resolution_rejected"] if config.rlhf_enabled and config.adaptive_resolution else None
            ),
            "adaptive_resolution_max": (
                max(
                    target_resolutions_for_spec(
                        config.resolution,
                        True,
                        config.adaptive_resolution_ladder,
                    ),
                    default=None,
                )
                if config.adaptive_resolution
                else None
            ),
        }

        text_cache_kwargs = {
            "split_names": text_split_names,
            "aggregate_names": [],
            "variations_in_name": "concept.text_variations",
            "balancing_in_name": "concept.balancing",
            "balancing_strategy_in_name": "concept.balancing_strategy",
            "variations_group_in_name": [
                "concept.path",
                "concept.seed",
                "concept.include_subdirectories",
                "concept.text",
            ]
            + variations_group_extra,
            "group_enabled_in_name": "concept.enabled",
            "before_cache_fun": before_cache_text_fun,
            "stop_check_fun": stop_check,
            "modeltype": config.model_type.value,
            "source_path_in_name": "sample_prompt_path",
            "sourceless": sourceless,
            "trust_cache": config.skip_cache_validation,
            # A caption .txt is optional: an image with no caption trains with
            # an empty prompt. Tolerate the missing source file so the text
            # cache still builds an (empty-caption) entry for it instead of
            # skipping it — otherwise the image is cached but its text
            # counterpart is not, leaving the pair misaligned for sourceless
            # training. Captions that DO exist are still edit-checked normally.
            "tolerate_missing_source": True,
            # Content-addressed reuse: the final post-augmentation prompt
            # string fully determines the cached text payload, so identical
            # caption lines are encoded once and reused everywhere — across
            # variations, files and concepts. Editing one line of a
            # multi-line caption re-encodes only that line; a bulk edit that
            # appends the same line to every file encodes it once total.
            "content_key_in_name": "prompt",
            "build_max_workers": text_cache_build_workers,
            # cache_scrub_captions: blank stamped prompts, zero token ids.
            **(text_cache_scrub_kwargs(config, text_split_names) if text_caching else {}),
        }

        # Concepts with their own cache_dir are routed to a SmartDiskCache per
        # root. Runs without overrides keep the plain single-directory caches.
        concepts = getattr(self, "_concepts_for_cache", None) or []
        if uses_concept_cache_roots(concepts, config.cache_dir):
            sourceless_roots = concept_cache_roots(concepts, config.cache_dir)
            image_disk_cache = ConceptCacheRouter(
                config.cache_dir, "image", image_cache_kwargs, sourceless_roots=sourceless_roots
            )
            text_disk_cache = ConceptCacheRouter(
                config.cache_dir, "text", text_cache_kwargs, sourceless_roots=sourceless_roots
            )
        else:
            image_disk_cache = SmartDiskCache(cache_dir=image_cache_dir, **image_cache_kwargs)
            text_disk_cache = SmartDiskCache(cache_dir=text_cache_dir, **text_cache_kwargs)

        modules = []

        if config.latent_caching:
            if not config.prompt_only_data:
                modules.append(image_disk_cache)

                sort_names = [x for x in sort_names if x not in image_aggregate_names]
                sort_names = [x for x in sort_names if x not in image_split_names]
            # Prompt-only: no image cache (zeros are cheaper to regenerate than
            # to read, and its edit checks would stat nonexistent files). The
            # image names stay in sort_names so VariationSorting still carries
            # the live zero-latent rows.

            if text_caching:
                modules.append(text_disk_cache)
                sort_names = [x for x in sort_names if x not in text_split_names]

        if len(sort_names) > 0:
            variation_sorting = VariationSorting(
                names=sort_names,
                balancing_in_name="concept.balancing",
                balancing_strategy_in_name="concept.balancing_strategy",
                variations_group_in_name=[
                    "concept.path",
                    "concept.seed",
                    "concept.include_subdirectories",
                    "concept.text",
                ]
                + variations_group_extra,
                group_enabled_in_name="concept.enabled",
            )

            modules.append(variation_sorting)

        return modules

    def _create_dataset(
        self,
        config: TrainConfig,
        model: BaseModel,
        model_setup: ModelSetupText2ImageMixin,
        train_progress: TrainProgress,
        is_validation: bool,
        aspect_bucketing_quantization: int,
        frame_dim_enabled: bool = False,
        allow_video_files: bool = False,
        vae_frame_dim: bool = False,
        supports_inpainting: bool = True,  # TODO many models probably don't support inpainting, but this has been enabled in most dataloaders before refactoring, too
    ):
        # Reset before each invocation; _aspect_bucketing_in stashes the
        # current AspectBucketing instance here so _cache_modules can wire
        # bucket-method drift detection into SmartDiskCache.
        self._aspect_bucketing_for_cache = None
        # The concepts this loader trains on; _cache_modules routes the ones
        # with a cache_dir override to their own cache roots.
        concepts = self._select_concepts(config, is_validation)
        self._concepts_for_cache = concepts

        output_modules = self._output_modules(config, model, model_setup, is_validation=is_validation)

        enumerate_input = self._enumerate_input_modules(config, allow_videos=allow_video_files)
        load_input = self._load_input_modules(config, model.train_dtype, vae_frame_dim=vae_frame_dim)
        mask_augmentation = self._mask_augmentation_modules(config)
        # Build aspect-bucketing first so its instance is stashed on self
        # before _cache_modules constructs SmartDiskCache.
        aspect_bucketing_in = self._aspect_bucketing_in(config, aspect_bucketing_quantization, frame_dim_enabled)
        cache_modules = self._cache_modules(config, model, model_setup)
        crop_modules = self._crop_modules(config)
        augmentation_modules = self._augmentation_modules(config)
        if supports_inpainting:
            inpainting_modules = self._inpainting_modules(config)
        preparation_modules = (
            self._preparation_modules(config, model)
            + self._dpo_rejected_preparation_modules(config, model)
            + self._dpo_rejected_mask_modules(config)
        )

        debug_modules = self._debug_modules(config, model)

        pre_cache_modules = [
            enumerate_input,
            load_input,
            mask_augmentation,
            aspect_bucketing_in,
            crop_modules,
            augmentation_modules,
        ]
        if supports_inpainting:
            pre_cache_modules.append(inpainting_modules)
        pre_cache_modules.append(preparation_modules)

        if config.sourceless_training and config.latent_caching:
            if hasattr(config, "train_text_encoder_or_embedding") and config.train_text_encoder_or_embedding():
                raise RuntimeError(
                    "Sourceless training cannot be used with text encoder training. "
                    "Disable sourceless_training or disable text encoder training."
                )
            # Sourceless mode stubs out the source-reading pipeline, so the
            # cache index must already carry the per-sample metadata. Fail fast
            # with a clear, UI-surfaced message rather than letting mgds raise
            # deep in __init_sourceless (or, worse, silently train on misaligned
            # data).
            problems = []
            for root in concept_cache_roots(concepts, config.cache_dir) or [config.cache_dir]:
                problems += sourceless_cache_problems(root)
            if problems:
                raise RuntimeError(
                    "Sourceless training cannot start — the cache is not metadata-complete:\n  - "
                    + "\n  - ".join(problems)
                )
            return self._create_mgds(
                config,
                [self._placeholder_modules_for(pre_cache_modules), cache_modules, output_modules],
                train_progress,
                is_validation,
                concepts=concepts,
            )

        return self._create_mgds(
            config,
            pre_cache_modules
            + [
                cache_modules,
                output_modules,
                debug_modules if config.debug_mode else None,
            ],
            train_progress,
            is_validation,
            concepts=concepts,
        )

    def _dpo_rejected_mask_modules(self, config: TrainConfig) -> list:
        # The rejected mask at latent resolution. Sized from the rejected latent
        # itself, so the per-model VAE factor (and the loaders that override
        # _dpo_rejected_preparation_modules) need no mirror of downscale_mask.
        if not (config.rlhf_enabled and config.masked_training):
            return []

        return [
            ScaleMaskToLatent(
                mask_in_name="mask_rejected",
                latent_in_name="latent_image_rejected",
                out_name="latent_mask_rejected",
            )
        ]

    def _dpo_rejected_preparation_modules(self, config: TrainConfig, model: BaseModel) -> list:
        # Mirrors the standard chosen-image VAE encode for 'image_rejected'.
        # Dataloaders whose encode deviates from this pattern override it.
        if not config.rlhf_enabled:
            return []

        rescale_image = RescaleImageChannels(
            image_in_name="image_rejected",
            image_out_name="image_rejected",
            in_range_min=0,
            in_range_max=1,
            out_range_min=-1,
            out_range_max=1,
        )
        encode_image = EncodeVAE(
            in_name="image_rejected",
            out_name="latent_image_rejected_distribution",
            vae=model.vae,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )
        image_sample = SampleVAEDistribution(
            in_name="latent_image_rejected_distribution", out_name="latent_image_rejected", mode="mean"
        )

        return [rescale_image, encode_image, image_sample]

    @abstractmethod
    def _preparation_modules(self, config: TrainConfig, model: BaseModel):
        pass

    @abstractmethod
    def _cache_modules(self, config: TrainConfig, model: BaseModel, model_setup: BaseModelSetup):
        pass

    @abstractmethod
    def _output_modules(
        self, config: TrainConfig, model: BaseModel, model_setup: BaseModelSetup, is_validation: bool = False
    ):
        pass

    @abstractmethod
    def _debug_modules(self, config: TrainConfig, model: BaseModel):
        pass
