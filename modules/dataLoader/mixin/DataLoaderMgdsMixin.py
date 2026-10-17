import json
from abc import ABCMeta

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_pattern_util import validate_dpo_patterns
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState
from mgds.pipelineModules.PlaceholderModule import PlaceholderModule

import torch


class DataLoaderMgdsMixin(metaclass=ABCMeta):
    @staticmethod
    def _placeholder_modules_for(definition: list | object) -> list[PlaceholderModule]:
        if definition is None:
            return []
        if isinstance(definition, list):
            placeholders = []
            for item in definition:
                placeholders.extend(DataLoaderMgdsMixin._placeholder_modules_for(item))
            return placeholders
        return [PlaceholderModule()]

    @staticmethod
    def __filter_dpo_concepts(concepts: list[ConceptConfig], config: TrainConfig) -> list[ConceptConfig]:
        # An RLHF DPO run trains on chosen/rejected pairs only. Concepts without
        # patterns have no pair data, so they are dropped here; in a DPO run
        # they feed the SFT-anchor loader instead (when enabled).
        for concept in concepts:
            if concept.enabled:
                validate_dpo_patterns(concept.dpo_chosen_pattern, concept.dpo_rejected_pattern)

        dpo_concepts = [c for c in concepts if c.is_dpo()]
        skipped = [c.name or c.path for c in concepts if c.enabled and not c.is_dpo()]

        if not any(c.enabled for c in dpo_concepts):
            raise RuntimeError(
                "RLHF DPO requires at least one enabled concept with chosen/rejected patterns "
                "(set 'DPO Chosen Pattern' and 'DPO Rejected Pattern' in the concept window)."
            )
        if skipped:
            print(
                f"RLHF DPO: skipping {len(skipped)} concepts without DPO patterns "
                "(they feed the SFT-anchor loader when enabled): " + ", ".join(skipped)
            )

        # IPO regresses the margin toward 1/(2*tau) and never multiplies by beta,
        # so a per-concept beta there is a silent no-op. Say so rather than let
        # the setting look effective.
        if config.rlhf_dpo_objective == DPOObjective.IPO:
            overriding = [c.name or c.path for c in dpo_concepts if c.enabled and c.dpo_beta_override > 0]
            if overriding:
                print(
                    "RLHF DPO: the IPO objective ignores beta (tau plays its role), so the per-concept "
                    f"beta override on {len(overriding)} concepts has no effect: " + ", ".join(overriding)
                )

        if config.rlhf_dpo_adaptive_beta:
            opted_out = [c.name or c.path for c in dpo_concepts if c.enabled and not c.dpo_adaptive_beta]
            if opted_out:
                print(
                    f"RLHF DPO: adaptive beta leaves {len(opted_out)} concepts alone (their configured beta is "
                    "applied unscaled and their margins stay out of the controller): " + ", ".join(opted_out)
                )

        return dpo_concepts

    @staticmethod
    def __select_kto_concepts(concepts: list[ConceptConfig]) -> list[ConceptConfig]:
        """The KTO loader reads KTO concepts only: plain folders of single
        images, each concept labelled Liked or Disliked as a whole."""
        kto_concepts = [c for c in concepts if ConceptType(c.type) == ConceptType.KTO]
        patterned = [
            c.name or c.path for c in kto_concepts if c.enabled and (c.dpo_chosen_pattern or c.dpo_rejected_pattern)
        ]
        if patterned:
            raise RuntimeError(
                "KTO concepts are single Liked / Disliked images and cannot carry DPO chosen/rejected "
                "patterns; clear the patterns or use a STANDARD concept for the pairs: " + ", ".join(patterned)
            )
        if not any(c.enabled for c in kto_concepts):
            raise RuntimeError("The KTO loader needs at least one enabled KTO concept.")
        return kto_concepts

    def _select_concepts(self, config: TrainConfig, is_validation: bool = False) -> list[ConceptConfig]:
        concepts = config.concepts
        if concepts is None:
            with open(config.concept_file_name, "r") as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]

        # The standard-side loader of an RLHF run (SFT-anchor loader, or the
        # main loader in SFT + DPO mode) reaches this method with
        # config.rlhf_enabled forced to False by BaseDataLoader, so it falls
        # into the standard branch; DPO pattern concepts are excluded there so
        # their chosen images aren't double-trained as supervised data.
        # A plain (non-DPO) validation loader excludes them for the same
        # reason: a VALIDATION concept with chosen/rejected patterns belongs to
        # the DPO validation loader, and reading its whole tree as ordinary
        # validation images would score the rejected images as targets.
        if getattr(self, "is_kto", False):
            return self.__select_kto_concepts(concepts)
        # KTO concepts are never in valid_types: only the KTO loader reads them.
        valid_types = (
            {ConceptType.VALIDATION} if is_validation else {ConceptType.STANDARD, ConceptType.PRIOR_PREDICTION}
        )
        concepts = [concept for concept in concepts if ConceptType(concept.type) in valid_types]
        if config.rlhf_enabled:
            concepts = self.__filter_dpo_concepts(concepts, config)
        elif is_validation or getattr(self, "is_sft_anchor", False):
            concepts = [c for c in concepts if not c.is_dpo()]
        return concepts

    def _create_mgds(
        self,
        config: TrainConfig,
        definition: list,
        train_progress: TrainProgress,
        is_validation: bool = False,
        concepts: list[ConceptConfig] | None = None,
    ):
        if concepts is None:
            concepts = self._select_concepts(config, is_validation)
        concepts = [c.to_dict() for c in concepts]

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size,
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
