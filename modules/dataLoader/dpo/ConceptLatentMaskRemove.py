from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class ConceptLatentMaskRemove(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Trains a sample unmasked (all-ones latent mask) at the run's unmasked
    probability, or at its concept's own when the concept overrides it.

    mgds' RandomLatentMaskRemove applies the run-wide probability only for
    inpainting models (it also blanks the conditioning image), so before this
    module the setting silently did nothing for every other model. A negative
    concept override inherits ``default_probability``. One draw per sample
    covers every listed mask, so the chosen and rejected halves of a DPO pair
    are unmasked together.
    """

    def __init__(self, latent_mask_names: list[str], probability_in_name: str, default_probability: float):
        super().__init__()
        self.latent_mask_names = latent_mask_names
        self.probability_in_name = probability_in_name
        self.default_probability = default_probability

    def length(self) -> int:
        return self._get_previous_length(self.latent_mask_names[0])

    def get_inputs(self) -> list[str]:
        return self.latent_mask_names + [self.probability_in_name]

    def get_outputs(self) -> list[str]:
        return self.latent_mask_names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        masks = {name: self._get_previous_item(variation, name, index) for name in self.latent_mask_names}

        probability = self._get_previous_item(variation, self.probability_in_name, index)
        if probability is None or probability < 0:
            probability = self.default_probability
        if probability <= 0:
            return masks

        if self._get_rand(variation, index).random() < probability:
            masks = {name: None if mask is None else torch.ones_like(mask) for name, mask in masks.items()}

        return masks
