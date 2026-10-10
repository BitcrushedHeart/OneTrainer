from random import Random

from mgds.pipelineModules.RandomCircularMaskShrink import RandomCircularMaskShrink


class PairedCircularMaskShrink(RandomCircularMaskShrink):
    """Circular mask shrink for the rejected half of a DPO pair that draws the
    same random numbers as the chosen half's shrink.

    mgds seeds every module's draws with its own pipeline position, so two
    plain RandomCircularMaskShrink modules would shrink an aligned pair (same
    mask on both images) to two different discs and the pair would no longer
    compare the same region. Sharing the chosen module's draw keeps identical
    masks identical; a rejected image with its own mask still gets a centre
    inside that mask and a radius scaled to it.
    """

    def __init__(self, paired_with: RandomCircularMaskShrink, **kwargs):
        super().__init__(**kwargs)
        self.paired_with = paired_with

    def _get_rand(self, variation: int, index: int = -1) -> Random:
        return self.paired_with._get_rand(variation, index)
