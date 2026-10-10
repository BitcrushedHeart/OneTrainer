from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from torchvision import transforms


class ScaleMaskToLatent(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Resizes a mask to the spatial size of a latent.

    The chosen mask is downscaled by each dataloader's own VAE factor. The
    rejected half of a DPO pair is prepared by the shared mixin, which does not
    know that factor, so its mask is sized from the rejected latent itself and
    lands at the same resolution whatever the VAE.
    """

    def __init__(self, mask_in_name: str, latent_in_name: str, out_name: str):
        super().__init__()
        self.mask_in_name = mask_in_name
        self.latent_in_name = latent_in_name
        self.out_name = out_name

    def length(self) -> int:
        return self._get_previous_length(self.mask_in_name)

    def get_inputs(self) -> list[str]:
        return [self.mask_in_name, self.latent_in_name]

    def get_outputs(self) -> list[str]:
        return [self.out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        mask = self._get_previous_item(variation, self.mask_in_name, index)
        latent = self._get_previous_item(variation, self.latent_in_name, index)

        size = (int(latent.shape[-2]), int(latent.shape[-1]))
        if tuple(mask.shape[-2:]) != size:
            mask = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)(mask)

        return {
            self.out_name: mask,
        }
