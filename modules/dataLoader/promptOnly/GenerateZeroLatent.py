from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

import torch


class GenerateZeroLatent(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Synthesizes a zero `latent_image` of the bucketed latent shape.

    The distillation losses only read latent_image's shape/device/dtype (the
    initial noise and every target are computed live), so a zeros tensor
    replaces the whole load->VAE-encode path. Also emits the constant
    `crop_offset` the batch schema expects, since no crop stage runs.
    """

    def __init__(
        self,
        crop_resolution_in_name: str,
        latent_out_name: str,
        crop_offset_out_name: str,
        channels: int,
        downscale_factor: int,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.crop_resolution_in_name = crop_resolution_in_name
        self.latent_out_name = latent_out_name
        self.crop_offset_out_name = crop_offset_out_name
        self.channels = channels
        self.downscale_factor = downscale_factor
        self.dtype = dtype

    def length(self) -> int:
        return self._get_previous_length(self.crop_resolution_in_name)

    def get_inputs(self) -> list[str]:
        return [self.crop_resolution_in_name]

    def get_outputs(self) -> list[str]:
        return [self.latent_out_name, self.crop_offset_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        crop_resolution = self._get_previous_item(variation, self.crop_resolution_in_name, index)
        height, width = crop_resolution[-2], crop_resolution[-1]
        latent = torch.zeros(
            (self.channels, height // self.downscale_factor, width // self.downscale_factor),
            dtype=self.dtype,
        )
        return {
            self.latent_out_name: latent,
            self.crop_offset_out_name: (0, 0),
        }
