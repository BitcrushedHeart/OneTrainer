from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

from torchvision import transforms
from torchvision.transforms import functional


def cover_scale_size(image_hw: tuple[int, int], crop_hw: tuple[int, int]) -> tuple[int, int]:
    """Smallest aspect-preserving (h, w) that covers ``crop_hw``.

    Both sides scale by the same factor, chosen so neither ends up shorter
    than the crop box; the longer side overshoots and is cropped away. The
    result is never smaller than the crop box in either dimension, even under
    rounding, so the subsequent crop always has enough pixels."""
    h, w = image_hw
    crop_h, crop_w = crop_hw
    scale = max(crop_h / h, crop_w / w)
    return max(crop_h, round(h * scale)), max(crop_w, round(w * scale))


def relative_crop_offset(
    chosen_offset: tuple[int, int],
    chosen_scale_hw: tuple[int, int],
    crop_hw: tuple[int, int],
    cover_hw: tuple[int, int],
) -> tuple[int, int]:
    """Map the chosen image's crop offset onto the cover-scaled image.

    The chosen image was cropped at ``chosen_offset`` out of a slack of
    ``chosen_scale - crop`` pixels per axis (0 when the bucket matched its
    aspect exactly). The same *fraction* of the slack is used here, so a
    jittered pair keeps the same relative framing: jitter that pushed the
    photo's crop to its left edge pushes the generation's crop to its left
    edge too. An axis with no slack on the chosen side centres."""
    result = []
    for axis in range(2):
        slack_here = cover_hw[axis] - crop_hw[axis]
        slack_chosen = chosen_scale_hw[axis] - crop_hw[axis]
        if slack_here <= 0:
            result.append(0)
        elif slack_chosen <= 0:
            result.append(slack_here // 2)
        else:
            fraction = min(1.0, max(0.0, chosen_offset[axis] / slack_chosen))
            result.append(min(slack_here, round(fraction * slack_here)))
    return result[0], result[1]


class CoverScaleCropImage(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Scale-and-crop for the rejected half of a DPO pair that keeps its aspect.

    ``ScaleCropImage`` resizes every listed name to the chosen image's
    ``scale_resolution`` before the shared crop — for the rejected image, whose
    aspect need not match the chosen image's bucket, that is a stretch, and a
    stretched generation is trivially discriminable from the photo it is paired
    against. This module instead scales the rejected image by one factor so it
    *covers* the crop box, then crops with the chosen image's relative crop
    offset. A rejected image in the same bucket as its chosen image comes out
    pixel-for-pixel as ``ScaleCropImage`` would have produced it.
    """

    def __init__(
        self,
        names: list[str],
        scale_resolution_in_name: str,
        crop_resolution_in_name: str,
        crop_offset_in_name: str,
    ):
        super().__init__()
        self.names = names
        self.scale_resolution_in_name = scale_resolution_in_name
        self.crop_resolution_in_name = crop_resolution_in_name
        self.crop_offset_in_name = crop_offset_in_name

    def length(self) -> int:
        return self._get_previous_length(self.names[0])

    def get_inputs(self) -> list[str]:
        return self.names + [self.scale_resolution_in_name, self.crop_resolution_in_name, self.crop_offset_in_name]

    def get_outputs(self) -> list[str]:
        return self.names

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        scale_resolution = self._get_previous_item(variation, self.scale_resolution_in_name, index)
        crop_resolution = self._get_previous_item(variation, self.crop_resolution_in_name, index)
        chosen_offset = self._get_previous_item(variation, self.crop_offset_in_name, index)

        crop_hw = (int(crop_resolution[-2]), int(crop_resolution[-1]))
        chosen_scale_hw = (int(scale_resolution[-2]), int(scale_resolution[-1]))

        item = {}
        for name in self.names:
            image = self._get_previous_item(variation, name, index)
            if image is not None:
                cover_hw = cover_scale_size((int(image.shape[-2]), int(image.shape[-1])), crop_hw)
                image = transforms.Resize(
                    cover_hw,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )(image)
                y_offset, x_offset = relative_crop_offset(
                    (int(chosen_offset[0]), int(chosen_offset[1])), chosen_scale_hw, crop_hw, cover_hw
                )
                image = functional.crop(image, y_offset, x_offset, crop_hw[0], crop_hw[1])
            item[name] = image
        return item
