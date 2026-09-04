from modules.util.distill_metadata_util import extract_distill_metadata

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModules.ResolutionPolicy import target_resolutions_for_spec
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


def _spec_to_resolution(spec) -> tuple[int, int] | None:
    """Best square/fixed resolution for a target spec ("1024", "768,1024", "832x1216")."""
    if spec is None:
        return None
    if isinstance(spec, int):
        return (spec, spec)
    spec = str(spec)
    if "x" in spec and "," not in spec:
        width, height = (int(part.strip()) for part in spec.split("x"))
        return (height, width)
    targets = target_resolutions_for_spec(spec, False, None)
    if not targets:
        return None
    target = max(targets)
    return (target, target)


class SynthesizeResolution(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Emits `original_resolution` without decoding an image.

    Priority: distill sidecar width/height -> concept resolution override ->
    settings.target_resolution. The synthesized resolution always covers the
    chosen target tier, so the downstream no-upscale bucketing policy keeps
    every prompt row trainable at its intended resolution.
    """

    def __init__(
        self,
        path_in_name: str,
        resolution_out_name: str,
        target_resolution_in_name: str,
        enable_resolution_override_in_name: str,
        resolution_override_in_name: str,
    ):
        super().__init__()

        self.path_in_name = path_in_name
        self.resolution_out_name = resolution_out_name
        self.target_resolution_in_name = target_resolution_in_name
        self.enable_resolution_override_in_name = enable_resolution_override_in_name
        self.resolution_override_in_name = resolution_override_in_name

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.resolution_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        image_path = self._get_previous_item(variation, self.path_in_name, index)

        resolution = None
        metadata = extract_distill_metadata(image_path)
        if metadata is not None and metadata.width and metadata.height:
            resolution = (metadata.height, metadata.width)

        if resolution is None and self._get_previous_item(variation, self.enable_resolution_override_in_name, index):
            resolution = _spec_to_resolution(
                self._get_previous_item(variation, self.resolution_override_in_name, index)
            )

        if resolution is None:
            resolution = _spec_to_resolution(self._get_previous_item(variation, self.target_resolution_in_name, index))

        if resolution is None:
            raise RuntimeError(
                f"Could not synthesize a resolution for '{image_path}': no sidecar width/height, "
                "no concept resolution override, and the target resolution spec is empty."
            )

        return {self.resolution_out_name: resolution}
