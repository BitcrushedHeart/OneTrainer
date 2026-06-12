import argparse
from typing import Any

from modules.util.args.BaseArgs import BaseArgs
from modules.util.enum.DataType import DataType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class MergeOFTArgs(BaseArgs):
    """Arguments for the OFT / DoRA-OFT merge tool.

    Bakes an OFT adapter into the base model checkpoint by routing the
    rotation through OneTrainer's training math (``OFTModule.apply_to_module``)
    rather than the divergent ComfyUI merge path.

    Base + transformer override paths and the model type are inherited from
    the active Train UI's TrainConfig (the same fields the Model tab fills in).
    The user only picks the adapter, output destination, and a few
    quality-of-life knobs.
    """

    input_model_path: str
    oft_adapter_path: str
    output_path: str
    output_dtype: DataType
    output_model_format: ModelFormat
    strength: float

    def __init__(self, data: list[(str, Any, type, bool)]):
        super().__init__(data)

    def weight_dtypes(self) -> ModelWeightDtypes:
        return ModelWeightDtypes.from_single_dtype(self.output_dtype)

    @staticmethod
    def parse_args() -> "MergeOFTArgs":
        parser = argparse.ArgumentParser(description="OneTrainer OFT merge tool.")

        parser.add_argument(
            "--oft-adapter-path",
            type=str,
            required=True,
            dest="oft_adapter_path",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            required=True,
            dest="output_path",
            help="Output path for the merged checkpoint.",
        )
        parser.add_argument(
            "--output-dtype",
            type=DataType,
            required=False,
            default=DataType.BFLOAT_16,
            dest="output_dtype",
            choices=list(DataType),
        )
        parser.add_argument(
            "--output-model-format",
            type=ModelFormat,
            required=False,
            default=ModelFormat.SAFETENSORS,
            dest="output_model_format",
            choices=list(ModelFormat),
        )
        parser.add_argument(
            "--strength",
            type=float,
            required=False,
            default=1.0,
            dest="strength",
            help="Merge strength: W_final = (1-s)*W_base + s*W_dora. 1.0 = full bake.",
        )

        args = MergeOFTArgs.default_values()
        args.from_dict(vars(parser.parse_args()))
        return args

    @staticmethod
    def default_values():
        data = []

        data.append(("input_model_path", "", str, False))
        data.append(("oft_adapter_path", "", str, False))
        data.append(("output_path", "", str, False))
        data.append(("output_dtype", DataType.BFLOAT_16, DataType, False))
        data.append(("output_model_format", ModelFormat.SAFETENSORS, ModelFormat, False))
        data.append(("strength", 1.0, float, False))

        return MergeOFTArgs(data)
