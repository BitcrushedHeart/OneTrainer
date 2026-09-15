import copy
import hashlib
import json
from collections import OrderedDict
from datetime import datetime

from modules.model.BaseModel import BaseModel
from modules.util import git_util
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.modelSpec.ModelSpec import ModelSpec

import torch
from torch import Tensor


class DtypeModelSaverMixin:
    def __init__(self):
        super().__init__()

    def _convert_state_dict_dtype(
        self,
        state_dict: dict,
        dtype: torch.dtype | None,
    ) -> dict:
        converted_state_dict = {}

        for key, value in state_dict.items():
            if isinstance(value, dict):
                converted_state_dict[key] = self._convert_state_dict_dtype(value, dtype)
            else:
                converted_state_dict[key] = value.to(device="cpu", dtype=dtype)

        return converted_state_dict

    def _convert_state_dict_to_contiguous(
        self,
        state_dict: dict,
    ):
        for key, value in state_dict.items():
            if isinstance(value, dict):
                self._convert_state_dict_to_contiguous(value)
            else:
                state_dict[key] = value.contiguous()

    def __calculate_safetensors_hash(
        self,
        state_dict: dict[str, Tensor] | None = None,
    ) -> str | None:
        if state_dict is None:
            return None

        sha256_hash = hashlib.sha256()

        ordered_state_dict = OrderedDict(sorted(state_dict.items()))
        for tensor in ordered_state_dict.values():
            data = tensor.contiguous().cpu().flatten().view(torch.uint8).numpy().tobytes()
            sha256_hash.update(data)

        return f"0x{sha256_hash.hexdigest()}"

    def _create_safetensors_header(
        self,
        model: BaseModel,
        state_dict: dict[str, Tensor] | None = None,
        internal_format: bool = False,
    ) -> dict[str, str]:
        """Build the safetensors metadata header.

        ``internal_format`` marks a ModelFormat.INTERNAL backup: it is only read back by
        OneTrainer's resume path, which ignores provenance, so the SHA-256 over every tensor
        byte and the git subprocess calls are skipped (no ``modelspec.hash_sha256``,
        ``ot_branch`` or ``ot_revision``). Exported formats keep them.
        """
        model_spec = copy.deepcopy(model.model_spec) if model.model_spec is not None else ModelSpec()

        if model.train_config is not None and model.train_config.include_train_config == ConfigPart.SETTINGS:
            config = json.dumps(model.train_config.to_settings_dict(secrets=False))
        elif model.train_config is not None and model.train_config.include_train_config == ConfigPart.ALL:
            config = json.dumps(model.train_config.to_pack_dict(secrets=False))
        else:
            config = None

        # update calculated fields
        model_spec.date = datetime.now().strftime("%Y-%m-%d")
        # None drops the key, including any hash inherited from the loaded model spec
        model_spec.hash_sha256 = None if internal_format else self.__calculate_safetensors_hash(state_dict)

        # assemble the header
        model_spec_dict = model_spec.to_dict()
        one_trainer_header = {}
        if not internal_format:
            one_trainer_header["ot_branch"] = git_util.get_git_branch()
            one_trainer_header["ot_revision"] = git_util.get_git_revision()
        if config is not None:
            one_trainer_header["ot_config"] = config

        kohya_header = {}  # needed for the Automatic1111 webui to pick up model versions
        if model.model_type.is_stable_diffusion_xl():
            kohya_header["ss_base_model_version"] = "sdxl_"
        elif model.model_type.is_sd_v2():
            kohya_header["ss_v2"] = "True"
        return model_spec_dict | one_trainer_header | kohya_header
