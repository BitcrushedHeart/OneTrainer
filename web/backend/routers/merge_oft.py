"""FastAPI router for the OFT / DoRA-OFT merge tool.

The frontend passes the active TrainConfig's base_model_name /
transformer_model_name / vae_model_name in the request body so the merge
inherits the same Model tab settings the user would use for training.
"""

import logging
import threading

from web.backend.utils.path_security import validate_path

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tools"])

_merge_lock = threading.Lock()


class MergeOFTRequest(BaseModel):
    oft_adapter_path: str
    output_path: str
    output_dtype: str = "BFLOAT_16"
    output_model_format: str = "SAFETENSORS"
    strength: float = 1.0
    compute_device: str | None = None  # "cuda" | "cpu" | None (auto)
    # Inherited from the active Train UI's Model tab.
    base_model_name: str
    transformer_model_name: str = ""
    vae_model_name: str = ""


class MergeOFTResponse(BaseModel):
    ok: bool
    error: str | None = None
    modules_merged: int = 0
    orthogonality_max_err: dict[str, float] | None = None
    pre_zero_rows: dict[str, int] | None = None
    post_zero_rows: dict[str, int] | None = None


@router.post("/tools/merge-oft", response_model=MergeOFTResponse)
def merge_oft(req: MergeOFTRequest) -> MergeOFTResponse:
    validate_path(req.oft_adapter_path, must_exist=True)
    validate_path(req.output_path, must_exist=False)
    # base_model_name may be a HF repo id (e.g., "Tongyi-MAI/Z-Image"), so we
    # only path-validate when it actually looks like a local path.
    if (
        req.base_model_name
        and any(req.base_model_name.startswith(p) for p in ("/", "\\", "./", ".\\"))
        or (len(req.base_model_name) > 1 and req.base_model_name[1] == ":")
    ):
        validate_path(req.base_model_name, must_exist=True)
    if req.transformer_model_name:
        validate_path(req.transformer_model_name, must_exist=True)
    if req.vae_model_name:
        validate_path(req.vae_model_name, must_exist=True)

    if not _merge_lock.acquire(blocking=False):
        return MergeOFTResponse(ok=False, error="A merge is already in progress")
    try:
        from modules.util.config.TrainConfig import TrainConfig
        from modules.util.enum.DataType import DataType
        from modules.util.enum.ModelFormat import ModelFormat
        from modules.util.oft_merge import merge_oft_adapter
        from modules.util.oft_verify import MergeVerificationError
        from modules.util.torch_util import torch_gc

        output_dtype = DataType(req.output_dtype)
        output_format = ModelFormat(req.output_model_format)

        # Build a stub TrainConfig that carries the inherited path fields.
        # The orchestrator's _train_config_from_adapter call will override the
        # training-time settings from the adapter's ot_config; only the path
        # fields from this stub flow through. transformer/vae model names live
        # on sub-configs in modern TrainConfig.
        ui_train_config = TrainConfig.default_values()
        ui_train_config.base_model_name = req.base_model_name
        ui_train_config.transformer.model_name = req.transformer_model_name
        ui_train_config.vae.model_name = req.vae_model_name

        logger.info(
            "Merging OFT adapter %s -> %s (base=%s, transformer=%s, strength=%s)",
            req.oft_adapter_path,
            req.output_path,
            req.base_model_name,
            req.transformer_model_name or "(stock)",
            req.strength,
        )

        report = merge_oft_adapter(
            oft_adapter_path=req.oft_adapter_path,
            output_path=req.output_path,
            output_dtype=output_dtype,
            output_format=output_format,
            ui_train_config=ui_train_config,
            strength=float(req.strength),
            compute_device=req.compute_device,
        )
        torch_gc()
        return MergeOFTResponse(
            ok=True,
            modules_merged=len(report.post_zero_rows),
            orthogonality_max_err=report.orthogonality_max_err,
            pre_zero_rows=report.pre_zero_rows,
            post_zero_rows=report.post_zero_rows,
        )
    except MergeVerificationError as exc:
        logger.warning("Merge verification failed: %s", exc)
        try:
            from modules.util.torch_util import torch_gc

            torch_gc()
        except Exception:  # noqa: BLE001
            pass
        return MergeOFTResponse(ok=False, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("OFT merge failed")
        try:
            from modules.util.torch_util import torch_gc

            torch_gc()
        except Exception:  # noqa: BLE001
            pass
        return MergeOFTResponse(ok=False, error=str(exc))
    finally:
        _merge_lock.release()
