from web.backend.services.runpod_setup_service import RunpodSetupService
from web.backend.services.trainer_service import TrainerService

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/cloud", tags=["cloud"])


class RunpodPrepareRequest(BaseModel):
    api_key: str
    ssh_user: str = "root"
    ssh_key_file: str = ""
    ssh_password: str = ""
    pod_name: str = "OneTrainer"
    run_id: str = "job1"
    min_download: int = 0
    # upload with the pod on zero GPUs (CPU-only rate), then resume it with its GPU to train
    upload_without_gpu: bool = False
    start: bool = False
    epochs: int = 3
    # Any RunPod GPU type id (e.g. "NVIDIA A100 80GB PCIe"); empty keeps the RTX 5090 default.
    gpu_type: str = ""
    cloud_type: str = "SECURE"


class RunpodGpuRequest(BaseModel):
    # Falls back to the saved cloud secret when empty.
    api_key: str = ""


@router.get("/runpod/preview")
def runpod_preview() -> dict:
    return RunpodSetupService.get_instance().preview()


@router.post("/runpod/gpus")
def runpod_gpus(req: RunpodGpuRequest) -> dict:
    # POST so the API key travels in the body, not the URL. Failures come back as
    # ok=False + error (not an HTTP error) so the modal can fall back to free text.
    return RunpodSetupService.get_instance().list_gpu_types(req.api_key)


@router.post("/runpod/git-preflight")
def runpod_git_preflight() -> dict:
    try:
        return RunpodSetupService.get_instance().run_git_preflight()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/runpod/prepare")
def runpod_prepare(req: RunpodPrepareRequest) -> dict:
    result = RunpodSetupService.get_instance().prepare(req.model_dump())
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error", "RunPod prepare failed"))

    if req.start:
        start_result = TrainerService.get_instance().start_training(reattach=False)
        result["start"] = start_result
        if not start_result.get("ok"):
            raise HTTPException(status_code=422, detail=start_result.get("error", "Training start failed"))

    return result


@router.post("/runpod/live-test")
def runpod_live_test(req: RunpodPrepareRequest) -> dict:
    result = RunpodSetupService.get_instance().start_live_test(req.model_dump())
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("error", "RunPod live test failed"))
    return result
