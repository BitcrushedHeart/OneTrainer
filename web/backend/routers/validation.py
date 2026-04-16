from web.backend.services.validation_service import ValidationService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["validation"])


class ActionResponse(BaseModel):
    ok: bool
    error: str | None = None


class DeepScanRequest(BaseModel):
    threshold: float = 0.93


class MatchActionRequest(BaseModel):
    val_image_path: str


@router.get("/validation/status")
def get_status():
    service = ValidationService.get_instance()
    return service.get_status()


@router.get("/validation/results")
def get_results():
    service = ValidationService.get_instance()
    return service.get_results()


@router.post("/validation/scan/basic", response_model=ActionResponse)
def scan_basic():
    service = ValidationService.get_instance()
    result = service.scan_basic()
    return ActionResponse(**result)


@router.post("/validation/scan/deep", response_model=ActionResponse)
def scan_deep(req: DeepScanRequest | None = None):
    threshold = req.threshold if req else 0.93
    service = ValidationService.get_instance()
    result = service.scan_deep(threshold=threshold)
    return ActionResponse(**result)


@router.post("/validation/cancel", response_model=ActionResponse)
def cancel_scan():
    service = ValidationService.get_instance()
    result = service.cancel()
    return ActionResponse(**result)


@router.post("/validation/match/remove", response_model=ActionResponse)
def remove_match(req: MatchActionRequest):
    service = ValidationService.get_instance()
    return ActionResponse(**service.remove_match(req.val_image_path))


@router.post("/validation/match/move")
def move_match(req: MatchActionRequest):
    service = ValidationService.get_instance()
    return service.move_match(req.val_image_path)
