from web.backend.services.dpo_service import DPOService

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(tags=["dpo"])


class ActionResponse(BaseModel):
    ok: bool
    error: str | None = None


class RemovePairRequest(BaseModel):
    chosen_path: str | None = None
    rejected_path: str | None = None


class StartSessionRequest(BaseModel):
    source_folder: str
    output_dir: str
    pairs_per_group: int = 1


class SelectImageRequest(BaseModel):
    path: str


class FinalizeRequest(BaseModel):
    val_percentage: float = 0.0


@router.post("/dpo/check-pairs")
def check_pairs():
    service = DPOService.get_instance()
    return service.check_pairs()


@router.post("/dpo/remove-strays")
def remove_strays():
    service = DPOService.get_instance()
    return service.remove_strays()


@router.get("/dpo/review")
def review_pairs():
    service = DPOService.get_instance()
    return service.review_pairs()


@router.post("/dpo/remove-pair", response_model=ActionResponse)
def remove_pair(req: RemovePairRequest):
    service = DPOService.get_instance()
    return ActionResponse(**service.remove_pair(req.chosen_path, req.rejected_path))


@router.post("/dpo/fix-captions")
def fix_multiline_captions():
    service = DPOService.get_instance()
    return service.fix_multiline_captions()


# ---- Curation Session ----

@router.post("/dpo/session/start")
def start_session(req: StartSessionRequest):
    service = DPOService.get_instance()
    return service.start_session(req.source_folder, req.output_dir, req.pairs_per_group)


@router.get("/dpo/session/status")
def session_status():
    service = DPOService.get_instance()
    return service.get_session_status()


@router.post("/dpo/session/next-group")
def next_group():
    service = DPOService.get_instance()
    return service.next_group()


@router.post("/dpo/session/select")
def select_image(req: SelectImageRequest):
    service = DPOService.get_instance()
    return service.select_image(req.path)


@router.post("/dpo/session/skip-group", response_model=ActionResponse)
def skip_group():
    service = DPOService.get_instance()
    return ActionResponse(**service.skip_group())


@router.post("/dpo/session/finalize")
def finalize_session(req: FinalizeRequest):
    service = DPOService.get_instance()
    return service.finalize_session(req.val_percentage)


@router.post("/dpo/session/cancel", response_model=ActionResponse)
def cancel_session():
    service = DPOService.get_instance()
    return ActionResponse(**service.cancel_session())


@router.get("/dpo/session/image")
def serve_image(path: str):
    service = DPOService.get_instance()
    abs_path = service.serve_image(path)
    if abs_path is None:
        return ActionResponse(ok=False, error="Image not found")
    return FileResponse(abs_path)
