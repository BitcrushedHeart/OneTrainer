from web.backend.services.dpo_replace_service import MAX_IMAGES_PER_PAIR, DPOReplacePairsService

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["dpo-replace"])


class StartRequest(BaseModel):
    concept_dir: str
    generated_folder: str = ""
    base_url: str = "http://localhost:7801"
    token: str = ""
    images_per_pair: int = Field(default=2, ge=1, le=MAX_IMAGES_PER_PAIR)
    new_pass: bool = False


class SessionRequest(BaseModel):
    session_id: str


class ResumeRequest(SessionRequest):
    token: str = ""


class GroupRequest(SessionRequest):
    group_id: str


class BestRequest(GroupRequest):
    best_id: str | None = None


class ApplyRequest(GroupRequest):
    best_id: str
    worst_id: str
    override_reversal: bool = False


class PairModel(BaseModel):
    chosen_id: str
    rejected_id: str


class ApplyPairsRequest(GroupRequest):
    # The first pair replaces the reviewed pair; the rest are added. No cap.
    pairs: list[PairModel] = Field(min_length=1)
    override_reversal: bool = False


def service():
    return DPOReplacePairsService.get_instance()


@router.get("/dpo/replace/concepts")
def concepts():
    return service().concepts()


@router.post("/dpo/replace/start")
def start(req: StartRequest):
    return service().start(**req.model_dump())


@router.post("/dpo/replace/restore")
def restore(req: StartRequest):
    return service().start(**{**req.model_dump(), "new_pass": False, "restore_only": True})


@router.get("/dpo/replace/status")
def status():
    return service().status()


@router.post("/dpo/replace/pause")
def pause(req: SessionRequest):
    return service().pause(req.session_id)


@router.post("/dpo/replace/resume")
def resume(req: ResumeRequest):
    return service().resume(req.session_id, req.token)


@router.post("/dpo/replace/close")
def close(req: SessionRequest):
    return service().pause(req.session_id, close=True)


@router.get("/dpo/replace/current")
def current():
    return service().current()


@router.post("/dpo/replace/best")
def best(req: BestRequest):
    return service().select_best(**req.model_dump())


@router.post("/dpo/replace/apply")
def apply(req: ApplyRequest):
    return service().apply(**req.model_dump())


@router.post("/dpo/replace/apply-pairs")
def apply_pairs(req: ApplyPairsRequest):
    return service().apply_pairs(
        req.session_id,
        req.group_id,
        [(pair.chosen_id, pair.rejected_id) for pair in req.pairs],
        req.override_reversal,
    )


@router.post("/dpo/replace/skip")
def skip(req: GroupRequest):
    return service().skip(**req.model_dump())


@router.post("/dpo/replace/retry")
def retry(req: GroupRequest):
    return service().retry(**req.model_dump())


@router.post("/dpo/replace/undo")
def undo(req: SessionRequest):
    return service().undo(req.session_id)


@router.get("/dpo/replace/image/{session_id}/{group_id}/{candidate_id}")
def image(session_id: str, group_id: str, candidate_id: str):
    try:
        path = service().image_path(session_id, group_id, candidate_id)
        # No origin filename in either the URL or response headers.
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
    except (ValueError, KeyError, StopIteration):
        raise HTTPException(status_code=404, detail="Candidate not found") from None
