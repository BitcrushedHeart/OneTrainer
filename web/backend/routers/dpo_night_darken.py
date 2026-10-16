from web.backend.services.dpo_night_darken_service import DPONightDarkenService

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter(tags=["dpo-night-darken"])


class OpenRequest(BaseModel):
    concept_dir: str


class ScanRequest(BaseModel):
    # Luma thresholds under which an image already counts as dark.
    thresholds: dict | None = None
    # Caption phrases; empty = the built-in list.
    phrases: list[str] = []


class AdjustModel(BaseModel):
    strength: float = 0.0
    noise: float = 0.0
    hue: float = 0.0
    saturation: float = 1.0


class DecideRequest(BaseModel):
    rel: str
    state: str
    adjust: AdjustModel | None = None


class UndoRequest(BaseModel):
    rel: str


class CommitRequest(BaseModel):
    # Empty = "<workspace>/DPO Night Edits" beside the source concept.
    night_edits_dir: str = ""
    fraction: float = 0.33


class RestoreRequest(BaseModel):
    commit_id: str


@router.post("/dpo/night-darken/open")
def open_concept(req: OpenRequest):
    return DPONightDarkenService.get_instance().open(req.concept_dir)


@router.get("/dpo/night-darken/state")
def state():
    return DPONightDarkenService.get_instance().state()


@router.post("/dpo/night-darken/scan")
def scan(req: ScanRequest):
    return DPONightDarkenService.get_instance().scan(thresholds=req.thresholds, phrases=req.phrases)


@router.get("/dpo/night-darken/status")
def status():
    return DPONightDarkenService.get_instance().status()


@router.post("/dpo/night-darken/cancel")
def cancel():
    return DPONightDarkenService.get_instance().cancel()


@router.get("/dpo/night-darken/candidates")
def candidates():
    return DPONightDarkenService.get_instance().candidates()


@router.get("/dpo/night-darken/caption")
def caption(rel: str):
    return DPONightDarkenService.get_instance().caption(rel)


@router.get("/dpo/night-darken/preview")
def preview(rel: str, strength: float = 0.0, noise: float = 0.0, hue: float = 0.0, saturation: float = 1.0):
    data = DPONightDarkenService.get_instance().preview(
        rel, {"strength": strength, "noise": noise, "hue": hue, "saturation": saturation}
    )
    if data is None:
        return Response(status_code=404, content="Not a candidate")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/dpo/night-darken/decide")
def decide(req: DecideRequest):
    adjust = req.adjust.model_dump() if req.adjust is not None else None
    return DPONightDarkenService.get_instance().decide(req.rel, req.state, adjust)


@router.post("/dpo/night-darken/undo")
def undo(req: UndoRequest):
    return DPONightDarkenService.get_instance().undo(req.rel)


@router.post("/dpo/night-darken/commit")
def commit(req: CommitRequest):
    return DPONightDarkenService.get_instance().commit(req.night_edits_dir, req.fraction)


@router.post("/dpo/night-darken/restore")
def restore(req: RestoreRequest):
    return DPONightDarkenService.get_instance().restore(req.commit_id)
