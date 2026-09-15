from typing import Literal

from web.backend.services.dpo_replace_service import DPOReplacePairsService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["dpo-replace"])


class StartRequest(BaseModel):
    # A DPO concept folder with manifest.json, chosen/ and rejected/.
    concept_dir: str
    # SwarmUI's raw output root: where fresh generations land and where the
    # settings of the most recent generation are read from.
    generated_folder: str
    base_url: str = "http://localhost:7801"
    token: str = ""
    # stale_first: corrected captions first, then oldest rejected generation.
    order: Literal["stale_first", "oldest", "least_realistic", "path"] = "stale_first"
    # Let an image other than the current chosen become the new chosen (the
    # old chosen may become rejected). Off: only the rejected side is picked.
    allow_promote: bool = True
    target_resolution: int = 1024
    quantization: int = 64


class GenerateRequest(BaseModel):
    # How many of the next-ranked pairs to regenerate, unless pair_ids is given.
    count: int = 10
    pair_ids: list[int] | None = None
    # Fresh images per pair; the most photoreal one is proposed.
    images_per_pair: int = 2
    concurrency: int = 3
    # Apply a scored suggestion that keeps the chosen side without review;
    # anything that would move the chosen side always waits for a person.
    auto_apply: bool = False


class ApplyRequest(BaseModel):
    # Two images of the group on screen; omitted = the suggestion.
    chosen_path: str | None = None
    rejected_path: str | None = None


@router.get("/dpo/replace/concepts")
def concepts():
    """Configured DPO concept folders that carry a manifest."""
    return DPOReplacePairsService.get_instance().concepts()


@router.post("/dpo/replace/start")
def start(req: StartRequest):
    return DPOReplacePairsService.get_instance().start(
        req.concept_dir,
        req.generated_folder,
        base_url=req.base_url,
        token=req.token,
        order=req.order,
        allow_promote=req.allow_promote,
        target_resolution=req.target_resolution,
        quantization=req.quantization,
    )


@router.get("/dpo/replace/status")
def status():
    return DPOReplacePairsService.get_instance().status()


@router.get("/dpo/replace/pairs")
def pairs(offset: int = 0, limit: int = 200):
    """The ranked pair list and the per-model summary."""
    return DPOReplacePairsService.get_instance().pairs(offset=offset, limit=limit)


@router.get("/dpo/replace/params")
def params():
    """The settings the next batch would use (the user's last generation)."""
    return DPOReplacePairsService.get_instance().params()


@router.post("/dpo/replace/generate")
def generate(req: GenerateRequest):
    return DPOReplacePairsService.get_instance().generate(
        count=req.count,
        pair_ids=req.pair_ids,
        images_per_pair=req.images_per_pair,
        concurrency=req.concurrency,
        auto_apply=req.auto_apply,
    )


@router.post("/dpo/replace/next")
def next_proposal():
    return DPOReplacePairsService.get_instance().next()


@router.get("/dpo/replace/current")
def current():
    return DPOReplacePairsService.get_instance().current()


@router.post("/dpo/replace/apply")
def apply(req: ApplyRequest = ApplyRequest()):  # noqa: B008 — pydantic default body
    """Make the two picked images the pair's chosen and rejected sides."""
    return DPOReplacePairsService.get_instance().apply(req.chosen_path, req.rejected_path)


@router.post("/dpo/replace/keep")
def keep():
    """Leave the pair as it is."""
    return DPOReplacePairsService.get_instance().keep()


@router.post("/dpo/replace/undo")
def undo():
    return DPOReplacePairsService.get_instance().undo()


@router.post("/dpo/replace/cancel")
def cancel():
    return DPOReplacePairsService.get_instance().cancel()
