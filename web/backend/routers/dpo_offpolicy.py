from typing import Literal

from web.backend.services.dpo_offpolicy_service import DPOOffPolicyService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["dpo-offpolicy"])


class StartRequest(BaseModel):
    generated_folder: str
    real_folder: str
    output_dir: str
    # Allowed aspect mismatch between photo and generation, as a fraction.
    # The trainer cover-scales the generation (aspect kept) and crops it to the
    # photo's crop box, so this bounds how much of the generation's long axis
    # is trimmed away.
    aspect_tolerance: float = 0.15
    # How much the photo may be upscaled relative to its generation, per
    # axis, before it is excluded (1.10 = 10%).
    max_upscale: float = 1.10
    # Pixel budget (side of a square) for the suggested generation size.
    base_resolution: int = 1024
    # Candidate presentation order: "random" (default) or "path" (sorted).
    order: Literal["random", "path"] = "random"


@router.post("/dpo/offpolicy/start")
def start(req: StartRequest):
    return DPOOffPolicyService.get_instance().start(
        req.generated_folder,
        req.real_folder,
        req.output_dir,
        aspect_tolerance=req.aspect_tolerance,
        max_upscale=req.max_upscale,
        base_resolution=req.base_resolution,
        order=req.order,
    )


@router.get("/dpo/offpolicy/status")
def status():
    return DPOOffPolicyService.get_instance().status()


@router.post("/dpo/offpolicy/next")
def next_candidate():
    return DPOOffPolicyService.get_instance().next()


@router.get("/dpo/offpolicy/current")
def current_candidate():
    """The candidate on screen, for a UI that reloaded mid-session."""
    return DPOOffPolicyService.get_instance().current()


@router.post("/dpo/offpolicy/confirm")
def confirm():
    """Caption fits: export photo-as-chosen / generation-as-rejected."""
    return DPOOffPolicyService.get_instance().confirm()


@router.post("/dpo/offpolicy/skip")
def skip():
    return DPOOffPolicyService.get_instance().skip()


@router.post("/dpo/offpolicy/skip-forever")
def skip_forever():
    """Never offer this photo again for this output folder (persisted)."""
    return DPOOffPolicyService.get_instance().skip_forever()


@router.post("/dpo/offpolicy/remove-caption")
def remove_caption():
    """Caption is wrong: remove that line from the photo's sidecar (photo kept)."""
    return DPOOffPolicyService.get_instance().remove_caption()


class SuggestRequest(BaseModel):
    # Trainer bucket label ("3:2") to draw from; empty/None for any bucket.
    aspect_bucket: str | None = None
    count: int = 1
    # Comma-separated terms that must all appear in the caption line.
    keywords: str = ""


@router.post("/dpo/offpolicy/reroll")
def reroll():
    """Show a different generation for the photo on screen."""
    return DPOOffPolicyService.get_instance().reroll()


@router.post("/dpo/offpolicy/undo")
def undo():
    """Reverse the last confirm / skip / delete and show that candidate again."""
    return DPOOffPolicyService.get_instance().undo()


@router.post("/dpo/offpolicy/suggest")
def suggest(req: SuggestRequest = SuggestRequest()):  # noqa: B008 — pydantic default body
    """Random captions with no usable generation yet, the size to generate
    each at, and the per-bucket counts still open."""
    return DPOOffPolicyService.get_instance().suggest(
        aspect_bucket=req.aspect_bucket, count=req.count, keywords=req.keywords
    )


@router.get("/dpo/offpolicy/buckets")
def buckets(keywords: str = ""):
    """Trainer buckets that still have photos to generate for, optionally
    counting only photos with a caption line containing every keyword."""
    return DPOOffPolicyService.get_instance().buckets(keywords=keywords)


class LookupRequest(BaseModel):
    filename: str


@router.post("/dpo/offpolicy/lookup")
def lookup(req: LookupRequest):
    """Indexed photos matching a file name, with the size to generate at."""
    return DPOOffPolicyService.get_instance().lookup(req.filename)


@router.post("/dpo/offpolicy/cancel")
def cancel():
    return DPOOffPolicyService.get_instance().cancel()
