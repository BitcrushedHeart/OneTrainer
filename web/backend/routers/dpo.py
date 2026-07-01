from typing import Literal

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
    mode: Literal["selection", "swiss", "triage"] = "selection"


class SelectImageRequest(BaseModel):
    path: str


class FinalizeRequest(BaseModel):
    val_percentage: float = 0.0


class SwissVoteRequest(BaseModel):
    a: str
    b: str
    winner: Literal["a", "b", "tie"]


class SwissReorderRequest(BaseModel):
    order: list[str]


class SwissExportRequest(BaseModel):
    pair_count: int


class ConfirmPairRequest(BaseModel):
    continue_scoring: bool = False


class TriagePair(BaseModel):
    chosen: str
    rejected: str


class CommitPairsRequest(BaseModel):
    pairs: list[TriagePair]
    discard_rest: bool = False


class TriageAlignRequest(BaseModel):
    good: list[str]
    bad: list[str]


class BucketAnalysisRequest(BaseModel):
    concept_path: str
    batch_size: int
    target_resolutions: list[int]
    quantization: int


class ApplyCaptionRequest(BaseModel):
    chosen_image: str
    rejected_image: str
    caption: str


class RepairRejectedRequest(BaseModel):
    quality_floor: float = 0.85


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


@router.post("/dpo/check-caption-mismatches")
def check_caption_mismatches():
    service = DPOService.get_instance()
    return service.check_caption_mismatches()


@router.post("/dpo/correct-all-to-chosen")
def correct_all_to_chosen():
    service = DPOService.get_instance()
    return service.correct_all_captions_to_chosen()


@router.post("/dpo/apply-caption", response_model=ActionResponse)
def apply_caption(req: ApplyCaptionRequest):
    service = DPOService.get_instance()
    return ActionResponse(
        **service.apply_caption(
            req.chosen_image,
            req.rejected_image,
            req.caption,
        )
    )


@router.post("/dpo/repair-rejected")
def repair_rejected(req: RepairRejectedRequest):
    service = DPOService.get_instance()
    return service.repair_rejected(quality_floor=req.quality_floor)


@router.get("/dpo/repair-status")
def repair_status():
    service = DPOService.get_instance()
    return service.repair_status()


@router.post("/dpo/bucket-analysis")
def bucket_analysis(req: BucketAnalysisRequest):
    service = DPOService.get_instance()
    return service.bucket_analysis(
        concept_path=req.concept_path,
        batch_size=req.batch_size,
        target_resolutions=req.target_resolutions,
        quantization=req.quantization,
    )


# ---- Curation Session ----


@router.post("/dpo/session/start")
def start_session(req: StartSessionRequest):
    service = DPOService.get_instance()
    return service.start_session(req.source_folder, req.output_dir, req.pairs_per_group, req.mode)


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


@router.post("/dpo/session/confirm-pair")
def confirm_pair(req: ConfirmPairRequest):
    service = DPOService.get_instance()
    return service.confirm_pair(req.continue_scoring)


@router.post("/dpo/session/cancel-pair")
def cancel_pair():
    service = DPOService.get_instance()
    return service.cancel_pending_pair()


@router.post("/dpo/session/commit-pairs")
def commit_pairs(req: CommitPairsRequest):
    service = DPOService.get_instance()
    return service.commit_triage_pairs([(p.chosen, p.rejected) for p in req.pairs], discard_rest=req.discard_rest)


@router.post("/dpo/session/discard-group")
def discard_group():
    service = DPOService.get_instance()
    return service.discard_group()


@router.post("/dpo/session/triage-align")
def triage_align(req: TriageAlignRequest):
    service = DPOService.get_instance()
    return service.triage_align(req.good, req.bad)


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


# ---- Swiss tournament endpoints ----


@router.get("/dpo/session/swiss-state")
def swiss_state():
    service = DPOService.get_instance()
    return service.swiss_state()


@router.post("/dpo/session/swiss-vote")
def swiss_vote(req: SwissVoteRequest):
    service = DPOService.get_instance()
    return service.swiss_vote(req.a, req.b, req.winner)


@router.post("/dpo/session/swiss-finish-early")
def swiss_finish_early():
    service = DPOService.get_instance()
    return service.swiss_finish_early()


@router.get("/dpo/session/swiss-ranking")
def swiss_ranking():
    service = DPOService.get_instance()
    return service.swiss_ranking()


@router.post("/dpo/session/swiss-reorder")
def swiss_reorder(req: SwissReorderRequest):
    service = DPOService.get_instance()
    return service.swiss_reorder(req.order)


@router.post("/dpo/session/swiss-export")
def swiss_export(req: SwissExportRequest):
    service = DPOService.get_instance()
    return service.swiss_export(req.pair_count)
