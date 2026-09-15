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
    # Presentation order of the scanned groups. The realism orders rank on
    # cached P(AI) scores (largest_gap on the spread between a group's best
    # and worst image) and fall back to the shuffle when scoring is off.
    group_order: Literal["random", "least_realistic", "most_realistic", "largest_gap"] = "most_realistic"
    # Comma-separated caption keywords; every term must appear in a group's
    # prompt for the group to be shown. Empty means no filtering.
    caption_filter: str = ""
    # Smallest group to present. 2 (the floor a pair needs) shows everything;
    # raise it to skip prompts with only a handful of generations.
    min_images_per_group: int = 2


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
    discard_rest: bool = False


class ConfirmPairRequest(BaseModel):
    continue_scoring: bool = False
    discard_rest: bool = False


class TriagePair(BaseModel):
    chosen: str
    rejected: str


class CommitPairsRequest(BaseModel):
    pairs: list[TriagePair]
    discard_rest: bool = False


class TriageAlignRequest(BaseModel):
    good: list[str]
    bad: list[str]


class RealismScoresRequest(BaseModel):
    paths: list[str]


class ScoreBreakdownRequest(BaseModel):
    paths: list[str]


class RealismDeviceRequest(BaseModel):
    # "off" is not a device: it disables scoring entirely, for when the GPU is
    # busy and even the CPU pass is not wanted. Kept on this one setting so the
    # service has a single thing to consult before touching the model.
    device: Literal["auto", "cpu", "off"] = "auto"


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
    return service.start_session(
        req.source_folder,
        req.output_dir,
        req.pairs_per_group,
        req.mode,
        group_order=req.group_order,
        caption_filter=req.caption_filter,
        min_images_per_group=req.min_images_per_group,
    )


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
    return service.confirm_pair(req.continue_scoring, discard_rest=req.discard_rest)


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


@router.post("/dpo/session/realism-scores")
def realism_scores(req: RealismScoresRequest):
    service = DPOService.get_instance()
    return service.realism_scores(req.paths)


@router.post("/dpo/session/score-breakdown")
def score_breakdown(req: ScoreBreakdownRequest):
    """Every scorer's verdict per image of the current group: realism,
    prompt adherence (with its propositions), age agreement, scalar quality
    and the combined score. Feeds the per-tile panel in Selection mode."""
    service = DPOService.get_instance()
    return service.score_breakdown(req.paths)


@router.post("/dpo/session/score-backfill")
def start_score_backfill():
    """Score the whole source tree on every axis once so Selection never
    loads a model.

    Returns immediately; poll ``/dpo/session/score-backfill/status``. Safe to
    re-run — anything already cached under the current model versions is a
    hit, so a second pass only costs the images added since the first.
    """
    service = DPOService.get_instance()
    return service.start_score_backfill()


@router.get("/dpo/session/score-backfill/status")
def score_backfill_status():
    service = DPOService.get_instance()
    return service.score_backfill_status()


@router.post("/dpo/session/score-backfill/stop")
def stop_score_backfill():
    service = DPOService.get_instance()
    return service.stop_score_backfill()


@router.get("/dpo/realism/device")
def realism_device():
    """Where the realism head currently runs: ``auto`` (CUDA when usable),
    ``cpu``, or ``off`` when scoring is disabled."""
    service = DPOService.get_instance()
    return service.realism_device()


@router.post("/dpo/realism/device")
def set_realism_device(req: RealismDeviceRequest):
    """Move scoring on or off the GPU, or turn it off altogether.

    Applies from the next scoring call onwards, so a backfill already in flight
    switches at its next chunk rather than having to be restarted. Not tied to a
    curation session — it can be set before one starts, which is how the setup
    screen's toggle keeps a whole session off the model.
    """
    service = DPOService.get_instance()
    return service.set_realism_device(req.device)


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
    return service.swiss_export(req.pair_count, discard_rest=req.discard_rest)
