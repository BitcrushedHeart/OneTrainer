from typing import Any, Literal

from modules.util.dpo_triage_fix_util import MAIN_SESSION
from web.backend.services.dpo_triage_fix_service import DPOTriageFixService

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["dpo-triage-fix"])


def _session(session: str) -> DPOTriageFixService:
    """The Triage & Fix session a request is for (the ``session`` query
    parameter; the main session when it is left out)."""
    try:
        return DPOTriageFixService.session(session)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex


class FixModel(BaseModel):
    # A triage category id: faces, hands_feet, limbs, text, background_people, other.
    id: str
    # The segment target; blank takes the category's. Required for "other".
    target: str = ""
    creativity: float | None = None
    threshold: float | None = None
    # Leads the segment's sub-prompt (Mode A) and is the whole prompt of the
    # edit fallback (Mode B). Blank = the category's default.
    instruction: str = ""
    include_caption: bool | None = None

    def payload(self) -> dict:
        out: dict[str, Any] = {"id": self.id, "target": self.target, "instruction": self.instruction}
        if self.creativity is not None:
            out["creativity"] = self.creativity
        if self.threshold is not None:
            out["threshold"] = self.threshold
        if self.include_caption is not None:
            out["include_caption"] = self.include_caption
        return out


class CategoryModel(BaseModel):
    id: str
    target: str = ""
    instruction: str = ""
    creativity: float | None = None
    threshold: float | None = None
    include_caption: bool | None = None
    # Detection ladder for yolo targets: retry at low_threshold, then this
    # CLIPSeg word. None = keep the category default.
    fallback_target: str | None = None
    low_threshold: float | None = None


class LoraModel(BaseModel):
    name: str
    weight: float = 1.0


class StartRequest(BaseModel):
    # An exported DPO dataset: manifest.json, chosen/ and rejected/.
    concept_dir: str
    base_url: str = "http://localhost:7801"
    token: str = ""
    # The fixer model as SwarmUI names it (Flux.2 Klein).
    model: str
    # Klein (distilled) is a 4-step, CFG-1 model.
    steps: int = 4
    cfgscale: float = 1.0
    # Blank / None = SwarmUI's defaults for the model class.
    sampler: str = ""
    scheduler: str = ""
    sigmashift: float | None = None
    # SwarmUI "Color Correction Behavior" for the pasted-back crop: None / Uniform / Linear / Linear2.
    color_correction: str = "None"
    # SwarmUI "Crop Reference Strength" (local patch, Flux.2 / Klein): 1 = the original crop as reference throughout.
    crop_reference: float = 1.0
    # LoRAs / adapters applied on top of the fixer model, named as SwarmUI lists them.
    loras: list[LoraModel] = []
    options_per_image: int = 2
    concurrency: int = 2
    segment_params: dict[str, Any] = {}
    # Quick Mask knobs: maskshrinkgrow (context margin px), maskblur, maskgrow.
    mask_params: dict[str, Any] = {}
    categories: list[CategoryModel] = []
    prompt_template: str = ""
    # Optional second dataset receiving (fixed = chosen, original = rejected).
    fix_pair_dir: str = ""
    shuffle_seed: int | None = None
    detector: str = ""
    # Start with sending held: confirms queue up but nothing goes to SwarmUI yet.
    hold: bool = False
    # What the session's tab is called; shown in the other sessions' messages.
    name: str = ""


class ConfirmRequest(BaseModel):
    stem: str
    fixes: list[FixModel]
    # From the second round: a fixed, skipped or kept image is taken up again.
    second_round: bool = False


class ConfirmManyRequest(BaseModel):
    stems: list[str]
    fixes: list[FixModel]


class ConfirmMaskRequest(BaseModel):
    stem: str
    # The painted mask as a data URI (white / opaque = repaint).
    mask: str
    instruction: str = ""
    creativity: float | None = None
    second_round: bool = False
    # Overrides the session's colour correction for this image; None = the session's.
    color_correction: str | None = None
    crop_reference: float | None = None


class ConfirmAllRequest(BaseModel):
    mode: Literal["default", "detected"] = "default"
    fixes: list[FixModel] = []


class HoldRequest(BaseModel):
    held: bool


class StemRequest(BaseModel):
    stem: str


class StemsRequest(BaseModel):
    stems: list[str]


class PickRequest(BaseModel):
    stem: str
    k: int


class RequeueRequest(BaseModel):
    stem: str
    fixes: list[FixModel] | None = None
    # A plain re-roll only changes the seed. ``lower`` also turns every fix's
    # creativity down one notch (one sampling step unless ``lower_by`` says).
    lower: bool = False
    lower_by: float | None = None


def _categories(models: list[CategoryModel]) -> list[dict]:
    out = []
    for cat in models:
        item: dict[str, Any] = {"id": cat.id, "target": cat.target, "instruction": cat.instruction}
        if cat.creativity is not None:
            item["creativity"] = cat.creativity
        if cat.threshold is not None:
            item["threshold"] = cat.threshold
        if cat.include_caption is not None:
            item["include_caption"] = cat.include_caption
        if cat.fallback_target is not None:
            item["fallback_target"] = cat.fallback_target
        if cat.low_threshold is not None:
            item["low_threshold"] = cat.low_threshold
        out.append(item)
    return out


@router.get("/dpo/triage-fix/sessions")
def sessions():
    return DPOTriageFixService.overview()


@router.get("/dpo/triage-fix/concepts")
def concepts(session: str = MAIN_SESSION):
    return _session(session).concepts()


@router.get("/dpo/triage-fix/inspect")
def inspect(concept_dir: str, session: str = MAIN_SESSION):
    return _session(session).inspect(concept_dir)


@router.post("/dpo/triage-fix/start")
def start(req: StartRequest, session: str = MAIN_SESSION):
    settings = {
        "model": req.model,
        "steps": req.steps,
        "cfgscale": req.cfgscale,
        "sampler": req.sampler,
        "scheduler": req.scheduler,
        "sigmashift": req.sigmashift,
        "color_correction": req.color_correction,
        "crop_reference": req.crop_reference,
        "loras": [entry.model_dump() for entry in req.loras],
        "options_per_image": req.options_per_image,
        "concurrency": req.concurrency,
        "segment_params": req.segment_params,
        "mask_params": req.mask_params,
        "categories": _categories(req.categories),
        "prompt_template": req.prompt_template,
    }
    return _session(session).start(
        req.concept_dir,
        base_url=req.base_url,
        token=req.token,
        settings=settings,
        fix_pair_dir=req.fix_pair_dir,
        shuffle_seed=req.shuffle_seed,
        detector=req.detector,
        hold=req.hold,
        name=req.name,
    )


@router.get("/dpo/triage-fix/status")
def status(session: str = MAIN_SESSION):
    return _session(session).status()


@router.post("/dpo/triage-fix/hold")
def hold(req: HoldRequest, session: str = MAIN_SESSION):
    return _session(session).set_hold(req.held)


@router.get("/dpo/triage-fix/items")
def items(session: str = MAIN_SESSION):
    return _session(session).items()


@router.post("/dpo/triage-fix/confirm")
def confirm(req: ConfirmRequest, session: str = MAIN_SESSION):
    return _session(session).confirm(req.stem, [f.payload() for f in req.fixes], second_round=req.second_round)


@router.post("/dpo/triage-fix/confirm-mask")
def confirm_mask(req: ConfirmMaskRequest, session: str = MAIN_SESSION):
    return _session(session).confirm_mask(
        req.stem,
        req.mask,
        instruction=req.instruction,
        creativity=req.creativity,
        second_round=req.second_round,
        color_correction=req.color_correction,
        crop_reference=req.crop_reference,
    )


@router.post("/dpo/triage-fix/confirm-all")
def confirm_all(req: ConfirmAllRequest, session: str = MAIN_SESSION):
    return _session(session).confirm_all(req.mode, [f.payload() for f in req.fixes])


@router.post("/dpo/triage-fix/skip")
def skip(req: StemRequest, session: str = MAIN_SESSION):
    return _session(session).skip(req.stem)


@router.post("/dpo/triage-fix/skip-many")
def skip_many(req: StemsRequest, session: str = MAIN_SESSION):
    return _session(session).skip_many(req.stems)


@router.post("/dpo/triage-fix/confirm-many")
def confirm_many_first(req: ConfirmManyRequest, session: str = MAIN_SESSION):
    return _session(session).confirm_many(req.stems, [f.payload() for f in req.fixes], second_round=False)


@router.post("/dpo/triage-fix/undo-triage")
def undo_triage(req: StemRequest, session: str = MAIN_SESSION):
    return _session(session).undo_triage(req.stem)


@router.post("/dpo/triage-fix/next-ready")
def next_ready(session: str = MAIN_SESSION):
    return _session(session).next_ready()


@router.post("/dpo/triage-fix/pick")
def pick(req: PickRequest, session: str = MAIN_SESSION):
    return _session(session).pick(req.stem, req.k)


@router.post("/dpo/triage-fix/keep")
def keep(req: StemRequest, session: str = MAIN_SESSION):
    return _session(session).keep(req.stem)


@router.post("/dpo/triage-fix/requeue")
def requeue(req: RequeueRequest, session: str = MAIN_SESSION):
    fixes = [f.payload() for f in req.fixes] if req.fixes else None
    return _session(session).requeue(req.stem, fixes, lower=req.lower, lower_by=req.lower_by)


@router.post("/dpo/triage-fix/delete-pair")
def delete_pair(req: StemRequest, session: str = MAIN_SESSION):
    return _session(session).delete_pair(req.stem)


@router.post("/dpo/triage-fix/retriage")
def retriage(req: StemRequest, session: str = MAIN_SESSION):
    return _session(session).retriage(req.stem)


@router.post("/dpo/triage-fix/second/confirm-many")
def confirm_many(req: ConfirmManyRequest, session: str = MAIN_SESSION):
    return _session(session).confirm_many(req.stems, [f.payload() for f in req.fixes])


@router.post("/dpo/triage-fix/second/skip")
def skip_second(req: StemsRequest, session: str = MAIN_SESSION):
    return _session(session).skip_second(req.stems)


@router.post("/dpo/triage-fix/second/clear")
def clear_second(session: str = MAIN_SESSION):
    return _session(session).clear_second()


@router.post("/dpo/triage-fix/second/undo")
def undo_second(req: StemRequest, session: str = MAIN_SESSION):
    return _session(session).undo_second(req.stem)


@router.post("/dpo/triage-fix/add-fix-pairs")
def add_fix_pairs(req: StemsRequest, session: str = MAIN_SESSION):
    return _session(session).add_fix_pairs(req.stems)


@router.post("/dpo/triage-fix/undo-decision")
def undo_decision(session: str = MAIN_SESSION):
    return _session(session).undo_decision()


@router.post("/dpo/triage-fix/cancel")
def cancel(session: str = MAIN_SESSION):
    return _session(session).cancel()
