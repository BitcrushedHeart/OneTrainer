from typing import Literal

from modules.util.dpo_character_util import DEFAULT_DPO_RESOLUTION
from web.backend.services.character_dpo_service import DEFAULT_MASK_MODEL, CharacterDPOService

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(tags=["dpo-character"])

# ``models_dir`` on every route here is one folder or several separated by
# ``;``, searched in order for each character's LoRA (a best anywhere beats a
# patience-second anywhere). Blank means the service default: the repo's
# ``models`` folder, then the finished-LoRA folder.


@router.get("/dpo/character/list")
def list_characters(
    root: str = "",
    models_dir: str | None = None,
    swarm_url: str | None = None,
    swarm_token: str | None = None,
):
    return CharacterDPOService.get_instance().list_characters(
        root, models_dir=models_dir or None, swarm_url=swarm_url or None, swarm_token=swarm_token or None
    )


class PublishLoraRequest(BaseModel):
    name: str
    root: str
    models_dir: str | None = None
    swarm_lora_dir: str
    force: bool = False


@router.post("/dpo/character/lora/publish")
def publish_lora(req: PublishLoraRequest):
    return CharacterDPOService.get_instance().publish_lora(
        req.name, req.root, req.swarm_lora_dir, models_dir=req.models_dir or None, force=req.force
    )


class ImportScanRequest(BaseModel):
    name: str
    root: str
    concept_files: list[str]
    models_dir: str | None = None


@router.post("/dpo/character/import/scan")
def import_scan(req: ImportScanRequest):
    return CharacterDPOService.get_instance().scan_imports(
        req.name, req.root, req.concept_files, models_dir=req.models_dir or None
    )


class ImportPick(BaseModel):
    concept_dir: str
    pair_id: int


class ImportApplyRequest(BaseModel):
    name: str
    root: str
    picks: list[ImportPick]


@router.post("/dpo/character/import/apply")
def import_apply(req: ImportApplyRequest):
    return CharacterDPOService.get_instance().apply_imports(
        req.name, req.root, [pick.model_dump() for pick in req.picks]
    )


class FinalizeRequest(BaseModel):
    name: str
    root: str
    val_pct: float = 10.0
    mask_model: str = DEFAULT_MASK_MODEL
    mask_options: dict | None = None
    # Resolution override written on the four DPO concepts (the pairs' own
    # size; the global run trains lower).
    dpo_resolution: str = DEFAULT_DPO_RESOLUTION


@router.post("/dpo/character/finalize")
def finalize(req: FinalizeRequest):
    result = CharacterDPOService.get_instance().finalize(
        req.name,
        req.root,
        val_pct=req.val_pct,
        mask_model=req.mask_model,
        mask_options=req.mask_options,
        dpo_resolution=req.dpo_resolution,
    )
    if result.get("conflict"):
        raise HTTPException(status_code=409, detail=result.get("error"))
    return result


@router.get("/dpo/character/finalize/status")
def finalize_status(name: str, root: str):
    return CharacterDPOService.get_instance().finalize_status(name, root)


class GenerateRequest(BaseModel):
    names: list[str]
    root: str
    models_dir: str | None = None
    swarm_url: str
    swarm_token: str = ""
    # SwarmUI's LoRA folder: each character's LoRA is published there first.
    swarm_lora_dir: str
    # Generation settings come from the newest image here, minus its LoRAs.
    param_folder: str
    images_per_caption: int = 2
    # 0 = every caption sidecar in the character's train folder.
    captions: int = 0
    concurrency: int | None = None


@router.post("/dpo/character/generate")
def generate(req: GenerateRequest):
    """Queue one SwarmUI negatives run per character with its own LoRA.
    Progress is the shared ``GET /swarm/status``."""
    return CharacterDPOService.get_instance().generate_negatives(
        req.names,
        req.root,
        swarm_url=req.swarm_url,
        swarm_lora_dir=req.swarm_lora_dir,
        param_folder=req.param_folder,
        models_dir=req.models_dir or None,
        swarm_token=req.swarm_token,
        images_per_caption=req.images_per_caption,
        captions=req.captions,
        concurrency=req.concurrency,
    )


class PrepareRequest(BaseModel):
    names: list[str]
    root: str
    models_dir: str | None = None
    val_pct: float = 10.0
    mask_model: str = DEFAULT_MASK_MODEL
    mask_options: dict | None = None
    dpo_resolution: str = DEFAULT_DPO_RESOLUTION


@router.post("/dpo/character/prepare")
def prepare(req: PrepareRequest):
    """Finalize (split + masks + concepts) every named character in one background job."""
    result = CharacterDPOService.get_instance().prepare(
        req.names,
        req.root,
        models_dir=req.models_dir or None,
        val_pct=req.val_pct,
        mask_model=req.mask_model,
        mask_options=req.mask_options,
        dpo_resolution=req.dpo_resolution,
    )
    if result.get("conflict"):
        raise HTTPException(status_code=409, detail=result.get("error"))
    return result


@router.get("/dpo/character/prepare/status")
def prepare_status():
    return CharacterDPOService.get_instance().prepare_status()


@router.get("/dpo/character/train-captions")
def train_captions(name: str, root: str):
    """Caption sidecars in the character's train folder (what a generate run covers)."""
    return CharacterDPOService.train_captions(name, root)


class QueueBuildRequest(BaseModel):
    names: list[str]
    root: str
    models_dir: str | None = None
    add_on_weight: float = 0.7
    dpo_beta: float = 200.0
    mode: Literal["preview", "add", "export"] = "preview"


@router.post("/dpo/character/queue/build")
def queue_build(req: QueueBuildRequest):
    return CharacterDPOService.get_instance().build_queue(
        req.names,
        req.root,
        models_dir=req.models_dir or None,
        add_on_weight=req.add_on_weight,
        mode=req.mode,
        dpo_beta=req.dpo_beta,
    )


@router.get("/dpo/character/image")
def serve_image(path: str):
    abs_path = CharacterDPOService.get_instance().serve_image(path)
    if abs_path is None:
        raise HTTPException(status_code=403, detail="Image is not under a character root or a scanned concept")
    return FileResponse(abs_path)
