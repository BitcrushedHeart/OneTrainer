from typing import Literal

from web.backend.services.swarm_generate_service import SwarmGenerateService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["swarm-generate"])


class CheckRequest(BaseModel):
    base_url: str = "http://localhost:7801"
    token: str = ""


class LoraEntry(BaseModel):
    name: str
    weight: float = 1.0


class GenerateRequest(BaseModel):
    real_folder: str
    generated_folder: str
    output_dir: str
    model: str = ""
    base_url: str = "http://localhost:7801"
    token: str = ""
    # How many distinct captions to generate for, and how many images each.
    captions: int = 10
    images_per_caption: int = 1
    keywords: str = ""
    # Tag lists make poor realism prompts; short captions likewise.
    exclude_tag_style: bool = True
    min_caption_words: int = 0
    steps: int | None = None
    cfg_scale: float | None = None
    negative_prompt: str = ""
    loras: list[LoraEntry] = []
    # "last" = reuse the newest generation's settings (sampler, sigma shift,
    # LoRAs, ...), "preset" = a saved SwarmUI preset, "manual" = only the
    # fields above.
    param_source: Literal["last", "preset", "manual"] = "last"
    preset_name: str = ""
    # Requests kept in flight so SwarmUI always has the next one queued.
    concurrency: int = 3
    aspect_tolerance: float = 0.15
    target_resolution: int = 1024
    quantization: int = 64
    # Plan the batch and return it without asking SwarmUI for anything.
    dry_run: bool = False


@router.post("/swarm/check")
def check(req: CheckRequest):
    """Confirm SwarmUI is reachable and list its models."""
    return SwarmGenerateService.get_instance().check(req.base_url, req.token)


@router.post("/swarm/generate")
def generate(req: GenerateRequest):
    """Start a background generation run (or plan one with dry_run)."""
    return SwarmGenerateService.get_instance().start(
        real_folder=req.real_folder,
        generated_folder=req.generated_folder,
        output_dir=req.output_dir,
        model=req.model,
        base_url=req.base_url,
        token=req.token,
        captions=req.captions,
        images_per_caption=req.images_per_caption,
        keywords=req.keywords,
        exclude_tag_style=req.exclude_tag_style,
        min_caption_words=req.min_caption_words,
        steps=req.steps,
        cfg_scale=req.cfg_scale,
        negative_prompt=req.negative_prompt,
        loras=[entry.model_dump() for entry in req.loras],
        param_source=req.param_source,
        preset_name=req.preset_name,
        concurrency=req.concurrency,
        aspect_tolerance=req.aspect_tolerance,
        target_resolution=req.target_resolution,
        quantization=req.quantization,
        dry_run=req.dry_run,
    )


@router.get("/swarm/status")
def status():
    return SwarmGenerateService.get_instance().status()


@router.post("/swarm/cancel")
def cancel():
    """Stop the current run and drop everything queued behind it."""
    return SwarmGenerateService.get_instance().cancel()


@router.post("/swarm/queue/clear")
def clear_queue():
    """Drop the waiting runs; the one in flight finishes."""
    return SwarmGenerateService.get_instance().clear_queue()


@router.get("/swarm/last-params")
def last_params(generated_folder: str):
    """The settings the most recent generation used, as this tool would reuse them."""
    return SwarmGenerateService.get_instance().preview_params(generated_folder)


@router.get("/swarm/buckets")
def buckets(target_resolution: int = 1024, quantization: int = 64):
    """The trainer bucket sizes generations are requested at."""
    from modules.util.swarm_generate_util import bucket_sizes

    return {"ok": True, "buckets": [list(size) for size in bucket_sizes(target_resolution, quantization)]}
