from web.backend.services.dpo_saturation_service import DPOSaturationService

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(tags=["dpo-saturation"])


class StartRequest(BaseModel):
    # Current checkpoint (transformer safetensors); empty in adapter mode.
    current: str = ""
    # Origin (pre-DPO lineage base); empty = the config's transformer model.
    origin: str = ""
    # Adapter mode: origin = base without it, current = base with it.
    adapter: str = ""
    # Draws per pair from the training timestep distribution.
    draws: int = 8
    # Fixed noise levels "0.1,0.5,0.9" instead of --draws; each gets `seeds` noise draws.
    timesteps: str = ""
    seeds: int = 2
    seed_base: int = 1_000_000
    batch_size: int = 1
    # Latent cache to use; empty = the config's cache_dir.
    cache_dir: str = ""
    force_origin: bool = False
    # Only concepts whose name/path contains one of these; empty = all.
    concepts: list[str] = []


class SummaryRequest(BaseModel):
    concept_path: str
    # None = the concept's beta override, else the config's global beta.
    beta: float | None = None
    dead_weight: float = 0.05
    require_all_buckets: bool = False
    min_draws: int = 1
    tails: int = 24
    bins: int = 40


class PairRequest(BaseModel):
    concept_path: str
    key: str
    beta: float | None = None


class RetireRequest(BaseModel):
    concept_path: str
    # Explicit keys, or None = every pair saturated under the rule below.
    keys: list[str] | None = None
    beta: float | None = None
    dead_weight: float = 0.05
    require_all_buckets: bool = False
    min_draws: int = 1
    reason: str = ""
    dry_run: bool = False


class RestoreRequest(BaseModel):
    concept_path: str
    # None = every retired pair of the concept.
    keys: list[str] | None = None


@router.get("/dpo/saturation/concepts")
def concepts():
    """Configured DPO concepts and the state of their saturation tables."""
    return DPOSaturationService.get_instance().concepts()


@router.post("/dpo/saturation/start")
def start(req: StartRequest):
    return DPOSaturationService.get_instance().start(
        current=req.current,
        origin=req.origin,
        adapter=req.adapter,
        draws=req.draws,
        timesteps=req.timesteps,
        seeds=req.seeds,
        seed_base=req.seed_base,
        batch_size=req.batch_size,
        cache_dir=req.cache_dir,
        force_origin=req.force_origin,
        concepts=req.concepts,
    )


@router.get("/dpo/saturation/status")
def status():
    return DPOSaturationService.get_instance().status()


@router.post("/dpo/saturation/cancel")
def cancel():
    return DPOSaturationService.get_instance().cancel()


@router.post("/dpo/saturation/summary")
def summary(req: SummaryRequest):
    """Saturation summary of one concept at a beta and dead-weight budget."""
    return DPOSaturationService.get_instance().summary(
        req.concept_path,
        beta=req.beta,
        dead_weight=req.dead_weight,
        require_all_buckets=req.require_all_buckets,
        min_draws=req.min_draws,
        tails=req.tails,
        bins=req.bins,
    )


@router.post("/dpo/saturation/pair")
def pair(req: PairRequest):
    return DPOSaturationService.get_instance().pair(req.concept_path, req.key, beta=req.beta)


@router.post("/dpo/saturation/retire")
def retire(req: RetireRequest):
    """Move pairs out of the training tree (reversible)."""
    return DPOSaturationService.get_instance().retire(
        req.concept_path,
        keys=req.keys,
        beta=req.beta,
        dead_weight=req.dead_weight,
        require_all_buckets=req.require_all_buckets,
        min_draws=req.min_draws,
        reason=req.reason,
        dry_run=req.dry_run,
    )


@router.post("/dpo/saturation/restore")
def restore(req: RestoreRequest):
    return DPOSaturationService.get_instance().restore(req.concept_path, req.keys)


@router.get("/dpo/saturation/retired")
def retired(concept_path: str):
    return DPOSaturationService.get_instance().retired(concept_path)


@router.get("/dpo/saturation/image")
def serve_image(path: str):
    abs_path = DPOSaturationService.get_instance().serve_image(path)
    if abs_path is None:
        return {"ok": False, "error": "Image not found"}
    return FileResponse(abs_path)
