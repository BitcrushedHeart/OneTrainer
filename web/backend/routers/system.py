from web.backend.services.monitor_service import MonitorService

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/system", tags=["system"])


class GpuMetrics(BaseModel):
    index: int
    name: str
    vram_used_mb: float
    vram_total_mb: float
    vram_percent: float
    temperature: float | None = None
    utilization: float | None = None


class SystemMetricsResponse(BaseModel):
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpus: list[GpuMetrics]


class GpuInfo(BaseModel):
    index: int
    name: str
    vram_total_mb: float


class SystemInfoResponse(BaseModel):
    cpu_count: int
    cpu_count_physical: int | None = None
    ram_total_gb: float
    gpus: list[GpuInfo]


@router.get("/metrics", response_model=SystemMetricsResponse)
def get_metrics():
    monitor = MonitorService.get_instance()
    return SystemMetricsResponse(**monitor.get_metrics())


@router.get("/info", response_model=SystemInfoResponse)
def get_info():
    monitor = MonitorService.get_instance()
    return SystemInfoResponse(**monitor.get_system_info())


class CacheStatusResponse(BaseModel):
    cache_dir: str
    exists: bool
    size_mb: float
    file_count: int


class CacheActionResponse(BaseModel):
    ok: bool
    error: str | None = None


@router.get("/cache/status", response_model=CacheStatusResponse)
def get_cache_status():
    import os

    from web.backend.services.config_service import ConfigService

    config_service = ConfigService.get_instance()
    config = config_service.get_config_for_training()
    cache_dir = getattr(config, "cache_dir", "") or "workspace-cache/run"

    if not os.path.isdir(cache_dir):
        return CacheStatusResponse(cache_dir=cache_dir, exists=False, size_mb=0, file_count=0)

    total_size = 0
    file_count = 0
    for dirpath, _dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    return CacheStatusResponse(
        cache_dir=cache_dir,
        exists=True,
        size_mb=round(total_size / (1024 * 1024), 2),
        file_count=file_count,
    )


@router.post("/cache/clear", response_model=CacheActionResponse)
def clear_cache():
    import os
    import shutil

    from web.backend.services.config_service import ConfigService

    config_service = ConfigService.get_instance()
    config = config_service.get_config_for_training()
    cache_dir = getattr(config, "cache_dir", "") or "workspace-cache/run"

    if not os.path.isdir(cache_dir):
        return CacheActionResponse(ok=True)

    try:
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        return CacheActionResponse(ok=True)
    except Exception as e:
        return CacheActionResponse(ok=False, error=str(e))


class GcPreviewResponse(BaseModel):
    ok: bool
    text_orphans: int = 0
    text_bytes: float = 0
    image_orphans: int = 0
    image_bytes: float = 0
    total_files: int = 0
    total_mb: float = 0
    error: str | None = None


@router.post("/cache/gc-preview", response_model=GcPreviewResponse)
def cache_gc_preview():
    import os

    from web.backend.services.config_service import ConfigService

    config_service = ConfigService.get_instance()
    config = config_service.get_config_for_training()
    cache_dirs = [d for d in _gc_cache_dirs(config) if os.path.isdir(d)]

    if not cache_dirs:
        return GcPreviewResponse(ok=True)

    try:
        from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

        text_orphans = text_bytes = image_orphans = image_bytes = 0
        for cache_dir in cache_dirs:
            text_stats = SmartDiskCache.gc_preview(os.path.join(cache_dir, "text"))
            image_stats = SmartDiskCache.gc_preview(os.path.join(cache_dir, "image"))
            text_orphans += text_stats["orphan_count"]
            text_bytes += text_stats["orphan_bytes"]
            image_orphans += image_stats["orphan_count"]
            image_bytes += image_stats["orphan_bytes"]
        return GcPreviewResponse(
            ok=True,
            text_orphans=text_orphans,
            text_bytes=text_bytes,
            image_orphans=image_orphans,
            image_bytes=image_bytes,
            total_files=text_orphans + image_orphans,
            total_mb=round((text_bytes + image_bytes) / (1024 * 1024), 1),
        )
    except Exception as e:
        return GcPreviewResponse(ok=False, error=str(e))


@router.post("/cache/gc-clean", response_model=CacheActionResponse)
def cache_gc_clean():
    import os

    from web.backend.services.config_service import ConfigService

    config_service = ConfigService.get_instance()
    config = config_service.get_config_for_training()
    cache_dirs = [d for d in _gc_cache_dirs(config) if os.path.isdir(d)]

    if not cache_dirs:
        return CacheActionResponse(ok=True)

    try:
        from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

        for cache_dir in cache_dirs:
            SmartDiskCache.gc_clean(os.path.join(cache_dir, "text"))
            SmartDiskCache.gc_clean(os.path.join(cache_dir, "image"))
        return CacheActionResponse(ok=True)
    except Exception as e:
        return CacheActionResponse(ok=False, error=str(e))


class AdaptivePruneResponse(BaseModel):
    ok: bool
    affected_entries: int = 0
    affected_entry_paths: list[str] = Field(default_factory=list)
    removed_variants: int = 0
    variant_records: list[dict] = Field(default_factory=list)
    files: int = 0
    file_records: list[dict] = Field(default_factory=list)
    bytes: int = 0
    skipped_missing_metadata: int = 0
    skipped_entries: list[dict] = Field(default_factory=list)
    entries_without_usable_variant: int = 0
    entries_without_usable_variant_paths: list[str] = Field(default_factory=list)
    deleted_files: int = 0
    delete_errors: list[dict] = Field(default_factory=list)
    message: str | None = None
    error: str | None = None


def _image_cache_dir(cache_dir: str) -> str | None:
    import os

    nested = os.path.join(cache_dir, "image")
    if os.path.isfile(os.path.join(nested, "cache.json")):
        return nested
    if os.path.isfile(os.path.join(cache_dir, "cache.json")):
        return cache_dir
    return None


def _training_mutation_error() -> str | None:
    from web.backend.services.trainer_service import TrainerService

    status = TrainerService.get_instance().get_status().get("status", "idle")
    if status in {"starting", "running", "stopping"}:
        return f"Cache cleanup is unavailable while training is {status}."
    return None


def _is_exact_resolution(spec) -> bool:
    spec = str(spec or "")
    return "x" in spec.lower() and "," not in spec


def _load_concepts(config):
    concepts = getattr(config, "concepts", None)
    if concepts:
        return concepts
    concept_file = getattr(config, "concept_file_name", "")
    if not concept_file:
        return []
    try:
        from web.backend.services.concept_service import ConceptService

        return ConceptService().load_concepts(concept_file)
    except Exception:
        return []


def _value(obj, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _gc_cache_dirs(config) -> list[str]:
    """The run cache plus every concept's cache_dir override.

    A concept cache is swept exactly like the run cache: unreferenced .pt files
    and entries whose source file is gone. Clear Cache never touches it.
    """
    from mgds.pipelineModules.ConceptCacheRouter import normalize_cache_root, resolve_cache_root

    default_root = getattr(config, "cache_dir", "") or "workspace-cache/run"
    roots = {normalize_cache_root(default_root): default_root}
    for concept in _load_concepts(config):
        root = resolve_cache_root(_value(concept, "cache_dir", ""), default_root)
        roots.setdefault(normalize_cache_root(root), root)
    return list(roots.values())


def _adaptive_policy_has_scalar_spec(config, concepts) -> bool:
    if not _is_exact_resolution(getattr(config, "resolution", "")):
        return True
    for concept in concepts:
        image = _value(concept, "image", {})
        if (
            _value(concept, "enabled", True)
            and _value(image, "enable_resolution_override", False)
            and not _is_exact_resolution(_value(image, "resolution_override", ""))
        ):
            return True
    return False


def _image_header_resolution(path: str) -> list[int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            orientation = image.getexif().get(274)
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
        return [int(height), int(width)]
    except Exception:
        return None


def _dpo_resolution_provider(config, concepts):
    import os

    from modules.util.dpo_pattern_util import build_rejected_index, match_chosen, resolve_rejected

    index_cache = {}

    def resolve(filepath: str, _entry: dict) -> dict[str, list[int]]:
        result = {}
        chosen_resolution = _image_header_resolution(filepath)
        if chosen_resolution is not None:
            result["original_resolution"] = chosen_resolution
        for concept in concepts:
            if not _value(concept, "enabled", True):
                continue
            chosen_pattern = _value(concept, "dpo_chosen_pattern", "")
            rejected_pattern = _value(concept, "dpo_rejected_pattern", "")
            concept_path = _value(concept, "path", "")
            if not chosen_pattern or not rejected_pattern or not concept_path:
                continue
            stem = match_chosen(chosen_pattern, concept_path, filepath)
            if stem is None:
                continue
            if concept_path not in index_cache:
                index_cache[concept_path] = build_rejected_index(
                    concept_path, _value(concept, "include_subdirectories", False)
                )
            try:
                rejected_path = resolve_rejected(
                    rejected_pattern,
                    concept_path,
                    stem,
                    os.path.splitext(filepath)[1],
                    index_cache[concept_path],
                )
            except FileNotFoundError:
                return result
            rejected_resolution = _image_header_resolution(rejected_path)
            if rejected_resolution is not None:
                result["original_resolution_rejected"] = rejected_resolution
            return result
        return result

    return resolve


def _adaptive_prune(clean: bool) -> AdaptivePruneResponse:
    import os

    from web.backend.services.config_service import ConfigService

    mutation_error = _training_mutation_error()
    if mutation_error:
        return AdaptivePruneResponse(ok=False, error=mutation_error)

    config = ConfigService.get_instance().get_config_for_training()
    if not getattr(config, "adaptive_resolution", False):
        return AdaptivePruneResponse(ok=False, error="Adaptive Resolution is not enabled.")

    concepts = _load_concepts(config)
    if not _adaptive_policy_has_scalar_spec(config, concepts):
        return AdaptivePruneResponse(ok=True, message="Exact WxH resolutions are unchanged; there is nothing to prune.")

    cache_dir = getattr(config, "cache_dir", "") or "workspace-cache/run"
    image_cache_dir = _image_cache_dir(cache_dir)
    if image_cache_dir is None or not os.path.isdir(image_cache_dir):
        return AdaptivePruneResponse(ok=True, message="No image cache index was found.")

    try:
        from mgds.pipelineModules.ResolutionPolicy import parse_scalar_resolutions
        from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

        additional_names = ["original_resolution_rejected"] if getattr(config, "rlhf_enabled", False) else None
        provider = _dpo_resolution_provider(config, concepts) if additional_names else None
        method = SmartDiskCache.clean_adaptive_prune if clean else SmartDiskCache.preview_adaptive_prune
        maximum = max(parse_scalar_resolutions(str(getattr(config, "resolution", ""))), default=None)
        result = method(
            image_cache_dir,
            additional_resolution_names=additional_names,
            additional_resolution_provider=provider,
            adaptive_resolution_ladder=getattr(config, "adaptive_resolution_ladder", None),
            adaptive_resolution_max=maximum,
        )
        return AdaptivePruneResponse(ok=True, **result)
    except Exception as exc:
        return AdaptivePruneResponse(ok=False, error=str(exc))


@router.post("/cache/adaptive-prune-preview", response_model=AdaptivePruneResponse)
def adaptive_cache_prune_preview():
    return _adaptive_prune(clean=False)


@router.post("/cache/adaptive-prune-clean", response_model=AdaptivePruneResponse)
def adaptive_cache_prune_clean():
    return _adaptive_prune(clean=True)


# ---- Profiling ----


class ProfilingDumpResponse(BaseModel):
    ok: bool
    path: str | None = None
    error: str | None = None


class ProfilingToggleResponse(BaseModel):
    ok: bool
    active: bool = False
    error: str | None = None


_profiling_active = False


@router.post("/profiling/dump-stacks", response_model=ProfilingDumpResponse)
def profiling_dump_stacks():
    """Dump current Python thread stacks to <workspace>/profiling/stacks_<ts>.txt."""
    import datetime
    import pathlib
    import sys
    import traceback

    from web.backend.services.config_service import ConfigService

    try:
        config_service = ConfigService.get_instance()
        config = config_service.get_config_for_training()
        workspace = getattr(config, "workspace_dir", "") or "workspace"
        out_dir = pathlib.Path(workspace) / "profiling"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"stacks_{ts}.txt"

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(f"Stack dump captured at {datetime.datetime.now().isoformat()}\n")
            for tid, frame in sys._current_frames().items():
                fh.write(f"\n--- thread {tid} ---\n")
                traceback.print_stack(frame, file=fh)

        return ProfilingDumpResponse(ok=True, path=str(out_path))
    except Exception as exc:
        return ProfilingDumpResponse(ok=False, error=str(exc))


@router.post("/profiling/toggle", response_model=ProfilingToggleResponse)
def profiling_toggle():
    """Toggle Scalene profiling. Requires OneTrainer to be launched with --profile."""
    global _profiling_active
    try:
        import scalene  # noqa: F401
    except ImportError:
        return ProfilingToggleResponse(
            ok=False,
            active=False,
            error="Scalene not installed - launch OneTrainer with --profile to enable profiling",
        )
    _profiling_active = not _profiling_active
    return ProfilingToggleResponse(ok=True, active=_profiling_active)
