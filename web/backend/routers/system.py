from web.backend.services.monitor_service import MonitorService

from fastapi import APIRouter
from pydantic import BaseModel

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
    cache_dir = getattr(config, "cache_dir", "") or "workspace-cache/run"

    if not os.path.isdir(cache_dir):
        return GcPreviewResponse(ok=True)

    try:
        from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
        text_stats = SmartDiskCache.gc_preview(os.path.join(cache_dir, "text"))
        image_stats = SmartDiskCache.gc_preview(os.path.join(cache_dir, "image"))
        total_files = text_stats["orphan_count"] + image_stats["orphan_count"]
        total_mb = (text_stats["orphan_bytes"] + image_stats["orphan_bytes"]) / (1024 * 1024)
        return GcPreviewResponse(
            ok=True,
            text_orphans=text_stats["orphan_count"],
            text_bytes=text_stats["orphan_bytes"],
            image_orphans=image_stats["orphan_count"],
            image_bytes=image_stats["orphan_bytes"],
            total_files=total_files,
            total_mb=round(total_mb, 1),
        )
    except Exception as e:
        return GcPreviewResponse(ok=False, error=str(e))


@router.post("/cache/gc-clean", response_model=CacheActionResponse)
def cache_gc_clean():
    import os
    from web.backend.services.config_service import ConfigService
    config_service = ConfigService.get_instance()
    config = config_service.get_config_for_training()
    cache_dir = getattr(config, "cache_dir", "") or "workspace-cache/run"

    if not os.path.isdir(cache_dir):
        return CacheActionResponse(ok=True)

    try:
        from mgds.pipelineModules.SmartDiskCache import SmartDiskCache
        SmartDiskCache.gc_clean(os.path.join(cache_dir, "text"))
        SmartDiskCache.gc_clean(os.path.join(cache_dir, "image"))
        return CacheActionResponse(ok=True)
    except Exception as e:
        return CacheActionResponse(ok=False, error=str(e))
