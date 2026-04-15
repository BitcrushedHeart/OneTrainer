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
