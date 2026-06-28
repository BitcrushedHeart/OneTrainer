from modules.util.distill_metadata_util import DistillBuildFilters, DistillBuildRequest as DistillDatasetBuildRequest
from web.backend.services.distill_service import DistillService

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["distill"])


class DistillBuildFiltersRequest(BaseModel):
    model_contains: str | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    scheduler: str | None = None
    width: int | None = None
    height: int | None = None


class DistillBuildDatasetRequest(BaseModel):
    source_folder: str
    output_folder: str
    include_subdirectories: bool = True
    val_percentage: float = 0.0
    filters: DistillBuildFiltersRequest = DistillBuildFiltersRequest()


@router.post("/distill/build-dataset")
def build_dataset(req: DistillBuildDatasetRequest):
    service = DistillService.get_instance()
    return service.build_dataset(
        DistillDatasetBuildRequest(
            source_folder=req.source_folder,
            output_folder=req.output_folder,
            include_subdirectories=req.include_subdirectories,
            val_percentage=req.val_percentage,
            filters=DistillBuildFilters(
                model_contains=req.filters.model_contains,
                steps=req.filters.steps,
                cfg_scale=req.filters.cfg_scale,
                scheduler=req.filters.scheduler,
                width=req.filters.width,
                height=req.filters.height,
            ),
        )
    )
