from modules.util.distill_metadata_util import (
    DistillBuildFilters,
    DistillBuildRequest as DistillDatasetBuildRequest,
    build_distillation_dataset,
)
from web.backend.services._singleton import SingletonMixin


class DistillService(SingletonMixin):
    def build_dataset(self, request: DistillDatasetBuildRequest) -> dict:
        return build_distillation_dataset(request)
