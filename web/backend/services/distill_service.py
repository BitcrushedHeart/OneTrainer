from modules.util.distill_metadata_util import (
    DistillBuildRequest as DistillDatasetBuildRequest,
)
from modules.util.distill_metadata_util import (
    build_distillation_dataset,
)
from web.backend.services._singleton import SingletonMixin


class DistillService(SingletonMixin):
    def build_dataset(self, request: DistillDatasetBuildRequest) -> dict:
        return build_distillation_dataset(request)
