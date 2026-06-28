from enum import Enum


class DistillCfgMode(Enum):
    AUTO = "AUTO"
    FORCE_1 = "FORCE_1"
    METADATA = "METADATA"

    def __str__(self):
        return self.value
