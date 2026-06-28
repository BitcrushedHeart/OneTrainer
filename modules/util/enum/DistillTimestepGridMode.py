from enum import Enum


class DistillTimestepGridMode(Enum):
    AUTO = "AUTO"
    TRAILING = "TRAILING"
    UNIFORM = "UNIFORM"

    def __str__(self):
        return self.value
