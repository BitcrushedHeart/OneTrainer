from enum import Enum


class DistillVramMode(Enum):
    SPLIT = "SPLIT"
    CONCURRENT = "CONCURRENT"

    def __str__(self):
        return self.value
