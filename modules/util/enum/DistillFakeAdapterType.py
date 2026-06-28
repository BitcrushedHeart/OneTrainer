from enum import Enum


class DistillFakeAdapterType(Enum):
    LORA = "LORA"
    DORA = "DORA"
    OFT = "OFT"

    def __str__(self):
        return self.value
