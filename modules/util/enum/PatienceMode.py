from enum import Enum


class PatienceMode(Enum):
    STOP_TRAINING = "STOP_TRAINING"
    SAVE_ALTERNATIVE_AT_END = "SAVE_ALTERNATIVE_AT_END"

    def __str__(self):
        return self.value
