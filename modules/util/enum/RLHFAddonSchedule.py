from enum import Enum


class RLHFAddonSchedule(Enum):
    # Every micro-step is one standard batch; DPO pairs, KTO batches and text
    # reward rollouts ride along on some of them (one extra term per step,
    # lowest peak memory).
    PIGGYBACK = "PIGGYBACK"
    # Standard batches, DPO pairs, KTO batches and text reward rollouts each
    # get their own micro-step, interleaved evenly through the epoch; every
    # source is consumed at its full configured length and repeats.
    INDEPENDENT = "INDEPENDENT"

    def __str__(self):
        return self.value
