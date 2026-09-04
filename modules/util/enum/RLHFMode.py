from enum import Enum


class RLHFMode(Enum):
    # DPO pairs are the primary loader; STANDARD concepts (if any) become a
    # weighted SFT-anchor term.
    DPO = "DPO"
    # The standard finetune is the primary loader at full weight; DPO pairs
    # are interleaved as a weighted add-on term.
    SFT_DPO = "SFT_DPO"

    def __str__(self):
        return self.value
