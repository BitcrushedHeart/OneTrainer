from enum import Enum


class ConceptType(Enum):
    STANDARD = "STANDARD"
    VALIDATION = "VALIDATION"
    PRIOR_PREDICTION = "PRIOR_PREDICTION"
    # Single images labelled Liked / Disliked (ConceptConfig.kto_label) for the
    # unpaired KTO preference loss. Only the KTO loader of an RLHF run reads
    # them; the standard, validation and DPO pair loaders skip them.
    KTO = "KTO"

    def __str__(self):
        return self.value
