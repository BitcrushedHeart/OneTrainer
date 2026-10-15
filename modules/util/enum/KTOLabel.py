from enum import Enum


class KTOLabel(Enum):
    # Label of a KTO concept's images: Liked images are pushed above the
    # reference, Disliked images below it (unpaired KTO).
    LIKED = "LIKED"
    DISLIKED = "DISLIKED"

    def __str__(self):
        return self.value
