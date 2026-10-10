from enum import Enum


class TextRewardSource(Enum):
    # Roll the policy out from noise on synthetic prompts every reward step
    # (costs a multi-step prefix per sample).
    ONLINE_ROLLOUT = "ONLINE_ROLLOUT"
    # Start from a pre-generated image (Turbo + adapter via SwarmUI, curated
    # by the Text Alignment Dataset Tool): VAE-encode, re-noise to a late
    # sigma, and take the single grad-bearing step from there. With
    # text_reward_init_sigma > 0 the image is instead re-noised to that high
    # sigma and the policy rolls down to the grad step itself (hybrid).
    OFFLINE_IMAGES = "OFFLINE_IMAGES"

    def __str__(self):
        return self.value
