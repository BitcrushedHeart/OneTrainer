from enum import Enum


class AttentionBackend(Enum):
    DEFAULT = "DEFAULT"
    NATIVE_FLASH = "NATIVE_FLASH"
    NATIVE_EFFICIENT = "NATIVE_EFFICIENT"
    NATIVE_CUDNN = "NATIVE_CUDNN"
    NATIVE_MATH = "NATIVE_MATH"
    FLEX = "FLEX"
    # The memory-efficient kernel behind custom ops, so torch.compile never copies the
    # broadcast key-padding mask into a dense (B, H, L, L) buffer in the backward.
    # Z-Image only. See modules/util/attention_nocopy.py.
    EFFICIENT_NOCOPY = "EFFICIENT_NOCOPY"

    def __str__(self):
        return self.value

    def diffusers_backend_name(self) -> str | None:
        """The `diffusers.models.attention_dispatch.AttentionBackendName` value this
        maps to, or None when diffusers' dispatch is not what selects the kernel
        (DEFAULT leaves diffusers' own default in place; EFFICIENT_NOCOPY swaps the
        processor instead)."""
        return _DIFFUSERS_BACKEND_NAMES[self]

    def replaces_processor(self) -> bool:
        return self is AttentionBackend.EFFICIENT_NOCOPY


_DIFFUSERS_BACKEND_NAMES: dict[AttentionBackend, str | None] = {
    AttentionBackend.DEFAULT: None,
    AttentionBackend.NATIVE_FLASH: "_native_flash",
    AttentionBackend.NATIVE_EFFICIENT: "_native_efficient",
    AttentionBackend.NATIVE_CUDNN: "_native_cudnn",
    AttentionBackend.NATIVE_MATH: "_native_math",
    AttentionBackend.FLEX: "flex",
    AttentionBackend.EFFICIENT_NOCOPY: None,
}
