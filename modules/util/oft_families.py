"""The canonical late-family stems for the SoReal! lineup.

Leaf module (no OneTrainer imports) so oft_align / transfer_util / model-setup
code can share the definition without pulling in oft_merge's loader/saver
dependency chain (oft_merge -> create -> BaseDataLoader -> BaseModelSetup),
which would be circular from BaseModelSetup.

``attention.to_out.0`` is deliberately NOT a late family: it is covered by the
adapters but does not differ between Base and the distilled variants, so it is
never conjugated, never sentinel-aligned, and never trained in transfer mode.
"""

DEFAULT_LATE_FAMILY_STEMS: dict[str, tuple[str, ...]] = {
    "qkv": ("attention.to_q", "attention.to_k", "attention.to_v"),
    "mlp": ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
}
