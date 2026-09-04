from modules.util.enum.AttentionBackend import AttentionBackend

from torch import nn

from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor


def _validate_diffusers_backend(name: str) -> None:
    """Fail at setup time rather than mid-step: diffusers only resolves the backend
    string inside `dispatch_attention_fn`, and only runs its own requirement checks
    from the `attention_backend()` context manager, which we don't use."""
    from diffusers.models import attention_dispatch

    backend_name_cls = attention_dispatch.AttentionBackendName
    try:
        backend_name = backend_name_cls(name)
    except ValueError as e:
        valid = sorted(b.value for b in backend_name_cls)
        raise ValueError(f"Unknown diffusers attention backend '{name}'. Available backends: {valid}") from e

    registry = getattr(attention_dispatch, "_AttentionBackendRegistry", None)
    if registry is not None and backend_name not in registry._backends:
        raise ValueError(f"Attention backend '{name}' is not registered in the installed diffusers.")

    check_requirements = getattr(attention_dispatch, "_check_attention_backend_requirements", None)
    if check_requirements is not None:
        check_requirements(backend_name)


def _nocopy_processor_class():
    # imported lazily: registering the custom ops is only needed when they are selected
    from modules.util.attention_nocopy import ZImageNoCopyEfficientAttnProcessor

    return ZImageNoCopyEfficientAttnProcessor


def apply_attention_backend(module: nn.Module | None, backend: AttentionBackend) -> int:
    """Pin the attention kernel used by every attention processor inside `module`.

    For the diffusers-dispatched backends the name is set per processor *instance*
    (`processor._attention_backend`), never on the processor class, so two models
    loaded in the same process can't clobber each other. `AttentionBackend.DEFAULT`
    writes None, which restores diffusers' own default (the `DIFFUSERS_ATTN_BACKEND`
    env var, "native").

    `EFFICIENT_NOCOPY` instead swaps each Z-Image processor for
    `ZImageNoCopyEfficientAttnProcessor` (and any other backend swaps it back), so
    the kernel choice is a config change in either direction.

    Returns the number of distinct processor instances that were configured.
    """
    name = backend.diffusers_backend_name()
    if name is not None:
        _validate_diffusers_backend(name)

    if module is None:
        return 0

    nocopy_cls = _nocopy_processor_class() if backend.replaces_processor() else None

    seen: set[int] = set()
    for child in module.modules():
        processor = getattr(child, "processor", None)
        if processor is None or not hasattr(processor, "_attention_backend"):
            continue
        if id(processor) in seen:
            continue

        if nocopy_cls is not None:
            if not isinstance(processor, nocopy_cls):
                if not isinstance(processor, ZSingleStreamAttnProcessor):
                    raise ValueError(
                        f"{backend} is only implemented for Z-Image's ZSingleStreamAttnProcessor, "
                        f"found {type(processor).__name__}"
                    )
                processor = nocopy_cls()
                child.set_processor(processor)
        elif type(processor) is not ZSingleStreamAttnProcessor and isinstance(processor, ZSingleStreamAttnProcessor):
            # a processor subclass we installed earlier (EFFICIENT_NOCOPY): restore diffusers' own
            processor = ZSingleStreamAttnProcessor()
            child.set_processor(processor)

        seen.add(id(processor))
        processor._attention_backend = name

    return len(seen)
