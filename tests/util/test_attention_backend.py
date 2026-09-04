"""CPU-only tests for the Z-Image attention backend selection.

Nothing here touches CUDA: the tiny transformer is built on CPU in fp32 and the
backends that only have CUDA kernels are probed and skipped rather than run.
"""

# Plain-import form so isort keeps the priming import before the direct
# modelSetup imports (create -> factory.import_dir resolves the circular
# optimizer_util dependency once, in bootstrap order).
import modules.util.create  # noqa: F401 -- import-side-effect required
from modules.model.ZImageModel import ZImageModel
from modules.modelSetup.ZImageLoRASetup import ZImageLoRASetup
from modules.util.attention_backend_util import _validate_diffusers_backend, apply_attention_backend
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AttentionBackend import AttentionBackend
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
from torch import nn

from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    dispatch_attention_fn,
)
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel,
    ZSingleStreamAttnProcessor,
)

import pytest

# Small enough to stay fast on CPU. head_dim = dim // n_heads must equal sum(axes_dims).
TINY_CONFIG = {
    "all_patch_size": (2,),
    "all_f_patch_size": (1,),
    "in_channels": 4,
    "dim": 64,
    "n_layers": 2,
    "n_refiner_layers": 1,
    "n_heads": 4,
    "n_kv_heads": 4,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 16,
    "rope_theta": 256.0,
    "axes_dims": [4, 6, 6],
    "axes_lens": [64, 32, 32],
}

# Expected processor count: noise_refiner + context_refiner + n_layers.
EXPECTED_PROCESSORS = 1 + 1 + 2


def _build_tiny_transformer(seed: int = 0, **config_overrides) -> ZImageTransformer2DModel:
    torch.manual_seed(seed)
    model = ZImageTransformer2DModel(**{**TINY_CONFIG, **config_overrides})
    model = model.to(torch.float32).eval()

    # `x_pad_token` / `cap_pad_token` are `torch.empty` in the diffusers constructor and
    # are only filled by a checkpoint load, so an unloaded model carries uninitialized
    # memory there (NaN often enough to make comparisons flaky). Seed them explicitly.
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 1000)
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
        model.x_pad_token.copy_(torch.randn(model.x_pad_token.shape, generator=generator) * 0.02)
        model.cap_pad_token.copy_(torch.randn(model.cap_pad_token.shape, generator=generator) * 0.02)
        assert all(torch.isfinite(p).all() for p in model.parameters())

    return model


def _tiny_inputs(seed: int = 1):
    """Batch size 1, matching the OneTrainer call shape: a list of (C, F, H, W)
    latents, a timestep tensor and a list of (seq, cap_feat_dim) caption tensors."""
    generator = torch.Generator().manual_seed(seed)
    latents = [torch.randn(4, 1, 8, 8, generator=generator)]
    caps = [torch.randn(7, 16, generator=generator)]
    timestep = torch.tensor([0.3])
    return latents, timestep, caps


def _processors(module: nn.Module) -> list:
    found = []
    for child in module.modules():
        processor = getattr(child, "processor", None)
        if processor is not None and hasattr(processor, "_attention_backend"):
            found.append(processor)
    return found


def _backend_runs_on_cpu(backend: AttentionBackend, *, requires_grad: bool = False) -> bool:
    """Probe the kernel with a throwaway tensor; the fused SDPA backends other than
    flash have no CPU implementation and raise instead of falling back, and flex
    attention refuses the CPU backward pass outright."""
    name = backend.diffusers_backend_name()
    if name is None:
        return True
    query = torch.randn(1, 4, 2, 8, requires_grad=requires_grad)
    try:
        out = dispatch_attention_fn(query, query.clone(), query.clone(), attn_mask=None, backend=name)
        if requires_grad:
            out.sum().backward()
    except Exception:
        return False
    return True


CANDIDATE_ARMS = (
    AttentionBackend.NATIVE_MATH,
    AttentionBackend.NATIVE_EFFICIENT,
    AttentionBackend.NATIVE_FLASH,
    AttentionBackend.FLEX,
)

CPU_FORWARD_ARMS = [backend for backend in CANDIDATE_ARMS if _backend_runs_on_cpu(backend)]
CPU_BACKWARD_ARMS = [backend for backend in CPU_FORWARD_ARMS if _backend_runs_on_cpu(backend, requires_grad=True)]


# ---------------------------------------------------------------- enum mapping


def test_enum_maps_onto_backends_that_exist_in_installed_diffusers():
    registered = {backend.value for backend in _AttentionBackendRegistry.list_backends()}
    declared = {member.value for member in AttentionBackendName}

    assert AttentionBackend.DEFAULT.diffusers_backend_name() is None

    for member in AttentionBackend:
        if member is AttentionBackend.DEFAULT:
            continue
        name = member.diffusers_backend_name()
        if member.replaces_processor():
            assert name is None, f"{member} swaps the processor and must not also pin a diffusers backend"
            continue
        assert name in declared, f"{member} maps to {name!r}, not an AttentionBackendName value"
        assert name in registered, f"{member} maps to {name!r}, which is not a registered backend"


def test_every_enum_member_has_a_mapping():
    for member in AttentionBackend:
        # raises KeyError if a member was added without a mapping
        member.diffusers_backend_name()


def test_validate_rejects_an_unknown_backend_name():
    with pytest.raises(ValueError, match="Unknown diffusers attention backend"):
        _validate_diffusers_backend("definitely_not_a_backend")


# ------------------------------------------------------------- apply mechanics


def test_apply_sets_every_processor_instance_and_leaves_the_class_alone():
    model = _build_tiny_transformer()
    processors = _processors(model)
    assert len(processors) == EXPECTED_PROCESSORS
    assert len({id(p) for p in processors}) == EXPECTED_PROCESSORS, (
        "processors are shared between attention modules; per-instance assignment would be redundant"
    )
    assert ZSingleStreamAttnProcessor._attention_backend is None

    count = apply_attention_backend(model, AttentionBackend.NATIVE_MATH)

    assert count == EXPECTED_PROCESSORS
    for processor in processors:
        assert processor.__dict__["_attention_backend"] == "_native_math"
    assert ZSingleStreamAttnProcessor._attention_backend is None, "class attribute must not be mutated"

    assert apply_attention_backend(model, AttentionBackend.DEFAULT) == EXPECTED_PROCESSORS
    for processor in processors:
        assert processor._attention_backend is None
    assert ZSingleStreamAttnProcessor._attention_backend is None


def test_apply_does_not_leak_across_models():
    first = _build_tiny_transformer()
    second = _build_tiny_transformer()

    apply_attention_backend(first, AttentionBackend.NATIVE_MATH)

    assert all(p._attention_backend == "_native_math" for p in _processors(first))
    assert all(p._attention_backend is None for p in _processors(second))


def test_apply_tolerates_a_module_without_processors():
    assert apply_attention_backend(nn.Linear(2, 2), AttentionBackend.NATIVE_MATH) == 0
    assert apply_attention_backend(None, AttentionBackend.NATIVE_MATH) == 0


def test_apply_reaches_the_dispatch_call(monkeypatch):
    """The pinned name is what `dispatch_attention_fn` actually receives, and at
    batch size 1 the transformer passes no attention mask."""
    from diffusers.models.transformers import transformer_z_image

    calls = []
    original = transformer_z_image.dispatch_attention_fn

    def spy(query, key, value, attn_mask=None, **kwargs):
        calls.append((kwargs.get("backend"), attn_mask is None))
        return original(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(transformer_z_image, "dispatch_attention_fn", spy)

    model = _build_tiny_transformer()
    apply_attention_backend(model, AttentionBackend.NATIVE_MATH)
    latents, timestep, caps = _tiny_inputs()
    with torch.no_grad():
        model(latents, timestep, caps, return_dict=True)

    assert len(calls) == EXPECTED_PROCESSORS
    assert all(call == ("_native_math", True) for call in calls)


# ------------------------------------------------------- numerical equivalence


def _forward(model: ZImageTransformer2DModel, backend: AttentionBackend) -> torch.Tensor:
    apply_attention_backend(model, backend)
    latents, timestep, caps = _tiny_inputs()
    with torch.no_grad():
        output = model(latents, timestep, caps, return_dict=True).sample
    return torch.stack(output, dim=0)


def _forward_backward(model: ZImageTransformer2DModel, backend: AttentionBackend):
    apply_attention_backend(model, backend)
    model.zero_grad(set_to_none=True)
    latents, timestep, caps = _tiny_inputs()
    latents = [latent.clone().requires_grad_(True) for latent in latents]
    output = model(latents, timestep, caps, return_dict=True).sample
    loss = torch.stack(output, dim=0).square().mean()
    loss.backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    grads["__input__"] = latents[0].grad.detach().clone()
    return torch.stack(output, dim=0).detach(), grads


@pytest.mark.parametrize("backend", CPU_FORWARD_ARMS, ids=str)
def test_forward_matches_the_default_backend(backend):
    model = _build_tiny_transformer()
    reference = _forward(model, AttentionBackend.DEFAULT)
    candidate = _forward(model, backend)
    assert torch.allclose(reference, candidate, atol=1e-5, rtol=1e-5), (
        f"{backend} max abs diff {(reference - candidate).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("backend", CPU_BACKWARD_ARMS, ids=str)
def test_backward_matches_the_default_backend(backend):
    model = _build_tiny_transformer()
    reference_out, reference_grads = _forward_backward(model, AttentionBackend.DEFAULT)
    candidate_out, candidate_grads = _forward_backward(model, backend)

    assert torch.allclose(reference_out, candidate_out, atol=1e-5, rtol=1e-5)
    assert reference_grads.keys() == candidate_grads.keys()
    assert len(reference_grads) > 1
    for name, reference_grad in reference_grads.items():
        candidate_grad = candidate_grads[name]
        assert torch.allclose(reference_grad, candidate_grad, atol=1e-5, rtol=1e-5), (
            f"{backend} grad {name} max abs diff {(reference_grad - candidate_grad).abs().max().item():.3e}"
        )


@pytest.mark.parametrize(
    "backend",
    [AttentionBackend.NATIVE_EFFICIENT, AttentionBackend.NATIVE_CUDNN],
    ids=str,
)
def test_cuda_only_backends_are_selectable_even_without_a_cpu_kernel(backend):
    if _backend_runs_on_cpu(backend):
        pytest.skip(f"{backend} runs on CPU in this torch build; already covered by the equivalence arms")
    # Selection must still validate and apply: these are registered diffusers
    # backends, they just have no CPU kernel, so only a forward pass would raise.
    model = _build_tiny_transformer()
    assert apply_attention_backend(model, backend) == EXPECTED_PROCESSORS


# ---------------------------------------------------------- setup_optimizations


def _make_setup_model(transformer: nn.Module) -> ZImageModel:
    model = ZImageModel(ModelType.Z_IMAGE)
    model.train_config = TrainConfig.default_values()
    model.text_encoder = None
    model.transformer = transformer
    model.vae = None
    return model


def _make_setup_config(backend: AttentionBackend) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.gradient_checkpointing = GradientCheckpointingMethod.OFF
    config.attention_backend = backend
    return config


def test_train_config_default_is_the_default_backend():
    assert TrainConfig.default_values().attention_backend is AttentionBackend.DEFAULT


def test_setup_optimizations_applies_the_configured_backend():
    transformer = _build_tiny_transformer()
    model = _make_setup_model(transformer)
    setup = ZImageLoRASetup(torch.device("cpu"), torch.device("cpu"), False)

    setup.setup_optimizations(model, _make_setup_config(AttentionBackend.NATIVE_MATH))

    processors = _processors(transformer)
    assert len(processors) == EXPECTED_PROCESSORS
    assert all(p._attention_backend == "_native_math" for p in processors)


def test_setup_optimizations_default_resets_the_backend():
    transformer = _build_tiny_transformer()
    apply_attention_backend(transformer, AttentionBackend.NATIVE_MATH)
    model = _make_setup_model(transformer)
    setup = ZImageLoRASetup(torch.device("cpu"), torch.device("cpu"), False)

    setup.setup_optimizations(model, _make_setup_config(AttentionBackend.DEFAULT))

    assert all(p._attention_backend is None for p in _processors(transformer))
