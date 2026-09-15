"""OFT spectral-clip state: read-only forwards, one advance per optimizer update.

CPU-only. u_state/v_state hold the power-iteration state behind oft_clipped_norm.
Forwards only read it (one read-only step per forward), so a forward and its
non-reentrant checkpoint recompute, DPO reference passes and validation all see
the same clip; the trainer advances it once per optimizer update through
BaseModelSetup.after_optimizer_step_adapters -> OFTRotationModule.advance_spectral_state.

Covers:
- forwards never write the state (train, eval, no_grad, bf16 DPO param swap);
- checkpointed and uncheckpointed hooked DoRA-OFT stacks get identical gradients
  with an engaged clip and an unconverged state;
- advance_spectral_state converges to the top singular pair, honours the
  degeneracy guard, is a no-op without a clip, and warm_start_spectral_state
  keeps its pre-refactor numbers;
- SPECTRAL_STEPS_PER_UPDATE tracks a drifting Q (the evidence for its value);
- the shared after_optimizer_step_adapters hook reaches every trainable OFT module.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import DoRAOFTModule, LoRAModuleWrapper, OFTModule
from modules.module.oft_utils import SPECTRAL_STEPS_PER_UPDATE, OFTRotationModule
from modules.util.checkpointing_util import CheckpointLayer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
from torch import nn

import pytest


def _rotation(block_size: int, r: int = 3, clip: float | None = 0.95, **kwargs) -> OFTRotationModule:
    return OFTRotationModule(
        r=r,
        n_elements=block_size * (block_size - 1) // 2,
        block_size=block_size,
        in_features=r * block_size,
        oft_clipped_norm=clip,
        **kwargs,
    )


def _state(module: OFTRotationModule) -> tuple[torch.Tensor, torch.Tensor]:
    return module.u_state.clone(), module.v_state.clone()


def _assert_state_equal(module: OFTRotationModule, state: tuple[torch.Tensor, torch.Tensor]):
    assert torch.equal(module.u_state, state[0])
    assert torch.equal(module.v_state, state[1])


def _scaled_skew(module: OFTRotationModule) -> torch.Tensor:
    scaling = 2 * math.sqrt(module.block_size - 1) if module.oft_scaled else 1
    weight = module.weight.detach().to(torch.float32) / scaling
    return module._pytorch_skew_symmetric(weight, module.block_size)


def _forward_sigma(module: OFTRotationModule, Q_skew: torch.Tensor) -> torch.Tensor:
    """The sigma _cayley_batch clips with: one read-only step from the stored state."""
    with torch.no_grad():
        v_vec, u_vec = module._spectral_norm(Q_skew)
        return torch.sum(u_vec * torch.bmm(Q_skew, v_vec), dim=1).squeeze(-1)


def _hooked_stack(cls=DoRAOFTModule, *, seed: int, std: float = 0.5, clip: float = 0.95):
    """Three Linears hooked by (DoRA-)OFT, block 8, weights large enough to engage the clip."""
    torch.manual_seed(seed)
    seq = nn.Sequential(nn.Linear(24, 32, bias=True), nn.Linear(32, 24, bias=False), nn.Linear(24, 16, bias=True))
    adapters = []
    for layer in seq:
        adapter = cls(
            prefix="test",
            orig_module=layer,
            oft_block_size=8,
            block_share=False,
            oft_scaled=False,
            oft_clipped_norm=clip,
        )
        with torch.no_grad():
            adapter.oft_R.weight.normal_(0, std)
            if isinstance(adapter, DoRAOFTModule):
                adapter.dora_multiplier.uniform_(0.8, 1.2)
        adapter.hook_to_module()
        adapters.append(adapter)
    return seq, adapters


# --------------------------------------------------------------------------- #
# Forwards are read-only
# --------------------------------------------------------------------------- #


def _dpo_swap_forward(seq, adapters, x):
    """What the EXISTING_ADAPTER reference pass does: param.data -> bf16 temporaries, no_grad forward."""
    saved = []
    for adapter in adapters:
        for param in adapter.parameters():
            saved.append((param, param.data))
            param.data = param.data.to(torch.bfloat16)
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            return seq(x)
    finally:
        for param, data in saved:
            param.data = data


@pytest.mark.parametrize("mode", ["train", "train_backward", "train_autocast", "eval", "no_grad", "dpo_bf16_swap"])
@pytest.mark.parametrize("cls", [OFTModule, DoRAOFTModule])
def test_forward_never_writes_spectral_state(mode, cls):
    seq, adapters = _hooked_stack(cls, seed=1)
    seq.train(mode != "eval")
    states = [_state(adapter.oft_R) for adapter in adapters]
    # The clip is engaged, so the state is actually read on this path.
    for adapter in adapters:
        assert (_forward_sigma(adapter.oft_R, _scaled_skew(adapter.oft_R)) > adapter.oft_R.oft_clipped_norm).all()
    x = torch.randn(4, 24)

    for _ in range(3):
        if mode == "train_backward":
            seq(x).square().sum().backward()
        elif mode == "train_autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = seq(x)
            y.float().sum().backward()
        elif mode == "no_grad":
            with torch.no_grad():
                seq(x)
        elif mode == "dpo_bf16_swap":
            _dpo_swap_forward(seq, adapters, x)
        else:
            seq(x)

    for adapter, state in zip(adapters, states, strict=True):
        _assert_state_equal(adapter.oft_R, state)
        assert adapter.oft_R.u_state.dtype == torch.float32
    for adapter in adapters:
        adapter.remove_hook_from_module()


def test_rotation_module_forward_is_read_only_in_both_modes():
    module = _rotation(16, r=4)
    with torch.no_grad():
        module.weight.normal_(0, 0.3)
    state = _state(module)
    x = torch.randn(2, 64)
    module.train()
    y_train = module(x)
    module.eval()
    y_eval = module(x)
    _assert_state_equal(module, state)
    assert torch.equal(y_train, y_eval)  # same state -> same clip in either mode


@pytest.mark.parametrize("cls", [OFTModule, DoRAOFTModule])
def test_checkpoint_recompute_sees_the_forward_clip(cls):
    """Non-reentrant checkpoint re-runs the forward during backward. With a
    forward-advanced state the recompute clipped with a different sigma than the
    forward; read-only forwards make both gradients identical."""
    plain_seq, plain_adapters = _hooked_stack(cls, seed=7)
    ckpt_seq, ckpt_adapters = _hooked_stack(cls, seed=7)
    checkpointed = CheckpointLayer(ckpt_seq, None, torch.device("cpu"))
    plain_seq.train()
    checkpointed.train()

    # Unconverged (freshly random) state with an engaged clip: one more
    # power-iteration step would move sigma, so a stale recompute would show.
    # Probed on an eval-mode copy so the stacks' own state is never touched here.
    for adapter in plain_adapters:
        rot = adapter.oft_R
        probe = _rotation(rot.block_size, r=rot.r, clip=rot.oft_clipped_norm).eval()
        with torch.no_grad():
            probe.weight.copy_(rot.weight)
        probe.u_state.copy_(rot.u_state)
        probe.v_state.copy_(rot.v_state)
        Q_skew = _scaled_skew(probe)
        sigma = _forward_sigma(probe, Q_skew)
        assert (sigma > rot.oft_clipped_norm).all()
        probe.advance_spectral_state(steps=1)
        assert ((_forward_sigma(probe, Q_skew) - sigma).abs() > 1e-4).any()

    states = [_state(adapter.oft_R) for adapter in ckpt_adapters]
    torch.manual_seed(99)
    x = torch.randn(5, 24)
    upstream = torch.randn(5, 16)

    y_plain = plain_seq(x)
    (y_plain * upstream).sum().backward()
    y_ckpt = checkpointed(x)
    (y_ckpt * upstream).sum().backward()

    for a_plain, a_ckpt, state in zip(plain_adapters, ckpt_adapters, states, strict=True):
        for p_plain, p_ckpt in zip(a_plain.parameters(), a_ckpt.parameters(), strict=True):
            assert p_ckpt.grad is not None
            assert (p_plain.grad - p_ckpt.grad).abs().max().item() <= 1e-7
        _assert_state_equal(a_ckpt.oft_R, state)
    assert torch.equal(y_plain, y_ckpt)
    for adapter in (*plain_adapters, *ckpt_adapters):
        adapter.remove_hook_from_module()


# --------------------------------------------------------------------------- #
# advance_spectral_state
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scaled", [False, True])
def test_advance_converges_to_top_singular_value(scaled):
    torch.manual_seed(3)
    module = _rotation(16, r=4, oft_scaled=scaled)
    with torch.no_grad():
        module.weight.normal_(0, 0.4)
    Q_skew = _scaled_skew(module)
    true_sigma = torch.linalg.matrix_norm(Q_skew.double(), ord=2).float()

    errors = []
    for _ in range(6):
        errors.append(((true_sigma - _forward_sigma(module, Q_skew)) / true_sigma).max().item())
        module.advance_spectral_state()
    assert errors[-1] < errors[0]
    assert all(e >= -1e-6 for e in errors)  # one power step never overestimates sigma

    module.advance_spectral_state(steps=500)
    torch.testing.assert_close(_forward_sigma(module, Q_skew), true_sigma, rtol=1e-5, atol=0)
    # The left vector is a top singular vector: ||Q^T u|| == sigma_1.
    u = module.u_state.unsqueeze(-1)
    torch.testing.assert_close(
        torch.linalg.vector_norm(torch.bmm(Q_skew.mT, u), dim=1).squeeze(-1), true_sigma, rtol=1e-5, atol=0
    )
    assert module.u_state.dtype == torch.float32 and module.v_state.dtype == torch.float32


def test_advance_default_steps_is_the_module_constant():
    torch.manual_seed(4)
    a, b = _rotation(8), _rotation(8)
    b.u_state.copy_(a.u_state)
    b.v_state.copy_(a.v_state)
    with torch.no_grad():
        a.weight.normal_(0, 0.3)
        b.weight.copy_(a.weight)
    a.advance_spectral_state()
    b.advance_spectral_state(steps=SPECTRAL_STEPS_PER_UPDATE)
    _assert_state_equal(a, _state(b))


def test_advance_respects_degeneracy_guard():
    torch.manual_seed(5)
    module = _rotation(16, r=3)
    with torch.no_grad():
        module.weight.normal_(0, 0.3)
        module.weight[1].zero_()  # a block the optimizer has not moved yet
    before = _state(module)

    module.advance_spectral_state(steps=10)

    assert torch.equal(module.u_state[1], before[0][1]) and torch.equal(module.v_state[1], before[1][1])
    assert not torch.equal(module.u_state[0], before[0][0])
    assert torch.isfinite(module.u_state).all() and torch.isfinite(module.v_state).all()
    torch.testing.assert_close(module.u_state.norm(dim=1), torch.ones(3))

    fresh = _rotation(16, r=3)  # Q == 0 at init (OFTModule zero-inits): the whole state is kept
    with torch.no_grad():
        fresh.weight.zero_()
    state = _state(fresh)
    fresh.advance_spectral_state()
    _assert_state_equal(fresh, state)


@pytest.mark.parametrize(
    "kwargs",
    [{"clip": None}, {"clip": 0.95, "use_cayley_neumann": False}, {"clip": 0.95, "oft_cans": True}],
    ids=["no_clip", "exact_solver", "cans"],
)
def test_advance_is_a_noop_without_clip(kwargs):
    module = _rotation(8, **kwargs)
    assert module.oft_clipped_norm is None
    with torch.no_grad():
        module.weight.normal_(0, 0.3)
    weight = module.weight.detach().clone()
    module.advance_spectral_state()
    module.warm_start_spectral_state()
    assert not hasattr(module, "u_state")
    assert torch.equal(module.weight, weight)


def test_advance_from_bf16_weight_keeps_float32_state():
    torch.manual_seed(6)
    module = _rotation(16, r=2)
    with torch.no_grad():
        module.weight.normal_(0, 0.3)
    module.weight.data = module.weight.data.to(torch.bfloat16)
    module.advance_spectral_state(steps=200)
    assert module.u_state.dtype == torch.float32 and module.v_state.dtype == torch.float32
    Q_skew = _scaled_skew(module)
    torch.testing.assert_close(
        _forward_sigma(module, Q_skew), torch.linalg.matrix_norm(Q_skew, ord=2), rtol=1e-4, atol=0
    )


def _legacy_warm_start(module: OFTRotationModule, steps: int):
    """warm_start_spectral_state before the advance/warm-start refactor, verbatim."""
    scaling_factor = 2 * math.sqrt(module.block_size - 1) if module.oft_scaled else 1
    Q = (module.weight.detach() / scaling_factor).to(torch.float32)
    Q_skew = module._pytorch_skew_symmetric(Q, module.block_size)
    u = module.u_state.to(torch.float32).unsqueeze(-1)
    v = module.v_state.to(torch.float32).unsqueeze(-1)
    for _ in range(steps):
        v_raw = torch.bmm(Q_skew.mT, u)
        v_norm = torch.linalg.vector_norm(v_raw, dim=1, keepdim=True)
        v = torch.where(v_norm >= 1e-6, v_raw / v_norm.clamp_min(1e-8), v)
        u_raw = torch.bmm(Q_skew, v)
        u_norm = torch.linalg.vector_norm(u_raw, dim=1, keepdim=True)
        u = torch.where(u_norm >= 1e-6, u_raw / u_norm.clamp_min(1e-8), u)
    return u.squeeze(-1), v.squeeze(-1)


@pytest.mark.parametrize("scaled", [False, True])
def test_warm_start_matches_pre_refactor_loop(scaled):
    torch.manual_seed(8)
    module = _rotation(16, r=4, oft_scaled=scaled)
    with torch.no_grad():
        module.weight.normal_(0, 0.3)
        module.weight[2].zero_()
    expected_u, expected_v = _legacy_warm_start(module, 37)
    module.warm_start_spectral_state(steps=37)
    assert torch.equal(module.u_state, expected_u)
    assert torch.equal(module.v_state, expected_v)


# --------------------------------------------------------------------------- #
# SPECTRAL_STEPS_PER_UPDATE tracks a drifting Q
# --------------------------------------------------------------------------- #

# Each simulated optimizer update moves every block by DRIFT_PER_UPDATE of its
# Frobenius norm, i.e. an Adam-like per-element step of ~1e-5 on weights with
# rms ~1e-2. "fixed" drift keeps one direction for all updates (a consistent
# gradient), so over 100 updates the top singular pair migrates and crosses
# over -- the hard case; "random" drift is an unbiased random walk.
DRIFT_PER_UPDATE = 1e-3
DRIFT_UPDATES = 100
# Relative underestimate of sigma_1 by the forward's one-step estimate. At
# oft_clipped_norm 0.95 a 1e-3 underestimate leaves ||Q_clipped||_2 <= 0.951.
SIGMA_TOLERANCE = 1e-3


def _worst_tracking_error(seed: int, steps: int, drift: str, block: int = 96, r: int = 4) -> float:
    torch.manual_seed(seed)
    module = _rotation(block, r=r, clip=0.95)
    with torch.no_grad():
        w = torch.randn(r, module.n_elements)
        sigma = torch.linalg.matrix_norm(module._pytorch_skew_symmetric(w, block), ord=2)
        # True sigma 1.0..2.0 per block: every block is clipped.
        module.weight.copy_(w * (torch.linspace(1.0, 2.0, r) / sigma).unsqueeze(-1))
    module.warm_start_spectral_state(steps=3000)  # converged, as after a long run or a resume

    gen = torch.Generator().manual_seed(seed + 1)
    direction = torch.randn(r, module.n_elements, generator=gen)
    worst = 0.0
    for _ in range(DRIFT_UPDATES):
        with torch.no_grad():
            d = direction if drift == "fixed" else torch.randn(r, module.n_elements, generator=gen)
            module.weight.add_(
                d / d.norm(dim=1, keepdim=True) * module.weight.norm(dim=1, keepdim=True) * DRIFT_PER_UPDATE
            )
        module.advance_spectral_state(steps=steps)
        Q_skew = _scaled_skew(module)
        true_sigma = torch.linalg.matrix_norm(Q_skew, ord=2)
        worst = max(worst, ((true_sigma - _forward_sigma(module, Q_skew)) / true_sigma).max().item())
    return worst


@pytest.mark.parametrize(("seed", "drift"), [(0, "fixed"), (1, "fixed"), (3, "fixed"), (0, "random")])
def test_steps_per_update_tracks_drifting_sigma(seed, drift):
    assert SPECTRAL_STEPS_PER_UPDATE == 2
    assert _worst_tracking_error(seed, SPECTRAL_STEPS_PER_UPDATE, drift) <= SIGMA_TOLERANCE


def test_fewer_steps_per_update_do_not_track():
    """Why 2: with the state frozen (0) or one step per update, seed 1's fixed
    drift (the hardest of the fixtures above) exceeds the tolerance."""
    assert _worst_tracking_error(1, 0, "fixed") > SIGMA_TOLERANCE
    assert _worst_tracking_error(1, 1, "fixed") > SIGMA_TOLERANCE


# --------------------------------------------------------------------------- #
# BaseModelSetup.after_optimizer_step_adapters
# --------------------------------------------------------------------------- #


def _toy_transformer(hidden: int = 32) -> nn.Module:
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attn()

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(2)])

    return Toy()


def _wrapper(prefix: str, *, peft_type: PeftType = PeftType.OFT_2, dora_oft: bool = True) -> LoRAModuleWrapper:
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = peft_type
    config.dora_oft = dora_oft
    config.oft_scaled = False
    config.oft_block_size = 8
    config.oft_clipped_norm = 0.95
    config.oft_block_share = False
    config.oft_cans = False
    config.dropout_probability = 0.0
    wrapper = LoRAModuleWrapper(_toy_transformer(), prefix, config, [])
    with torch.no_grad():
        for module in wrapper.modules():
            if isinstance(module, OFTRotationModule):
                module.weight.normal_(0, 0.3)
    return wrapper


def _rotations(wrapper: LoRAModuleWrapper) -> list[OFTRotationModule]:
    return [m for m in wrapper.modules() if isinstance(m, OFTRotationModule)]


def test_after_optimizer_step_adapters_advances_trainable_oft_modules_once():
    torch.manual_seed(10)
    policy, fake, teacher = _wrapper("transformer"), _wrapper("transformer_fake"), _wrapper("transformer_teacher")
    lora = _wrapper("transformer_lora", peft_type=PeftType.LORA)
    model = SimpleNamespace(
        adapters=lambda: [policy, lora], fake_adapters=lambda: [fake], teacher_adapters=lambda: [teacher]
    )
    config = TrainConfig.default_values()

    trainable = [*_rotations(policy), *_rotations(fake)]
    assert len(trainable) == 8
    expected = []
    for rot in trainable:
        twin = _rotation(rot.block_size, r=rot.r)
        with torch.no_grad():
            twin.weight.copy_(rot.weight)
        twin.u_state.copy_(rot.u_state)
        twin.v_state.copy_(rot.v_state)
        twin.advance_spectral_state()
        expected.append(_state(twin))
    teacher_states = [_state(rot) for rot in _rotations(teacher)]

    BaseModelSetup.after_optimizer_step_adapters(None, model, config)

    for rot, state in zip(trainable, expected, strict=True):
        _assert_state_equal(rot, state)  # exactly SPECTRAL_STEPS_PER_UPDATE steps, once
    for rot, state in zip(_rotations(teacher), teacher_states, strict=True):
        _assert_state_equal(rot, state)  # frozen teacher untouched


def test_after_optimizer_step_adapters_without_adapters_is_a_noop():
    model = SimpleNamespace(adapters=list, fake_adapters=list)
    BaseModelSetup.after_optimizer_step_adapters(None, model, TrainConfig.default_values())
