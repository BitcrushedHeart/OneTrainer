"""OFT rotation cache (config oft_rotation_cache): module and model-setup behaviour.

CPU-only. The cache keeps R(Q) of every Linear OFT rotation of the policy adapters in a
plain-attribute leaf refreshed once per optimizer update; forwards read the leaf, dL/dR
accumulates through a post-accumulate-grad hook (bf16 with stochastic rounding, or
exact for a float32 leaf, which the exactness tests use), and
BaseModelSetup.before_clip_adapters pulls it back into Q.grad through one float32 R(Q).

Covers: flag-off bit identity with the Phase-1 forward; cached vs uncached training
over accumulated micro-steps on a checkpointed hooked DoRA-OFT model (exact with a
float32 leaf, bounded with bf16 + autocast); Cayley call counts; the DPO reference
bypass; invalidation (load, restore helper, resume/partial materialization);
aot_eager compile without recompiles and with Phase-1's graph count; _apply;
requires_grad sync; which adapters get a cache; the stale-cache guard; the dedicated
rounding generator.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import DoRAOFTModule, LoRAModuleWrapper, OFTModule
from modules.module.oft_utils import (
    ROTATION_CACHE_STALE_RATIO,
    OFTRotationModule,
    set_rotation_cache_seed,
)
from modules.util import bf16_stochastic_rounding
from modules.util.checkpointing_util import CheckpointLayer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
import torch._dynamo
import torch._dynamo.testing
from torch import nn

import pytest

HIDDEN = 32
BLOCK = 8


class _Setup(BaseModelSetup):
    def __init__(self, debug_mode: bool = False):
        super().__init__(torch.device("cpu"), torch.device("cpu"), debug_mode)

    def create_parameters(self, model, config):
        return None

    def setup_optimizations(self, model, config):
        pass

    def setup_model(self, model, config):
        pass

    def setup_train_device(self, model, config):
        pass

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        return {}

    def calculate_loss(self, model, batch, data, config):
        return None

    def after_optimizer_step(self, model, config, train_progress):
        pass


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.to_k = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        return self.to_k(torch.tanh(self.to_q(x)))


class _Toy(nn.Module):
    def __init__(self, depth: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(depth)])
        self.proj = nn.Conv2d(4, 4, 1)  # a Conv2d OFT target; never run by forward

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def _config(*, block_share=False, dropout=0.0, debug=False) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = ModelType.Z_IMAGE
    config.training_method = TrainingMethod.LORA
    config.peft_type = PeftType.OFT_2
    config.dora_oft = True
    config.oft_scaled = True
    config.oft_block_size = BLOCK
    config.oft_clipped_norm = 0.95
    config.oft_block_share = block_share
    config.oft_cans = False
    config.dropout_probability = dropout
    config.oft_rotation_cache = True
    config.train_dtype = DataType.BFLOAT_16
    config.debug_mode = debug
    config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
    return config


def _build(config: TrainConfig, *, seed=0, prefix="transformer", std=0.3, checkpoint=True, compile_counter=None):
    torch.manual_seed(seed)
    toy = _Toy()
    wrapper = LoRAModuleWrapper(toy, prefix, config, [])
    wrapper.set_dropout(config.dropout_probability)
    with torch.no_grad():
        for module in wrapper.lora_modules.values():
            module.oft_R.weight.normal_(0, std)
            module.dora_multiplier.uniform_(0.8, 1.2)
            module.oft_R.warm_start_spectral_state()
    wrapper.hook_to_module()
    if checkpoint:
        for i, block in enumerate(toy.layers):
            layer = CheckpointLayer(block, None, torch.device("cpu"))
            if compile_counter is not None:
                layer.compile(fullgraph=True, backend=compile_counter)
            toy.layers[i] = layer
    toy.train()
    return toy, wrapper


def _model(policy, fake=(), teacher=()):
    return SimpleNamespace(
        adapters=lambda: [policy], fake_adapters=lambda: list(fake), teacher_adapters=lambda: list(teacher)
    )


def _linear_rotations(wrapper) -> list[OFTRotationModule]:
    return [m.oft_R for m in wrapper.lora_modules.values() if isinstance(m.orig_module, nn.Linear)]


def _enable(wrapper, dtype):
    for rotation in _linear_rotations(wrapper):
        rotation.enable_rotation_cache(dtype)


def _micro_step(toy, gen, *, autocast=False, dropout_seed=0):
    x = torch.randn(3, 5, HIDDEN, generator=gen)
    upstream = torch.randn(3, 5, HIDDEN, generator=gen)
    torch.manual_seed(dropout_seed)  # same dropout masks in twin runs (checkpoint replays the RNG)
    if autocast:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = toy(x)
    else:
        y = toy(x)
    (y.float() * upstream).sum().backward()


def _update(wrapper, setup, config, optimizer, *, cached):
    model = _model(wrapper)
    if cached:
        setup.before_clip_adapters(model, config)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    setup.after_optimizer_step_adapters(model, config)


def _phase1_forward(module: OFTRotationModule, x: torch.Tensor) -> torch.Tensor:
    """OFTRotationModule.forward before the rotation cache, verbatim."""
    required_dtype = x.dtype
    orig_shape = x.shape
    scaling_factor = 2 * math.sqrt(module.block_size - 1) if module.oft_scaled else 1
    effective_weight = module.weight / scaling_factor
    orth_rotate = module._cayley_batch(
        effective_weight, module.block_size, module.use_cayley_neumann, module.num_cayley_neumann_terms, module.oft_cans
    )
    orth_rotate = module.dropout(orth_rotate)
    orth_rotate = orth_rotate.to(required_dtype)
    rank = module.in_features // module.block_size if module.block_share else module.r
    x_reshaped = x.reshape(*x.shape[:-1], rank, module.block_size)
    if module.block_share:
        orth_rotate = orth_rotate.repeat(rank, 1, 1)
    x_rotated_reshaped = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)
    return x_rotated_reshaped.reshape(*orig_shape).to(required_dtype)


# --------------------------------------------------------------------------- #
# 1. flag off: bit-identical to Phase 1
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("use_autocast", [False, True])
@pytest.mark.parametrize(("block_share", "dropout"), [(False, 0.0), (True, 0.0), (False, 0.3)])
def test_flag_off_forward_and_grads_bit_identical_to_phase1(use_autocast, block_share, dropout):
    def rotation(seed):
        torch.manual_seed(seed)
        module = OFTRotationModule(
            r=1 if block_share else 4,
            n_elements=BLOCK * (BLOCK - 1) // 2,
            block_size=BLOCK,
            in_features=4 * BLOCK,
            block_share=block_share,
            oft_scaled=True,
            dropout_probability=dropout,
            oft_clipped_norm=0.95,
        )
        with torch.no_grad():
            module.weight.normal_(0, 0.4)
        return module

    new, old = rotation(1), rotation(1)
    old.u_state.copy_(new.u_state)
    old.v_state.copy_(new.v_state)
    x = torch.randn(2, 6, 4 * BLOCK)
    for fn, module in ((new.forward, new), (lambda t: _phase1_forward(old, t), old)):
        torch.manual_seed(5)
        if use_autocast:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = fn(x)
        else:
            y = fn(x)
        (y.float() * torch.linspace(-1, 1, y.numel()).reshape(y.shape)).sum().backward()
        module.y = y.detach()
    assert not new.rotation_cache_enabled and new.rotation_leaf is None
    assert torch.equal(new.y, old.y)
    assert torch.equal(new.weight.grad, old.weight.grad)


def test_flag_off_setup_allocates_nothing_and_hooks_are_no_ops():
    config = _config()
    config.oft_rotation_cache = False
    toy, wrapper = _build(config)
    setup, model = _Setup(), _model(wrapper)
    setup.setup_adapter_rotation_cache(model, config)
    assert all(r.rotation_leaf is None and not r.rotation_cache_active for r in _linear_rotations(wrapper))
    _micro_step(toy, torch.Generator().manual_seed(0))
    grads = [(p, p.grad.clone()) for p in wrapper.parameters() if p.grad is not None]
    assert grads
    setup.before_clip_adapters(model, config)
    assert all(torch.equal(p.grad, g) for p, g in grads)


# --------------------------------------------------------------------------- #
# 2. cached vs uncached training
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("block_share", "dropout"), [(False, 0.0), (True, 0.0), (False, 0.25)])
def test_cached_training_matches_uncached_exactly_with_float32_leaf(block_share, dropout):
    config = _config(block_share=block_share, dropout=dropout, debug=True)
    toy_on, on = _build(config)
    toy_off, off = _build(config)
    _enable(on, torch.float32)
    setup = _Setup(debug_mode=True)
    opt_on = torch.optim.SGD(on.parameters(), lr=1e-2)
    opt_off = torch.optim.SGD(off.parameters(), lr=1e-2)
    gen_on, gen_off = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)

    for update in range(2):
        for micro in range(4):
            _micro_step(toy_on, gen_on, dropout_seed=100 * update + micro)
            _micro_step(toy_off, gen_off, dropout_seed=100 * update + micro)
        _update(on, setup, config, opt_on, cached=True)
        _update(off, setup, config, opt_off, cached=False)

    state_on, state_off = on.state_dict(), off.state_dict()
    assert state_on.keys() == state_off.keys()
    for key in state_on:
        assert (state_on[key].float() - state_off[key].float()).abs().max().item() <= 1e-6, key
    for rotation in _linear_rotations(on):
        assert rotation.rotation_grad_acc is None  # released at materialization


def test_cached_bf16_accumulation_stays_close_to_uncached_under_autocast():
    config = _config()
    toy_on, on = _build(config)
    toy_off, off = _build(config)
    _enable(on, torch.bfloat16)
    set_rotation_cache_seed(0, "cpu")
    gen_on, gen_off = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    for micro in range(4):
        _micro_step(toy_on, gen_on, autocast=True, dropout_seed=micro)
        _micro_step(toy_off, gen_off, autocast=True, dropout_seed=micro)
    _Setup(debug_mode=True).before_clip_adapters(_model(on), config)
    for rot_on, rot_off in zip(_linear_rotations(on), _linear_rotations(off), strict=True):
        rel = (rot_on.weight.grad - rot_off.weight.grad).norm() / rot_off.weight.grad.norm()
        assert rel.item() <= 2e-2


# --------------------------------------------------------------------------- #
# 3. Cayley call counts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cached", [True, False])
def test_cayley_calls_per_micro_step_and_update(monkeypatch, cached):
    config = _config(debug=True)
    toy, wrapper = _build(config)
    if cached:
        _enable(wrapper, torch.float32)
    calls = []
    original = OFTRotationModule._cayley_batch
    monkeypatch.setattr(
        OFTRotationModule, "_cayley_batch", lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1]
    )
    setup = _Setup(debug_mode=True)
    opt = torch.optim.SGD(wrapper.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    linear = len(_linear_rotations(wrapper))

    _micro_step(toy, gen)
    per_micro = len(calls)
    calls.clear()
    _micro_step(toy, gen)
    _update(wrapper, setup, config, opt, cached=cached)
    micro_and_update = len(calls)

    if cached:
        assert per_micro == 0
        assert micro_and_update == 2 * linear  # one materialize + one refresh per module
    else:
        assert per_micro == 2 * linear  # forward + checkpoint recompute
        assert micro_and_update == 2 * linear


# --------------------------------------------------------------------------- #
# 4. DPO EXISTING_ADAPTER reference passes bypass the cache
# --------------------------------------------------------------------------- #


def test_reference_pass_bypasses_cache_and_matches_uncached_reference():
    config = _config(debug=True)
    toy_on, on = _build(config)
    toy_off, off = _build(config)
    _enable(on, torch.bfloat16)
    setup_on, setup_off = _Setup(debug_mode=True), _Setup(debug_mode=True)
    x = torch.randn(2, 5, HIDDEN)

    def reference_output(setup, toy, wrapper):
        with (
            setup.reference_model(_model(wrapper), config),
            torch.no_grad(),
            torch.autocast("cpu", dtype=torch.bfloat16),
        ):
            inside = [r.rotation_cache_active for r in _linear_rotations(wrapper)]
            return toy(x), inside

    reference_output(setup_on, toy_on, on)  # captures the reference = current policy
    reference_output(setup_off, toy_off, off)
    # The policy moves on; the leaf follows it, the reference must not.
    for wrapper in (on, off):
        with torch.no_grad():
            for rotation in _linear_rotations(wrapper):
                rotation.weight.add_(0.05)
        setup_on.after_optimizer_step_adapters(_model(wrapper), config)

    ref_on, inside = reference_output(setup_on, toy_on, on)
    ref_off, _ = reference_output(setup_off, toy_off, off)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        policy_on = toy_on(x)

    assert inside == [False] * len(inside)
    assert all(r.rotation_cache_active for r in _linear_rotations(on))
    assert torch.equal(ref_on, ref_off)
    assert not torch.allclose(ref_on.float(), policy_on.float(), atol=1e-3)
    assert all(r.rotation_grad_acc is None for r in _linear_rotations(on))

    with pytest.raises(ValueError), setup_on.reference_model(_model(on), config):
        raise ValueError("boom")
    assert all(r.rotation_cache_active for r in _linear_rotations(on))


# --------------------------------------------------------------------------- #
# 5. invalidation points at module / setup level
# --------------------------------------------------------------------------- #


def _staleness(rotation):
    return rotation.rotation_cache_staleness(rotation.rotation_leaf, rotation._fp32_rotation(rotation.weight.detach()))


def test_load_state_dict_refreshes_the_cache():
    config = _config()
    _, wrapper = _build(config, seed=0)
    _, donor = _build(config, seed=1, prefix="transformer")
    _enable(wrapper, torch.float32)
    old_leaf = _linear_rotations(wrapper)[0].rotation_leaf.detach().clone()
    wrapper.load_state_dict(donor.state_dict())
    rotation = _linear_rotations(wrapper)[0]
    assert not torch.equal(rotation.rotation_leaf, old_leaf)
    assert _staleness(rotation).item() == 0.0


def test_refresh_refuses_pending_rotation_gradients():
    config = _config()
    toy, wrapper = _build(config)
    _enable(wrapper, torch.float32)
    _micro_step(toy, torch.Generator().manual_seed(0))
    with pytest.raises(RuntimeError, match="materialize"):
        _linear_rotations(wrapper)[0].refresh_rotation_cache()


@pytest.mark.parametrize("warm", [True, False])
def test_restore_helper_warms_on_request_and_refreshes(warm):
    config = _config()
    _, wrapper = _build(config)
    _, fake = _build(config, seed=3, prefix="transformer_fake")
    _enable(wrapper, torch.float32)
    rotations = [*_linear_rotations(wrapper), *_linear_rotations(fake)]
    versions = [r._spectral_state_version for r in rotations]
    states = [r.u_state.clone() for r in rotations]
    with torch.no_grad():
        for rotation in rotations:
            rotation.weight.add_(torch.randn_like(rotation.weight) * 0.2)  # e.g. param.data.copy_(best)
    assert _staleness(rotations[0]).item() > ROTATION_CACHE_STALE_RATIO

    _Setup().refresh_adapters_after_param_restore(_model(wrapper, fake=[fake]), config, warm=warm)

    for rotation, version, state in zip(rotations, versions, states, strict=True):
        assert (rotation._spectral_state_version > version) == warm
        assert torch.equal(rotation.u_state, state) != warm
    for rotation in _linear_rotations(wrapper):
        assert _staleness(rotation).item() == 0.0


def test_partial_materialization_and_resume_match_an_uninterrupted_window():
    """Mid-window staging materializes into Q.grad; a resumed run restores Q, Q.grad and
    the clip state, refills the cache and finishes the window: same update as without the stop."""
    config = _config(debug=True)
    setup = _Setup(debug_mode=True)
    toy_ref, ref = _build(config)
    toy_a, run_a = _build(config)
    _enable(ref, torch.float32)
    _enable(run_a, torch.float32)
    gen_ref, gen_a = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)

    for micro in range(4):
        _micro_step(toy_ref, gen_ref, dropout_seed=micro)
    setup.before_clip_adapters(_model(ref), config)

    for micro in range(2):
        _micro_step(toy_a, gen_a, dropout_seed=micro)
    setup.before_clip_adapters(_model(run_a), config)  # _stage_accumulator_state_for_save
    saved_grads = [None if p.grad is None else p.grad.clone() for p in run_a.parameters()]
    saved_state = [(r.u_state.clone(), r.v_state.clone()) for r in _linear_rotations(run_a)]

    toy_b, run_b = _build(config, seed=0)  # a fresh process: same base model...
    with torch.no_grad():
        for p in run_b.parameters():
            p.add_(torch.randn_like(p))  # ...whose adapter only the backup restores
    run_b.load_state_dict(run_a.state_dict())
    for rotation, (u, v) in zip(_linear_rotations(run_b), saved_state, strict=True):
        rotation.u_state.copy_(u)
        rotation.v_state.copy_(v)
    for p, grad in zip(run_b.parameters(), saved_grads, strict=True):
        p.grad = None if grad is None else grad.clone()  # _restore_accumulator_state
    _enable(run_b, torch.float32)  # start(): setup_adapter_rotation_cache after load
    gen_b = torch.Generator().manual_seed(4)
    for _ in range(2):  # replay the data stream position
        torch.randn(3, 5, HIDDEN, generator=gen_b)
        torch.randn(3, 5, HIDDEN, generator=gen_b)
    for micro in range(2, 4):
        _micro_step(toy_b, gen_b, dropout_seed=micro)
    setup.before_clip_adapters(_model(run_b), config)

    for p_ref, p_b in zip(ref.parameters(), run_b.parameters(), strict=True):
        if p_ref.grad is None:  # the Conv2d adapter never runs
            assert p_b.grad is None
            continue
        assert (p_ref.grad - p_b.grad).abs().max().item() <= 1e-6


# --------------------------------------------------------------------------- #
# 8. torch.compile
# --------------------------------------------------------------------------- #


def _compiled_run(cached: bool) -> tuple[int, int]:
    torch._dynamo.reset()
    config = _config(debug=True)
    config.oft_rotation_cache = cached
    counter = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
    toy, wrapper = _build(config, compile_counter=counter)
    if cached:
        _enable(wrapper, torch.bfloat16)
    setup = _Setup(debug_mode=True)
    opt = torch.optim.SGD(wrapper.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    x_val = torch.randn(2, 5, HIDDEN)
    frames = []
    with torch._functorch.config.patch(backward_pass_autocast="off"):
        for update in range(2):
            for micro in range(3):
                set_rotation_cache_seed(update * 10 + micro, "cpu")
                _micro_step(toy, gen, autocast=True, dropout_seed=micro)
                with (
                    setup.reference_model(_model(wrapper), config),
                    torch.no_grad(),
                    torch.autocast("cpu", dtype=torch.bfloat16),
                ):
                    toy(x_val)
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                toy(x_val)  # validation
            _update(wrapper, setup, config, opt, cached=cached)
            frames.append(counter.frame_count)
    torch._dynamo.reset()
    return frames[0], frames[1]


def test_compiled_checkpointed_cache_no_recompile_and_phase1_graph_count():
    cached_first, cached_second = _compiled_run(cached=True)
    phase1_first, phase1_second = _compiled_run(cached=False)
    assert cached_first == cached_second  # no recompile across refreshes / updates
    assert phase1_first == phase1_second
    assert cached_second == phase1_second  # same variants: grad, no_grad, reference


# --------------------------------------------------------------------------- #
# 9. _apply, 11. requires_grad, 13. which adapters are cached
# --------------------------------------------------------------------------- #


def test_apply_moves_device_only_and_rehooks_an_incompatible_move():
    config = _config()
    toy, wrapper = _build(config)
    _enable(wrapper, torch.bfloat16)
    _micro_step(toy, torch.Generator().manual_seed(0))
    module = next(iter(wrapper.lora_modules.values()))
    rotation = module.oft_R
    leaf, hook = rotation.rotation_leaf, rotation._rotation_leaf_hook

    module.to(dtype=torch.float64)  # LoRAModuleWrapper.to(dtype=lora_weight_dtype)
    assert rotation.weight.dtype == torch.float64
    assert rotation.rotation_leaf is leaf and rotation.rotation_leaf.dtype == torch.bfloat16
    assert rotation._rotation_leaf_hook is hook

    # cpu -> meta is not shallow-copy compatible (cpu <-> cuda is: that path keeps the
    # object via .data =, checked on GPU by scripts/util/_perf_verify_rotation_cache.py).
    module.to("meta")
    moved = rotation.rotation_leaf
    assert moved.device.type == "meta" and moved.is_leaf and moved.requires_grad
    assert moved.dtype == torch.bfloat16
    assert rotation._rotation_leaf_hook is not hook
    assert rotation.rotation_grad_acc.device.type == "meta"


def test_requires_grad_syncs_at_refresh_and_frozen_modules_skip_accumulation():
    config = _config(debug=True)
    toy, wrapper = _build(config)
    frozen = _linear_rotations(wrapper)[0]
    frozen.weight.requires_grad_(False)
    _enable(wrapper, torch.float32)
    setup = _Setup(debug_mode=True)
    gen = torch.Generator().manual_seed(0)

    assert not frozen.rotation_leaf.requires_grad
    _micro_step(toy, gen)
    assert frozen.rotation_grad_acc is None
    setup.before_clip_adapters(_model(wrapper), config)
    assert frozen.weight.grad is None

    frozen.weight.requires_grad_(True)  # e.g. after_optimizer_step re-enabling a part
    setup.after_optimizer_step_adapters(_model(wrapper), config)
    assert frozen.rotation_leaf.requires_grad
    _micro_step(toy, gen)
    assert frozen.rotation_grad_acc is not None

    frozen.weight.requires_grad_(False)  # frozen mid-window: pending gradient is dropped
    setup.before_clip_adapters(_model(wrapper), config)
    assert frozen.rotation_grad_acc is None and frozen.weight.grad is None


def test_setup_caches_only_policy_linear_oft_modules(capsys):
    config = _config()
    _, policy = _build(config)
    _, fake = _build(config, seed=1, prefix="transformer_fake")
    _, teacher = _build(config, seed=2, prefix="transformer_teacher")
    model = _model(policy, fake=[fake], teacher=[teacher])

    _Setup().setup_adapter_rotation_cache(model, config)

    for name, module in policy.lora_modules.items():
        is_linear = isinstance(module.orig_module, nn.Linear)
        assert module.oft_R.rotation_cache_enabled == is_linear, name
        assert module.oft_R.rotation_cache_active == is_linear, name
        assert module.oft_R.rotation_leaf is None or module.oft_R.rotation_leaf.dtype == torch.bfloat16
    assert any(isinstance(m.orig_module, nn.Conv2d) for m in policy.lora_modules.values())
    for wrapper in (fake, teacher):
        assert all(not m.oft_R.rotation_cache_enabled for m in wrapper.lora_modules.values())
    assert "OFT rotation cache: 4 modules" in capsys.readouterr().out


def test_non_oft_run_is_ignored_by_setup():
    config = _config()
    config.peft_type = PeftType.LORA
    assert not config.oft_rotation_cache_applies()
    policy = SimpleNamespace(lora_modules={"x": object()})
    _Setup().setup_adapter_rotation_cache(_model(policy), config)  # must not touch the wrapper


# --------------------------------------------------------------------------- #
# stale-cache guard (amendment 3)
# --------------------------------------------------------------------------- #


def test_staleness_score_bounds_fresh_bf16_rounding_and_flags_small_changes():
    torch.manual_seed(0)
    for std in (0.0, 1e-4, 2e-3, 0.05, 1.0):
        rotation = OFTRotationModule(
            r=4, n_elements=16 * 15 // 2, block_size=16, in_features=64, oft_scaled=True, oft_clipped_norm=0.95
        )
        with torch.no_grad():
            rotation.weight.normal_(0, std) if std else rotation.weight.zero_()
        rotation.warm_start_spectral_state()
        rotation.enable_rotation_cache(torch.bfloat16)
        assert _staleness(rotation).item() <= 1.0 + 2e-3, std  # round-to-nearest: at most half an ulp
        with torch.no_grad():
            rotation.weight.add_(torch.randn_like(rotation.weight) * 1e-5)  # a tiny optimizer step
        assert _staleness(rotation).item() > ROTATION_CACHE_STALE_RATIO, std


def _stale_window(debug: bool):
    config = _config(debug=debug)
    toy, wrapper = _build(config)
    _enable(wrapper, torch.bfloat16)
    setup = _Setup(debug_mode=False)
    with torch.no_grad():
        _linear_rotations(wrapper)[1].weight.add_(1e-3)  # a restore that forgot to refresh
    _micro_step(toy, torch.Generator().manual_seed(0), autocast=True)
    return setup, wrapper, config, toy


def test_stale_cache_raises_in_debug_mode_naming_the_module():
    setup, wrapper, config, _ = _stale_window(debug=True)
    name = list(wrapper.lora_modules)[1]
    with pytest.raises(RuntimeError, match=name.replace(".", r"\.")):
        setup.before_clip_adapters(_model(wrapper), config)


def test_stale_cache_warns_once_per_run_otherwise():
    setup, wrapper, config, toy = _stale_window(debug=False)
    with pytest.warns(UserWarning, match="stale"):
        setup.before_clip_adapters(_model(wrapper), config)
    # Still stale in the next window (never refreshed): no second warning.
    for rotation in _linear_rotations(wrapper):
        rotation.rotation_grad_acc = None
    _micro_step(toy, torch.Generator().manual_seed(1), autocast=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        setup.before_clip_adapters(_model(wrapper), config)
    assert not [w for w in caught if "stale" in str(w.message)]


def test_clip_state_change_without_refresh_is_stale_and_fresh_training_never_is():
    config = _config(debug=True)
    toy, wrapper = _build(config)
    _enable(wrapper, torch.bfloat16)
    setup = _Setup(debug_mode=True)
    opt = torch.optim.SGD(wrapper.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    for update in range(3):  # correct bookkeeping: never trips the guard
        for micro in range(2):
            set_rotation_cache_seed(update * 10 + micro, "cpu")
            _micro_step(toy, gen, autocast=True)
        _update(wrapper, setup, config, opt, cached=True)

    _linear_rotations(wrapper)[0].advance_spectral_state()  # clip state moved, cache not refreshed
    _micro_step(toy, gen, autocast=True)
    with pytest.raises(RuntimeError, match="stale"):
        setup.before_clip_adapters(_model(wrapper), config)


# --------------------------------------------------------------------------- #
# dedicated rounding generator
# --------------------------------------------------------------------------- #


def test_accumulator_rounding_uses_its_own_seeded_generator():
    config = _config()

    def accumulate(seed):
        toy, wrapper = _build(config)
        _enable(wrapper, torch.bfloat16)
        bf16_stochastic_rounding.set_seed(42, torch.device("cpu"))
        optimizer_state = bf16_stochastic_rounding.generator.get_state().clone()
        set_rotation_cache_seed(seed, "cpu")
        gen = torch.Generator().manual_seed(0)
        for _ in range(3):
            _micro_step(toy, gen, autocast=True)
        assert torch.equal(bf16_stochastic_rounding.generator.get_state(), optimizer_state)
        return [r.rotation_grad_acc.clone() for r in _linear_rotations(wrapper)]

    first, again, other = accumulate(7), accumulate(7), accumulate(8)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_dora_oft_and_plain_oft_both_cache():
    for cls in (OFTModule, DoRAOFTModule):
        linear = nn.Linear(16, 16)
        module = cls(
            prefix="t", orig_module=linear, oft_block_size=8, block_share=False, oft_scaled=False, oft_clipped_norm=0.95
        )
        with torch.no_grad():
            module.oft_R.weight.normal_(0, 0.2)
        module.hook_to_module()
        x = torch.randn(3, 16)
        before = linear(x)
        module.oft_R.enable_rotation_cache(torch.float32)
        torch.testing.assert_close(linear(x), before, rtol=0, atol=1e-6)
        module.remove_hook_from_module()
