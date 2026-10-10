"""Masked DPO: each half's log-prob is the per-pixel average error inside its
own clamped mask, in both the fused [chosen; rejected] path and the light path.

Drives the real methods on CPU with a per-pixel (1x1 conv) net standing in for
the transformer. Because the net is per-pixel, a masked loss over a region must
equal the unmasked loss over that region cropped out, which gives exact
reference values without reimplementing the objective."""

from contextlib import contextmanager

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.ModelType import ModelType
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TrainingMethod import TrainingMethod

import torch

import pytest

BATCH, CHANNELS, H, W = 2, 3, 4, 6


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(CHANNELS, 5, 1)
        self.c2 = torch.nn.Conv2d(5, CHANNELS, 1)

    def forward(self, x):
        return self.c2(torch.tanh(self.c1(x)))


class _Model:
    def __init__(self):
        torch.manual_seed(0)
        self.net = _Net()
        torch.manual_seed(1)
        self.reference_net = _Net().requires_grad_(False)


class _FakeSetup(BaseModelSetup):
    def __init__(self):
        super().__init__(torch.device("cpu"), torch.device("cpu"), False)
        self.reference_depth = 0

    def create_parameters(self, model, config):
        raise NotImplementedError

    def setup_optimizations(self, model, config):
        raise NotImplementedError

    def setup_model(self, model, config):
        raise NotImplementedError

    def setup_train_device(self, model, config):
        raise NotImplementedError

    def after_optimizer_step(self, model, config, train_progress):
        raise NotImplementedError

    @contextmanager
    def reference_model(self, model, config):
        self.reference_depth += 1
        try:
            yield
        finally:
            self.reference_depth -= 1

    def predict(self, model, batch, config, train_progress, *, deterministic=False):
        net = model.reference_net if self.reference_depth > 0 else model.net
        return {
            "predicted": net(batch["latent_image"]),
            "target": batch["target"],
            "timestep": batch["timestep"][: batch["latent_image"].shape[0]],
        }

    def calculate_loss(self, model, batch, data, config):
        return (data["predicted"] - data["target"]).pow(2).mean()


def _batch(seed: int = 7) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "latent_image": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "latent_image_rejected": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "target": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "target_rejected": torch.randn(BATCH, CHANNELS, H, W, generator=generator),
        "timestep": torch.tensor([0.1, 0.6, 0.1, 0.6]),
    }


def _left_mask() -> torch.Tensor:
    mask = torch.zeros(BATCH, 1, H, W)
    mask[..., : W // 2] = 1.0
    return mask


def _right_mask() -> torch.Tensor:
    mask = torch.zeros(BATCH, 1, H, W)
    mask[..., W // 2 :] = 1.0
    return mask


def _crop(batch: dict, chosen_cols: slice, rejected_cols: slice) -> dict:
    cropped = dict(batch)
    for key in ("latent_image", "target"):
        cropped[key] = batch[key][..., chosen_cols]
        cropped[key + "_rejected"] = batch[key + "_rejected"][..., rejected_cols]
    return cropped


def _config(*, masked: bool, unmasked_weight: float = 0.0, reuse: bool = True, mix: float = 0.0) -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_beta = 2.0
    config.rlhf_dpo_objective = DPOObjective.SIGMOID
    config.rlhf_supervised_mix = mix
    config.rlhf_dpo_light_reuse_rejected_graph = reuse
    config.gradient_checkpointing = GradientCheckpointingMethod.ON
    config.masked_training = masked
    config.unmasked_weight = unmasked_weight
    for te in (config.text_encoder, config.text_encoder_2, config.text_encoder_3, config.text_encoder_4):
        te.dropout_probability = 0.0
    return config


def _grads(model: _Model) -> list[torch.Tensor]:
    return [p.grad.clone() for p in model.net.parameters()]


def _run_fused(config: TrainConfig, batch: dict):
    setup, model = _FakeSetup(), _Model()
    loss = setup.calculate_dpo_loss(model, batch, config, None)
    loss.backward()
    return loss.detach(), _grads(model)


def _run_light(config: TrainConfig, batch: dict):
    setup, model = _FakeSetup(), _Model()
    loss = setup.calculate_dpo_loss_light(model, batch, config, None, lambda term: term.backward())
    return loss.detach(), _grads(model)


def _assert_same(a, b):
    loss_a, grads_a = a
    loss_b, grads_b = b
    assert torch.allclose(loss_a, loss_b, atol=1e-6)
    for ga, gb in zip(grads_a, grads_b, strict=True):
        assert torch.allclose(ga, gb, atol=1e-6)


@pytest.mark.parametrize("run", [_run_fused, _run_light], ids=["fused", "light"])
def test_an_all_ones_mask_reproduces_unmasked_dpo(run):
    masked = dict(_batch(), latent_mask=torch.ones(BATCH, 1, H, W))
    _assert_same(run(_config(masked=True), masked), run(_config(masked=False), _batch()))


@pytest.mark.parametrize("run", [_run_fused, _run_light], ids=["fused", "light"])
def test_the_chosen_mask_covers_both_halves_when_the_rejected_has_none(run):
    masked = dict(_batch(), latent_mask=_left_mask())
    left = slice(0, W // 2)
    _assert_same(run(_config(masked=True), masked), run(_config(masked=False), _crop(_batch(), left, left)))


@pytest.mark.parametrize("run", [_run_fused, _run_light], ids=["fused", "light"])
def test_each_half_uses_its_own_mask(run):
    masked = dict(_batch(), latent_mask=_left_mask(), latent_mask_rejected=_right_mask())
    reference = _crop(_batch(), slice(0, W // 2), slice(W // 2, W))
    _assert_same(run(_config(masked=True), masked), run(_config(masked=False), reference))


@pytest.mark.parametrize("reuse", [True, False], ids=["held_graph", "five_forwards"])
@pytest.mark.parametrize("mix", [0.0, 0.3], ids=["no_mix", "supervised_mix"])
def test_light_path_matches_the_fused_path_under_masks(reuse, mix):
    batch = dict(_batch(), latent_mask=_left_mask(), latent_mask_rejected=_right_mask())
    batch["unmasked_weight_override"] = torch.tensor([-1.0, 0.25])
    config = _config(masked=True, unmasked_weight=0.1, reuse=reuse, mix=mix)
    _assert_same(_run_light(config, batch), _run_fused(config, batch))


def test_the_margin_does_not_scale_with_the_mask_area():
    # Same per-pixel content, mask covering a third vs the whole latent: the
    # per-pixel average inside the region, and so the margin, must agree.
    batch = _batch()
    for key in ("latent_image", "target"):
        batch[key] = batch[key][..., :2].repeat(1, 1, 1, 3)
        batch[key + "_rejected"] = batch[key + "_rejected"][..., :2].repeat(1, 1, 1, 3)
    small = torch.zeros(BATCH, 1, H, W)
    small[..., :2] = 1.0
    config = _config(masked=True)
    _assert_same(
        _run_fused(config, dict(batch, latent_mask=small)),
        _run_fused(config, dict(batch, latent_mask=torch.ones(BATCH, 1, H, W))),
    )


def test_mask_weights_apply_the_per_concept_floor():
    batch = {"latent_mask": _left_mask(), "unmasked_weight_override": torch.tensor([-1.0, 0.4])}
    weights = _FakeSetup._dpo_mask_weights(batch, _config(masked=True, unmasked_weight=0.1))
    assert weights.dtype == torch.float32
    assert weights[0].min().item() == pytest.approx(0.1)
    assert weights[1].min().item() == pytest.approx(0.4)
    assert weights.max().item() == 1.0


def test_mask_weights_are_off_without_masked_training_or_for_inpainting_models():
    batch = {"latent_mask": _left_mask()}
    assert _FakeSetup._dpo_mask_weights(batch, _config(masked=False)) is None
    assert _FakeSetup._dpo_mask_weights({}, _config(masked=True)) is None
    inpainting = _config(masked=True)
    inpainting.model_type = ModelType.STABLE_DIFFUSION_15_INPAINTING
    assert _FakeSetup._dpo_mask_weights(batch, inpainting) is None
