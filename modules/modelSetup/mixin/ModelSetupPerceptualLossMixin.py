"""Low-sigma perceptual term on the one-step clean estimate.

Plain v-MSE has the posterior mean as its minimiser; at low sigma, where the
model decides texture and edges, that mean is a soft image. This mixin *adds*
(never replaces) a perceptual distance between the clean estimate
``x0_hat = x_t - sigma * v_hat`` and the dataset latent ``x0``, measured in
the VAE decoder's intermediate feature maps (LPL, arXiv 2411.04873): no RGB,
no external feature network, no VGG checkerboard.

Weight per sample::

    w(sigma) = strength * min(((1 - sigma) / sigma)^2, gamma) * g(sigma)

``min(SNR, gamma)`` is the min-SNR cap (arXiv 2303.09556). It matters: for
``|v_hat - v|^2 + w * L(x0_hat)`` the optimum moves ``x0_hat`` by
``-(w * sigma^2 / 2) * grad L``, so a flat ``w`` pulls hardest at *high*
sigma. ``SNR ~ 1/sigma^2`` cancels that; the cap fades the term out below
``sigma ~ 1/sqrt(gamma)`` where the one-step estimate is already ``x0``.
``g`` is 1 up to ``perceptual_sigma_full`` and a cosine taper to 0 at
``perceptual_sigma_max``. The gate is deliberately wide: PixelGen
(arXiv 2602.02493) loses most of the gain below sigma <= 0.4, and under the
logit-normal + shift-3 sampler a sigma <= 0.3 gate would see ~2% of samples.

Features: ``conv_in -> mid_block -> up_blocks`` of the diffusers decoder, the
output of every up block taken as one feature map. Each map is
cross-normalized with the *prediction's* per-channel statistics (LPL), target
positions whose normalized energy exceeds ``_OUTLIER_ENERGY_FACTOR`` times
the median are masked (LPL's outlier mask, stand-in rule), and the masked
mean squared difference is averaged over blocks. The prediction and target
walks run in lockstep so only one target map is alive at a time; prediction
blocks run under non-reentrant activation checkpointing.

Cost: only samples inside the gate decode, on a random ``perceptual_crop``
window (latent edge ``crop // 8``) rather than the frame. The VAE has to be on
the train device (``setup_train_device`` keeps it there when the term is on;
latent caching would otherwise page it out).

Diagnostics accumulate in the mixin and are drained once per optimiser step
by :meth:`pop_perceptual_metrics`: ``loss`` (weighted term), ``raw`` (unweighted
feature distance), ``fraction`` (share of samples inside the gate), and on the
first micro-step of each optimiser step ``grad_norm_v`` / ``mse_grad_norm_v``,
both the gradient on the velocity output (the MSE one analytic). Set
``perceptual_strength`` so their ratio sits around 0.1-0.2 (LPL: a fifth).
"""

from __future__ import annotations

import math
import random
from abc import ABCMeta

from modules.model.BaseModel import BaseModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LossWeight import LossWeight

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

_OUTLIER_ENERGY_FACTOR = 8.0
_LATENT_DOWNSCALE = 8


def perceptual_weight(
    sigma: Tensor,
    strength: float,
    gamma: float,
    sigma_full: float,
    sigma_max: float,
) -> Tensor:
    """Per-sample weight ``strength * min(SNR, gamma) * g(sigma)`` (see module doc)."""
    sigma = sigma.reshape(-1).float().clamp(1e-4, 1.0)
    snr = ((1.0 - sigma) / sigma) ** 2
    if sigma_max > sigma_full:
        ramp = ((sigma - sigma_full) / (sigma_max - sigma_full)).clamp(0.0, 1.0)
        gate = 0.5 * (1.0 + torch.cos(math.pi * ramp))
    else:
        gate = (sigma <= sigma_full).float()
    gate = torch.where(sigma >= sigma_max, torch.zeros_like(gate), gate)
    return strength * torch.minimum(snr, torch.full_like(snr, gamma)) * gate


def random_latent_window(height: int, width: int, edge: int, rng: random.Random) -> tuple[int, int, int, int]:
    """``(y0, y1, x0, x1)`` of a random ``edge``-sized window; the full frame when ``edge`` is 0 or covers it."""
    if edge <= 0 or (edge >= height and edge >= width):
        return 0, height, 0, width
    edge_y = min(edge, height)
    edge_x = min(edge, width)
    y0 = rng.randrange(0, height - edge_y + 1)
    x0 = rng.randrange(0, width - edge_x + 1)
    return y0, y0 + edge_y, x0, x0 + edge_x


def cross_normalized_distance(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Per-sample masked mean squared distance between feature maps, both
    normalized with the prediction's per-channel spatial statistics (detached).
    ``mask`` is ``(B, 1, h, w)`` in latent resolution, resized to the map.

    The shared mean cancels, ``pred_n - target_n == (pred - target) / std``, so
    the normalized maps are never built: at the last up block of a 512 crop each
    full fp32 map is 128 MiB, and this keeps at most two alive at once (and
    only one saved for backward) at a point where the transformer's activations
    are all still held."""
    with torch.no_grad():
        pred32 = pred.detach().float()
        mean = pred32.mean(dim=(2, 3), keepdim=True)
        std = pred32.std(dim=(2, 3), keepdim=True) + 1e-6
        del pred32
        # copy=True: .float() on an fp32 map is the map itself, and this is in place
        energy = target.to(torch.float32, copy=True).sub_(mean).div_(std).pow_(2).sum(dim=1, keepdim=True)
        median = energy.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
        keep = (energy <= _OUTLIER_ENERGY_FACTOR * median).float()
        del energy
        if mask is not None:
            keep = keep * F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")

    diff = ((pred.float() - target) / std).pow(2).mean(dim=1, keepdim=True) * keep
    return diff.sum(dim=(1, 2, 3)) / keep.sum(dim=(1, 2, 3)).clamp_min(1.0)


def _run_block(block, sample: Tensor, use_checkpoint: bool) -> Tensor:
    if use_checkpoint and torch.is_grad_enabled() and sample.requires_grad:
        return checkpoint(block, sample, None, use_reentrant=False)
    return block(sample, None)


def decoder_feature_distance(
    decoder,
    z_pred: Tensor,
    z_target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Average cross-normalized distance over the decoder's up-block outputs,
    prediction and target walked in lockstep; the target under ``no_grad``."""
    with torch.no_grad():
        sample_t = decoder.mid_block(decoder.conv_in(z_target), None)
    sample_p = _run_block(decoder.mid_block, decoder.conv_in(z_pred), True)

    total = torch.zeros(z_pred.shape[0], device=z_pred.device, dtype=torch.float32)
    blocks = 0
    for block in decoder.up_blocks:
        with torch.no_grad():
            sample_t = block(sample_t, None)
        sample_p = _run_block(block, sample_p, True)
        total = total + cross_normalized_distance(sample_p, sample_t, mask)
        blocks += 1
    return total / max(blocks, 1)


class ModelSetupPerceptualLossMixin(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._perceptual_sums: dict[str, float] = {}
        self._perceptual_counts: dict[str, int] = {}
        self._perceptual_calls = 0
        self._perceptual_warned = False
        self._perceptual_rng = random.Random()

    # -- metrics -----------------------------------------------------------------

    def _perceptual_record(self, **values: float) -> None:
        for key, value in values.items():
            self._perceptual_sums[key] = self._perceptual_sums.get(key, 0.0) + float(value)
            self._perceptual_counts[key] = self._perceptual_counts.get(key, 0) + 1

    def pop_perceptual_metrics(self) -> dict[str, float]:
        """Averages since the last call, then reset. Empty when the term never ran."""
        metrics = {key: self._perceptual_sums[key] / self._perceptual_counts[key] for key in self._perceptual_sums}
        self._perceptual_sums = {}
        self._perceptual_counts = {}
        return metrics

    # -- guards ------------------------------------------------------------------

    @staticmethod
    def _perceptual_applies(batch: dict, data: dict, config: TrainConfig) -> bool:
        if config.perceptual_strength <= 0.0:
            return False
        if not torch.is_grad_enabled():
            # validation calls calculate_loss under no_grad; nothing to shape there
            return False
        if "_distill_latent_input" in batch:
            # DMD branch: the target is zeros, latent_image is not what is fitted
            return False
        if getattr(config, "distillation_teacher_lora_model_name", None):
            # teacher-target branch: the model is fitted to the teacher's velocity
            return False
        return all(key in data for key in ("predicted_latent", "sigma", "latent_image"))

    def _perceptual_warn_once(self, message: str) -> None:
        if not self._perceptual_warned:
            print(f"Perceptual loss skipped: {message}")
            self._perceptual_warned = True

    # -- the term ----------------------------------------------------------------

    def _add_perceptual_loss(
        self,
        model: BaseModel,
        batch: dict,
        data: dict,
        config: TrainConfig,
        losses: Tensor,
    ) -> Tensor:
        """Add ``w(sigma) * L`` to the per-sample ``losses`` (shape ``(B,)``) for
        every sample inside the gate. Returns ``losses`` unchanged when the term
        does not apply."""
        if not self._perceptual_applies(batch, data, config):
            return losses

        self._perceptual_calls += 1
        sigma = data["sigma"].reshape(-1).float()
        weight = perceptual_weight(
            sigma,
            config.perceptual_strength,
            config.perceptual_snr_gamma,
            config.perceptual_sigma_full,
            config.perceptual_sigma_max,
        )
        active = weight > 0
        self._perceptual_record(fraction=active.float().mean().item())
        if not bool(active.any()):
            return losses

        vae = getattr(model, "vae", None)
        decoder = getattr(vae, "decoder", None)
        if decoder is None or not all(hasattr(decoder, name) for name in ("conv_in", "mid_block", "up_blocks")):
            self._perceptual_warn_once("the model's VAE has no diffusers-style decoder")
            return losses
        vae_param = next(vae.parameters())
        if vae_param.device.type != losses.device.type:
            self._perceptual_warn_once(
                f"VAE is on {vae_param.device} while training on {losses.device}; "
                "setup_train_device keeps it resident when perceptual_strength > 0"
            )
            return losses

        idx = active.nonzero().flatten()
        pred_latent = data["predicted_latent"][idx]
        target_latent = data["latent_image"][idx].detach()
        latent_mask = batch.get("latent_mask") if config.masked_training else None
        if latent_mask is not None:
            latent_mask = latent_mask[idx].detach()

        height, width = pred_latent.shape[-2:]
        edge = config.perceptual_crop // _LATENT_DOWNSCALE if config.perceptual_crop > 0 else 0
        pred_crops, target_crops, mask_crops = [], [], []
        for i in range(pred_latent.shape[0]):
            y0, y1, x0, x1 = random_latent_window(height, width, edge, self._perceptual_rng)
            pred_crops.append(pred_latent[i : i + 1, :, y0:y1, x0:x1])
            target_crops.append(target_latent[i : i + 1, :, y0:y1, x0:x1])
            if latent_mask is not None:
                mask_crops.append(latent_mask[i : i + 1, :, y0:y1, x0:x1])
        pred_crop = torch.cat(pred_crops, dim=0)
        target_crop = torch.cat(target_crops, dim=0)
        mask_crop = torch.cat(mask_crops, dim=0) if mask_crops else None

        vae_dtype = vae_param.dtype
        z_pred = model.unscale_latents(pred_crop).to(dtype=vae_dtype)
        z_target = model.unscale_latents(target_crop).to(dtype=vae_dtype)
        post_quant_conv = getattr(vae, "post_quant_conv", None)
        if post_quant_conv is not None:
            z_pred = post_quant_conv(z_pred)
            with torch.no_grad():
                z_target = post_quant_conv(z_target)

        raw = decoder_feature_distance(decoder, z_pred, z_target, mask_crop)

        scale = config.loss_scaler.get_scale(config.batch_size, config.gradient_accumulation_steps)
        loss_weight = batch["loss_weight"].reshape(-1).float()[idx] if "loss_weight" in batch else 1.0
        weighted = raw * weight[idx] * scale * loss_weight

        self._perceptual_record(loss=weighted.sum().item(), raw=raw.mean().item())
        if (self._perceptual_calls - 1) % max(config.gradient_accumulation_steps, 1) == 0:
            self._perceptual_grad_norms(batch, data, config, weighted, scale)

        return losses.index_add(0, idx, weighted.to(dtype=losses.dtype))

    def _perceptual_grad_norms(self, batch: dict, data: dict, config: TrainConfig, weighted: Tensor, scale: float):
        """Gradient of the perceptual term and of the MSE term on the velocity
        output. Perceptual via autograd (re-runs decoder features only, the graph
        stops at the transformer output); MSE analytic."""
        predicted = data.get("predicted")
        if not isinstance(predicted, Tensor) or not predicted.requires_grad:
            return
        grad = torch.autograd.grad(weighted.sum(), predicted, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            return
        target = data["target"].to(dtype=torch.float32)
        residual = predicted.detach().float() - target
        numel = residual[0].numel()
        per_sample = torch.full((residual.shape[0],), 2.0 * config.mse_strength * scale / numel, device=residual.device)
        if "loss_weight" in batch:
            per_sample = per_sample * batch["loss_weight"].reshape(-1).float()
        if config.loss_weight_fn == LossWeight.SIGMA:
            per_sample = per_sample * data["sigma"].reshape(-1).float()
        mse_grad = residual * per_sample.view(-1, *([1] * (residual.dim() - 1)))
        self._perceptual_record(
            grad_norm_v=float(grad.detach().float().norm()),
            mse_grad_norm_v=float(mse_grad.norm()),
        )
