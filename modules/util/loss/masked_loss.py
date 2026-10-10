import torch
from torch import Tensor


def clamp_mask(mask: Tensor, unmasked_weight: float | Tensor) -> Tensor:
    """Lower-bound the mask by the unmasked weight: a scalar, or one weight per
    sample shaped to broadcast against the mask (per-concept overrides)."""
    if isinstance(unmasked_weight, Tensor):
        return torch.clamp(torch.maximum(mask, unmasked_weight.to(dtype=mask.dtype, device=mask.device)), max=1)
    return torch.clamp(mask, unmasked_weight, 1)


def resolve_unmasked_weight(batch: dict, config, ndim: int) -> float | Tensor:
    """The run's unmasked weight, or a per-sample tensor where a concept overrides
    it. The dataloader carries ``concept.unmasked_weight_override`` per sample
    (a batch mixes concepts); a negative override inherits the global value.
    The tensor is shaped ``[B, 1, ...]`` with ``ndim`` dims so it broadcasts
    against a ``[B, C, H, W]`` mask or loss map."""
    override = batch.get("unmasked_weight_override")
    if not isinstance(override, Tensor):
        return config.unmasked_weight
    override = override.to(dtype=torch.float32)
    weight = torch.where(override < 0, torch.full_like(override, float(config.unmasked_weight)), override)
    return weight.view(-1, *([1] * (ndim - 1)))


def masked_mse_per_sample(pred: Tensor, target: Tensor, clamped_mask: Tensor) -> Tensor:
    """Squared error averaged per pixel inside the weighted region, one value
    per sample in fp32.

    Dividing by the mask's own weight sum (not the pixel count) makes the
    result independent of how much of the image the mask covers. Used for the
    DPO log-probs, where that matters more than for a supervised loss: the
    margin is a difference of these values and beta * margin runs through a
    sigmoid, so an area-scaled margin would change each pair's effective beta
    rather than merely reweight the pair. With an all-ones mask this equals the
    plain mean over C, H, W."""
    squared_error = (pred - target).pow(2)
    dims = list(range(1, squared_error.ndim))
    weighted = (squared_error * clamped_mask.to(dtype=squared_error.dtype)).sum(dim=dims, dtype=torch.float32)
    area = clamped_mask.to(dtype=torch.float32).expand_as(squared_error).sum(dim=dims)
    return weighted / area.clamp_min(torch.finfo(torch.float32).tiny)


def masked_losses(
    losses: Tensor,
    mask: Tensor,
    unmasked_weight: float | Tensor,
    normalize_masked_area_loss: bool,
) -> Tensor:
    clamped_mask = clamp_mask(mask, unmasked_weight)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    return losses


def masked_losses_with_prior(
    losses: Tensor,
    prior_losses: Tensor | None,
    mask: Tensor,
    unmasked_weight: float | Tensor,
    normalize_masked_area_loss: bool,
    masked_prior_preservation_weight: float,
) -> Tensor:
    clamped_mask = clamp_mask(mask, unmasked_weight)

    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    if masked_prior_preservation_weight == 0 or prior_losses is None:
        return losses

    clamped_mask = 1 - clamped_mask
    prior_losses *= clamped_mask * masked_prior_preservation_weight

    if normalize_masked_area_loss:
        prior_losses = prior_losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    return losses + prior_losses
