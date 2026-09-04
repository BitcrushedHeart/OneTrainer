"""DPO objective helpers shared by the fused [chosen; rejected] DPO path and the
memory-light path used by SFT + DPO add-on steps.

The light path evaluates a pair as single-sample forwards and backwards each
half on its own, so only one sample's autograd graph is alive at a time. That
is exact: the loss is a scalar function of the two per-sample log-probs, so
backwarding (dL/dlogp_c * logp_c_live) and then (dL/dlogp_r * logp_r_live)
accumulates precisely dL/dθ.
"""

from modules.util.enum.DPOObjective import DPOObjective

import torch
import torch.nn.functional as F
from torch import Tensor


def dpo_objective(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    ref_chosen_logp: Tensor,
    ref_rejected_logp: Tensor,
    *,
    beta: float | Tensor,
    objective: DPOObjective,
    ipo_tau: float,
    label_smoothing: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns (loss, dpo_loss, chosen_ratio, rejected_ratio, margin).

    ``loss`` and ``dpo_loss`` are scalars; the ratios and margin are per-sample.
    ``beta`` may be a scalar or a per-sample ``[B]`` tensor (per-concept beta
    overrides); it multiplies the per-sample margin elementwise either way, so
    the reduction below is unchanged.
    """
    chosen_ratio = policy_chosen_logp - ref_chosen_logp.detach()
    rejected_ratio = policy_rejected_logp - ref_rejected_logp.detach()
    margin = chosen_ratio - rejected_ratio

    if objective == DPOObjective.IPO:
        # IPO regresses the raw margin toward the fixed target 1/(2*tau)
        # instead of pushing it to infinity, which structurally resists
        # reward hacking. tau plays beta's role; label smoothing and beta
        # do not apply.
        dpo_loss = (margin - 1.0 / (2.0 * ipo_tau)).pow(2).mean()
        loss = dpo_loss
    else:
        logits = beta * margin
        dpo_loss = -F.logsigmoid(logits).mean()
        loss = dpo_loss
        if label_smoothing > 0:
            s = label_smoothing
            loss = (1 - s) * loss + s * (-F.logsigmoid(-logits).mean())

    return loss, dpo_loss, chosen_ratio, rejected_ratio, margin


def dpo_surrogate_weights(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    ref_chosen_logp: Tensor,
    ref_rejected_logp: Tensor,
    **objective_kwargs,
) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
    """Chain-rule weights for backwarding the two halves separately.

    Returns ``(g_chosen, g_rejected, outputs)`` where ``g_* = dL/d(logp_*)`` per
    sample (detached) and ``outputs`` is the ``dpo_objective`` tuple evaluated at
    the given values, detached, for metrics. Backwarding
    ``(g_chosen * logp_chosen_live).sum()`` and then
    ``(g_rejected * logp_rejected_live).sum()`` accumulates exactly ``dL/dθ``.
    """
    pc = policy_chosen_logp.detach().float().requires_grad_(True)
    pr = policy_rejected_logp.detach().float().requires_grad_(True)
    with torch.enable_grad():
        outputs = dpo_objective(
            pc,
            pr,
            ref_chosen_logp.detach().float(),
            ref_rejected_logp.detach().float(),
            **objective_kwargs,
        )
        g_c, g_r = torch.autograd.grad(outputs[0], [pc, pr])
    return g_c.detach(), g_r.detach(), tuple(o.detach() for o in outputs)


def split_dpo_pair_batch(batch: dict) -> tuple[dict, dict]:
    """Splits a paired batch into a chosen-only and a rejected-only batch of the
    same B. In the rejected batch every ``<key>_rejected`` takes the place of
    ``<key>``; keys without a rejected twin are shared by both."""
    chosen: dict = {}
    rejected: dict = {}
    for key, value in batch.items():
        if key.endswith("_rejected"):
            continue
        chosen[key] = value
        rejected[key] = batch.get(key + "_rejected", value)
    return chosen, rejected
