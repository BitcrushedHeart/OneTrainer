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


def dpo_chosen_anchor_active(chosen_anchor: float | Tensor) -> bool:
    """Whether a resolved chosen anchor adds a term. A per-pair ``[B]`` tensor
    only comes back when the batch's pairs resolve to different coefficients
    (see BaseModelSetup._resolve_chosen_anchor), so at least one of them is > 0."""
    return isinstance(chosen_anchor, Tensor) or chosen_anchor > 0


def dpo_chosen_anchor_penalties(
    chosen_ratio: Tensor,
    *,
    beta: float | Tensor,
    objective: DPOObjective,
    chosen_anchor: float | Tensor,
) -> Tensor:
    """Per-pair DPOP chosen-likelihood anchor, ``[B]``:
    ``chosen_anchor * scale * relu(-chosen_ratio)`` with ``scale`` the pair's
    beta (1 for IPO). See :func:`dpo_chosen_anchor_penalty` for the term;
    this is its per-pair form, whose mean is that term. ``chosen_anchor`` may
    be a per-pair ``[B]`` tensor (per-concept ``dpo_chosen_anchor_override``)."""
    scale = 1.0 if objective == DPOObjective.IPO else beta
    if isinstance(chosen_anchor, Tensor):
        chosen_anchor = chosen_anchor.to(chosen_ratio.device, chosen_ratio.dtype).reshape(chosen_ratio.shape)
    return chosen_anchor * (scale * F.relu(-chosen_ratio))


def dpo_chosen_anchor_penalty(
    chosen_ratio: Tensor,
    *,
    beta: float | Tensor,
    objective: DPOObjective,
    chosen_anchor: float | Tensor,
) -> Tensor:
    """DPOP's chosen-likelihood anchor (Pal et al. 2024, "Smaug"), as a separate
    additive term: ``chosen_anchor * mean(beta * relu(-chosen_ratio))``.

    ``-chosen_ratio`` is ``err_policy(chosen) - err_ref(chosen)``, so the term is
    one-sided: it is zero (and has zero gradient) while the policy fits the
    chosen half at least as well as the reference, and only pushes back when the
    margin is being won by making the chosen half worse too. Unlike an SFT term
    on the chosen image it never pulls the policy toward copying it.

    Scaled by the pair's beta because DPO pushes each half with weight
    ``beta * sigmoid(-beta * margin)`` (about beta/2 near zero margin). Its direct
    push on the chosen log-prob is always upward; the chosen half falls because
    the rejected half's push-down spills onto it through shared features. An
    unscaled anchor (like rlhf_supervised_mix) loses to that by a factor of beta;
    ``chosen_anchor = 0.5`` matches the rejected push near zero margin, 1.0 doubles
    it. IPO has no beta, so its anchor is unscaled. Not applied under KTO, whose
    desirable term already is a chosen anchor (``1 - sigmoid(beta * chosen_ratio)``
    rises as soon as the chosen half drops below the reference).

    A concept's ``dpo_chosen_anchor_override`` replaces the coefficient for its
    own pairs (``chosen_anchor`` is then a per-pair tensor). The beta scaling
    stays, so a beta-2 concept needs ~100x the coefficient of a beta-200 one
    for the same pull.
    """
    return dpo_chosen_anchor_penalties(chosen_ratio, beta=beta, objective=objective, chosen_anchor=chosen_anchor).mean()


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
    chosen_anchor: float | Tensor = 0.0,
    kto_desirable_weight: float = 1.0,
    kto_undesirable_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns (loss, dpo_loss, chosen_ratio, rejected_ratio, margin).

    ``loss`` and ``dpo_loss`` are scalars; the ratios and margin are per-sample.
    ``beta`` may be a scalar or a per-sample ``[B]`` tensor (per-concept beta
    overrides); it multiplies the per-sample margin elementwise either way, so
    the reduction below is unchanged. ``loss`` includes the chosen anchor
    (``dpo_chosen_anchor_penalty``) when ``chosen_anchor > 0`` (a scalar, or a
    per-pair ``[B]`` tensor from per-concept overrides) under the sigmoid and
    IPO objectives; ``dpo_loss`` stays the bare preference term. Because
    the anchor sits inside ``loss``, the light path's surrogate weights
    (``dpo_surrogate_weights`` differentiates ``loss``) carry its gradient onto
    the chosen half automatically.
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
    elif objective == DPOObjective.KTO:
        # KTO (Ethayarajh et al. 2024) scores the two halves as independent
        # examples against the reference: the chosen half is desirable and
        # wants beta*ratio > 0, the rejected half is undesirable and wants
        # beta*ratio < 0, each through its own saturating sigmoid utility. The
        # desirable term can only be paid by the chosen likelihood rising above
        # the reference, so a collapsing rejected half does not satisfy it the
        # way it does in DPO (no likelihood displacement). The KTO reference
        # point (a KL estimate from mismatched pairs in the batch) is 0 here:
        # at batch size 1 there is nothing to estimate it from, and it only
        # shifts where the utility saturates. Beta per-sample works as in DPO;
        # label smoothing does not apply.
        desirable = kto_desirable_weight * (1.0 - torch.sigmoid(beta * chosen_ratio))
        undesirable = kto_undesirable_weight * (1.0 - torch.sigmoid(-beta * rejected_ratio))
        dpo_loss = torch.cat([desirable, undesirable]).mean()
        loss = dpo_loss
    else:
        logits = beta * margin
        dpo_loss = -F.logsigmoid(logits).mean()
        loss = dpo_loss
        if label_smoothing > 0:
            s = label_smoothing
            loss = (1 - s) * loss + s * (-F.logsigmoid(-logits).mean())

    if dpo_chosen_anchor_active(chosen_anchor) and objective != DPOObjective.KTO:
        loss = loss + dpo_chosen_anchor_penalty(
            chosen_ratio, beta=beta, objective=objective, chosen_anchor=chosen_anchor
        )

    return loss, dpo_loss, chosen_ratio, rejected_ratio, margin


def dpo_per_sample_losses(
    chosen_ratio: Tensor,
    rejected_ratio: Tensor,
    *,
    beta: float | Tensor,
    objective: DPOObjective,
    ipo_tau: float,
    label_smoothing: float,
    chosen_anchor: float | Tensor = 0.0,
    kto_desirable_weight: float = 1.0,
    kto_undesirable_weight: float = 1.0,
) -> Tensor:
    """Each pair's own loss, ``[B]``, such that ``dpo_objective``'s ``loss``
    is their mean (for KTO, the mean of the pair's two half-terms). Includes
    the pair's DPOP chosen anchor when ``chosen_anchor > 0``, like ``loss``.
    Detached; for the per-concept loss trend of the gradient diagnostics."""
    chosen_ratio = chosen_ratio.detach().float()
    rejected_ratio = rejected_ratio.detach().float()
    margin = chosen_ratio - rejected_ratio
    beta_t = beta.detach().float() if isinstance(beta, Tensor) else beta
    if objective == DPOObjective.KTO:
        desirable = kto_desirable_weight * (1.0 - torch.sigmoid(beta_t * chosen_ratio))
        undesirable = kto_undesirable_weight * (1.0 - torch.sigmoid(-beta_t * rejected_ratio))
        return (desirable + undesirable) / 2.0
    if objective == DPOObjective.IPO:
        losses = (margin - 1.0 / (2.0 * ipo_tau)).pow(2)
    else:
        logits = beta_t * margin
        losses = -F.logsigmoid(logits)
        if label_smoothing > 0:
            s = label_smoothing
            losses = (1 - s) * losses + s * (-F.logsigmoid(-logits))
    if dpo_chosen_anchor_active(chosen_anchor):
        losses = losses + dpo_chosen_anchor_penalties(
            chosen_ratio, beta=beta_t, objective=objective, chosen_anchor=chosen_anchor
        )
    return losses


def kto_unpaired_objective(
    policy_logp: Tensor,
    ref_logp: Tensor,
    liked: Tensor,
    *,
    beta: float | Tensor,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """KTO on single labelled images, no pair involved. Returns ``(loss, ratio)``.

    ``ratio`` is each sample's reference-relative log-prob. A Liked image wants
    ``beta * ratio > 0``, a Disliked one ``< 0``, each through the same
    saturating sigmoid utility as the paired KTO objective (reference point 0;
    see ``dpo_objective``). ``liked`` is a bool ``[B]`` tensor; ``beta`` may be a
    scalar or a per-sample ``[B]`` tensor.
    """
    ratio = policy_logp - ref_logp.detach()
    liked_f = liked.to(ratio.dtype)
    sign = liked_f * 2.0 - 1.0
    weight = liked_f * desirable_weight + (1.0 - liked_f) * undesirable_weight
    loss = (weight * (1.0 - torch.sigmoid(sign * beta * ratio))).mean()
    return loss, ratio


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
