"""Align an OFT adapter's late qkv/mlp blocks to a *sentinel* reference model.

Standing policy for the SoReal! lineup: every adapter baked across Base/Turbo/
Lightning must land its late ``qkv_late`` / ``mlp_late`` blocks (layers >= 22) as
close to **Turbo** as a scalar merge strength allows. Turbo is the sentinel -- its
merged late blocks define the target, because the DPO/preference data is developed
against Turbo's outputs. Every other variant differs from Turbo only in those late
rotations, so matching them there keeps the whole lineup visually consistent while
leaving each model's step-count identity (the rest of the late rotation) intact.

This module derives, per family, the least-squares merge strength that best matches
the target model's merged late block to the sentinel's:

    merged_X(s) = W_X + s * C_X,   C_X = W_X @ (R^T - I)     (linear-blend strength)
    target      = W_sentinel @ R^T                          (sentinel at s=1)
    s* = <C_X, T0_X> / <C_X, C_X>,   T0_X = target - W_X     (pooled over the family)

See scripts/util/_dpo_strength_match.py for the full derivation, the granularity
analysis, and why a scalar can only close part of the gap. A family whose strength
leaves the ``[warn_low, warn_high]`` band means a scalar can't sensibly align that
block -- the caller is warned so a human looks before baking.

Faithful to modules/module/oft_utils.py OFTRotationModule for the Cayley-Neumann
path (the lineup's config: OFT_2, block 128, clip 0.99, cans off, scaled off).
Unsupported orthogonalizations (CANS, exact solver, block_share) raise rather than
silently mis-predict the strength.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass

from modules.util.oft_families import DEFAULT_LATE_FAMILY_STEMS

import torch

from safetensors import safe_open

NEUMANN_TERMS = 5  # OFTRotationModule.num_cayley_neumann_terms default
_ADAPTER_KEY = re.compile(r"^(?:transformer\.)?layers\.(\d+)\.(.+)\.oft_R\.weight$")


@dataclass
class AdapterOFTConfig:
    block_size: int  # as REQUESTED at training time -- see block_size_from_elements
    clip: float | None
    scaled: bool


def block_size_from_elements(n_elements: int) -> int:
    """Recover a block's true width from the stored upper-triangular element count.

    ``n_elements == b * (b - 1) // 2``, so ``b = (1 + sqrt(1 + 8n)) / 2``.

    This must be derived per tensor rather than read from ``ot_config``:
    ``OFTModule.initialize_weights`` (LoRAModule.py:709-713) silently adjusts the
    requested ``oft_block_size`` to a divisor of ``in_features``. An adapter trained
    with a requested 400 therefore stores 384-wide blocks for 3840-dim inputs and
    320-wide blocks for 10240-dim inputs -- two different block sizes inside the SAME
    adapter, which the single ``ot_config`` field cannot represent. Reconstructing at
    the requested size raises in ``index_put`` (mismatched triu index count) rather
    than mis-predicting quietly, so this is a hard failure on any adapter whose block
    size was adjusted, not a silent accuracy loss.
    """
    b = (1.0 + math.sqrt(1.0 + 8.0 * n_elements)) / 2.0
    block = round(b)
    if abs(b - block) > 1e-6 or block * (block - 1) // 2 != n_elements:
        raise ValueError(f"oft_R row width {n_elements} is not b*(b-1)/2 for any integer b")
    return block


def _read_oft_config(adapter_path: str) -> AdapterOFTConfig:
    """Pull the OFT hyperparameters from the adapter's ot_config metadata."""
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
    cfg = json.loads(md.get("ot_config", "{}"))
    if cfg.get("oft_cans"):
        raise NotImplementedError("oft_align does not model the CANS (Newton-Schulz) path; add it before use.")
    if cfg.get("oft_block_share"):
        raise NotImplementedError("oft_align does not model block_share rotations; add it before use.")
    return AdapterOFTConfig(
        block_size=int(cfg.get("oft_block_size", 128)),
        clip=cfg.get("oft_clipped_norm", None),
        scaled=bool(cfg.get("oft_scaled", False)),
    )


def reconstruct_oft_rotation(vec: torch.Tensor, cfg: AdapterOFTConfig) -> torch.Tensor:
    """Rebuild the per-block orthogonal R from oft_R params, faithful to oft_utils.

    vec: (r, n_elements) upper-tri skew params -> R: (r, block, block).
    Uses the exact spectral norm (eigh on -Q@Q) for the clip instead of the
    module's 1-step power iteration from non-persistent random state; on trained
    adapters the learned angle is far below the clip so the two agree, and eigh is
    deterministic and dodges a batched-SVD segfault under Windows MKL.

    The block size comes from ``vec``'s own width, not from ``cfg.block_size``, which
    holds the pre-adjustment request -- see block_size_from_elements.
    """
    bs = block_size_from_elements(vec.shape[-1])
    scaling = 2 * (bs - 1) ** 0.5 if cfg.scaled else 1.0
    vec = vec / scaling
    r = vec.shape[0]
    rows, cols = torch.triu_indices(bs, bs, 1, device=vec.device)
    Q = torch.zeros(r, bs, bs, device=vec.device, dtype=vec.dtype)
    bidx = torch.arange(r, device=vec.device)[:, None]
    Q = Q.index_put((bidx, rows, cols), vec)
    Q = Q - Q.transpose(-2, -1)  # skew-symmetric

    if cfg.clip is not None:
        ev = torch.linalg.eigvalsh(-torch.bmm(Q, Q))  # sigma^2 of skew Q, ascending
        sigma = ev[:, -1].clamp_min(0).sqrt().reshape(r, 1, 1)
        Q = Q * (cfg.clip / torch.clamp(sigma, min=cfg.clip))

    R = torch.eye(bs, device=vec.device, dtype=vec.dtype).repeat(r, 1, 1)
    R.add_(Q, alpha=2.0)
    Q2 = torch.bmm(Q, Q)
    R.add_(Q2, alpha=2.0)
    Qp = Q2
    for _ in range(3, NEUMANN_TERMS - 1):
        Qp = torch.bmm(Qp, Q)
        R.add_(Qp, alpha=2.0)
    Qp = torch.bmm(Qp, Q)
    R.add_(Qp)  # last term coeff 1
    return R


def merged_full(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """W @ R^T, block-diagonal on the input dim (matches apply_to_module 'orc,rkc->ork')."""
    out, inf = W.shape
    Wr = W.reshape(out, R.shape[0], R.shape[1])
    return torch.einsum("orc,rkc->ork", Wr, R).reshape(out, inf)


def _base_key_prefix(target_path: str) -> str:
    """Detect whether the checkpoint prefixes weight keys with 'transformer.'."""
    with safe_open(target_path, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            if k.endswith(".weight") and ".layers." in k:
                return "transformer." if k.startswith("transformer.") else ""
    return ""


def family_match_strengths(
    adapter_path: str,
    sentinel_path: str,
    target_path: str,
    *,
    late_start: int = 22,
    families: dict[str, tuple[str, ...]] | None = None,
    sentinel_strength: float = 1.0,
    warn_low: float = 0.5,
    warn_high: float = 1.5,
    device: str | None = None,
    label: str = "",
) -> tuple[dict[str, float], list[str]]:
    """Per-family merge strength that best aligns ``target``'s late blocks to ``sentinel``.

    Returns ``({family: strength}, warnings)``. A family whose strength falls to
    ``warn_low`` or rises to ``warn_high`` gets a warning (also emitted via
    ``warnings.warn``) -- the scalar can't sensibly align that block; look before baking.

    ``sentinel_strength`` is the merge strength the sentinel itself is baked at (its
    late blocks land at ``W_sentinel + s_sent * (W_sentinel @ R^T - W_sentinel)``).
    It defaults to 1.0 (the standing full-strength policy); pass the sentinel's actual
    strength when it is baked partially, so the target the others align to is the
    sentinel's *real* merged block rather than a hypothetical full-strength one.
    """
    fams = families or DEFAULT_LATE_FAMILY_STEMS
    stem_to_fam = {s: f for f, ss in fams.items() for s in ss}
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _read_oft_config(adapter_path)
    prefix = _base_key_prefix(target_path)

    # Phase 1: read every needed tensor up front (mmap only), then close the files.
    # Interleaving safetensors mmap with LAPACK/einsum segfaults under Windows MKL.
    recs = []  # (fam, vec, W_target, W_sentinel)
    with (
        safe_open(adapter_path, framework="pt", device="cpu") as fa,
        safe_open(sentinel_path, framework="pt", device="cpu") as fs,
        safe_open(target_path, framework="pt", device="cpu") as ftg,
    ):
        for k in fa.keys():  # noqa: SIM118
            m = _ADAPTER_KEY.match(k)
            if not m:
                continue
            layer, stem = int(m.group(1)), m.group(2)
            fam = stem_to_fam.get(stem)
            if fam is None or layer < late_start:
                continue
            bk = f"{prefix}layers.{layer}.{stem}.weight"
            recs.append(
                (
                    fam,
                    fa.get_tensor(k).to(torch.float32).clone(),
                    ftg.get_tensor(bk).to(torch.float32).clone(),
                    fs.get_tensor(bk).to(torch.float32).clone(),
                )
            )

    # Phase 2: pooled least-squares per family.
    acc: dict[str, list[float]] = {f: [0.0, 0.0] for f in fams}  # fam -> [num, den]
    for fam, vec, Wx, Ws in recs:
        vec, Wx, Ws = vec.to(dev), Wx.to(dev), Ws.to(dev)
        R = reconstruct_oft_rotation(vec, cfg)
        # sentinel merged at its own strength: Ws + s_sent * (Ws @ R^T - Ws)
        target = Ws + sentinel_strength * (merged_full(Ws, R) - Ws)
        C = merged_full(Wx, R) - Wx
        T0 = target - Wx
        acc[fam][0] += torch.sum(C * T0, dtype=torch.float64).item()
        acc[fam][1] += torch.sum(C * C, dtype=torch.float64).item()

    strengths: dict[str, float] = {}
    msgs: list[str] = []
    tag = f"{label}: " if label else ""
    for fam, (num, den) in acc.items():
        if den <= 0.0:  # adapter doesn't rotate this family in the late blocks
            strengths[fam] = 1.0
            continue
        s = num / den
        strengths[fam] = s
        if s <= warn_low or s >= warn_high:
            msg = (
                f"{tag}{fam}_late align strength {s:.3f} is outside [{warn_low}, {warn_high}] "
                f"vs sentinel -- a scalar can't align this block to Turbo; inspect before baking."
            )
            msgs.append(msg)
            warnings.warn(msg, stacklevel=2)
    return strengths, msgs
