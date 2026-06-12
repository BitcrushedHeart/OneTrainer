"""Verification gates for OFT / DoRA-OFT merge.

Pure functions consumed by ``MergeOFTUI``. Each helper returns its raw
measurement; the caller decides which deltas should hard-fail and assembles
a ``MergeVerificationError`` with the structured failure list.

The gates exist because we shipped a base model with 869 dead V rows on
layer 1 from a ComfyUI Save Checkpoint merge that silently mishandled
``scaled_oft`` and used the current (post-merge) norm rather than the
training-time ``initial_norm``. Catching the same class of bug requires:

- per-block orthogonality of R before applying (catches scaling drift)
- per-key NaN/Inf after applying
- no *new* zero-norm output rows vs the pre-merge baseline
- for DoRA-OFT, ||merged_row[i]|| == |dora_multiplier[i]| * ||base_row[i]||
  (the rotation-preserves-norm invariant) within bf16 tolerance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from modules.module.LoRAModule import DoRAOFTModule, OFTModule

import torch
from torch import nn


@dataclass
class VerificationFailure:
    """One structured failure from a verification gate."""

    gate: str
    location: str
    detail: str


class MergeVerificationError(RuntimeError):
    """Raised when a merge gate fails. ``failures`` is structured for logging."""

    def __init__(self, failures: list[VerificationFailure]):
        self.failures = failures
        lines = [f"  [{f.gate}] {f.location}: {f.detail}" for f in failures]
        message = f"Merge verification failed ({len(failures)} issue(s)):\n" + "\n".join(lines)
        super().__init__(message)


@dataclass
class MergeReport:
    """Aggregated per-key measurements collected over the merge."""

    pre_zero_rows: dict[str, int] = field(default_factory=dict)
    post_zero_rows: dict[str, int] = field(default_factory=dict)
    nan_counts: dict[str, int] = field(default_factory=dict)
    inf_counts: dict[str, int] = field(default_factory=dict)
    orthogonality_max_err: dict[str, float] = field(default_factory=dict)
    dora_invariant_max_residual: dict[str, float] = field(default_factory=dict)


def compute_orthogonality_error(oft_module: OFTModule) -> torch.Tensor:
    """Compute per-block ``||R^T R - I||_F / sqrt(n)`` for the module's rotation.

    Uses the same Cayley-Neumann + scaled_oft path as ``forward`` /
    ``apply_to_module`` so the value reflects what will actually be applied.
    Returns a 1D tensor of length ``num_blocks``.
    """
    block_size = oft_module.oft_R.block_size
    scaling_factor = 2 * math.sqrt(block_size - 1) if oft_module.oft_scaled else 1
    effective_weight = oft_module.oft_R.weight.detach().to(torch.float32) / scaling_factor

    R = oft_module.oft_R._cayley_batch(
        effective_weight,
        block_size,
        oft_module.oft_R.use_cayley_neumann,
        oft_module.oft_R.num_cayley_neumann_terms,
        oft_module.oft_R.oft_cans,
    )
    eye = torch.eye(block_size, dtype=R.dtype, device=R.device).unsqueeze(0)
    return (R.transpose(-1, -2) @ R - eye).norm(dim=(1, 2)) / math.sqrt(block_size)


def count_zero_rows(weight: torch.Tensor) -> int:
    """Count output-axis rows whose per-row L2 norm is exactly zero.

    Linear weights have shape ``(out, in)``; Conv2d has ``(out, in, kH, kW)``.
    Output axis is dim 0 in both cases.
    """
    flat = weight.reshape(weight.shape[0], -1).to(torch.float32)
    return int((flat.norm(dim=1) == 0).sum().item())


def scan_nan_inf(weight: torch.Tensor) -> tuple[int, int]:
    """Return (nan_count, inf_count) for a single weight tensor."""
    nan = int(torch.isnan(weight).sum().item())
    inf = int(torch.isinf(weight).sum().item())
    return nan, inf


def check_dora_invariant(
    merged_weight: torch.Tensor,
    dora_multiplier: torch.Tensor,
    base_row_norm: torch.Tensor,
) -> torch.Tensor:
    """For DoRA-OFT, ``||merged_row[i]|| == |dora_multiplier[i]| * ||base_row[i]||``.

    OFT's rotation is norm-preserving, so baking ``W @ R^T`` then scaling each
    output row by ``dora_multiplier`` gives a row norm of
    ``|dora_multiplier| * ||base_row||`` (``base_row`` = the pre-merge base
    weight row). Returns the per-row absolute residual; caller compares against
    a tolerance scaled to the working dtype.
    """
    flat = merged_weight.reshape(merged_weight.shape[0], -1).to(torch.float32)
    row_norms = flat.norm(dim=1)
    expected = dora_multiplier.detach().to(torch.float32).abs().reshape(-1) * base_row_norm.detach().to(
        torch.float32
    ).reshape(-1)
    return (row_norms - expected).abs()


def gate_orthogonality(
    modules_by_key: dict[str, OFTModule],
    threshold: float = 1e-2,
) -> tuple[list[VerificationFailure], dict[str, float]]:
    """Run the orthogonality gate over every OFTModule prior to apply.

    Returns (failures, max_err_by_key). The threshold sits in the wide gap
    between Cayley-Neumann truncation noise (well below 1e-3 for trained
    adapters, the same residual training sees with num_terms=5) and a
    scaled_oft-mishandling failure mode (O(1) per-block error, e.g. the
    ComfyUI Save Checkpoint bug we shipped against).

    Threshold 1e-2 is comfortably above bf16's relative precision (~7.8e-3)
    so on-disk storage noise can't push a healthy adapter over the line,
    while still being 100x below catastrophic-divergence values.
    """
    failures: list[VerificationFailure] = []
    max_err_by_key: dict[str, float] = {}
    for key, module in modules_by_key.items():
        err = compute_orthogonality_error(module)
        max_err = float(err.max().item())
        max_err_by_key[key] = max_err
        if max_err > threshold:
            failures.append(
                VerificationFailure(
                    gate="orthogonality",
                    location=key,
                    detail=(
                        f"max per-block ||R^T R - I||_F / sqrt(n) = {max_err:.3e} "
                        f"(threshold {threshold:.0e}); R has lost orthogonality, "
                        f"check scaled_oft / Cayley-Neumann term count"
                    ),
                )
            )
    return failures, max_err_by_key


def _bf16_tolerance(reference: torch.Tensor) -> float:
    """Per-row tolerance for the DoRA-OFT norm invariant.

    Sized for a base saved in bf16 (the common case). Combines bf16's
    relative epsilon (~7.8e-3) against the per-row magnitude with a small
    absolute floor for rows whose expected norm is near zero.
    """
    rel = 0.05  # 5% -- bf16 norm reconstruction loses some precision over many entries
    abs_floor = 1e-4
    return float((rel * reference.abs() + abs_floor).max().item())


def gate_post_merge(
    base_module_by_key: dict[str, nn.Module],
    pre_zero_rows: dict[str, int],
    dora_multiplier_by_key: dict[str, torch.Tensor] | None = None,
    base_row_norm_by_key: dict[str, torch.Tensor] | None = None,
) -> tuple[list[VerificationFailure], MergeReport]:
    """Run post-merge gates: NaN/Inf, zero-row delta, DoRA invariant.

    ``base_module_by_key`` maps each adapter key to the underlying
    ``nn.Linear``/``nn.Conv2d`` whose weight has just been baked.
    ``dora_multiplier_by_key`` / ``base_row_norm_by_key`` are populated only for
    DoRA-OFT entries -- the module's ``dora_multiplier`` parameter and the
    per-output-row L2 norm of the base weight captured *before* the bake. The
    invariant checked is ``||merged_row|| == |dora_multiplier| * ||base_row||``.
    """
    failures: list[VerificationFailure] = []
    report = MergeReport(pre_zero_rows=dict(pre_zero_rows))

    for key, base_module in base_module_by_key.items():
        weight = base_module.weight.data

        nan, inf = scan_nan_inf(weight)
        report.nan_counts[key] = nan
        report.inf_counts[key] = inf
        if nan > 0 or inf > 0:
            failures.append(
                VerificationFailure(
                    gate="nan_inf",
                    location=key,
                    detail=f"{nan} NaN, {inf} Inf entries after merge",
                )
            )

        post_zeros = count_zero_rows(weight)
        report.post_zero_rows[key] = post_zeros
        baseline = pre_zero_rows.get(key, 0)
        if post_zeros > baseline:
            failures.append(
                VerificationFailure(
                    gate="dead_rows",
                    location=key,
                    detail=(
                        f"{post_zeros - baseline} new zero-norm output rows (baseline {baseline}, post {post_zeros})"
                    ),
                )
            )

        if dora_multiplier_by_key is not None and base_row_norm_by_key is not None and key in dora_multiplier_by_key:
            mult = dora_multiplier_by_key[key]
            base_norm = base_row_norm_by_key[key]
            residuals = check_dora_invariant(weight, mult, base_norm)
            max_res = float(residuals.max().item())
            report.dora_invariant_max_residual[key] = max_res
            expected = mult.detach().to(torch.float32).abs().reshape(-1) * base_norm.detach().to(torch.float32).reshape(
                -1
            )
            tol = _bf16_tolerance(expected)
            if max_res > tol:
                failures.append(
                    VerificationFailure(
                        gate="dora_invariant",
                        location=key,
                        detail=(
                            f"max | ||merged_row|| - |mult|*||base_row|| | = {max_res:.3e} "
                            f"(tolerance {tol:.3e}); rotation likely lost orthogonality"
                        ),
                    )
                )

    return failures, report


def collect_oft_modules(lora_wrapper) -> tuple[dict[str, OFTModule], dict[str, nn.Module], dict[str, torch.Tensor]]:
    """Extract per-key OFT modules, base modules, and DoRA multipliers from a wrapper.

    Returns three parallel dicts keyed by ``module.prefix.rstrip(".")``:
    - the OFT/DoRA-OFT adapter module
    - its bound base ``nn.Module`` (the orig_module)
    - the ``dora_multiplier`` parameter, only present for DoRA-OFT entries
    """
    oft_modules: dict[str, OFTModule] = {}
    base_modules: dict[str, nn.Module] = {}
    dora_multipliers: dict[str, torch.Tensor] = {}

    for adapter in lora_wrapper.lora_modules.values():
        if not isinstance(adapter, OFTModule):
            continue
        key = adapter.prefix.rstrip(".")
        oft_modules[key] = adapter
        base_modules[key] = adapter.orig_module
        if isinstance(adapter, DoRAOFTModule):
            dora_multipliers[key] = adapter.dora_multiplier

    return oft_modules, base_modules, dora_multipliers
