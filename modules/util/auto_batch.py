"""Pure algorithm for resolving Auto-Batch settings into a concrete batch_size and
gradient_accumulation_steps for a queue entry.

Reuses the bucketing analyser from `modules.util.dpo_bucket_analysis_util` so that
drop counts match what the trainer actually sees at run time.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from modules.util.dpo_bucket_analysis_util import (
    analyze_concept,
    parse_target_resolutions,
    quantization_for_model,
)

if TYPE_CHECKING:
    from modules.util.config.TrainConfig import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class AutoBatchCandidate:
    batch_size: int
    total_pairs: int
    total_drops: int
    drop_pct: float
    valid: bool


@dataclass
class AutoBatchResult:
    batch_size: int
    accum: int
    effective_sample_count: int
    dropped: int
    candidates: list[AutoBatchCandidate] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "accum": self.accum,
            "effective_sample_count": self.effective_sample_count,
            "dropped": self.dropped,
            "candidates": [asdict(c) for c in self.candidates],
            "warning": self.warning,
        }


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_standard_concepts(concept_file_path: str) -> list[dict]:
    """Load the concept file and return enabled STANDARD concepts as dicts."""
    if not concept_file_path:
        return []
    from web.backend.services.concept_service import ConceptService

    try:
        concepts = ConceptService().load_concepts(concept_file_path)
    except (OSError, ValueError) as exc:
        logger.warning("Auto-batch: failed to load concept file %s: %s", concept_file_path, exc)
        return []

    out: list[dict] = []
    for c in concepts:
        if not c.get("enabled", True):
            continue
        ctype = c.get("type")
        ctype_str = str(ctype) if ctype is not None else "STANDARD"
        if ctype_str != "STANDARD":
            continue
        out.append(c)
    return out


def compute_auto_batch(
    *,
    merged_config: TrainConfig,
    min_batch_size: int,
    max_batch_size: int,
    target_pct: float,
    max_drop_pct: float,
) -> AutoBatchResult:
    """Resolve batch_size and accumulation_steps for the given merged train config.

    The user-confirmed algorithm:
      1. Pick the largest batch_size in [min,max] whose total bucket drops are
         <= max_drop_pct of standard samples; fall back to min if none qualify.
      2. accum = max(1, round(target_required / batch_size)) where
         target_required = max(1, round(effective * target_pct / 100)).
    """
    # Validation / clamping ---------------------------------------------------
    if min_batch_size < 1:
        min_batch_size = 1
    if max_batch_size < min_batch_size:
        max_batch_size = min_batch_size
    target_pct = _clamp(target_pct, 1.0, 100.0)
    max_drop_pct = _clamp(max_drop_pct, 0.0, 100.0)
    tolerance = max_drop_pct / 100.0

    warnings: list[str] = []

    # Resolution targets / quantization ---------------------------------------
    target_resolutions = parse_target_resolutions(getattr(merged_config, "resolution", "") or "")
    quantization = quantization_for_model(str(getattr(merged_config, "model_type", "")))

    if not target_resolutions:
        warnings.append(
            "resolution does not contain a bucketable target (e.g. 512); "
            "Auto-Batch defaulted to min batch size with accum=1."
        )
        return AutoBatchResult(
            batch_size=min_batch_size,
            accum=1,
            effective_sample_count=0,
            dropped=0,
            candidates=[],
            warning="; ".join(warnings),
        )

    # Concept loading ---------------------------------------------------------
    concept_file_name = getattr(merged_config, "concept_file_name", "") or ""
    concepts = _load_standard_concepts(concept_file_name)
    if not concepts:
        warnings.append("No enabled STANDARD concepts found")
        return AutoBatchResult(
            batch_size=min_batch_size,
            accum=1,
            effective_sample_count=0,
            dropped=0,
            candidates=[],
            warning="; ".join(warnings),
        )

    # Per-batch-size candidates ----------------------------------------------
    candidates: list[AutoBatchCandidate] = []
    for bs in range(min_batch_size, max_batch_size + 1):
        total_pairs = 0
        total_drops = 0
        for concept in concepts:
            path = concept.get("path", "")
            if not path:
                continue
            try:
                result = analyze_concept(path, bs, target_resolutions, quantization)
            except (OSError, ValueError) as exc:
                warnings.append(f"concept {concept.get('name') or path}: {exc}")
                continue
            for target in result.get("targets", []):
                total_pairs += int(target.get("total_pairs", 0))
                total_drops += int(target.get("total_drops", 0))
        drop_pct = (total_drops / total_pairs) if total_pairs > 0 else 1.0
        candidates.append(AutoBatchCandidate(
            batch_size=bs,
            total_pairs=total_pairs,
            total_drops=total_drops,
            drop_pct=drop_pct,
            valid=drop_pct <= tolerance,
        ))

    if not candidates:
        warnings.append("No candidates evaluated (all concepts skipped)")
        return AutoBatchResult(
            batch_size=min_batch_size,
            accum=1,
            effective_sample_count=0,
            dropped=0,
            candidates=[],
            warning="; ".join(warnings),
        )

    valid_candidates = [c for c in candidates if c.valid]
    if valid_candidates:
        chosen = max(valid_candidates, key=lambda c: c.batch_size)
    else:
        chosen = candidates[0]
        warnings.append(
            f"No batch size in [{min_batch_size},{max_batch_size}] kept drops <= {max_drop_pct:.1f}%; "
            f"falling back to min batch size {chosen.batch_size}."
        )

    effective = chosen.total_pairs
    if effective == 0:
        warnings.append("No standard samples found in any concept folder")
        return AutoBatchResult(
            batch_size=min_batch_size,
            accum=1,
            effective_sample_count=0,
            dropped=chosen.total_drops,
            candidates=candidates,
            warning="; ".join(warnings),
        )

    if chosen.batch_size > effective:
        warnings.append(
            f"min batch size {chosen.batch_size} exceeds dataset size {effective}; "
            "forcing accum=1."
        )
        return AutoBatchResult(
            batch_size=min_batch_size,
            accum=1,
            effective_sample_count=effective,
            dropped=chosen.total_drops,
            candidates=candidates,
            warning="; ".join(warnings),
        )

    target_required = max(1, round(effective * target_pct / 100.0))
    accum = max(1, round(target_required / chosen.batch_size))

    return AutoBatchResult(
        batch_size=chosen.batch_size,
        accum=accum,
        effective_sample_count=effective,
        dropped=chosen.total_drops,
        candidates=candidates,
        warning="; ".join(warnings) if warnings else None,
    )
