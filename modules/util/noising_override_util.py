"""Per-concept timestep sampling overrides (ConceptConfig.*_override fields).

The dataloader hands each sample its concept's override columns next to
loss_weight; this module turns them into the per-sample parameters
ModelSetupNoiseMixin draws timesteps with. A column holds the inherit value
(NaN for floats, "" for the distribution, False for the range-after-shift flag)
for samples whose concept leaves the setting alone, and a missing column means
every sample inherits.
"""

from dataclasses import dataclass
from math import isnan

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Tensor

FLOAT_OVERRIDE_COLUMNS = (
    "min_noising_strength_override",
    "max_noising_strength_override",
    "noising_weight_override",
    "noising_bias_override",
    "timestep_shift_override",
)
DISTRIBUTION_OVERRIDE_COLUMN = "timestep_distribution_override"
RANGE_AFTER_SHIFT_COLUMN = "noising_range_after_shift"
NOISING_OVERRIDE_COLUMNS = (*FLOAT_OVERRIDE_COLUMNS, DISTRIBUTION_OVERRIDE_COLUMN, RANGE_AFTER_SHIFT_COLUMN)

INHERIT_FLOAT = float("nan")
INHERIT_DISTRIBUTION = ""

# Batch column -> the value a row cached before the column existed takes
# (BaseDataLoader._INHERIT_CONCEPT_COLUMNS).
INHERIT_COLUMN_VALUES = {
    **dict.fromkeys(FLOAT_OVERRIDE_COLUMNS, INHERIT_FLOAT),
    DISTRIBUTION_OVERRIDE_COLUMN: INHERIT_DISTRIBUTION,
    RANGE_AFTER_SHIFT_COLUMN: False,
}


@dataclass
class NoisingParams:
    """Resolved per-sample timestep sampling settings (host lists, length B)."""

    min_strength: list[float]
    max_strength: list[float]
    weight: list[float]
    bias: list[float]
    shift: list[float]
    distribution: list[TimestepDistribution]
    range_after_shift: list[bool]


def _as_list(value, batch_size: int, name: str) -> list:
    values = value.reshape(-1).tolist() if isinstance(value, Tensor) else list(value)
    if len(values) == batch_size:
        return values
    if batch_size == 2 * len(values):
        # The fused DPO forward stacks [chosen; rejected]; the per-concept
        # columns live on the chosen rows and the rejected half shares them.
        return values + values
    raise RuntimeError(f"Concept column {name} has {len(values)} entries for a batch of {batch_size} samples.")


def _is_inherit_float(value: float) -> bool:
    return value is None or isnan(value)


def resolve_noising_overrides(
    config: TrainConfig,
    batch: dict | None,
    batch_size: int,
    shift: float,
) -> NoisingParams | None:
    """Per-sample settings for this batch, or None when no sample overrides
    anything (the caller then keeps the run-wide draw, bit for bit)."""
    if not batch:
        return None
    columns = {name: batch.get(name) for name in NOISING_OVERRIDE_COLUMNS}
    columns = {name: value for name, value in columns.items() if value is not None}
    if not columns:
        return None

    floats = {}
    float_names = [name for name in FLOAT_OVERRIDE_COLUMNS if name in columns]
    if float_names:
        # One host sync for every float column.
        stacked = torch.stack([torch.as_tensor(columns[name]).reshape(-1).float() for name in float_names])
        for name, values in zip(float_names, stacked.tolist(), strict=True):
            floats[name] = _as_list(values, batch_size, name)
    distributions = (
        [
            str(value)
            for value in _as_list(columns[DISTRIBUTION_OVERRIDE_COLUMN], batch_size, DISTRIBUTION_OVERRIDE_COLUMN)
        ]
        if DISTRIBUTION_OVERRIDE_COLUMN in columns
        else [INHERIT_DISTRIBUTION] * batch_size
    )
    after_shift = (
        [bool(value) for value in _as_list(columns[RANGE_AFTER_SHIFT_COLUMN], batch_size, RANGE_AFTER_SHIFT_COLUMN)]
        if RANGE_AFTER_SHIFT_COLUMN in columns
        else [False] * batch_size
    )

    overriding = (
        any(not _is_inherit_float(value) for values in floats.values() for value in values)
        or any(value != INHERIT_DISTRIBUTION for value in distributions)
        or any(after_shift)
    )
    if not overriding:
        return None

    def resolved(name: str, run_value: float) -> list[float]:
        values = floats.get(name)
        if values is None:
            return [float(run_value)] * batch_size
        return [float(run_value) if _is_inherit_float(value) else float(value) for value in values]

    return NoisingParams(
        min_strength=resolved("min_noising_strength_override", config.min_noising_strength),
        max_strength=resolved("max_noising_strength_override", config.max_noising_strength),
        weight=resolved("noising_weight_override", config.noising_weight),
        bias=resolved("noising_bias_override", config.noising_bias),
        shift=resolved("timestep_shift_override", shift),
        distribution=[
            config.timestep_distribution if value == INHERIT_DISTRIBUTION else TimestepDistribution(value)
            for value in distributions
        ],
        range_after_shift=after_shift,
    )


def describe_noising_overrides(concept: ConceptConfig, config: TrainConfig) -> str | None:
    """Short text of the settings this concept overrides with their effective
    values, or None when it inherits everything."""
    parts = []
    for label, field, run_value in (
        ("min", "min_noising_strength_override", config.min_noising_strength),
        ("max", "max_noising_strength_override", config.max_noising_strength),
        ("weight", "noising_weight_override", config.noising_weight),
        ("bias", "noising_bias_override", config.noising_bias),
        ("shift", "timestep_shift_override", config.timestep_shift),
    ):
        value = getattr(concept, field, None)
        if value is not None:
            parts.append(f"{label} {value:g} (run {run_value:g})")
    distribution = getattr(concept, "timestep_distribution_override", None)
    if distribution is not None:
        parts.append(f"distribution {distribution} (run {config.timestep_distribution})")
    if getattr(concept, "noising_range_after_shift", False):
        parts.append("range applied after the shift")
    return ", ".join(parts) if parts else None
