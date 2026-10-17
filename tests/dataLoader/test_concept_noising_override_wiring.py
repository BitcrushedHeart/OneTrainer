"""Per-concept timestep sampling overrides: ConceptConfig fields, the output
columns of every loader kind, the collate filler for rows cached before the
fields existed, and the startup notice."""

import math

from modules.dataLoader.BaseDataLoader import _shape_safe_collate
from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.noising_override_util import NOISING_OVERRIDE_COLUMNS, describe_noising_overrides

from mgds.OutputPipelineModule import OutputPipelineModule

import torch

NULLABLE_FIELDS = (
    "min_noising_strength_override",
    "max_noising_strength_override",
    "timestep_distribution_override",
    "noising_weight_override",
    "noising_bias_override",
    "timestep_shift_override",
)


class DataLoaderHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


def _output_columns(*, rlhf_enabled: bool = False, is_kto: bool = False) -> list[tuple[str, str]]:
    loader = DataLoaderHarness()
    loader.is_kto = is_kto
    config = TrainConfig.default_values()
    config.rlhf_enabled = rlhf_enabled
    modules = loader._output_modules_from_out_names(
        model=None,
        model_setup=None,
        output_names=["latent_image"],
        config=config,
        train_dtype=DataType.FLOAT_32,
    )
    output = next(module for module in modules if isinstance(module, OutputPipelineModule))
    return list(zip(output.input_names, output.output_names, strict=True))


def test_every_loader_kind_carries_the_override_columns_per_sample():
    for kind in ({}, {"rlhf_enabled": True}, {"is_kto": True}):
        columns = _output_columns(**kind)
        for name in NOISING_OVERRIDE_COLUMNS:
            assert (f"concept.{name}", name) in columns, (kind, name)


def test_concept_defaults_inherit_and_round_trip():
    concept = ConceptConfig.default_values()
    for field in NULLABLE_FIELDS:
        assert getattr(concept, field) is None
        assert concept.to_dict()[field] is None  # the cache.json snapshot key
    assert concept.noising_range_after_shift is False

    concept.min_noising_strength_override = 0.1
    concept.max_noising_strength_override = 0.6
    concept.noising_bias_override = -0.25
    concept.timestep_distribution_override = TimestepDistribution.LOGIT_NORMAL
    concept.noising_range_after_shift = True
    snapshot = concept.to_dict()
    assert snapshot["timestep_distribution_override"] == "LOGIT_NORMAL"
    assert snapshot["noising_weight_override"] is None
    restored = ConceptConfig.default_values().from_dict(snapshot)
    assert restored.min_noising_strength_override == 0.1
    assert restored.max_noising_strength_override == 0.6
    assert restored.noising_bias_override == -0.25
    assert restored.noising_weight_override is None
    assert restored.timestep_distribution_override == TimestepDistribution.LOGIT_NORMAL
    assert restored.noising_range_after_shift is True

    # Concept files and cache snapshots written before the fields existed.
    legacy = ConceptConfig.default_values().from_dict({"name": "legacy", "dpo_beta_override": 0.0})
    for field in NULLABLE_FIELDS:
        assert getattr(legacy, field) is None
    assert legacy.noising_range_after_shift is False


def test_collate_fills_rows_without_the_columns_with_the_inherit_sentinels():
    # mgds drops None values, so a row whose concept inherits (or was cached
    # before the field existed) lacks the column; the filler must collate with
    # rows that carry it, in either order.
    new = {
        "latent_image": torch.zeros(2),
        "min_noising_strength_override": torch.tensor(0.5),
        "timestep_distribution_override": "LOGIT_NORMAL",
        "noising_range_after_shift": torch.tensor(True),
    }
    old = {"latent_image": torch.zeros(2)}

    batch = _shape_safe_collate([new, old])
    assert float(batch["min_noising_strength_override"][0]) == 0.5
    assert math.isnan(float(batch["min_noising_strength_override"][1]))
    assert batch["timestep_distribution_override"] == ["LOGIT_NORMAL", ""]
    assert batch["noising_range_after_shift"].tolist() == [True, False]

    batch = _shape_safe_collate([old, new])
    assert math.isnan(float(batch["min_noising_strength_override"][0]))
    assert batch["timestep_distribution_override"] == ["", "LOGIT_NORMAL"]

    # No row has them: the columns stay absent and the run's settings apply.
    plain = _shape_safe_collate([old, old])
    assert all(name not in plain for name in NOISING_OVERRIDE_COLUMNS)


def _concept(name: str, **overrides) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.name = name
    concept.path = f"/data/{name}"
    for field, value in overrides.items():
        setattr(concept, field, value)
    return concept


def test_describe_names_the_overrides_with_their_effective_values():
    config = TrainConfig.default_values()
    config.min_noising_strength = 0.0
    config.max_noising_strength = 1.0
    config.noising_bias = 0.0
    assert describe_noising_overrides(_concept("plain"), config) is None
    text = describe_noising_overrides(
        _concept(
            "photos",
            min_noising_strength_override=0.1,
            max_noising_strength_override=0.6,
            noising_bias_override=-0.2,
            timestep_distribution_override=TimestepDistribution.LOGIT_NORMAL,
            noising_range_after_shift=True,
        ),
        config,
    )
    assert text == (
        "min 0.1 (run 0), max 0.6 (run 1), bias -0.2 (run 0), distribution LOGIT_NORMAL (run UNIFORM), "
        "range applied after the shift"
    )


def test_startup_notice_lists_overriding_concepts_only(capsys):
    config = TrainConfig.default_values()
    config.concepts = [
        _concept("photos", min_noising_strength_override=0.1, max_noising_strength_override=0.6),
        _concept("edits", noising_bias_override=-0.2),
        _concept("plain"),
        _concept("disabled", enabled=False, noising_bias_override=0.5),
    ]
    harness = DataLoaderMgdsMixin()
    harness._select_concepts(config)
    notice = capsys.readouterr().out
    assert "2 concepts override the run's noising settings" in notice
    assert "photos: min 0.1 (run 0), max 0.6 (run 1)" in notice
    assert "edits: bias -0.2 (run 0)" in notice
    assert "plain" not in notice and "disabled" not in notice

    harness._select_concepts(config, is_validation=True)
    assert "noising settings" not in capsys.readouterr().out

    config.concepts = [_concept("plain")]
    harness._select_concepts(config)
    assert capsys.readouterr().out == ""
