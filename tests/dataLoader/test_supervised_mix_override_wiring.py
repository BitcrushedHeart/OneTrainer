from modules.dataLoader.BaseDataLoader import _shape_safe_collate
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType

from mgds.OutputPipelineModule import OutputPipelineModule

import torch


class DataLoaderHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


def _output_module(rlhf_enabled: bool) -> OutputPipelineModule:
    config = TrainConfig.default_values()
    config.rlhf_enabled = rlhf_enabled
    modules = DataLoaderHarness()._output_modules_from_out_names(
        model=None,
        model_setup=None,
        output_names=["latent_image"],
        config=config,
        train_dtype=DataType.FLOAT_32,
    )
    return next(module for module in modules if isinstance(module, OutputPipelineModule))


def test_supervised_mix_override_travels_per_sample_only_on_dpo_runs():
    dpo_output = _output_module(rlhf_enabled=True)
    columns = list(zip(dpo_output.input_names, dpo_output.output_names, strict=True))
    assert ("concept.supervised_mix_override", "supervised_mix_override") in columns
    assert ("concept.dpo_beta_override", "dpo_beta_override") in columns

    assert "supervised_mix_override" not in _output_module(rlhf_enabled=False).output_names


def test_concept_defaults_to_inheriting_the_global_mix():
    concept = ConceptConfig.default_values()
    assert concept.supervised_mix_override == -1.0
    assert concept.to_dict()["supervised_mix_override"] == -1.0

    concept.supervised_mix_override = 3.0
    assert ConceptConfig.default_values().from_dict(concept.to_dict()).supervised_mix_override == 3.0

    # Concept files saved before the field existed load as "inherit".
    legacy = ConceptConfig.default_values().from_dict({"name": "legacy", "dpo_beta_override": 0.0})
    assert legacy.supervised_mix_override == -1.0


def test_collate_fills_rows_cached_before_the_column_existed():
    # Sourceless runs serve each row's concept as stamped at caching time, so a
    # batch can mix rows with and without the column in either order.
    new = {"latent_image": torch.zeros(2), "supervised_mix_override": torch.tensor(3.0)}
    old = {"latent_image": torch.zeros(2)}

    batch = _shape_safe_collate([new, old])
    assert torch.equal(batch["supervised_mix_override"], torch.tensor([3.0, -1.0]))
    batch = _shape_safe_collate([old, new])
    assert torch.equal(batch["supervised_mix_override"], torch.tensor([-1.0, 3.0]))

    # No row has it: the column stays absent and the global mix applies.
    assert "supervised_mix_override" not in _shape_safe_collate([old, old])
    assert "supervised_mix_override" not in old
