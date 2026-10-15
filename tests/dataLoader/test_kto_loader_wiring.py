from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DataType import DataType
from modules.util.enum.KTOLabel import KTOLabel

from mgds.OutputPipelineModule import OutputPipelineModule

import pytest


class OutputHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


class SelectHarness(DataLoaderMgdsMixin):
    def __init__(self, is_kto: bool = False, is_sft_anchor: bool = False):
        self.is_kto = is_kto
        self.is_sft_anchor = is_sft_anchor


def _concept(name, concept_type, *, enabled=True, label=KTOLabel.LIKED, chosen="", rejected=""):
    concept = ConceptConfig.default_values()
    concept.name = name
    concept.path = f"/data/{name}"
    concept.type = concept_type
    concept.enabled = enabled
    concept.kto_label = label
    concept.dpo_chosen_pattern = chosen
    concept.dpo_rejected_pattern = rejected
    return concept


def _output_columns(is_kto: bool) -> list[tuple[str, str]]:
    loader = OutputHarness()
    loader.is_kto = is_kto
    config = TrainConfig.default_values()
    # The KTO loader is built with rlhf_enabled forced off (standard pipeline).
    config.rlhf_enabled = False
    modules = loader._output_modules_from_out_names(
        model=None,
        model_setup=None,
        output_names=["latent_image"],
        config=config,
        train_dtype=DataType.FLOAT_32,
    )
    output = next(module for module in modules if isinstance(module, OutputPipelineModule))
    return list(zip(output.input_names, output.output_names, strict=True))


def test_kto_loader_carries_label_and_beta_per_sample_without_a_rejected_half():
    columns = _output_columns(is_kto=True)
    assert ("concept.kto_label", "kto_label") in columns
    assert ("concept.dpo_beta_override", "dpo_beta_override") in columns
    assert "latent_image_rejected" not in [out for _, out in columns]

    plain = [out for _, out in _output_columns(is_kto=False)]
    assert "kto_label" not in plain
    assert "dpo_beta_override" not in plain


def test_kto_concepts_go_to_the_kto_loader_only():
    config = TrainConfig.default_values()
    config.concepts = [
        _concept("faces", ConceptType.STANDARD),
        _concept("liked", ConceptType.KTO, label=KTOLabel.LIKED),
        _concept("disliked", ConceptType.KTO, label=KTOLabel.DISLIKED),
        _concept("val", ConceptType.VALIDATION),
    ]

    kto = SelectHarness(is_kto=True)._select_concepts(config)
    assert [c.name for c in kto] == ["liked", "disliked"]

    config.rlhf_enabled = False
    standard = SelectHarness()._select_concepts(config)
    assert [c.name for c in standard] == ["faces"]
    anchor = SelectHarness(is_sft_anchor=True)._select_concepts(config)
    assert [c.name for c in anchor] == ["faces"]
    validation = SelectHarness()._select_concepts(config, is_validation=True)
    assert [c.name for c in validation] == ["val"]


def test_kto_loader_rejects_patterns_and_needs_an_enabled_concept():
    config = TrainConfig.default_values()
    config.concepts = [_concept("liked", ConceptType.KTO, chosen="chosen/{}", rejected="rejected/{}")]
    with pytest.raises(RuntimeError, match="cannot carry DPO chosen/rejected patterns"):
        SelectHarness(is_kto=True)._select_concepts(config)

    config.concepts = [_concept("liked", ConceptType.KTO, enabled=False), _concept("faces", ConceptType.STANDARD)]
    with pytest.raises(RuntimeError, match="at least one enabled KTO concept"):
        SelectHarness(is_kto=True)._select_concepts(config)


def test_kto_concept_defaults_round_trip():
    concept = _concept("liked", ConceptType.KTO, label=KTOLabel.DISLIKED)
    assert concept.is_kto()
    restored = ConceptConfig.default_values().from_dict(concept.to_dict())
    assert restored.kto_label == KTOLabel.DISLIKED
    legacy = concept.to_dict()
    legacy.pop("kto_label")
    assert ConceptConfig.default_values().from_dict(legacy).kto_label == KTOLabel.LIKED
