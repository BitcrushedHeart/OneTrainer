from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.DPOObjective import DPOObjective

from mgds.OutputPipelineModule import OutputPipelineModule


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
    loader = DataLoaderHarness()
    config = TrainConfig.default_values()
    config.rlhf_enabled = rlhf_enabled
    modules = loader._output_modules_from_out_names(
        model=None,
        model_setup=None,
        output_names=["latent_image"],
        config=config,
        train_dtype=DataType.FLOAT_32,
    )
    return next(module for module in modules if isinstance(module, OutputPipelineModule))


def test_beta_override_travels_per_sample_only_on_dpo_runs():
    # A batch mixes concepts (resolution bucketing sorts across the whole
    # dataset), so beta has to ride along per sample like loss_weight does.
    dpo_output = _output_module(rlhf_enabled=True)
    dpo_columns = list(zip(dpo_output.input_names, dpo_output.output_names, strict=True))
    assert ("concept.dpo_beta_override", "dpo_beta_override") in dpo_columns
    assert ("concept.loss_weight", "loss_weight") in dpo_columns

    # Non-DPO runs keep the upstream column set untouched.
    assert "dpo_beta_override" not in _output_module(rlhf_enabled=False).output_names


def test_concept_defaults_to_inheriting_the_global_beta():
    # 0 is the inherit sentinel, so existing concept files (which have no such
    # key) migrate into "use the global beta" without a migration step.
    concept = ConceptConfig.default_values()
    assert concept.dpo_beta_override == 0.0
    assert concept.to_dict()["dpo_beta_override"] == 0.0

    restored = ConceptConfig.default_values().from_dict({"name": "legacy", "dpo_chosen_pattern": "chosen/{}"})
    assert restored.dpo_beta_override == 0.0


def _dpo_concept(name: str, beta_override: float) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.name = name
    concept.path = f"/data/{name}"
    concept.dpo_chosen_pattern = "chosen/{}"
    concept.dpo_rejected_pattern = "rejected/{}"
    concept.dpo_beta_override = beta_override
    return concept


def _filter(concepts, config):
    return DataLoaderMgdsMixin._DataLoaderMgdsMixin__filter_dpo_concepts(concepts, config)


def test_ipo_objective_warns_that_beta_overrides_do_nothing(capsys):
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_objective = DPOObjective.IPO
    _filter([_dpo_concept("tight", 400.0), _dpo_concept("inherits", 0.0)], config)

    warning = capsys.readouterr().out
    assert "IPO objective ignores beta" in warning
    assert "tight" in warning
    assert "inherits" not in warning


def test_sigmoid_objective_stays_quiet_about_beta_overrides(capsys):
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_dpo_objective = DPOObjective.SIGMOID
    _filter([_dpo_concept("tight", 400.0)], config)

    assert "ignores beta" not in capsys.readouterr().out
