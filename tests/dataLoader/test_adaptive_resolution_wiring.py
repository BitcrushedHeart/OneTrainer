from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.util.config.TrainConfig import TrainConfig

from mgds.pipelineModules.AspectBucketing import AspectBucketing
from mgds.pipelineModules.CalcAspect import CalcAspect


class DataLoaderHarness(DataLoaderText2ImageMixin):
    def _preparation_modules(self, config, model):
        return []

    def _cache_modules(self, config, model, model_setup):
        return []

    def _output_modules(self, config, model, model_setup, is_validation=False):
        return []

    def _debug_modules(self, config, model):
        return []


def test_dpo_rejected_resolution_only_constrains_adaptive_mode():
    loader = DataLoaderHarness()
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.aspect_ratio_bucketing = True

    config.adaptive_resolution = False
    legacy_modules = loader._aspect_bucketing_in(config, 64)
    legacy_bucketing = next(module for module in legacy_modules if isinstance(module, AspectBucketing))
    assert len([module for module in legacy_modules if isinstance(module, CalcAspect)]) == 1
    assert legacy_bucketing.additional_resolution_in_names == []

    config.adaptive_resolution = True
    config.adaptive_resolution_ladder = "896,640,320,192"
    adaptive_modules = loader._aspect_bucketing_in(config, 64)
    adaptive_bucketing = next(module for module in adaptive_modules if isinstance(module, AspectBucketing))
    assert len([module for module in adaptive_modules if isinstance(module, CalcAspect)]) == 2
    assert adaptive_bucketing.additional_resolution_in_names == ["original_resolution_rejected"]
    assert adaptive_bucketing.adaptive_resolution_ladder == "896,640,320,192"
