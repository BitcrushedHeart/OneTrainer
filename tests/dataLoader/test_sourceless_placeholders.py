from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin

from mgds.pipelineModules.PlaceholderModule import PlaceholderModule


class DummyModule:
    pass


def test_placeholder_modules_preserve_flattened_pre_cache_module_count():
    pre_cache_modules = [
        DummyModule(),
        [DummyModule(), None, [DummyModule(), DummyModule()]],
        DummyModule(),
    ]

    placeholders = DataLoaderMgdsMixin._placeholder_modules_for(pre_cache_modules)

    assert len(placeholders) == 5
    assert all(isinstance(module, PlaceholderModule) for module in placeholders)
    assert all(module.get_inputs() == [] and module.get_outputs() == [] for module in placeholders)
