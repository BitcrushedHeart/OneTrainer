from modules.util.config.TrainConfig import TrainConfig


def test_adaptive_resolution_defaults_off_and_round_trips():
    config = TrainConfig.default_values()
    assert config.adaptive_resolution is False
    assert config.adaptive_resolution_ladder == "1024,768,512,256,128"

    config.adaptive_resolution = True
    config.adaptive_resolution_ladder = "896,640,320,192"
    loaded = TrainConfig.default_values().from_dict(config.to_dict())
    assert loaded.adaptive_resolution is True
    assert loaded.adaptive_resolution_ladder == "896,640,320,192"


def test_version_23_config_migrates_with_adaptive_resolution_disabled():
    loaded = TrainConfig.default_values().from_dict({"__version": 23})
    assert loaded.adaptive_resolution is False


def test_version_24_config_migrates_with_the_default_adaptive_ladder():
    loaded = TrainConfig.default_values().from_dict({"__version": 24})
    assert loaded.adaptive_resolution_ladder == "1024,768,512,256,128"
