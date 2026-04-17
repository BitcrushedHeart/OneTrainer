import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_presets_dir(tmp_path, monkeypatch):
    """Redirect PRESETS_DIR to an isolated tmp directory and reset singleton."""
    presets_dir = tmp_path / "training_presets"
    presets_dir.mkdir()
    monkeypatch.setattr("web.backend.paths.PRESETS_DIR", str(presets_dir))
    monkeypatch.setattr(
        "web.backend.services.config_service.PRESETS_DIR", str(presets_dir)
    )
    monkeypatch.setattr(
        "web.backend.services.config_service._DEFAULT_PRESET_PATH",
        str(presets_dir / "#.json"),
    )
    # Reset singleton so __init__ reruns with patched paths.
    from web.backend.services.config_service import ConfigService
    ConfigService._instance = None
    yield presets_dir
    ConfigService._instance = None


def _write_preset(path: Path, model_type: str = "Z_IMAGE") -> None:
    from modules.util.config.TrainConfig import TrainConfig
    cfg = TrainConfig.default_values()
    cfg.model_type = type(cfg.model_type)[model_type]
    data = cfg.to_settings_dict(secrets=False)
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def test_first_run_seeds_from_z_image_preset(temp_presets_dir):
    """When #.json is absent, ConfigService seeds from a z-image built-in preset and writes #.json."""
    _write_preset(temp_presets_dir / "#z-image LoRA 16GB.json", "Z_IMAGE")

    from web.backend.services.config_service import ConfigService
    ConfigService.get_instance()

    default_file = temp_presets_dir / "#.json"
    assert default_file.exists(), "#.json should be created on first run"

    saved = json.loads(default_file.read_text(encoding="utf-8"))
    assert saved["model_type"] == "Z_IMAGE"


def test_first_run_falls_back_to_any_builtin(temp_presets_dir):
    """When no z-image preset exists, fall back to first #*.json preset."""
    _write_preset(temp_presets_dir / "#flux LoRA.json", "FLUX_DEV_1")

    from web.backend.services.config_service import ConfigService
    ConfigService.get_instance()

    saved = json.loads((temp_presets_dir / "#.json").read_text(encoding="utf-8"))
    assert saved["model_type"] == "FLUX_DEV_1"


def test_existing_default_preset_is_loaded_unchanged(temp_presets_dir):
    """If #.json already exists, it is loaded and not overwritten by a Z-Image seed."""
    _write_preset(temp_presets_dir / "#.json", "STABLE_DIFFUSION_XL_10_BASE")
    _write_preset(temp_presets_dir / "#z-image LoRA 16GB.json", "Z_IMAGE")
    original_mtime = (temp_presets_dir / "#.json").stat().st_mtime_ns

    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()

    assert service.config.model_type.name == "STABLE_DIFFUSION_XL_10_BASE"
    assert (temp_presets_dir / "#.json").stat().st_mtime_ns == original_mtime


def test_no_presets_at_all_keeps_defaults(temp_presets_dir):
    """With no preset files, keep raw TrainConfig defaults and do not create #.json."""
    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()
    assert service.config is not None
    assert not (temp_presets_dir / "#.json").exists()


def test_update_config_writes_default_preset_atomically(temp_presets_dir):
    """Every update_config persists to #.json."""
    from web.backend.services.config_service import ConfigService
    service = ConfigService.get_instance()

    current = service.get_config_dict()
    current["learning_rate"] = 0.000123
    service.update_config(current)

    saved = json.loads((temp_presets_dir / "#.json").read_text(encoding="utf-8"))
    assert saved["learning_rate"] == pytest.approx(0.000123)
