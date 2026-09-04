import json
from types import SimpleNamespace

from web.backend.routers import system
from web.backend.services.config_service import ConfigService
from web.backend.services.trainer_service import TrainerService


def _config(cache_dir):
    return SimpleNamespace(
        cache_dir=str(cache_dir),
        adaptive_resolution=True,
        resolution="1024",
        rlhf_enabled=False,
        concept_file_name="",
    )


def test_adaptive_prune_preview_route_reports_targeted_variants(tmp_path, monkeypatch):
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / "cache.json").write_text(
        json.dumps(
            {
                "entries": {
                    "small.png": {
                        "original_resolution": [700, 700],
                        "source_resolutions": {"original_resolution": [700, 700]},
                        "variants": {"1024x1024": {"cache_file": "high", "crop_resolution": [1024, 1024]}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (image_dir / "high_1.pt").write_bytes(b"high")
    monkeypatch.setattr(
        ConfigService,
        "get_instance",
        classmethod(lambda cls: SimpleNamespace(get_config_for_training=lambda: _config(tmp_path))),
    )
    monkeypatch.setattr(
        TrainerService, "get_instance", classmethod(lambda cls: SimpleNamespace(get_status=lambda: {"status": "idle"}))
    )

    result = system.adaptive_cache_prune_preview()

    assert result.ok is True
    assert result.removed_variants == 1
    assert result.entries_without_usable_variant == 1


def test_adaptive_prune_refuses_while_training_is_active(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ConfigService,
        "get_instance",
        classmethod(lambda cls: SimpleNamespace(get_config_for_training=lambda: _config(tmp_path))),
    )
    monkeypatch.setattr(
        TrainerService,
        "get_instance",
        classmethod(lambda cls: SimpleNamespace(get_status=lambda: {"status": "running"})),
    )

    result = system.adaptive_cache_prune_clean()

    assert result.ok is False
    assert "running" in result.error.lower()
