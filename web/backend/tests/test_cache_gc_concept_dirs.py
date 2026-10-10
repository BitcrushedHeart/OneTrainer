import os
from types import SimpleNamespace

from web.backend.routers.system import _gc_cache_dirs


def test_gc_covers_concept_cache_overrides_once(tmp_path):
    run_cache = str(tmp_path / "run")
    shared = str(tmp_path / "shared")
    config = SimpleNamespace(
        cache_dir=run_cache,
        concepts=[
            {"name": "a", "cache_dir": ""},
            {"name": "b", "cache_dir": shared},
            {"name": "c", "cache_dir": shared + os.sep},
            {"name": "d", "cache_dir": run_cache},
            # Disabled concepts still own cache files worth sweeping.
            {"name": "e", "cache_dir": str(tmp_path / "off"), "enabled": False},
        ],
    )

    assert _gc_cache_dirs(config) == [run_cache, shared, str(tmp_path / "off")]


def test_gc_without_overrides_is_just_the_run_cache(tmp_path):
    config = SimpleNamespace(cache_dir=str(tmp_path / "run"), concepts=[{"name": "a"}])
    assert _gc_cache_dirs(config) == [str(tmp_path / "run")]
