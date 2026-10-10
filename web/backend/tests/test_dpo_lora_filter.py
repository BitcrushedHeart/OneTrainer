"""The DPO Pair Tool's LoRA filter: the term parser, the per-image matcher,
the scan cache's cached ``loras`` field, and the session scan applying it."""

import json
import os
import time

from modules.util.dpo_curation_util import DpoScanCache
from web.backend.services.dpo_service import DPOService, image_matches_lora_terms, parse_lora_terms

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class TestParseLoraTerms:
    def test_empty_means_no_filter(self):
        assert parse_lora_terms("") == []
        assert parse_lora_terms("  ,  , ") == []
        assert parse_lora_terms(None) == []

    def test_lowercases_strips_and_dedupes_in_order(self):
        assert parse_lora_terms(" MyChar , detail, mychar,Detail ") == ["mychar", "detail"]


class TestImageMatchesLoraTerms:
    def test_no_terms_matches_everything(self):
        assert image_matches_lora_terms([], [])
        assert image_matches_lora_terms(["anything"], [])

    def test_terms_but_no_loras_fails(self):
        assert not image_matches_lora_terms([], ["mychar"])
        assert not image_matches_lora_terms(None, ["mychar"])

    def test_substring_match_is_case_insensitive(self):
        assert image_matches_lora_terms(["Flux/MyChar_v2"], ["mychar"])
        assert image_matches_lora_terms(["flux/mychar_v2"], ["MYCHAR"])
        assert not image_matches_lora_terms(["flux/other"], ["mychar"])

    def test_any_term_matching_any_lora_suffices(self):
        assert image_matches_lora_terms(["detail_boost", "flux/mychar"], ["nope", "mychar"])

    def test_basename_match_ignores_folder_and_extension(self):
        assert image_matches_lora_terms(["Chars/Hero.safetensors"], ["hero"])
        assert image_matches_lora_terms(["chars\\hero.pt"], ["hero"])

    def test_ignores_non_string_and_blank_entries(self):
        assert image_matches_lora_terms([None, "", 3, "mychar"], ["mychar"])
        assert not image_matches_lora_terms([None, "", 3], ["mychar"])


def _png(path: str, params: dict) -> str:
    info = PngInfo()
    info.add_text("sui_image_params", json.dumps({"sui_image_params": params}))
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path, pnginfo=info)
    return path


def _extract_loras(path: str) -> list[str]:
    from modules.util.image_metadata_util import extract_metadata

    return extract_metadata(path)["loras"]


def _extract_key_and_loras(path: str):
    from modules.util.image_metadata_util import extract_metadata

    meta = extract_metadata(path)
    return (meta["prompt"], meta["aspectratio"]), meta["loras"]


class TestScanCacheLoras:
    def test_get_loras_caches_including_empty_list(self, tmp_path):
        with_lora = _png(str(tmp_path / "a.png"), {"prompt": "x", "loras": ["A/B"]})
        without = _png(str(tmp_path / "b.png"), {"prompt": "y"})
        cache = DpoScanCache(str(tmp_path / "cache.json"))
        assert cache.get_loras(with_lora, _extract_loras) == ["a/b"]
        assert cache.get_loras(without, _extract_loras) == []
        misses = cache.misses
        calls = []

        def spy(path):
            calls.append(path)
            return _extract_loras(path)

        assert cache.get_loras(with_lora, spy) == ["a/b"]
        assert cache.get_loras(without, spy) == []
        assert calls == []
        assert cache.misses == misses

    def test_group_key_and_loras_cached_together_from_one_read(self, tmp_path):
        path = _png(str(tmp_path / "a.png"), {"prompt": "p", "aspectratio": "1:1", "loras": "One, Two"})
        cache = DpoScanCache(str(tmp_path / "cache.json"))
        reads = []

        def compute(p):
            reads.append(p)
            return _extract_key_and_loras(p)

        assert cache.get_group_key_and_loras(path, compute) == (("p", "1:1"), ["one", "two"])
        assert len(reads) == 1
        # Both fields are now served, each through its own getter too.
        assert cache.get_group_key_and_loras(path, compute) == (("p", "1:1"), ["one", "two"])
        assert cache.get_group_key(path, lambda p: pytest.fail("group_key should be cached")) == ("p", "1:1")
        assert cache.get_loras(path, lambda p: pytest.fail("loras should be cached")) == ["one", "two"]
        assert len(reads) == 1

    def test_partial_entry_recomputes_both(self, tmp_path):
        path = _png(str(tmp_path / "a.png"), {"prompt": "p", "loras": ["L"]})
        cache = DpoScanCache(str(tmp_path / "cache.json"))
        assert cache.get_group_key(path, lambda p: _extract_key_and_loras(p)[0]) == ("p", "")
        reads = []

        def compute(p):
            reads.append(p)
            return _extract_key_and_loras(p)

        assert cache.get_group_key_and_loras(path, compute) == (("p", ""), ["l"])
        assert len(reads) == 1

    def test_v4_cache_entries_migrate_and_miss_on_loras(self, tmp_path):
        path = _png(str(tmp_path / "a.png"), {"prompt": "p", "loras": ["L"]})
        st = os.stat(path)
        key = os.path.normcase(os.path.abspath(path))
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "version": 4,
                    "entries": {
                        key: {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "hash": "h", "group_key": ["p", ""]}
                    },
                }
            ),
            encoding="utf-8",
        )
        cache = DpoScanCache(str(cache_path))
        # Hash and group key survive the migration ...
        assert cache.get_pixel_hash(path) == "h"
        assert cache.get_group_key(path, lambda p: pytest.fail("should be cached")) == ("p", "")
        # ... and the new field is simply computed on first use.
        assert cache.get_loras(path, _extract_loras) == ["l"]
        cache.save()
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        assert saved["version"] == DpoScanCache._VERSION
        assert saved["entries"][key]["loras"] == ["l"]


# ---- Session scan ----


@pytest.fixture
def curation(monkeypatch):
    svc = DPOService.__new__(DPOService)
    svc.__init__()
    monkeypatch.setattr(DPOService, "get_instance", classmethod(lambda cls: svc))
    svc._realism_device = "off"
    monkeypatch.setattr(DPOService, "_resolve_concepts", lambda self: [])
    yield svc
    svc.cancel_session()


def _wait(svc, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if svc.get_session_status()["worker_finished"]:
            return
        time.sleep(0.05)
    raise AssertionError("scan did not finish")


def _drain_prompts(svc) -> set[str]:
    prompts = set()
    while True:
        res = svc.next_group()
        if res.get("done"):
            return prompts
        if res.get("waiting"):
            time.sleep(0.05)
            continue
        prompts.add(res["group"]["prompt"])


def _gen(path, prompt: str, loras, color) -> str:
    info = PngInfo()
    info.add_text("sui_image_params", json.dumps({"sui_image_params": {"prompt": prompt, "loras": loras}}))
    Image.new("RGB", (64, 64), color).save(path, pnginfo=info)
    return str(path)


def _tree(tmp_path):
    source = tmp_path / "generated"
    source.mkdir()
    # Every image used the character LoRA.
    for i in range(2):
        _gen(source / f"char_{i}.png", "char prompt", ["Flux/MyChar"], (10 + i, 20, 30))
    # One base-model image slipped into this prompt's group.
    _gen(source / "mixed_0.png", "mixed prompt", ["Flux/MyChar"], (40, 50, 60))
    _gen(source / "mixed_1.png", "mixed prompt", [], (41, 50, 60))
    # A different LoRA entirely.
    for i in range(2):
        _gen(source / f"other_{i}.png", "other prompt", ["Style/Painterly"], (70 + i, 80, 90))
    return source, tmp_path / "out"


def test_no_lora_filter_shows_every_group(curation, tmp_path):
    source, out = _tree(tmp_path)
    assert curation.start_session(str(source), str(out), mode="selection", group_order="random")["ok"]
    _wait(curation)
    assert _drain_prompts(curation) == {"char prompt", "mixed prompt", "other prompt"}


def test_lora_filter_keeps_only_groups_whose_images_all_match(curation, tmp_path):
    source, out = _tree(tmp_path)
    assert curation.start_session(str(source), str(out), mode="selection", group_order="random", lora_filter="MYCHAR")[
        "ok"
    ]
    _wait(curation)
    assert _drain_prompts(curation) == {"char prompt"}


def test_lora_filter_any_term_matches(curation, tmp_path):
    source, out = _tree(tmp_path)
    assert curation.start_session(
        str(source), str(out), mode="selection", group_order="random", lora_filter="mychar, painterly"
    )["ok"]
    _wait(curation)
    assert _drain_prompts(curation) == {"char prompt", "other prompt"}


def test_lora_filter_is_served_from_a_prior_unfiltered_scan_cache(curation, tmp_path):
    """A cache written by a filter-less scan has group keys but no LoRA
    lists; the filtered scan fills them in rather than failing or reading
    stale data."""
    source, out = _tree(tmp_path)
    assert curation.start_session(str(source), str(out), mode="selection", group_order="random")["ok"]
    _wait(curation)
    _drain_prompts(curation)
    curation.cancel_session()
    assert curation.start_session(
        str(source), str(out), mode="selection", group_order="random", lora_filter="painterly"
    )["ok"]
    _wait(curation)
    assert _drain_prompts(curation) == {"other prompt"}
