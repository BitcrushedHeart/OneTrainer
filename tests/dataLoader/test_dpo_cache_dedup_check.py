"""scripts/util/dpo_cache_dedup_check.py: counts, from a cache.json alone,
the entries that train on another pair's tensors (a file shared by entries
of differing sidecar identity, minus the entry the .pt names as its builder),
and reports 0 once every entry has a file of its own."""

import importlib.util
import json
import os
import sys

import torch

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "util", "dpo_cache_dedup_check.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("dpo_cache_dedup_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(source_hash: str, cache_file: str, sidecar_hashes: dict, dedup_key: str | None = None) -> dict:
    entry = {
        "hash": source_hash,
        "mtime": 1.0,
        "modeltype": "testmodel",
        "variants": {"1024x1024": {"cache_file": cache_file}},
        "sidecar_hashes": sidecar_hashes,
        "sidecar_mtimes": dict.fromkeys(sidecar_hashes, 1.0),
    }
    if dedup_key is not None:
        entry["dedup_key"] = dedup_key
    return entry


def _write_cache(cache_dir, entries: dict, holders: dict[str, str]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "cache.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 3, "entries": entries, "hash_index": {}}, f)
    for cache_file, image_path in holders.items():
        torch.save(
            {"latent_image": torch.zeros(2), "image_path": image_path}, os.path.join(cache_dir, f"{cache_file}_1.pt")
        )


def _report(capsys, cache_dir, monkeypatch) -> str:
    monkeypatch.setattr(sys, "argv", ["dpo_cache_dedup_check.py", str(cache_dir), "--examples", "2"])
    assert _load_script().main() == 0
    return capsys.readouterr().out


def test_counts_victims_per_concept_and_names_the_holder(tmp_path, capsys, monkeypatch):
    root = str(tmp_path / "data")
    minor = os.path.join(root, "DPO Minor Edits", "chosen", "train", "pair_0001.png")
    rlhf = os.path.join(root, "RLHF_8", "chosen", "train", "pair_3439.png")
    off_a = os.path.join(root, "Off Policy 2", "chosen", "train", "pair_0039.jpg")
    off_b = os.path.join(root, "Off Policy 2", "chosen", "train", "pair_6906.jpg")
    off_c = os.path.join(root, "Off Policy 2", "chosen", "train", "pair_6907.jpg")
    lone = os.path.join(root, "Glorp", "chosen", "train", "img.png")
    entries = {
        # the live SFT layout: a masked entry built the source-named file, an unmasked twin points at it
        minor: _entry("a4633575d96dca34", "a4633575d96d_1024x1024", {minor + "-mask": "m1"}),
        rlhf: _entry("a4633575d96dca34", "a4633575d96d_1024x1024", {}),
        # one chosen image, three rejected images, one file (built from the first)
        off_a: _entry("9efc20641f3f0000", "9efc20641f3f_1024x1024", {off_a + "-rej": "r1"}),
        off_b: _entry("9efc20641f3f0000", "9efc20641f3f_1024x1024", {off_b + "-rej": "r2"}),
        off_c: _entry("9efc20641f3f0000", "9efc20641f3f_1024x1024", {off_c + "-rej": "r3"}),
        lone: _entry("1111111111110000", "111111111111_1024x1024", {}),
    }
    cache_dir = tmp_path / "cache"
    _write_cache(
        cache_dir,
        entries,
        {"a4633575d96d_1024x1024": minor, "9efc20641f3f_1024x1024": off_a, "111111111111_1024x1024": lone},
    )

    out = _report(capsys, cache_dir, monkeypatch)
    assert "mixed groups (one file, differing rejected/mask): 2 files, largest holds 3 entries" in out
    rows = {
        line.split("  ")[0].strip(): line.split()
        for line in out.splitlines()
        if line.startswith(("RLHF_8", "Off Policy 2", "DPO Minor Edits", "Glorp"))
    }
    assert rows["RLHF_8"][-4:] == ["1", "1", "1", "1"]  # entries, in mixed, on other pair, files
    assert rows["DPO Minor Edits"][-4:] == ["1", "1", "0", "1"]  # the holder trains on its own tensors
    assert rows["Off Policy 2"][-4:] == ["3", "3", "2", "1"]
    assert rows["Glorp"][-4:] == ["1", "0", "0", "0"]
    assert "3 entries train on another pair's tensors" in out
    assert (
        "pair_3439.png -> a4633575d96d_1024x1024 (holds DPO Minor Edits/pair_0001.png; 2 entries on this file)" in out
    )
    assert "pair_6906.jpg -> 9efc20641f3f_1024x1024 (holds Off Policy 2/pair_0039.jpg; 3 entries on this file)" in out


def test_reports_zero_once_every_entry_has_its_own_file(tmp_path, capsys, monkeypatch):
    # The layout a normal cache pass leaves behind: the masked holder's file
    # moved to its dedup-key name, the unmasked twin built into the freed
    # source-hash name, every pair file named by its own identity.
    root = str(tmp_path / "data")
    minor = os.path.join(root, "DPO Minor Edits", "chosen", "train", "pair_0001.png")
    rlhf = os.path.join(root, "RLHF_8", "chosen", "train", "pair_3439.png")
    twin_a = os.path.join(root, "A", "chosen", "train", "p1.png")
    twin_b = os.path.join(root, "B", "chosen", "train", "p1.png")  # identical identity: legitimately shared
    entries = {
        minor: _entry(
            "a4633575d96dca34", "3018efa885c9_1024x1024", {minor + "-mask": "m1"}, dedup_key="3018efa885c9b34b"
        ),
        rlhf: _entry("a4633575d96dca34", "a4633575d96d_1024x1024", {}, dedup_key="a4633575d96dca34"),
        twin_a: _entry(
            "2222222222220000", "333333333333_1024x1024", {twin_a + "-rej": "r"}, dedup_key="3333333333330000"
        ),
        twin_b: _entry(
            "2222222222220000", "333333333333_1024x1024", {twin_b + "-rej": "r"}, dedup_key="3333333333330000"
        ),
    }
    cache_dir = tmp_path / "cache"
    _write_cache(
        cache_dir,
        entries,
        {"3018efa885c9_1024x1024": minor, "a4633575d96d_1024x1024": rlhf, "333333333333_1024x1024": twin_a},
    )
    out = _report(capsys, cache_dir, monkeypatch)
    assert "mixed groups (one file, differing rejected/mask): 0 files" in out
    assert "0 entries train on another pair's tensors" in out


@pytest.mark.parametrize("no_peek", [False, True])
def test_unknown_holder_counts_every_member_as_a_victim(tmp_path, capsys, monkeypatch, no_peek):
    root = str(tmp_path / "data")
    a = os.path.join(root, "A", "chosen", "train", "p1.png")
    b = os.path.join(root, "B", "chosen", "train", "p1.png")
    entries = {
        a: _entry("4444444444440000", "444444444444_1024x1024", {a + "-rej": "x"}),
        b: _entry("4444444444440000", "444444444444_1024x1024", {b + "-rej": "y"}),
    }
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, entries, {})  # no .pt on disk: the holder cannot be identified
    if no_peek:
        monkeypatch.setattr(sys, "argv", ["dpo_cache_dedup_check.py", str(cache_dir), "--no-peek"])
        assert _load_script().main() == 0
        out = capsys.readouterr().out
    else:
        out = _report(capsys, cache_dir, monkeypatch)
    assert "holder unknown for some files" in out
    assert "2 entries train on another pair's tensors" in out
