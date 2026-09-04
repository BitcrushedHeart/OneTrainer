"""The curation tool must never offer an image that is already a chosen or
rejected sample in another configured DPO concept — e.g. a generation the
off-policy builder committed — whether by its path or a byte-identical copy."""

import json
import os
import shutil
import time
from types import SimpleNamespace

from modules.util.dpo_curation_util import concept_usage_for_exclusion
from web.backend.services.dpo_service import DPOService

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _png(path: str, size, prompt: str, color) -> str:
    info = PngInfo()
    info.add_text("prompt", prompt)
    Image.new("RGB", size, color).save(path, pnginfo=info)
    return path


@pytest.fixture
def curation(monkeypatch):
    svc = DPOService.__new__(DPOService)
    svc.__init__()
    monkeypatch.setattr(DPOService, "get_instance", classmethod(lambda cls: svc))
    svc._realism_device = "off"
    yield svc
    svc.cancel_session()


def _wait(svc, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if svc.get_session_status()["worker_finished"]:
            return
        time.sleep(0.05)
    raise AssertionError("scan did not finish")


def test_concept_usage_for_exclusion_reads_manifest_and_files(tmp_path):
    root = tmp_path / "concept"
    (root / "chosen").mkdir(parents=True)
    (root / "rejected" / "sub").mkdir(parents=True)
    rej = _png(str(root / "rejected" / "sub" / "pair_0000.png"), (32, 32), "x", (1, 2, 3))
    (root / "chosen" / "pair_0000.txt").write_text("x", encoding="utf-8")
    src = str(tmp_path / "src.png")
    shutil.copyfile(rej, src)
    manifest = {
        "pairs": [
            {
                "pair_id": 0,
                "prompt": "x",
                "aspectratio": "1:1",
                "chosen_file": "pair_0000.jpg",
                "rejected_file": "sub/pair_0000.png",
                "chosen_source": None,
                "rejected_source": os.path.normcase(os.path.abspath(src)),
            }
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pairs = [(str(root / "chosen"), str(root / "rejected"))]

    sources, fingerprints, paths = concept_usage_for_exclusion(pairs)
    assert sources == {os.path.normcase(os.path.abspath(src))}
    assert paths == [rej]
    assert fingerprints  # backfilled from the exported file
    # The caller's own output dir is skipped.
    assert concept_usage_for_exclusion(pairs, exclude_dir=str(root)) == (set(), {}, [])
    # A concept without a manifest still contributes its files.
    (root / "manifest.json").unlink()
    assert concept_usage_for_exclusion(pairs)[2] == [rej]


def test_curation_scan_drops_images_used_by_another_concept(curation, monkeypatch, tmp_path):
    source = tmp_path / "generated"
    out = tmp_path / "curation_out"
    foreign = tmp_path / "offpolicy"
    source.mkdir()
    (foreign / "chosen").mkdir(parents=True)
    (foreign / "rejected").mkdir()

    used = _png(str(source / "used.png"), (64, 64), "shared prompt", (1, 2, 3))
    _png(str(source / "used_copy.png"), (64, 64), "shared prompt", (1, 2, 3))  # pixel-identical copy
    _png(str(source / "free_a.png"), (64, 64), "shared prompt", (4, 5, 6))
    _png(str(source / "free_b.png"), (64, 64), "shared prompt", (7, 8, 9))
    # The off-policy concept committed used.png as its rejected sample.
    shutil.copyfile(used, str(foreign / "rejected" / "pair_0000.png"))
    manifest = {
        "pairs": [
            {
                "pair_id": 0,
                "prompt": "shared prompt",
                "aspectratio": "1:1",
                "chosen_file": "pair_0000.jpg",
                "rejected_file": "pair_0000.png",
                "chosen_source": None,
                "rejected_source": os.path.normcase(os.path.abspath(used)),
            }
        ]
    }
    (foreign / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    concept = SimpleNamespace(
        enabled=True, path=str(foreign), dpo_chosen_pattern="chosen/{}", dpo_rejected_pattern="rejected/{}"
    )
    monkeypatch.setattr(DPOService, "_resolve_concepts", lambda self: [concept])

    assert curation.start_session(str(source), str(out), pairs_per_group=1, mode="selection", group_order="random")[
        "ok"
    ]
    _wait(curation)
    group = curation.next_group()["group"]
    assert sorted(os.path.basename(p) for p in group["images"]) == ["free_a.png", "free_b.png"]
