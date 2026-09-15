"""DPO Saturation service end to end without a model: a concept on disk with
a hand-written saturation table, a fake child process that emits the
scorer's JSON progress lines, and the in-process summary/retire/restore."""

import json
import os
import subprocess
import sys
import tempfile
import time

from modules.util import dpo_saturation_util as sat
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from web.backend.services.dpo_saturation_service import DPOSaturationService

import pytest


def _wait(service, key, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = service.status()
        if key(status):
            return status
        time.sleep(0.02)
    raise AssertionError(f"timed out: {service.status()}")


@pytest.fixture
def concept_root(tmp_path):
    root = tmp_path / "RLHF"
    chosen_dir = root / "chosen" / "train"
    rejected_dir = root / "rejected" / "train"
    chosen_dir.mkdir(parents=True)
    rejected_dir.mkdir(parents=True)
    draws = sat.make_draws(draws=4, seed_base=0)
    table = sat.new_table(
        concept={
            "name": "dpo",
            "path": str(root),
            "chosen_dir": str(root / "chosen"),
            "rejected_dir": str(root / "rejected"),
            "beta_override": 150.0,
        },
        global_beta=750.0,
        num_train_timesteps=1000,
        origin={"path": str(tmp_path / "origin.safetensors"), "size": 1, "mtime": 1.0, "hash": "0xabc", "title": None},
        draws=draws,
    )
    ts = [100, 350, 600, 900]
    # margins: dead (0.04), fresh (0), flip (-0.02)
    specs = {"p_dead": 0.46, "p_fresh": 0.50, "p_flip": 0.52}
    for stem, current_chosen in specs.items():
        (chosen_dir / f"{stem}.png").write_bytes(b"c")
        (chosen_dir / f"{stem}.txt").write_text("caption")
        (rejected_dir / f"{stem}.png").write_bytes(b"r")
        key = f"train/{stem}"
        sat.ensure_pair(table, key, str(chosen_dir / f"{stem}.png"), str(rejected_dir / f"{stem}.png"))
        for draw, t in zip(draws, ts, strict=True):
            sat.record_draw(table, key, "origin", draw, t, 0.5, 0.5)
            sat.record_draw(table, key, "current", draw, t, current_chosen, 0.5)
    sat.mark_current(table, {"path": str(tmp_path / "current.safetensors"), "size": 2, "mtime": 2.0, "hash": "0xdef"})
    sat.save_table(str(root), table)
    (tmp_path / "origin.safetensors").write_bytes(b"o")
    (tmp_path / "current.safetensors").write_bytes(b"c")
    return root


@pytest.fixture
def service(monkeypatch, concept_root, tmp_path):
    svc = DPOSaturationService.__new__(DPOSaturationService)
    svc.__init__()
    monkeypatch.setattr(DPOSaturationService, "get_instance", classmethod(lambda cls: svc))

    concept = ConceptConfig.default_values()
    concept.name = "dpo"
    concept.path = str(concept_root)
    concept.enabled = True
    concept.include_subdirectories = True
    concept.dpo_chosen_pattern = "chosen/{}"
    concept.dpo_rejected_pattern = "rejected/{}"
    concept.dpo_beta_override = 150.0
    monkeypatch.setattr(DPOSaturationService, "_dpo_concepts", lambda self: [concept])

    config = TrainConfig.default_values()
    config.rlhf_dpo_beta = 750.0
    config.transformer.model_name = str(tmp_path / "origin.safetensors")
    config.cache_dir = str(tmp_path / "cache")
    monkeypatch.setattr(DPOSaturationService, "_train_config", lambda self: config)
    yield svc
    svc.cancel()


def test_concepts_reports_table_state(service, concept_root):
    res = service.concepts()
    assert res["ok"] and len(res["concepts"]) == 1
    c = res["concepts"][0]
    assert c["path"] == str(concept_root) and c["beta_override"] == 150.0 and c["retired"] == 0
    assert c["table"]["pairs"] == 3 and c["table"]["scored"] == 3 and c["table"]["draws"] == 4
    assert res["global_beta"] == 750.0 and res["default_origin"].endswith("origin.safetensors")


def test_summary_uses_concept_beta_and_marks_tails(service, concept_root):
    res = service.summary(str(concept_root), dead_weight=0.05)
    assert res["ok"], res
    assert res["beta"] == 150.0  # concept override beats the global 750
    assert res["pairs"] == 3 and res["saturated"] == 1
    assert res["saturated_rows"][0]["key"] == "train/p_dead"
    assert res["most_negative"][0]["key"] == "train/p_flip"
    assert len(res["logit_histogram"]["counts"]) == 40
    assert res["utility_curve"][-1]["pairs"] == 3 and res["utility_curve"][-1]["share"] == pytest.approx(1.0)
    assert "rows" not in res
    # explicit beta override
    assert service.summary(str(concept_root), beta=750.0)["beta"] == 750.0
    # unknown concept / missing table
    assert not service.summary(str(concept_root.parent / "nope"))["ok"]


def test_pair_detail(service, concept_root):
    res = service.pair(str(concept_root), "train/p_dead")
    assert res["ok"] and len(res["draws"]) == 4 and res["beta"] == 150.0
    assert all(d["weight"] < 0.01 for d in res["draws"])
    assert not service.pair(str(concept_root), "train/missing")["ok"]


def test_retire_and_restore_flow(service, concept_root):
    dry = service.retire(str(concept_root), dead_weight=0.05, dry_run=True)
    assert dry["ok"] and dry["moved"] == ["train/p_dead"] and dry["dry_run"]
    assert (concept_root / "chosen" / "train" / "p_dead.png").exists()

    res = service.retire(str(concept_root), dead_weight=0.05)
    assert res["ok"] and res["moved"] == ["train/p_dead"] and res["retired"] == 1
    assert not (concept_root / "chosen" / "train" / "p_dead.png").exists()
    assert (concept_root / "chosen" / sat.RETIRED_DIRNAME / "train" / "p_dead.png").exists()
    assert (concept_root / "rejected" / sat.RETIRED_DIRNAME / "train" / "p_dead.png").exists()

    # the summary now excludes the retired pair
    summary = service.summary(str(concept_root), dead_weight=0.05)
    assert summary["pairs"] == 2 and summary["saturated"] == 0 and summary["retired"] == 1

    listed = service.retired(str(concept_root))
    assert listed["ok"] and [p["key"] for p in listed["pairs"]] == ["train/p_dead"]
    assert listed["pairs"][0]["chosen"].endswith("p_dead.png") and sat.RETIRED_DIRNAME in listed["pairs"][0]["chosen"]
    assert listed["pairs"][0]["rejected"] != listed["pairs"][0]["chosen"]

    # retired images are servable, unrelated files are not
    assert service.serve_image(listed["pairs"][0]["chosen"]) is not None
    assert service.serve_image(str(concept_root / "chosen" / "train" / "p_fresh.png")) is not None
    stray = concept_root.parent / "stray.png"
    stray.write_bytes(b"x")
    assert service.serve_image(str(stray)) is None

    # explicit keys, then restore everything
    res = service.retire(str(concept_root), keys=["train/p_flip"], reason="label check")
    assert res["ok"] and res["moved"] == ["train/p_flip"] and res["retired"] == 2
    res = service.restore(str(concept_root), None)
    assert res["ok"] and sorted(res["restored"]) == ["train/p_dead", "train/p_flip"] and res["retired"] == 0
    assert (concept_root / "chosen" / "train" / "p_dead.png").exists()
    assert service.concepts()["concepts"][0]["retired"] == 0


def _fake_popen(script: str):
    def factory(cmd, **kwargs):
        assert "--progress-json" in cmd
        return subprocess.Popen([sys.executable, "-c", script], **kwargs)

    return factory


def test_start_pumps_progress_and_finishes(service, tmp_path):
    script = (
        "import json, sys\n"
        "print('loading model: x')\n"
        "print(json.dumps({'progress': {'phase': 'origin pass', 'done': 0, 'total': 3}}), flush=True)\n"
        "print(json.dumps({'progress': {'phase': 'origin pass', 'done': 3, 'total': 3}}), flush=True)\n"
        "print(json.dumps({'progress': {'phase': 'done', 'done': 1, 'total': 1}}), flush=True)\n"
        "print('done in 0.0 min')\n"
    )
    service.popen_factory = _fake_popen(script)
    res = service.start(current=str(tmp_path / "current.safetensors"), draws=2)
    assert res["ok"], res
    status = _wait(service, lambda s: not s["running"])
    assert status["phase"] == "done" and status["progress_phase"] == "done"
    assert status["done"] == 1 and status["total"] == 1
    assert any("loading model" in line for line in status["log"])
    assert status["request"]["draws"] == 2
    # the written config carries the concept file and no inline concepts
    written_path = os.path.join(tempfile.gettempdir(), "onetrainer_dpo_saturation", "train_config.json")
    with open(written_path, encoding="utf-8") as f:
        written = json.load(f)
    assert written["concepts"] is None


def test_start_validates_and_reports_failure(service, tmp_path):
    assert not service.start()["ok"]
    assert not service.start(current=str(tmp_path / "current.safetensors"), adapter="a")["ok"]
    assert not service.start(current=str(tmp_path / "missing.safetensors"))["ok"]
    assert not service.start(current=str(tmp_path / "current.safetensors"), draws=0)["ok"]

    service.popen_factory = _fake_popen("import sys; print('boom: no cache'); sys.exit(3)")
    assert service.start(current=str(tmp_path / "current.safetensors"))["ok"]
    status = _wait(service, lambda s: not s["running"])
    assert status["phase"] == "error" and "code 3" in status["error"] and "boom" in status["error"]
    # moves are refused only while running; after the failure they work again
    assert service.restore(str(service._dpo_concepts()[0].path), None)["ok"]


def test_cancel_terminates_child(service, tmp_path):
    service.popen_factory = _fake_popen("import time\nprint('x', flush=True)\ntime.sleep(60)")
    assert service.start(current=str(tmp_path / "current.safetensors"))["ok"]
    _wait(service, lambda s: s["running"])
    assert not service.start(current=str(tmp_path / "current.safetensors"))["ok"]
    assert not service.retire(str(service._dpo_concepts()[0].path), keys=["train/p_dead"])["ok"]
    res = service.cancel()
    assert res["ok"] and res["cancelled"]
    status = _wait(service, lambda s: not s["running"])
    assert status["phase"] == "cancelled"
    assert service.cancel() == {"ok": True, "cancelled": False}
