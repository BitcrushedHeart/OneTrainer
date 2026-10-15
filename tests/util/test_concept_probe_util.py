"""Per-image visit ledger and fixed probe set (concept_probe_util)."""

import json
import os

from modules.util.concept_probe_util import ConceptVisitLedger, ProbeSet, move_sample, slice_sample

import torch

import pytest


def test_ledger_reports_paired_deltas_over_images_seen_twice():
    ledger = ConceptVisitLedger()
    ledger.record("A", "a1", 1.0)
    ledger.record("A", "a2", 2.0)
    ledger.record("A", "a3", 1.5)  # seen once: not counted
    ledger.record("B", "b1", 1.0)
    assert ledger.scalars() == {}
    ledger.record("A", "a1", 0.8)  # improved by 0.2
    ledger.record("A", "a2", 2.4)  # worse by 0.4
    ledger.record("B", "b1", 1.0)  # flat
    scalars = ledger.scalars()
    assert scalars["A/revisited"] == 2.0
    assert scalars["A/visit_delta"] == pytest.approx((-0.2 + 0.4) / 2)
    assert scalars["A/since_first"] == pytest.approx((-0.2 + 0.4) / 2)
    assert scalars["A/frac_improving"] == 0.5
    assert scalars["B/revisited"] == 1.0 and scalars["B/frac_improving"] == 0.0
    # a third visit: the visit delta pairs the latest two, since_first keeps the first
    ledger.record("A", "a1", 0.5)
    scalars = ledger.scalars()
    assert scalars["A/visit_delta"] == pytest.approx((-0.3 + 0.4) / 2)
    assert scalars["A/since_first"] == pytest.approx((-0.5 + 0.4) / 2)


def test_ledger_caps_the_visits_but_keeps_the_first_value():
    ledger = ConceptVisitLedger(visits_kept=3)
    for value in (5.0, 4.0, 3.0, 2.0, 1.0):
        ledger.record("A", "a", value)
    assert ledger.visits("a") == [3.0, 2.0, 1.0]
    scalars = ledger.scalars()
    assert scalars["A/visit_delta"] == -1.0
    assert scalars["A/since_first"] == -4.0  # against the uncapped first visit
    assert scalars["A/frac_improving"] == 1.0


def test_ledger_ignores_nan_and_missing_paths():
    ledger = ConceptVisitLedger()
    ledger.record("A", None, 1.0)
    ledger.record("A", "a", float("nan"))
    assert len(ledger) == 0
    ledger.record("A", "a", 1.0)
    ledger.record("A", "a", float("nan"))
    assert ledger.visits("a") == [1.0]


def test_ledger_round_trips_through_disk_and_starts_empty_without_a_file(tmp_path):
    ledger = ConceptVisitLedger(visits_kept=4)
    for value in (3.0, 2.0, 1.5):
        ledger.record("A", "a", value, kind="sft")
    ledger.record("P", "p", 0.7, kind="dpo")
    path = str(tmp_path / "d" / "visit_ledger.json")
    ledger.save(path)
    loaded = ConceptVisitLedger.load(path, visits_kept=4)
    assert loaded.state_dict() == ledger.state_dict()
    assert loaded.scalars() == ledger.scalars()
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["images"]["p"]["kind"] == "dpo"
    assert len(ConceptVisitLedger.load(str(tmp_path / "missing.json"))) == 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not json")
    assert len(ConceptVisitLedger.load(path)) == 0


def test_slice_sample_takes_one_row_of_every_collated_value():
    batch = {
        "latent": torch.arange(6.0).reshape(3, 2),
        "shared": torch.tensor([7.0, 8.0]),  # leading dim is not the batch
        "scalar": torch.tensor(1.0),
        "image_path": ["a", "b", "c"],
        "concept": {"name": ["x", "y", "z"], "loss_weight": torch.tensor([1.0, 2.0, 3.0])},
        "per_field": [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
        "size": (512, 768),
    }
    sample = slice_sample(batch, 1, 3)
    assert torch.equal(sample["latent"], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(sample["shared"], batch["shared"])
    assert sample["scalar"].item() == 1.0
    assert sample["image_path"] == ["b"]
    assert sample["concept"]["name"] == ["y"] and torch.equal(sample["concept"]["loss_weight"], torch.tensor([2.0]))
    assert sample["per_field"] is batch["per_field"]  # a 2-long list is not the batch: passed through
    assert sample["size"] == (512, 768)
    moved = move_sample(sample, torch.device("cpu"))
    assert torch.equal(moved["latent"], sample["latent"])


def test_probe_set_captures_up_to_k_distinct_images_per_concept_and_persists(tmp_path):
    probes = ProbeSet(images_per_concept=2)
    batch = {"x": torch.arange(4.0), "image_path": ["a1", "a2", "b1", "a1"]}
    assert probes.capture("A", "sft", batch, ["a1", "a2", None, "a1"], 4) == 2
    assert probes.capture("B", "dpo", batch, [None, None, "b1", None], 4) == 1
    assert probes.capture("A", "sft", {"x": torch.arange(1.0), "image_path": ["a3"]}, ["a3"], 1) == 0  # full
    assert probes.count("A") == 2 and probes.count("B") == 1 and probes.total() == 3
    assert probes.kind("A") == "sft" and probes.kind("B") == "dpo"
    assert [p for p, _ in probes.samples("A")] == ["a1", "a2"]
    assert dict(probes.samples("A"))["a2"]["x"].tolist() == [1.0]
    assert not probes.wants("A", "a9") and probes.wants("B", "b2")

    assert probes.delta_since_first("A", 1.0) is None  # baseline over 2 images
    assert probes.delta_since_first("A", 0.75) == pytest.approx(-0.25)
    probes.capture("B", "dpo", {"x": torch.arange(1.0), "image_path": ["b2"]}, ["b2"], 1)
    assert probes.delta_since_first("B", 0.5) is None
    assert probes.delta_since_first("B", 0.4) == pytest.approx(-0.1)

    path = str(tmp_path / "d" / "probes.pt")
    probes.save(path)
    assert not probes.dirty
    loaded = ProbeSet.load(path, images_per_concept=2)
    assert loaded.total() == 4 and loaded.kind("B") == "dpo"
    assert loaded.delta_since_first("A", 1.5) == pytest.approx(0.5)  # the baseline survives
    assert ProbeSet.load(str(tmp_path / "missing.pt")).total() == 0
    assert os.path.isfile(path)
