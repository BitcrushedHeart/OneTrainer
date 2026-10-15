from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.dpo_live_saturation import LiveSaturationLedger, dpo_concept_infos, format_live_report


def _dpo_concept(root: str, mix_override: float) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.path = root
    concept.dpo_chosen_pattern = "chosen/{}"
    concept.dpo_rejected_pattern = "rejected/{}"
    concept.supervised_mix_override = mix_override
    return concept


def test_concept_infos_resolve_the_supervised_mix(tmp_path):
    concepts = [
        _dpo_concept(str(tmp_path / "offpolicy"), 3.0),
        _dpo_concept(str(tmp_path / "onpolicy"), -1.0),
        _dpo_concept(str(tmp_path / "edits"), 0.0),
    ]
    mixes = {info.label: info.supervised_mix for info in dpo_concept_infos(concepts, 0.25)}
    assert mixes == {"offpolicy": 3.0, "onpolicy": 0.25, "edits": 0.0}

    # A global mix of 0 or below is off, so inheriting it is off too.
    assert dpo_concept_infos(concepts[1:2], -1.0)[0].supervised_mix == 0.0
    assert dpo_concept_infos(concepts[1:2])[0].supervised_mix == 0.0


def test_report_lists_each_concepts_supervised_mix(tmp_path):
    infos = dpo_concept_infos([_dpo_concept(str(tmp_path / "offpolicy"), 3.0)], 0.25)
    ledger = LiveSaturationLedger(str(tmp_path / "live"), infos, global_beta=200.0, start_step=0)
    summaries = ledger.summaries([])
    assert summaries[0]["supervised_mix"] == 3.0

    report = format_live_report(ledger, summaries, [], step=0)
    header = next(line for line in report.splitlines() if line.startswith("| concept"))
    row = next(line for line in report.splitlines() if line.startswith("| offpolicy"))
    assert header.rstrip(" |").endswith("sup. mix")
    assert row.rstrip(" |").endswith("| 3")
    assert header.count("|") == row.count("|")
