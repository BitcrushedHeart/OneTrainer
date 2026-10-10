"""GenericTrainer's per-concept DPO metrics and live saturation ledger: set up
from the run's concepts, fed from the model setup's per-pair records, and
flushed into <workspace>/dpo_live/."""

from types import SimpleNamespace
from unittest.mock import Mock

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_live_saturation import LEDGER_FILENAME, LIVE_DIRNAME
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.RLHFMode import RLHFMode

import pytest


def _dpo_concept(path: str, beta: float) -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.path = path
    concept.dpo_chosen_pattern = "chosen/{}"
    concept.dpo_rejected_pattern = "rejected/{}"
    concept.dpo_beta_override = beta
    return concept


def _trainer(tmp_path, **overrides) -> GenericTrainer:
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_beta = 800.0
    config.workspace_dir = str(tmp_path / "workspace")
    standard = ConceptConfig.default_values()
    standard.path = str(tmp_path / "SoReal")
    config.concepts = [
        standard,
        _dpo_concept(str(tmp_path / "RLHF_5"), 120.0),
        _dpo_concept(str(tmp_path / "Off Policy RLHF"), 650.0),
    ]
    for key, value in overrides.items():
        setattr(config, key, value)
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = config
    trainer.model = SimpleNamespace(train_progress=SimpleNamespace(global_step=100))
    trainer.model_setup = Mock()
    trainer._dpo_unmapped_pair_warned = False
    return trainer


def _record(path: str, margin: float, beta: float = 120.0) -> dict:
    return {"path": path, "margin": margin, "chosen": -0.01, "rejected": -0.01 - margin, "beta": beta, "t": 0.4}


def test_flags_default_off_and_round_trip():
    config = TrainConfig.default_values()
    assert config.rlhf_dpo_concept_logging is False and config.rlhf_dpo_live_saturation is False
    config.rlhf_dpo_concept_logging = True
    config.rlhf_dpo_live_saturation = True
    loaded = TrainConfig.default_values().from_dict(config.to_dict())
    assert loaded.rlhf_dpo_concept_logging is True and loaded.rlhf_dpo_live_saturation is True
    # A config saved before the fields existed loads them off.
    old = config.to_dict()
    del old["rlhf_dpo_concept_logging"], old["rlhf_dpo_live_saturation"]
    loaded = TrainConfig.default_values().from_dict(old)
    assert loaded.rlhf_dpo_concept_logging is False and loaded.rlhf_dpo_live_saturation is False


def test_setup_does_nothing_while_both_flags_are_off(tmp_path):
    trainer = _trainer(tmp_path)
    trainer._GenericTrainer__setup_dpo_pair_logging()
    assert trainer._dpo_concept_resolver is None
    assert trainer._dpo_concept_window is None
    assert trainer._dpo_live_ledger is None


def test_records_feed_the_concept_window_and_the_ledger(tmp_path):
    trainer = _trainer(tmp_path, rlhf_dpo_concept_logging=True, rlhf_dpo_live_saturation=True)
    trainer._GenericTrainer__setup_dpo_pair_logging()
    chosen = str(tmp_path / "RLHF_5" / "chosen" / "a.png")
    trainer.model_setup.get_last_dpo_pair_records.return_value = [_record(chosen, 0.02)]
    trainer.model_setup.dpo_reference_fingerprint.return_value = "ref-a"

    trainer._GenericTrainer__collect_dpo_pairs(SimpleNamespace(global_step=101))

    assert trainer._dpo_concept_window.scalars()["dpo/concept/RLHF_5/margin"] == pytest.approx(0.02)
    ledger = trainer._dpo_live_ledger
    assert ledger.reference == "ref-a"
    assert [v[0] for v in ledger.pairs[chosen]["visits"]] == [101]
    assert ledger.flush(101, force=True)
    assert (tmp_path / "workspace" / LIVE_DIRNAME / LEDGER_FILENAME).is_file()


def test_unmapped_pairs_are_skipped_with_one_warning(tmp_path, capsys):
    trainer = _trainer(tmp_path, rlhf_dpo_concept_logging=True)
    trainer._GenericTrainer__setup_dpo_pair_logging()
    stray = _record(str(tmp_path / "elsewhere" / "x.png"), 0.1, beta=800.0)
    trainer.model_setup.get_last_dpo_pair_records.return_value = [stray, stray]

    trainer._GenericTrainer__collect_dpo_pairs(SimpleNamespace(global_step=1))
    trainer._GenericTrainer__collect_dpo_pairs(SimpleNamespace(global_step=2))

    assert trainer._dpo_concept_window.scalars() == {}
    assert capsys.readouterr().out.count("under no DPO concept") == 1
    # Without a ledger the reference is never hashed.
    trainer.model_setup.dpo_reference_fingerprint.assert_not_called()


def test_live_saturation_is_skipped_for_ipo(tmp_path, capsys):
    trainer = _trainer(
        tmp_path,
        rlhf_dpo_concept_logging=True,
        rlhf_dpo_live_saturation=True,
        rlhf_dpo_objective=DPOObjective.IPO,
    )
    trainer._GenericTrainer__setup_dpo_pair_logging()
    assert trainer._dpo_live_ledger is None
    assert trainer._dpo_concept_window is not None and trainer._dpo_concept_window.with_weight is False
    assert "sigmoid objective" in capsys.readouterr().out
