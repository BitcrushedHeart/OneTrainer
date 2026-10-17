"""GenericTrainer's per-concept adaptive beta: one controller per adaptive DPO
concept, each updated only on steps with a sample of its own and only from
those samples' margins, the ratios handed to the model setup and logged per
concept; the pooled controller keeps running for samples under no concept.
CPU-only, on the mocked DPO trainer of test_generic_trainer_rotation_cache.
"""

from types import SimpleNamespace
from unittest.mock import Mock, call

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_beta_controller import DPOBetaController
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.RLHFMode import RLHFMode

import pytest
from test_generic_trainer_rotation_cache import _ordering_trainer


class _Controller:
    def __init__(self, value: float):
        self.value = value
        self.inputs: list[float] = []
        self.margin_ema = 0.0

    def update(self, margin: float) -> float:
        self.inputs.append(margin)
        return self.value


_BASE = {
    "loss": 0.1,
    "dpo_loss": 0.1,
    "chosen_reward": 0.2,
    "rejected_reward": 0.1,
    "accuracy": 1.0,
    "reward_margin": 0.3,
    "beta_mean": 150.0,
}
KEYS = ["Off_Policy", "RLHF_5"]


def _run(tmp_path, monkeypatch, metrics: dict, *, adaptive: bool = True):
    # 6 micro-batches at gradient accumulation 3: two optimizer steps, each
    # accumulating three copies of ``metrics``.
    trainer, events, setup = _ordering_trainer(tmp_path, monkeypatch, use_scaler=False)
    trainer.config.rlhf_dpo_adaptive_beta = adaptive
    trainer._dpo_adaptive_keys = list(KEYS)
    pooled = trainer._dpo_beta_controller = _Controller(123.0)
    controllers = {"Off_Policy": _Controller(0.5), "RLHF_5": _Controller(2.0)}
    trainer._dpo_beta_controllers = dict(controllers)
    setup.get_last_dpo_metrics.return_value = metrics
    trainer.train()
    return trainer, pooled, controllers, setup


def test_each_controller_sees_only_its_own_concepts_mean(tmp_path, monkeypatch):
    metrics = {
        **_BASE,
        "adaptive_margin_sum": 0.0015,
        "adaptive_count": 3.0,
        "adaptive_margin_sum/RLHF_5": 0.0012,
        "adaptive_count/RLHF_5": 2.0,
        "adaptive_margin_sum/Off_Policy": 0.0003,
        "adaptive_count/Off_Policy": 1.0,
    }
    trainer, pooled, controllers, setup = _run(tmp_path, monkeypatch, metrics)
    assert controllers["RLHF_5"].inputs == pytest.approx([0.0006, 0.0006])
    assert controllers["Off_Policy"].inputs == pytest.approx([0.0003, 0.0003])
    assert pooled.inputs == pytest.approx([0.0005, 0.0005])
    assert setup.set_dpo_runtime_beta_ratios.call_args_list == [
        call({"Off_Policy": 0.5, "RLHF_5": 2.0}),
        call({"Off_Policy": 0.5, "RLHF_5": 2.0}),
    ]
    assert setup.set_dpo_runtime_beta.call_args_list == [call(123.0), call(123.0)]
    logged = {c.args[0]: c.args[1] for c in trainer.tensorboard.add_scalar.call_args_list}
    assert logged["dpo/concept/RLHF_5/adaptive_ratio"] == 2.0
    assert logged["dpo/concept/Off_Policy/adaptive_ratio"] == 0.5
    # dpo/beta follows the batch-mean applied beta once concepts adapt apart.
    assert logged["dpo/beta"] == 150.0


def test_a_concept_without_samples_this_step_is_left_alone(tmp_path, monkeypatch):
    metrics = {
        **_BASE,
        "adaptive_margin_sum": 0.0012,
        "adaptive_count": 2.0,
        "adaptive_margin_sum/RLHF_5": 0.0012,
        "adaptive_count/RLHF_5": 2.0,
    }
    trainer, pooled, controllers, setup = _run(tmp_path, monkeypatch, metrics)
    assert controllers["RLHF_5"].inputs == pytest.approx([0.0006, 0.0006])
    assert controllers["Off_Policy"].inputs == []
    # Its last ratio (none yet) is not in the dict handed over, so it keeps 1.0.
    assert setup.set_dpo_runtime_beta_ratios.call_args == call({"RLHF_5": 2.0})
    scalars = [c.args[0] for c in trainer.tensorboard.add_scalar.call_args_list]
    assert "dpo/concept/RLHF_5/adaptive_ratio" in scalars
    assert "dpo/concept/Off_Policy/adaptive_ratio" not in scalars


def test_controllers_are_created_on_first_use_with_a_ratio_beta0(tmp_path, monkeypatch):
    metrics = {
        **_BASE,
        "adaptive_margin_sum": 0.3,
        "adaptive_count": 1.0,
        "adaptive_margin_sum/RLHF_5": 0.3,
        "adaptive_count/RLHF_5": 1.0,
    }
    trainer, events, setup = _ordering_trainer(tmp_path, monkeypatch, use_scaler=False)
    trainer._dpo_adaptive_keys = list(KEYS)
    setup.get_last_dpo_metrics.return_value = metrics
    trainer.train()
    assert list(trainer._dpo_beta_controllers) == ["RLHF_5"]  # Off_Policy never had a sample
    controller = trainer._dpo_beta_controllers["RLHF_5"]
    assert isinstance(controller, DPOBetaController) and controller.beta0 == 1.0
    assert trainer._dpo_beta_ratios == {"RLHF_5": 1.0}  # still warming up


def test_metrics_without_the_per_concept_block_only_drive_the_pooled_controller(tmp_path, monkeypatch):
    metrics = {**_BASE, "adaptive_margin_sum": 0.3, "adaptive_count": 2.0}
    trainer, pooled, controllers, setup = _run(tmp_path, monkeypatch, metrics)
    assert pooled.inputs == pytest.approx([0.15, 0.15])
    assert controllers["RLHF_5"].inputs == [] and controllers["Off_Policy"].inputs == []
    setup.set_dpo_runtime_beta_ratios.assert_not_called()


def test_adaptive_beta_off_touches_nothing(tmp_path, monkeypatch):
    metrics = {
        **_BASE,
        "adaptive_margin_sum": 0.3,
        "adaptive_count": 1.0,
        "adaptive_margin_sum/RLHF_5": 0.3,
        "adaptive_count/RLHF_5": 1.0,
    }
    trainer, pooled, controllers, setup = _run(tmp_path, monkeypatch, metrics, adaptive=False)
    assert pooled.inputs == [] and controllers["RLHF_5"].inputs == []
    setup.set_dpo_runtime_beta.assert_not_called()
    setup.set_dpo_runtime_beta_ratios.assert_not_called()
    assert "dpo/beta" not in [c.args[0] for c in trainer.tensorboard.add_scalar.call_args_list]


# --------------------------------------------------------------------------- #
# startup: keys and the path -> concept mapping
# --------------------------------------------------------------------------- #


def _dpo_concept(path: str, *, adaptive: bool = True, name: str = "DPO") -> ConceptConfig:
    concept = ConceptConfig.default_values()
    concept.name = name
    concept.path = path
    concept.dpo_chosen_pattern = "chosen/{}"
    concept.dpo_rejected_pattern = "rejected/{}"
    concept.dpo_adaptive_beta = adaptive
    return concept


def _trainer(tmp_path, **overrides) -> GenericTrainer:
    config = TrainConfig.default_values()
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_adaptive_beta = True
    config.workspace_dir = str(tmp_path / "workspace")
    # Three concepts sharing one name: the key has to come from the folder.
    config.concepts = [
        _dpo_concept(str(tmp_path / "RLHF_5")),
        _dpo_concept(str(tmp_path / "Off Policy RLHF"), adaptive=False),
        _dpo_concept(str(tmp_path / "Night Edits")),
    ]
    for key, value in overrides.items():
        setattr(config, key, value)
    trainer = GenericTrainer.__new__(GenericTrainer)
    trainer.config = config
    trainer.model = SimpleNamespace(train_progress=SimpleNamespace(global_step=0))
    trainer.model_setup = Mock()
    trainer._dpo_concept_resolver = None
    return trainer


def test_setup_keys_the_adaptive_concepts_by_tag_and_maps_chosen_paths(tmp_path, capsys):
    trainer = _trainer(tmp_path)
    trainer._GenericTrainer__setup_adaptive_beta_concepts()
    assert trainer._dpo_adaptive_keys == ["Night_Edits", "RLHF_5"]
    (keys, key_of_path), _ = trainer.model_setup.set_dpo_concept_keys.call_args
    assert keys == ["Night_Edits", "RLHF_5"]
    assert key_of_path(str(tmp_path / "RLHF_5" / "chosen" / "a.png")) == "RLHF_5"
    assert key_of_path(str(tmp_path / "Night Edits" / "chosen" / "sub" / "b.png")) == "Night_Edits"
    # Opted out: no key, so its samples never reach a per-concept controller.
    assert key_of_path(str(tmp_path / "Off Policy RLHF" / "chosen" / "c.png")) is None
    assert key_of_path(str(tmp_path / "elsewhere" / "d.png")) is None
    assert key_of_path(None) is None
    assert "one controller per concept for Night_Edits, RLHF_5" in capsys.readouterr().out


@pytest.mark.parametrize(
    "overrides",
    [{"rlhf_dpo_adaptive_beta": False}, {"rlhf_dpo_objective": DPOObjective.IPO}, {"rlhf_enabled": False}],
    ids=["adaptive_off", "ipo", "no_rlhf"],
)
def test_setup_does_nothing_unless_adaptive_sigmoid_dpo(tmp_path, overrides):
    trainer = _trainer(tmp_path, **overrides)
    trainer._GenericTrainer__setup_adaptive_beta_concepts()
    assert trainer._dpo_adaptive_keys == []
    trainer.model_setup.set_dpo_concept_keys.assert_not_called()


def test_setup_with_every_concept_opted_out_keeps_the_pooled_controller_only(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.config.concepts = [_dpo_concept(str(tmp_path / "RLHF_5"), adaptive=False)]
    trainer._GenericTrainer__setup_adaptive_beta_concepts()
    assert trainer._dpo_adaptive_keys == []
    trainer.model_setup.set_dpo_concept_keys.assert_not_called()
