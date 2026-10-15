"""GenericTrainer under ``rlhf_addon_schedule = INDEPENDENT``: standard batches,
DPO pairs and text reward rollouts each on their own micro-step.

CPU-only. A real GenericTrainer.train() runs over mocked model/setup/loaders
with one real parameter, so the per-kind backward scaling, the gradient
accumulation windows across kinds, the progress counters and the mid-epoch
resume winding are all exercised for real. The PIGGYBACK path is driven the
same way to pin its (unchanged) behaviour, including the text reward
starvation that motivated the independent schedule.
"""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import modules.trainer.GenericTrainer as generic_trainer_module
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.RLHFAddonSchedule import RLHFAddonSchedule
from modules.util.enum.RLHFMode import RLHFMode
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.rlhf_schedule_util import IndependentSchedule, MicroStepKind
from modules.util.text_reward_util import TextRewardBank, TextRewardScheduler, TextSpec
from modules.util.TrainProgress import TrainProgress

import torch

import pytest

SFT, DPO, TEXT = MicroStepKind.SFT, MicroStepKind.DPO, MicroStepKind.TEXT
BATCH_SIZE = 2
SFT_GRAD, DPO_GRAD, TEXT_GRAD = 2.0, 3.0, 5.0  # d(term)/d(parameter) per element, before scaling
DPO_WEIGHT, TEXT_WEIGHT = 0.5, 0.25


class _FakeDataSet:
    """Enough of an mgds dataset for the trainer: an epoch counter, an
    approximate length and the private start offset ``wind_data_loader``
    rewrites, which the loader below honours like mgds does."""

    def __init__(self, length: int):
        self.length = length
        self.epochs_started = 0
        self.loading_pipeline = SimpleNamespace(
            _LoadingPipeline__initial_index=0, _LoadingPipeline__mark_reset_serial_modules=False
        )

    def start_next_epoch(self):
        self.epochs_started += 1

    def approximate_length(self) -> int:
        return self.length


class _FakeDataLoader:
    def __init__(self, name: str, length: int):
        self.name = name
        self.data_set = _FakeDataSet(length)
        self.actual = length  # batches really yielded; mgds lengths are approximate
        self.pulled: list[int] = []

    def get_data_set(self):
        return self.data_set

    def get_data_loader(self):
        start = self.data_set.loading_pipeline._LoadingPipeline__initial_index // BATCH_SIZE
        for index in range(start, self.actual):
            self.pulled.append(index)
            yield {"concept_type": ["STANDARD"] * BATCH_SIZE, "index": index, "source": self.name}


def _trainer(tmp_path, monkeypatch, *, schedule, ga, sft, dpo, text, progress=None, addon_interval=0):
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.train_device = "cpu"
    config.temp_device = "cpu"
    config.workspace_dir = str(tmp_path / "workspace")
    config.concept_file_name = str(tmp_path / "missing-concepts.json")
    config.sample_definition_file_name = str(tmp_path / "missing-samples.json")
    config.tensorboard = False
    config.validation = False
    config.epochs = 1
    config.batch_size = BATCH_SIZE
    config.gradient_accumulation_steps = ga
    config.sample_after_unit = TimeUnit.NEVER
    config.backup_after_unit = TimeUnit.NEVER
    config.save_every_unit = TimeUnit.NEVER
    config.rolling_backup = False
    config.clip_grad_norm = None
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    config.rlhf_dpo_addon_weight = DPO_WEIGHT
    config.rlhf_dpo_addon_interval = addon_interval
    config.rlhf_addon_schedule = schedule
    config.text_reward_weight = TEXT_WEIGHT
    config.text_reward_samples_per_epoch = text

    events: list[str] = []
    parameter = torch.nn.Parameter(torch.ones(3))
    trainer = GenericTrainer(config, Mock(), TrainCommands())
    # Flipped after construction: the startup validator would import doctr.
    config.text_reward_enabled = text > 0
    trainer.tensorboard = Mock()
    trainer.parameters = [parameter]
    trainer.sample_queue = []
    trainer.sft_anchor_source = None
    trainer.kto_source = None
    trainer._text_reward_scheduler = TextRewardScheduler(0)
    trainer._text_reward_bank = None

    optimizer = Mock(param_groups=[{"params": [parameter]}])
    grads_at_step: list[torch.Tensor] = []
    optimizer.step.side_effect = lambda: grads_at_step.append(parameter.grad.detach().clone())
    optimizer.zero_grad.side_effect = lambda set_to_none=True: setattr(parameter, "grad", None)
    trainer.model = Mock(
        ema=None,
        accumulator_state=None,
        parameters=None,
        train_progress=progress if progress is not None else TrainProgress(),
        optimizer=optimizer,
    )

    setup = trainer.model_setup = Mock()
    setup.pop_perceptual_metrics.return_value = {}
    setup.predict.side_effect = lambda *args, **kwargs: events.append("sft") or {}
    setup.calculate_loss.side_effect = lambda *args, **kwargs: (parameter * SFT_GRAD).sum()

    def dpo_light(model, batch, config, train_progress, backward_fn):
        events.append("dpo")
        backward_fn((parameter * DPO_GRAD).sum())
        return torch.tensor(DPO_GRAD * 3)

    setup.calculate_dpo_loss_light.side_effect = dpo_light
    setup.get_last_dpo_metrics.return_value = {
        "loss": 0.1,
        "dpo_loss": 0.1,
        "chosen_reward": 0.2,
        "rejected_reward": 0.1,
        "reward_margin": 0.1,
        "accuracy": 1.0,
    }

    def text_reward(model, sample, config, train_progress, backward_fn):
        events.append(f"text:{sample.spec.text}")
        backward_fn((parameter * TEXT_GRAD).sum())
        return torch.tensor(TEXT_GRAD * 3)

    setup.calculate_text_reward_loss.side_effect = text_reward
    setup.get_last_text_reward_metrics.return_value = {"ctc": 1.0}
    specs = [TextSpec("Sign", f"prompt{i}", (f"prompt{i}",), "words") for i in range(max(text, 1))]
    setup.build_text_reward_bank.side_effect = lambda model, config, epoch: TextRewardBank(
        specs, [None] * len(specs), None
    )

    trainer.model_saver = Mock()
    trainer.data_loader = _FakeDataLoader("sft", sft)
    dpo_loader = _FakeDataLoader("dpo", dpo)
    trainer.dpo_addon_source = generic_trainer_module.CyclingBatchSource(dpo_loader, "DPO add-on") if dpo else None

    monkeypatch.setattr(generic_trainer_module, "enable_grad_scaling", lambda *args: False)
    monkeypatch.setattr(generic_trainer_module, "gc", Mock())
    monkeypatch.setattr(generic_trainer_module, "torch_gc", lambda: None)
    monkeypatch.setattr("modules.util.create.create_lr_scheduler", Mock())
    return SimpleNamespace(
        trainer=trainer,
        config=config,
        events=events,
        grads=grads_at_step,
        sft_loader=trainer.data_loader,
        dpo_loader=dpo_loader,
        optimizer=optimizer,
        parameter=parameter,
    )


def _kind_of(event: str) -> MicroStepKind:
    return {"sft": SFT, "dpo": DPO}.get(event, TEXT)


def _expected_grad(kinds, ga) -> float:
    per_kind = {SFT: SFT_GRAD, DPO: DPO_GRAD * DPO_WEIGHT, TEXT: TEXT_GRAD * TEXT_WEIGHT}
    return sum(per_kind[kind] for kind in kinds) / ga


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_config_defaults_to_piggyback_and_round_trips():
    config = TrainConfig.default_values()
    assert config.rlhf_addon_schedule == RLHFAddonSchedule.PIGGYBACK
    assert not config.rlhf_addon_independent()
    config.rlhf_addon_schedule = RLHFAddonSchedule.INDEPENDENT
    loaded = TrainConfig.default_values().from_dict(config.to_dict())
    assert loaded.rlhf_addon_schedule == RLHFAddonSchedule.INDEPENDENT
    # a config saved before the field existed keeps today's behaviour
    stale = {k: v for k, v in config.to_dict().items() if k != "rlhf_addon_schedule"}
    assert TrainConfig.default_values().from_dict(stale).rlhf_addon_schedule == RLHFAddonSchedule.PIGGYBACK


def test_independent_only_applies_where_add_ons_ride_on_a_finetune():
    config = TrainConfig.default_values()
    config.rlhf_addon_schedule = RLHFAddonSchedule.INDEPENDENT
    assert config.rlhf_addon_independent()  # plain run (text reward)
    config.rlhf_enabled = True
    config.rlhf_mode = RLHFMode.SFT_DPO
    assert config.rlhf_addon_independent()
    config.rlhf_mode = RLHFMode.DPO
    assert not config.rlhf_addon_independent()  # DPO-primary has no add-on schedule


# --------------------------------------------------------------------------- #
# one epoch, fresh
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ga", [1, 2, 3])
def test_independent_epoch_runs_every_kind_on_its_own_micro_step(tmp_path, monkeypatch, ga, capsys):
    sft, dpo, text = 5, 4, 3
    run = _trainer(tmp_path, monkeypatch, schedule=RLHFAddonSchedule.INDEPENDENT, ga=ga, sft=sft, dpo=dpo, text=text)
    run.trainer.train()

    schedule = IndependentSchedule({SFT: sft, DPO: dpo, TEXT: text})
    kinds = [_kind_of(e) for e in run.events]
    # exactly one term per micro-step, in schedule order, every source at its full length
    assert kinds == list(schedule.kinds_from(0))
    assert Counter(kinds) == {SFT: sft, DPO: dpo, TEXT: text}
    assert run.sft_loader.pulled == list(range(sft))
    assert run.dpo_loader.pulled == list(range(dpo))
    assert run.dpo_loader.data_set.epochs_started == 1  # consumed exactly, never cycled
    assert [e for e in run.events if e.startswith("text")] == [f"text:prompt{i}" for i in range(text)]

    # gradient accumulation counts micro-steps of every kind
    total = sft + dpo + text
    assert run.optimizer.step.call_count == total // ga
    for window, grad in enumerate(run.grads):
        window_kinds = kinds[window * ga : (window + 1) * ga]
        assert torch.allclose(grad, torch.full((3,), _expected_grad(window_kinds, ga))), (window, window_kinds)

    # progress: every kind is a step; epoch_sample counts standard samples only
    progress = run.trainer.model.train_progress
    assert progress.global_step == total
    assert progress.epoch == 1 and progress.epoch_step == 0  # next_epoch() ran
    assert run.trainer.callbacks.on_update_train_progress.call_args_list[-1].args[1] == total

    out = capsys.readouterr().out
    assert f"{sft} standard + {dpo} DPO + {text} text reward = {total} micro-steps per epoch" in out
    assert f"{total // ga} optimizer steps (gradient accumulation {ga})" in out


def test_independent_epoch_sample_tracks_standard_batches_only(tmp_path, monkeypatch):
    run = _trainer(tmp_path, monkeypatch, schedule=RLHFAddonSchedule.INDEPENDENT, ga=1, sft=4, dpo=3, text=2)
    samples: list[int] = []
    run.trainer.callbacks.on_update_train_progress.side_effect = lambda p, *args: samples.append(p.epoch_sample)
    run.trainer.train()
    kinds = [_kind_of(e) for e in run.events]
    expected, seen = [], 0
    for kind in kinds:
        seen += BATCH_SIZE if kind is SFT else 0
        expected.append(seen)
    assert samples[: len(kinds)] == expected


def test_independent_logs_the_standard_loss_on_its_own(tmp_path, monkeypatch):
    run = _trainer(tmp_path, monkeypatch, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2, sft=4, dpo=2, text=0)
    run.trainer.train()
    tags = [call.args[0] for call in run.trainer.tensorboard.add_scalar.call_args_list]
    assert tags.count("loss/train_step") == 3
    # SFT appears in every GA=2 window of [S D S S D S]: logged each time, as the unscaled mean
    sft_logs = [
        call.args[1] for call in run.trainer.tensorboard.add_scalar.call_args_list if call.args[0] == "loss/sft_step"
    ]
    assert sft_logs == pytest.approx([SFT_GRAD * 3] * 3)
    assert "loss/dpo" in tags


def test_independent_with_only_a_text_reward_on_a_plain_run(tmp_path, monkeypatch):
    run = _trainer(tmp_path, monkeypatch, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2, sft=6, dpo=0, text=2)
    run.config.rlhf_enabled = False
    run.trainer.train()
    kinds = [_kind_of(e) for e in run.events]
    assert kinds == list(IndependentSchedule({SFT: 6, TEXT: 2}).kinds_from(0))
    assert run.optimizer.step.call_count == 4


def test_independent_ends_the_epoch_when_the_standard_loader_runs_dry(tmp_path, monkeypatch, capsys):
    run = _trainer(tmp_path, monkeypatch, schedule=RLHFAddonSchedule.INDEPENDENT, ga=1, sft=8, dpo=3, text=0)
    run.sft_loader.actual = 6  # approximate_length over-estimates by two batches
    run.trainer.train()
    kinds = [_kind_of(e) for e in run.events]
    assert Counter(kinds)[SFT] == 6
    assert Counter(kinds)[DPO] <= 3
    assert run.trainer.model.train_progress.epoch == 1
    assert "ran dry at micro-step" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


def test_independent_resume_continues_the_schedule_from_the_backup_position(tmp_path, monkeypatch, capsys):
    sft, dpo, text = 5, 4, 3
    schedule = IndependentSchedule({SFT: sft, DPO: dpo, TEXT: text})
    position = 7
    consumed = schedule.consumed_before(position)
    # exactly what a backup written under INDEPENDENT at this position carries
    progress = TrainProgress(
        epoch=0, epoch_step=position, epoch_sample=consumed[SFT] * BATCH_SIZE, global_step=position
    )
    run = _trainer(
        tmp_path,
        monkeypatch,
        schedule=RLHFAddonSchedule.INDEPENDENT,
        ga=2,
        sft=sft,
        dpo=dpo,
        text=text,
        progress=progress,
    )
    run.trainer.train()

    kinds = [_kind_of(e) for e in run.events]
    assert kinds == list(schedule.kinds_from(position))
    assert run.sft_loader.pulled == list(range(consumed[SFT], sft))
    assert run.dpo_loader.pulled == list(range(consumed[DPO], dpo))
    assert [e for e in run.events if e.startswith("text")] == [f"text:prompt{i}" for i in range(consumed[TEXT], text)]
    assert run.trainer.model.train_progress.global_step == len(schedule)
    # the GA windows are counted from the global step, so the first update lands
    # on the first even global step after the resume
    assert run.optimizer.step.call_count == len(schedule) // 2 - position // 2
    assert f"resuming at micro-step {position} of {len(schedule)}" in capsys.readouterr().out


def test_piggyback_backup_resumed_under_independent_is_reinterpreted_as_a_position(tmp_path, monkeypatch):
    sft, dpo, text = 6, 4, 2
    # A PIGGYBACK backup after 4 standard micro-steps: epoch_step 4, epoch_sample 4 * batch.
    progress = TrainProgress(epoch=0, epoch_step=4, epoch_sample=4 * BATCH_SIZE, global_step=4)
    run = _trainer(
        tmp_path,
        monkeypatch,
        schedule=RLHFAddonSchedule.INDEPENDENT,
        ga=2,
        sft=sft,
        dpo=dpo,
        text=text,
        progress=progress,
    )
    run.trainer.train()
    schedule = IndependentSchedule({SFT: sft, DPO: dpo, TEXT: text})
    consumed = schedule.consumed_before(4)
    kinds = [_kind_of(e) for e in run.events]
    assert kinds == list(schedule.kinds_from(4))
    # every loader was wound to the schedule's view of position 4, not the backup's SFT count
    assert run.sft_loader.pulled == list(range(consumed[SFT], sft))
    assert run.dpo_loader.pulled == list(range(consumed[DPO], dpo))
    assert run.trainer.model.train_progress.epoch == 1


def test_backup_past_the_end_of_the_schedule_moves_on_to_the_next_epoch(tmp_path, monkeypatch):
    # An INDEPENDENT backup at micro-step 40 resumed on a smaller dataset (or
    # vice versa): nothing to train in this epoch, no crash, no phantom tail.
    progress = TrainProgress(epoch=0, epoch_step=40, epoch_sample=10 * BATCH_SIZE, global_step=40)
    run = _trainer(
        tmp_path, monkeypatch, schedule=RLHFAddonSchedule.INDEPENDENT, ga=2, sft=5, dpo=4, text=0, progress=progress
    )
    run.trainer.train()
    assert run.events == []
    assert run.trainer.model.train_progress.epoch == 1
    assert run.trainer.model.train_progress.epoch_step == 0


def test_independent_backup_resumed_under_piggyback_keeps_the_standard_position(tmp_path, monkeypatch):
    # The INDEPENDENT epoch_sample counts standard samples, which is exactly
    # the main loader's PIGGYBACK resume offset; the epoch continues from the
    # standard batch it stopped at.
    progress = TrainProgress(epoch=0, epoch_step=9, epoch_sample=3 * BATCH_SIZE, global_step=9)
    run = _trainer(
        tmp_path,
        monkeypatch,
        schedule=RLHFAddonSchedule.PIGGYBACK,
        ga=1,
        sft=6,
        dpo=6,
        text=0,
        progress=progress,
    )
    run.sft_loader.data_set.loading_pipeline._LoadingPipeline__initial_index = progress.epoch_sample
    run.trainer.train()
    assert run.sft_loader.pulled == [3, 4, 5]
    assert run.trainer.model.train_progress.epoch == 1


# --------------------------------------------------------------------------- #
# PIGGYBACK is untouched
# --------------------------------------------------------------------------- #


def test_piggyback_still_rides_add_ons_on_standard_steps_and_starves_the_text_reward(tmp_path, monkeypatch):
    # Interval 1 (as many pairs as standard batches): every micro-step carries
    # a pair, so the text reward -- never allowed on a DPO step -- never fires.
    run = _trainer(tmp_path, monkeypatch, schedule=RLHFAddonSchedule.PIGGYBACK, ga=2, sft=4, dpo=4, text=2)
    run.trainer.train()
    assert run.events == ["sft", "dpo"] * 4
    assert run.optimizer.step.call_count == 2
    assert run.trainer.model.train_progress.global_step == 4
    for grad in run.grads:
        assert torch.allclose(grad, torch.full((3,), (2 * SFT_GRAD + 2 * DPO_GRAD * DPO_WEIGHT) / 2))
    assert run.trainer._independent_schedule is None


def test_piggyback_interval_two_visits_half_the_pairs(tmp_path, monkeypatch):
    run = _trainer(
        tmp_path, monkeypatch, schedule=RLHFAddonSchedule.PIGGYBACK, ga=1, sft=6, dpo=6, text=0, addon_interval=2
    )
    run.trainer.train()
    assert run.events == ["sft", "dpo", "sft"] * 3
    assert run.dpo_loader.pulled == [0, 1, 2]
