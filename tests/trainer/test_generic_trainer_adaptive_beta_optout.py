"""GenericTrainer feeds the adaptive beta controller the margin over the
samples whose concept takes part (pooled per sample across the window), and
skips the update on a step without such samples. CPU-only, on the mocked
DPO trainer of test_generic_trainer_rotation_cache.
"""

import pytest
from test_generic_trainer_rotation_cache import _ordering_trainer


class _Controller:
    def __init__(self):
        self.inputs: list[float] = []
        self.margin_ema = 0.0

    def update(self, margin: float) -> float:
        self.inputs.append(margin)
        return 123.0


def _run(tmp_path, monkeypatch, metrics: dict):
    # 6 micro-batches at gradient accumulation 3: two optimizer steps, each
    # accumulating three copies of ``metrics``.
    trainer, events, setup = _ordering_trainer(tmp_path, monkeypatch, use_scaler=False)
    setup.get_last_dpo_metrics.return_value = metrics
    controller = trainer._dpo_beta_controller = _Controller()
    trainer.train()
    return controller, events, setup


_BASE = {"loss": 0.1, "dpo_loss": 0.1, "chosen_reward": 0.2, "rejected_reward": 0.1, "accuracy": 1.0}


def test_controller_sees_the_mean_over_included_samples(tmp_path, monkeypatch):
    # Batch mean 4.0 (one off-policy pair at ~12), included mean 0.15.
    metrics = {**_BASE, "reward_margin": 4.0, "adaptive_margin_sum": 0.3, "adaptive_count": 2.0}
    controller, events, setup = _run(tmp_path, monkeypatch, metrics)
    assert controller.inputs == pytest.approx([0.15, 0.15])
    assert [call.args for call in setup.set_dpo_runtime_beta.call_args_list] == [(123.0,), (123.0,)]
    assert events.count("beta") == 2


def test_a_step_without_included_samples_leaves_the_runtime_beta_alone(tmp_path, monkeypatch):
    metrics = {**_BASE, "reward_margin": 4.0, "adaptive_margin_sum": 0.0, "adaptive_count": 0.0}
    controller, events, setup = _run(tmp_path, monkeypatch, metrics)
    assert controller.inputs == []
    setup.set_dpo_runtime_beta.assert_not_called()
    assert "beta" not in events


def test_metrics_without_the_mask_feed_the_batch_mean_as_before(tmp_path, monkeypatch):
    controller, events, setup = _run(tmp_path, monkeypatch, {**_BASE, "reward_margin": 0.1})
    assert controller.inputs == pytest.approx([0.1, 0.1])
    assert events.count("beta") == 2
