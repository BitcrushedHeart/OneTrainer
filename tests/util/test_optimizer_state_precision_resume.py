from types import SimpleNamespace

from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection

import torch


def _config(state_precision: str) -> TrainConfig:
    config = TrainConfig.default_values()
    config.optimizer.optimizer = Optimizer.ADAMW_ADV
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.999
    config.optimizer.weight_decay = 0.0
    config.optimizer.cautious_wd = False
    config.optimizer.centered_wd = 0.0
    config.optimizer.spectral_normalization = False
    config.optimizer.kourkoutas_beta = False
    config.optimizer.factored_2nd = True
    config.optimizer.state_precision = state_precision
    return config


def _groups(parameter: torch.nn.Parameter) -> NamedParameterGroupCollection:
    groups = NamedParameterGroupCollection()
    groups.add_group(NamedParameterGroup("test", [parameter], 1e-3))
    return groups


def _saved_fp32_state() -> dict:
    parameter = torch.nn.Parameter(torch.ones(32, 32))
    groups = _groups(parameter)
    optimizer = create.create_optimizer(
        groups,
        None,
        _config("auto"),
        model=SimpleNamespace(part=torch.nn.Linear(1, 1)),
    )
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state_dict()
    state["param_group_mapping"] = groups.unique_name_mapping
    state["param_group_optimizer_mapping"] = [str(Optimizer.ADAMW_ADV)]
    assert state["state"][0]["exp_avg"].dtype is torch.float32
    return state


def test_resume_converts_dense_optimizer_state_to_bf16_without_resetting_momentum():
    state = _saved_fp32_state()
    saved_momentum = state["state"][0]["exp_avg"].clone()

    resumed_parameter = torch.nn.Parameter(torch.ones(32, 32))
    resumed = create.create_optimizer(
        _groups(resumed_parameter),
        state,
        _config("bf16_sr"),
        model=SimpleNamespace(part=torch.nn.Linear(1, 1)),
    )

    group = resumed.param_groups[0]
    resumed_state = resumed.state[resumed_parameter]
    assert group["state_precision"] == "bf16_sr"
    assert group["actual_state_precision"] == "bf16_sr"
    assert resumed_state["exp_avg"].dtype is torch.bfloat16
    torch.testing.assert_close(resumed_state["exp_avg"].float(), saved_momentum, rtol=0, atol=1e-3)


def test_resume_does_not_apply_structural_factoring_without_conversion(capsys):
    state = _saved_fp32_state()
    resumed_parameter = torch.nn.Parameter(torch.ones(32, 32))

    resumed = create.create_optimizer(
        _groups(resumed_parameter),
        state,
        _config("factored"),
        model=SimpleNamespace(part=torch.nn.Linear(1, 1)),
    )

    assert resumed.param_groups[0]["state_precision"] == "auto"
    assert resumed.state[resumed_parameter]["factored"] is False
    assert "structural optimizer-state conversion" in capsys.readouterr().out
