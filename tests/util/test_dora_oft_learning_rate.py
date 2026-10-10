"""A separate learning rate for the DoRA magnitudes of a DoRA-OFT adapter
(config dora_oft_learning_rate): the optimizer group split, the parameter keys
that make the split invisible to saved state, and resuming a backup across it."""

from types import SimpleNamespace

# Plain-import form so isort keeps it FIRST: primes the factory import graph
# (create -> factory.import_dir) before any direct modelSetup import below.
import modules.util.create  # noqa: F401 -- import-side-effect required
from modules.modelSaver.mixin.InternalModelSaverMixin import InternalModelSaverMixin
from modules.modelSetup.ZImageLoRASetup import ZImageLoRASetup
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util import create
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType, PeftType
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection
from modules.util.optimizer.muon_util import split_parameters_for_muon
from modules.util.TrainProgress import TrainProgress

import torch
from torch import nn

HIDDEN = 8
LAYERS = 3
OFT_LR = 1e-3
DORA_LR = 1e-5


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(LAYERS)])


def _config(*, dora: bool = True, dora_lr: float | None = None) -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.model_type = ModelType.Z_IMAGE
    config.peft_type = PeftType.OFT_2
    config.dora_oft = dora
    config.dora_oft_learning_rate = dora_lr
    config.oft_block_size = 4
    config.learning_rate = OFT_LR
    config.transformer.train = True
    config.transformer.learning_rate = None
    config.optimizer.optimizer = Optimizer.ADAMW
    config.optimizer.weight_decay = 0.0
    config.optimizer.fused = False
    config.optimizer.foreach = False
    config.optimizer.stochastic_rounding = False
    config.optimizer.fused_back_pass = False
    return config


def _adapter(config: TrainConfig) -> LoRAModuleWrapper:
    return LoRAModuleWrapper(_Stack(), "transformer", config, [])


def _collection(adapter: LoRAModuleWrapper, config: TrainConfig) -> NamedParameterGroupCollection:
    """The real ZImageLoRASetup.create_parameters, with the adapter built for another
    DoRA learning rate re-pointed at this config's."""
    adapter.dora_learning_rate = config.dora_oft_learning_rate if config.dora_oft else None
    setup = ZImageLoRASetup(torch.device("cpu"), torch.device("cpu"), False)
    return setup.create_parameters(SimpleNamespace(transformer_lora=adapter, transformer_fake_lora=None), config)


def _multipliers(adapter: LoRAModuleWrapper) -> list[nn.Parameter]:
    return [module.dora_multiplier for module in adapter.lora_modules.values()]


def _rotations(adapter: LoRAModuleWrapper) -> list[nn.Parameter]:
    return [module.oft_R.weight for module in adapter.lora_modules.values()]


def _ids(parameters) -> list[int]:
    return [id(p) for p in parameters]


# ------------------------------------------------------------------
# the group split
# ------------------------------------------------------------------


def test_without_a_dora_learning_rate_the_adapter_stays_one_group():
    config = _config()
    adapter = _adapter(config)
    assert adapter.dora_learning_rate is None

    collection = _collection(adapter, config)

    assert collection.unique_name_mapping == ["transformer"]
    assert _ids(collection.parameters()) == _ids(adapter.parameters())
    assert collection.key_mapping == [("transformer", list(range(2 * LAYERS)))]


def test_a_dora_learning_rate_moves_the_magnitudes_into_their_own_group():
    config = _config(dora_lr=DORA_LR)
    adapter = _adapter(config)
    assert adapter.dora_learning_rate == DORA_LR

    groups = _collection(adapter, config).parameters_for_optimizer(config)

    assert [group["name"] for group in groups] == ["transformer", "transformer_dora"]
    assert [group["lr"] for group in groups] == [OFT_LR, DORA_LR]
    assert [group["initial_lr"] for group in groups] == [OFT_LR, DORA_LR]
    assert _ids(groups[0]["params"]) == _ids(_rotations(adapter))
    assert _ids(groups[1]["params"]) == _ids(_multipliers(adapter))


def test_the_rotations_keep_the_model_part_learning_rate():
    config = _config(dora_lr=DORA_LR)
    config.transformer.learning_rate = 5e-4

    groups = _collection(_adapter(config), config).parameters_for_optimizer(config)

    assert [group["lr"] for group in groups] == [5e-4, DORA_LR]


def test_a_dora_learning_rate_of_zero_still_splits():
    config = _config(dora_lr=0.0)

    groups = _collection(_adapter(config), config).parameters_for_optimizer(config)

    assert [(group["name"], group["lr"]) for group in groups] == [("transformer", OFT_LR), ("transformer_dora", 0.0)]


def test_a_dora_learning_rate_is_ignored_without_dora_oft():
    config = _config(dora=False, dora_lr=DORA_LR)
    adapter = _adapter(config)
    assert adapter.dora_learning_rate is None
    assert adapter.dora_multiplier_parameters() == []

    assert _collection(adapter, config).unique_name_mapping == ["transformer"]


def test_the_split_changes_neither_parameter_order_nor_keys():
    adapter = _adapter(_config())
    unsplit = _collection(adapter, _config())
    split = _collection(adapter, _config(dora_lr=DORA_LR))

    assert split.unique_name_mapping == ["transformer", "transformer_dora"]
    assert _ids(split.parameters()) == _ids(unsplit.parameters())
    assert [(key, id(p)) for key, p in split.iter_named_parameters()] == [
        (key, id(p)) for key, p in unsplit.iter_named_parameters()
    ]
    # both halves live in the unsplit group's key space, at their unsplit positions
    (rotation_space, rotation_keys), (dora_space, dora_keys) = split.key_mapping
    assert rotation_space == dora_space == "transformer"
    assert sorted(rotation_keys + dora_keys) == list(range(2 * LAYERS))
    unsplit_ids = _ids(unsplit.parameters())
    assert [unsplit_ids[i] for i in dora_keys] == _ids(_multipliers(adapter))


def test_split_returns_none_when_nothing_moves():
    parameters = [nn.Parameter(torch.zeros(1)) for _ in range(2)]
    group = NamedParameterGroup("part", parameters, 1e-3)

    assert group.split("part_other", [nn.Parameter(torch.zeros(1))], 1e-4) is None
    assert _ids(group.parameters) == _ids(parameters)
    assert group.key_indices == [0, 1]


def test_muon_aux_adam_keeps_the_dora_learning_rate():
    config = _config(dora_lr=DORA_LR)
    config.optimizer.MuonWithAuxAdam = True
    config.optimizer.muon_adam_lr = 3e-4
    config.optimizer.muon_hidden_layers = None
    adapter = _adapter(config)
    model = SimpleNamespace(model_type=ModelType.Z_IMAGE, transformer_lora=adapter)

    def _learning_rates(collection: NamedParameterGroupCollection) -> dict[tuple[str, str], float]:
        groups, _ = split_parameters_for_muon(model, collection.parameters_for_optimizer(config), config)
        return {(group["name"], group["optim_type"]): group["lr"] for group in groups}

    # 1-D magnitudes fall to the auxiliary Adam; with their own learning rate they keep it
    assert _learning_rates(_collection(adapter, config)) == {
        ("transformer", "muon"): OFT_LR,
        ("transformer_dora", "adam"): DORA_LR,
    }
    # without one, the Adam learning rate applies as before
    config.dora_oft_learning_rate = None
    assert _learning_rates(_collection(adapter, config)) == {
        ("transformer", "muon"): OFT_LR,
        ("transformer", "adam"): 3e-4,
    }


# ------------------------------------------------------------------
# resuming across the split
# ------------------------------------------------------------------


def _optimizer(collection: NamedParameterGroupCollection, state: dict | None, config: TrainConfig):
    return create.create_optimizer(collection, state, config, model=SimpleNamespace())


def _trained_state(collection: NamedParameterGroupCollection, config: TrainConfig, *, legacy: bool = False):
    """Optimizer state after one step with a distinct gradient per parameter, as a backup
    stores it, plus each parameter's momentum. ``legacy``: a backup from before the keys."""
    optimizer = _optimizer(collection, None, config)
    for i, p in enumerate(collection.parameters()):
        p.grad = torch.full_like(p, float(i + 1))
    optimizer.step()

    state = optimizer.state_dict()
    state["param_group_mapping"] = collection.unique_name_mapping
    state["param_group_optimizer_mapping"] = [str(config.optimizer.optimizer) for _ in collection.unique_name_mapping]
    if not legacy:
        state["param_group_keys"] = collection.key_mapping
    momentum = {id(p): optimizer.state[p]["exp_avg"].clone() for p in collection.parameters()}
    assert len({float(m.flatten()[0]) for m in momentum.values()}) == len(momentum)  # all distinct
    return state, momentum


def _assert_momentum_restored(optimizer, collection: NamedParameterGroupCollection, momentum: dict):
    for p in collection.parameters():
        torch.testing.assert_close(optimizer.state[p]["exp_avg"], momentum[id(p)], rtol=0, atol=0)


def test_adding_a_dora_learning_rate_on_resume_keeps_the_optimizer_state(capsys):
    adapter = _adapter(_config())
    state, momentum = _trained_state(_collection(adapter, _config()), _config(), legacy=True)

    config = _config(dora_lr=DORA_LR)
    collection = _collection(adapter, config)
    resumed = _optimizer(collection, state, config)

    _assert_momentum_restored(resumed, collection, momentum)
    assert [(group["name"], group["lr"]) for group in resumed.param_groups] == [
        ("transformer", OFT_LR),
        ("transformer_dora", DORA_LR),
    ]
    assert _ids(resumed.param_groups[1]["params"]) == _ids(_multipliers(adapter))
    assert "WARN" not in capsys.readouterr().out


def test_removing_the_dora_learning_rate_on_resume_keeps_the_optimizer_state(capsys):
    adapter = _adapter(_config())
    split_config = _config(dora_lr=DORA_LR)
    state, momentum = _trained_state(_collection(adapter, split_config), split_config)

    config = _config()
    collection = _collection(adapter, config)
    resumed = _optimizer(collection, state, config)

    _assert_momentum_restored(resumed, collection, momentum)
    assert [(group["name"], group["lr"]) for group in resumed.param_groups] == [("transformer", OFT_LR)]
    assert "WARN" not in capsys.readouterr().out


def test_changing_the_dora_learning_rate_on_resume_keeps_the_optimizer_state():
    adapter = _adapter(_config())
    split_config = _config(dora_lr=DORA_LR)
    state, momentum = _trained_state(_collection(adapter, split_config), split_config)

    config = _config(dora_lr=2e-6)
    collection = _collection(adapter, config)
    resumed = _optimizer(collection, state, config)

    _assert_momentum_restored(resumed, collection, momentum)
    assert [group["lr"] for group in resumed.param_groups] == [OFT_LR, 2e-6]


def test_a_backup_round_trip_carries_the_keys(tmp_path):
    adapter = _adapter(_config())
    split_config = _config(dora_lr=DORA_LR)
    collection = _collection(adapter, split_config)
    optimizer = _optimizer(collection, None, split_config)
    for i, p in enumerate(collection.parameters()):
        p.grad = torch.full_like(p, float(i + 1))
    optimizer.step()
    momentum = {id(p): optimizer.state[p]["exp_avg"].clone() for p in collection.parameters()}

    model = SimpleNamespace(
        optimizer=optimizer,
        parameters=collection,
        param_group_mapping=collection.unique_name_mapping,
        train_config=split_config,
        ema=None,
        train_progress=TrainProgress(),
        accumulator_state=None,
    )
    InternalModelSaverMixin()._save_internal_data(model, str(tmp_path))
    state = torch.load(tmp_path / "optimizer" / "optimizer.pt", weights_only=False)
    assert state["param_group_keys"] == collection.key_mapping

    config = _config()
    unsplit = _collection(adapter, config)
    _assert_momentum_restored(_optimizer(unsplit, state, config), unsplit, momentum)


def test_a_changed_adapter_structure_still_resets_the_optimizer_state(capsys):
    plain_config = _config(dora=False)
    state, _ = _trained_state(_collection(_adapter(plain_config), plain_config), plain_config)

    # dora_oft switched on since the backup: 3 saved rotations, 6 parameters now
    config = _config(dora_lr=DORA_LR)
    collection = _collection(_adapter(config), config)
    resumed = _optimizer(collection, state, config)

    assert len(resumed.state) == 0
    assert [(group["name"], group["lr"]) for group in resumed.param_groups] == [
        ("transformer", OFT_LR),
        ("transformer_dora", DORA_LR),
    ]
    out = capsys.readouterr().out
    assert "optimizer group 'transformer' changed parameter count (3 saved vs 6 current)" in out
