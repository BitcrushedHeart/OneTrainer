from types import SimpleNamespace

from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.module.LoRAModule import OFTModule
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
from torch import nn


def _setup_and_model():
    adapter = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        adapter.weight.fill_(1.0)
    model = SimpleNamespace(adapters=lambda: [adapter])
    setup = SimpleNamespace(_dpo_ref_params=None)
    return setup, model, adapter


def _existing_adapter_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
    return config


def test_rebase_replaces_the_frozen_reference_with_the_current_adapter():
    setup, model, adapter = _setup_and_model()
    config = _existing_adapter_config()

    # First DPO step: the lazy capture freezes the adapter at 1.0.
    with BaseModelSetup.reference_model(setup, model, config):
        pass
    with torch.no_grad():
        adapter.weight.fill_(3.0)
    with BaseModelSetup.reference_model(setup, model, config):
        assert torch.all(adapter.weight == 1.0)
    assert torch.all(adapter.weight == 3.0)

    # Rebase: the reference is now the drifted adapter, so the reference view
    # matches the policy and every implicit reward margin returns to zero.
    BaseModelSetup.rebase_dpo_reference(setup, model, config)
    with torch.no_grad():
        adapter.weight.fill_(5.0)
    with BaseModelSetup.reference_model(setup, model, config):
        assert torch.all(adapter.weight == 3.0)
    assert torch.all(adapter.weight == 5.0)


def test_rebase_takes_a_copy_not_a_view():
    setup, model, adapter = _setup_and_model()
    BaseModelSetup.rebase_dpo_reference(setup, model, _existing_adapter_config())
    with torch.no_grad():
        adapter.weight.fill_(9.0)
    assert torch.all(setup._dpo_ref_params[0][0] == 1.0)


def test_rebase_without_adapters_is_a_no_op():
    setup = SimpleNamespace(_dpo_ref_params="untouched")
    BaseModelSetup.rebase_dpo_reference(setup, SimpleNamespace(adapters=list), _existing_adapter_config())
    assert setup._dpo_ref_params == "untouched"


def test_rebase_reuses_the_existing_snapshot_tensors():
    setup, model, adapter = _setup_and_model()
    config = _existing_adapter_config()
    BaseModelSetup.rebase_dpo_reference(setup, model, config)
    first = setup._dpo_ref_params[0][0]

    with torch.no_grad():
        adapter.weight.fill_(7.0)
    BaseModelSetup.rebase_dpo_reference(setup, model, config)

    # Same storage, refreshed contents. Cloning a second snapshot before
    # dropping the first peaks at twice the reference, and the caching
    # allocator holds that extra block reserved for the rest of the run.
    assert setup._dpo_ref_params[0][0] is first
    assert torch.all(first == 7.0)


def test_rebase_reclones_when_the_snapshot_does_not_match_the_live_params():
    setup, model, adapter = _setup_and_model()
    # Stands in for a reference restored from a backup: right shape, still on
    # the CPU in its saved dtype, so it cannot be written into in place.
    stale = torch.zeros(4, 4, dtype=torch.float64)
    setup._dpo_ref_params = [[stale]]

    config = _existing_adapter_config()
    BaseModelSetup.rebase_dpo_reference(setup, model, config)

    assert setup._dpo_ref_params[0][0] is not stale
    assert setup._dpo_ref_params[0][0].dtype == config.train_dtype.torch_dtype()
    assert torch.all(setup._dpo_ref_params[0][0] == 1.0)


def test_rebase_reclones_when_the_adapter_gained_a_parameter():
    setup, model, _ = _setup_and_model()
    BaseModelSetup.rebase_dpo_reference(setup, model, _existing_adapter_config())
    setup._dpo_ref_params = [[]]

    BaseModelSetup.rebase_dpo_reference(setup, model, _existing_adapter_config())

    assert len(setup._dpo_ref_params[0]) == 1


def test_reference_snapshot_uses_train_dtype_without_changing_policy_dtype():
    setup, model, adapter = _setup_and_model()
    config = _existing_adapter_config()
    config.train_dtype = DataType.BFLOAT_16

    with BaseModelSetup.reference_model(setup, model, config):
        assert adapter.weight.dtype == torch.bfloat16

    assert setup._dpo_ref_params[0][0].dtype == torch.bfloat16
    assert adapter.weight.dtype == torch.float32


def test_loading_fp32_reference_converts_it_to_train_dtype(tmp_path):
    setup, model, adapter = _setup_and_model()
    setup._DPO_REFERENCE_FILENAME = BaseModelSetup._DPO_REFERENCE_FILENAME
    config = _existing_adapter_config()
    config.train_dtype = DataType.BFLOAT_16
    torch.save([[adapter.weight.detach().clone()]], tmp_path / "dpo_reference.pt")

    assert BaseModelSetup.load_dpo_reference(setup, str(tmp_path), model, config)
    assert setup._dpo_ref_params[0][0].dtype == torch.bfloat16

    BaseModelSetup.save_dpo_reference(setup, str(tmp_path))
    saved = torch.load(tmp_path / "dpo_reference.pt", map_location="cpu", weights_only=True)
    assert saved[0][0].dtype == torch.bfloat16


def test_bf16_reference_runs_an_oft_forward_under_autocast():
    linear = nn.Linear(8, 8, bias=False)
    adapter = OFTModule(
        "test",
        linear,
        oft_block_size=4,
        block_share=False,
        oft_scaled=False,
        oft_clipped_norm=None,
    )
    adapter.hook_to_module()
    model = SimpleNamespace(adapters=lambda: [adapter])
    setup = SimpleNamespace(_dpo_ref_params=None)
    config = _existing_adapter_config()
    config.train_dtype = DataType.BFLOAT_16
    policy_grad = torch.ones_like(adapter.oft_R.weight)
    adapter.oft_R.weight.grad = policy_grad

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16), BaseModelSetup.reference_model(setup, model, config):
        assert adapter.oft_R.weight.grad is None
        output = linear(torch.randn(2, 8))

    assert output.dtype == torch.bfloat16
    assert adapter.oft_R.weight.dtype == torch.float32
    assert adapter.oft_R.weight.grad is policy_grad
