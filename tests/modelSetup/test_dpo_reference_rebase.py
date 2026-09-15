import os
import shutil
from types import SimpleNamespace

import modules.modelSetup.BaseModelSetup as base_model_setup_module
from modules.modelSetup.BaseModelSetup import BaseModelSetup, _dpo_reference_matches, _pin_if_cuda
from modules.module.LoRAModule import OFTModule
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.TrainingMethod import TrainingMethod

import torch
from torch import nn

import pytest

# CPU-only suite. What it cannot reach, because no CUDA tensor can exist here:
# the real page-locking in _pin_if_cuda, the asynchronous host-to-device upload
# and its ordering against the reference forward, the blocking device-to-host
# copy_ of a rebase, and the VRAM actually freed between reference passes.
# Meta-device params stand in for "the param lives off the host" wherever the
# code only needs a non-CPU device; scripts/util/_perf_verify_dpo_reference.py
# measures the CUDA side on the GPU.

_FILENAME = BaseModelSetup._DPO_REFERENCE_FILENAME


def _fake_setup(**overrides):
    fields = {"_dpo_ref_params": None, "_DPO_REFERENCE_FILENAME": _FILENAME}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _setup_and_model():
    adapter = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        adapter.weight.fill_(1.0)
    model = SimpleNamespace(adapters=lambda: [adapter])
    return _fake_setup(), model, adapter


def _meta_setup_and_model():
    # The param lives off the host, so a reference pass has to upload the host
    # snapshot instead of pointing the param straight at it.
    adapter = nn.Linear(4, 4, bias=False, device="meta")
    model = SimpleNamespace(adapters=lambda: [adapter])
    return _fake_setup(), model, adapter


def _existing_adapter_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.training_method = TrainingMethod.LORA
    config.rlhf_enabled = True
    config.rlhf_dpo_ref_mode = DPORefMode.EXISTING_ADAPTER
    return config


def _saved_path(backup_dir) -> str:
    return os.path.join(str(backup_dir), _FILENAME)


def _saved_weight(backup_dir) -> torch.Tensor:
    return torch.load(_saved_path(backup_dir), map_location="cpu", weights_only=True)[0][0]


def _backup_dirs(tmp_path, *names):
    dirs = []
    for name in names:
        path = tmp_path / name
        path.mkdir()
        dirs.append(path)
    return dirs


def _captured_setup():
    # A reference frozen at 1.0 by the lazy capture of the first DPO step.
    setup, model, adapter = _setup_and_model()
    config = _existing_adapter_config()
    with BaseModelSetup.reference_model(setup, model, config):
        pass
    return setup, model, adapter, config


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
    assert not hasattr(setup, "_dpo_ref_version")


def test_rebase_reuses_the_existing_snapshot_tensors():
    setup, model, adapter = _setup_and_model()
    config = _existing_adapter_config()
    BaseModelSetup.rebase_dpo_reference(setup, model, config)
    first = setup._dpo_ref_params[0][0]
    version = setup._dpo_ref_version

    with torch.no_grad():
        adapter.weight.fill_(7.0)
    BaseModelSetup.rebase_dpo_reference(setup, model, config)

    # Same host storage, refreshed contents. Cloning a second snapshot before
    # dropping the first holds two references at once, and the caching host
    # allocator keeps pinned blocks it has handed out.
    assert setup._dpo_ref_params[0][0] is first
    assert first.device.type == "cpu"
    assert torch.all(first == 7.0)
    # Changed contents need a fresh backup file, not a link to the old one.
    assert setup._dpo_ref_version == version + 1


def test_rebase_reclones_when_the_snapshot_does_not_match_the_live_params():
    setup, model, adapter = _setup_and_model()
    # Right shape and on the host, but in a dtype that is not the configured
    # forward-pass dtype, so it cannot be written into in place.
    stale = torch.zeros(4, 4, dtype=torch.float64)
    setup._dpo_ref_params = [[stale]]

    config = _existing_adapter_config()
    BaseModelSetup.rebase_dpo_reference(setup, model, config)

    assert setup._dpo_ref_params[0][0] is not stale
    assert setup._dpo_ref_params[0][0].device.type == "cpu"
    assert setup._dpo_ref_params[0][0].dtype == config.train_dtype.torch_dtype()
    assert torch.all(setup._dpo_ref_params[0][0] == 1.0)


def test_rebase_reclones_when_the_adapter_gained_a_parameter():
    setup, model, _ = _setup_and_model()
    BaseModelSetup.rebase_dpo_reference(setup, model, _existing_adapter_config())
    setup._dpo_ref_params = [[]]

    BaseModelSetup.rebase_dpo_reference(setup, model, _existing_adapter_config())

    assert len(setup._dpo_ref_params[0]) == 1


def _fake_param(device: str, shape=(4, 4), dtype=torch.float32):
    return SimpleNamespace(data=SimpleNamespace(shape=torch.Size(shape), device=torch.device(device), dtype=dtype))


def _fake_ref(device: str, shape=(4, 4), dtype=torch.float16):
    return SimpleNamespace(shape=torch.Size(shape), device=torch.device(device), dtype=dtype)


def test_a_host_snapshot_matches_a_param_on_another_device():
    # The snapshot stays in host RAM while the adapter trains on the GPU, and
    # that must still count as lining up, or every rebase would re-clone.
    config = _existing_adapter_config()
    cuda_param = _fake_param("cuda:0")

    assert _dpo_reference_matches([[_fake_ref("cpu")]], [[cuda_param]], config)
    assert _dpo_reference_matches([[_fake_ref("cuda:0")]], [[cuda_param]], config)
    assert _dpo_reference_matches([[_fake_ref("cpu")]], [[_fake_param("cpu")]], config)

    # Shape and forward-pass dtype are still checked across devices.
    assert not _dpo_reference_matches([[_fake_ref("cpu", shape=(4, 5))]], [[cuda_param]], config)
    assert not _dpo_reference_matches([[_fake_ref("cpu", dtype=torch.float32)]], [[cuda_param]], config)
    # A copy parked on some other accelerator is not a host snapshot.
    assert not _dpo_reference_matches([[_fake_ref("meta")]], [[_fake_param("cpu")]], config)
    assert not _dpo_reference_matches([[_fake_ref("cuda:1")]], [[cuda_param]], config)


def test_reference_snapshot_uses_train_dtype_without_changing_policy_dtype():
    setup, model, adapter = _setup_and_model()
    config = _existing_adapter_config()
    config.train_dtype = DataType.BFLOAT_16

    with BaseModelSetup.reference_model(setup, model, config):
        assert adapter.weight.dtype == torch.bfloat16

    assert setup._dpo_ref_params[0][0].dtype == torch.bfloat16
    assert setup._dpo_ref_params[0][0].device.type == "cpu"
    assert adapter.weight.dtype == torch.float32
    assert setup._dpo_ref_version == 1


def test_reference_pass_uses_a_same_device_snapshot_without_copying():
    setup, model, adapter, config = _captured_setup()
    snapshot = setup._dpo_ref_params[0][0]

    with BaseModelSetup.reference_model(setup, model, config):
        # param.data hands out a fresh alias each access, so compare storage.
        assert adapter.weight.data_ptr() == snapshot.data_ptr()
    assert setup._dpo_ref_params[0][0] is snapshot


def test_reference_pass_uploads_a_temporary_and_keeps_the_host_snapshot():
    setup, model, adapter = _meta_setup_and_model()
    config = _existing_adapter_config()
    config.train_dtype = DataType.BFLOAT_16
    snapshot = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
    setup._dpo_ref_params = [[snapshot]]
    policy_grad = torch.zeros(4, 4, device="meta")
    adapter.weight.grad = policy_grad

    for _ in range(3):
        with BaseModelSetup.reference_model(setup, model, config):
            assert adapter.weight.grad is None
            assert adapter.weight.data.device.type == "meta"
            assert adapter.weight.dtype == torch.bfloat16
        # The upload is never written back: the snapshot object stays the host tensor.
        assert setup._dpo_ref_params[0][0] is snapshot
        assert snapshot.device.type == "cpu"
        assert torch.all(snapshot == 2.0)
        assert adapter.weight.device.type == "meta"
        assert adapter.weight.dtype == torch.float32
        assert adapter.weight.grad is policy_grad


@pytest.mark.parametrize("make", [_setup_and_model, _meta_setup_and_model])
def test_reference_pass_restores_the_policy_after_an_exception(make):
    setup, model, adapter = make()
    config = _existing_adapter_config()
    device = adapter.weight.device
    snapshot = torch.full((4, 4), 2.0, dtype=config.train_dtype.torch_dtype())
    setup._dpo_ref_params = [[snapshot]]
    policy_ptr = adapter.weight.data_ptr()
    policy_grad = torch.zeros(4, 4, device=device)
    adapter.weight.grad = policy_grad

    with pytest.raises(RuntimeError, match="boom"), BaseModelSetup.reference_model(setup, model, config):
        assert adapter.weight.dtype == snapshot.dtype
        raise RuntimeError("boom")

    assert adapter.weight.device == device
    assert adapter.weight.data_ptr() == policy_ptr
    assert adapter.weight.grad is policy_grad
    assert adapter.weight.dtype == torch.float32
    assert setup._dpo_ref_params[0][0] is snapshot


def test_pin_if_cuda_leaves_tensors_of_non_cuda_params_alone(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("pin_memory must not run without a CUDA param")

    monkeypatch.setattr(torch.Tensor, "pin_memory", refuse)
    monkeypatch.setattr(torch.Tensor, "is_pinned", refuse)
    tensor = torch.ones(3)

    assert _pin_if_cuda(tensor, torch.zeros(3)) is tensor
    assert _pin_if_cuda(tensor, torch.zeros(3, device="meta")) is tensor


def _fake_host_tensor(pinned=False, pin_result=None, pin_error=None):
    def pin_memory():
        if pin_error is not None:
            raise pin_error
        return pin_result

    return SimpleNamespace(device=torch.device("cpu"), is_pinned=lambda: pinned, pin_memory=pin_memory)


def test_pin_if_cuda_pins_host_tensors_of_cuda_params_once():
    cuda_like = SimpleNamespace(device=torch.device("cuda", 0))
    pinned = object()

    assert _pin_if_cuda(_fake_host_tensor(pin_result=pinned), cuda_like) is pinned
    already = _fake_host_tensor(pinned=True, pin_error=AssertionError("re-pinned"))
    assert _pin_if_cuda(already, cuda_like) is already
    on_device = SimpleNamespace(device=torch.device("cuda", 0))
    assert _pin_if_cuda(on_device, cuda_like) is on_device


def test_pin_if_cuda_falls_back_to_pageable_memory_when_pinning_fails():
    cuda_like = SimpleNamespace(device=torch.device("cuda", 0))
    tensor = _fake_host_tensor(pin_error=RuntimeError("CUDA error: out of memory"))

    with pytest.warns(UserWarning, match="pageable"):
        assert _pin_if_cuda(tensor, cuda_like) is tensor


def test_restored_reference_is_pinned_once_on_the_first_off_host_pass(monkeypatch, tmp_path):
    setup, model, adapter = _meta_setup_and_model()
    config = _existing_adapter_config()
    calls = []

    def fake_pin(tensor, like):
        # Stands in for pinning a tensor of a CUDA param: a new host tensor.
        calls.append(like.device.type)
        return tensor.clone()

    monkeypatch.setattr(base_model_setup_module, "_pin_if_cuda", fake_pin)
    torch.save([[torch.full((4, 4), 2.0, dtype=torch.float16)]], _saved_path(tmp_path))

    assert BaseModelSetup.load_dpo_reference(setup, str(tmp_path), model, config)
    assert calls == ["meta"]
    calls.clear()

    with BaseModelSetup.reference_model(setup, model, config):
        pass
    assert calls == ["meta"]
    pinned = setup._dpo_ref_params[0][0]
    assert pinned.device.type == "cpu"

    for _ in range(3):
        with BaseModelSetup.reference_model(setup, model, config):
            pass
    assert calls == ["meta"]
    assert setup._dpo_ref_params[0][0] is pinned

    # A new snapshot version gets its own single pinning pass.
    assert BaseModelSetup.load_dpo_reference(setup, str(tmp_path), model, config)
    calls.clear()
    for _ in range(2):
        with BaseModelSetup.reference_model(setup, model, config):
            pass
    assert calls == ["meta"]


def test_loading_fp32_reference_converts_it_to_train_dtype(tmp_path):
    setup, model, adapter = _setup_and_model()
    config = _existing_adapter_config()
    config.train_dtype = DataType.BFLOAT_16
    torch.save([[adapter.weight.detach().clone()]], tmp_path / "dpo_reference.pt")

    assert BaseModelSetup.load_dpo_reference(setup, str(tmp_path), model, config)
    assert setup._dpo_ref_params[0][0].dtype == torch.bfloat16
    assert setup._dpo_ref_params[0][0].device.type == "cpu"
    # The file does not hold the cast tensors, so it must not be linked.
    assert setup._dpo_ref_saved is None

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


def test_real_setup_starts_without_a_saved_reference():
    concrete = type("ConcreteSetup", (BaseModelSetup,), {})
    concrete.__abstractmethods__ = frozenset()
    setup = concrete(torch.device("cpu"), torch.device("cpu"), False)

    assert setup._dpo_ref_params is None
    assert setup._dpo_ref_version == 0
    assert setup._dpo_ref_saved is None
    assert setup._dpo_ref_pinned_version is None


def test_save_without_a_snapshot_writes_nothing(tmp_path):
    setup, _, _ = _setup_and_model()
    BaseModelSetup.save_dpo_reference(setup, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_unchanged_reference_is_hard_linked_into_the_next_backup(tmp_path):
    setup, _, _, _ = _captured_setup()
    first, second = _backup_dirs(tmp_path, "b1", "b2")

    BaseModelSetup.save_dpo_reference(setup, str(first))
    BaseModelSetup.save_dpo_reference(setup, str(second))

    assert os.path.samefile(_saved_path(first), _saved_path(second))
    assert os.stat(_saved_path(second)).st_nlink >= 2
    assert torch.all(_saved_weight(second) == 1.0)
    assert setup._dpo_ref_saved == (setup._dpo_ref_version, _saved_path(second))


def test_saving_the_same_backup_twice_keeps_one_file(tmp_path):
    setup, _, _, _ = _captured_setup()
    BaseModelSetup.save_dpo_reference(setup, str(tmp_path))
    BaseModelSetup.save_dpo_reference(setup, str(tmp_path))

    assert sorted(p.name for p in tmp_path.iterdir()) == [_FILENAME]
    assert torch.all(_saved_weight(tmp_path) == 1.0)


def test_rebased_reference_gets_its_own_file_and_older_backups_stay_intact(tmp_path):
    setup, model, adapter, config = _captured_setup()
    first, second, third = _backup_dirs(tmp_path, "b1", "b2", "b3")
    BaseModelSetup.save_dpo_reference(setup, str(first))
    BaseModelSetup.save_dpo_reference(setup, str(second))

    with torch.no_grad():
        adapter.weight.fill_(4.0)
    BaseModelSetup.rebase_dpo_reference(setup, model, config)
    BaseModelSetup.save_dpo_reference(setup, str(third))

    assert not os.path.samefile(_saved_path(second), _saved_path(third))
    assert os.stat(_saved_path(third)).st_nlink == 1
    assert torch.all(_saved_weight(third) == 4.0)
    assert torch.all(_saved_weight(first) == 1.0)
    assert torch.all(_saved_weight(second) == 1.0)
    assert not any(name.endswith(".tmp") for d in (first, second, third) for name in os.listdir(d))


def test_pruning_the_older_backup_keeps_the_linked_file_loadable(tmp_path):
    setup, _, _, _ = _captured_setup()
    first, second = _backup_dirs(tmp_path, "b1", "b2")
    BaseModelSetup.save_dpo_reference(setup, str(first))
    BaseModelSetup.save_dpo_reference(setup, str(second))

    shutil.rmtree(first)

    assert os.stat(_saved_path(second)).st_nlink == 1
    assert torch.all(_saved_weight(second) == 1.0)

    # With the recorded link source gone, the next backup is written for real.
    third = tmp_path / "b3"
    third.mkdir()
    BaseModelSetup.save_dpo_reference(setup, str(third))
    assert torch.all(_saved_weight(third) == 1.0)


def test_resume_round_trips_and_the_next_backup_links_to_the_restored_file(tmp_path):
    setup, model, adapter, config = _captured_setup()
    first, second = _backup_dirs(tmp_path, "b1", "b2")
    with torch.no_grad():
        adapter.weight.fill_(6.0)
    BaseModelSetup.rebase_dpo_reference(setup, model, config)
    BaseModelSetup.save_dpo_reference(setup, str(first))

    resumed = _fake_setup()
    with torch.no_grad():
        adapter.weight.fill_(8.0)
    assert BaseModelSetup.load_dpo_reference(resumed, str(first), model, config)
    assert torch.equal(resumed._dpo_ref_params[0][0], setup._dpo_ref_params[0][0])
    assert resumed._dpo_ref_saved == (resumed._dpo_ref_version, _saved_path(first))
    with BaseModelSetup.reference_model(resumed, model, config):
        assert torch.all(adapter.weight == 6.0)

    BaseModelSetup.save_dpo_reference(resumed, str(second))

    assert os.path.samefile(_saved_path(first), _saved_path(second))
    assert torch.all(_saved_weight(second) == 6.0)


def test_failed_hard_link_falls_back_to_a_real_write(monkeypatch, tmp_path):
    setup, _, _, _ = _captured_setup()
    first, second = _backup_dirs(tmp_path, "b1", "b2")
    BaseModelSetup.save_dpo_reference(setup, str(first))

    def no_links(*_args, **_kwargs):
        raise OSError("hard links not supported")

    monkeypatch.setattr(os, "link", no_links)
    BaseModelSetup.save_dpo_reference(setup, str(second))

    assert not os.path.samefile(_saved_path(first), _saved_path(second))
    assert torch.all(_saved_weight(second) == 1.0)
    assert setup._dpo_ref_saved == (setup._dpo_ref_version, _saved_path(second))


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("PytorchStreamWriter failed writing file")])
def test_interrupted_write_never_leaves_a_truncated_reference(monkeypatch, tmp_path, error):
    setup, model, adapter, config = _captured_setup()
    first, second, third = _backup_dirs(tmp_path, "b1", "b2", "b3")
    BaseModelSetup.save_dpo_reference(setup, str(first))
    BaseModelSetup.save_dpo_reference(setup, str(second))
    saved_before = setup._dpo_ref_saved

    with torch.no_grad():
        adapter.weight.fill_(5.0)
    BaseModelSetup.rebase_dpo_reference(setup, model, config)

    real_save = torch.save

    def interrupted_save(obj, path, *args, **kwargs):
        with open(path, "wb") as f:
            f.write(b"PK\x03\x04 truncated")
        raise error

    monkeypatch.setattr(torch, "save", interrupted_save)

    # OSError is logged and swallowed as before; anything else reaches the
    # trainer's backup handler, which deletes the partial backup.
    for backup in (third, second):
        if isinstance(error, OSError):
            BaseModelSetup.save_dpo_reference(setup, str(backup))
        else:
            with pytest.raises(RuntimeError):
                BaseModelSetup.save_dpo_reference(setup, str(backup))

    assert os.listdir(third) == []
    # The second backup still holds the intact link to the first backup's file.
    assert os.listdir(second) == [_FILENAME]
    assert os.path.samefile(_saved_path(first), _saved_path(second))
    monkeypatch.setattr(torch, "save", real_save)
    assert torch.all(_saved_weight(first) == 1.0)
    assert torch.all(_saved_weight(second) == 1.0)
    assert setup._dpo_ref_saved == saved_before

    BaseModelSetup.save_dpo_reference(setup, str(third))
    assert torch.all(_saved_weight(third) == 5.0)
    assert torch.all(_saved_weight(second) == 1.0)
