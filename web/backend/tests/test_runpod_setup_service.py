import json
from pathlib import Path

from modules.cloud.RunpodCloud import RUNPOD_DATA_CENTERS, RunpodCloud
from modules.trainer.CloudTrainer import CloudTrainer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType
from web.backend.services.runpod_setup_service import RunpodSetupService


def test_storage_rounding():
    # rounds up to the next ROUND_GB (5)
    assert RunpodSetupService._round_storage_gb(284.61) == 285
    assert RunpodSetupService._round_storage_gb(300.01) == 305


def test_latest_backup_uses_sorted_backup_name(tmp_path):
    cfg = TrainConfig.default_values()
    cfg.workspace_dir = str(tmp_path / "workspace")
    older = Path(cfg.workspace_dir) / "backup" / "2026-06-19_22-20-05-backup-3900-0-3900"
    newer = Path(cfg.workspace_dir) / "backup" / "2026-06-19_22-21-20-backup-3928-0-3928"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    assert RunpodSetupService._latest_backup(cfg) == str(newer)


def test_remote_path_mapping_matches_cloud_trainer():
    assert (
        CloudTrainer.adjust_path_to_remote_dir(r"F:\workspace\SoReal!\cache", "/workspace")
        == "/workspace/remote/workspace/SoReal!/cache"
    )


def _force_native_rsync(monkeypatch):
    import web.backend.services.runpod_setup_service as svc

    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/rsync" if name == "rsync" else None)


def test_mutate_config_for_runpod_resumes_when_user_opts_in_and_backup_exists(monkeypatch, tmp_path):
    _force_native_rsync(monkeypatch)
    cfg = TrainConfig.default_values()
    cfg.only_cache = True
    cfg.continue_last_backup = True  # user asked to continue
    cfg.clear_cache_before_training = True
    cfg.workspace_dir = str(tmp_path / "workspace")
    (Path(cfg.workspace_dir) / "backup" / "2026-06-19_22-21-20-backup-1-0-1").mkdir(parents=True)
    service = RunpodSetupService()

    service._mutate_config_for_runpod(cfg, 300, {"pod_name": "Blend", "run_id": "runpod-v27", "min_download": 500})

    assert cfg.cloud.enabled is True
    assert cfg.cloud.type == CloudType.RUNPOD
    assert cfg.cloud.file_sync == CloudFileSync.NATIVE_RSYNC
    assert cfg.cloud.gpu_type == "NVIDIA GeForce RTX 5090"
    assert cfg.cloud.volume_size == 300
    assert cfg.cloud.install_cmd == "git clone -b bitcrushed-blend https://github.com/BitcrushedHeart/OneTrainer.git"
    assert cfg.cloud.detach_trainer is True
    assert cfg.sourceless_training is True
    assert cfg.only_cache is False
    assert cfg.continue_last_backup is True
    assert cfg.clear_cache_before_training is False


def test_mutate_config_for_runpod_starts_fresh_without_backup(monkeypatch, tmp_path):
    _force_native_rsync(monkeypatch)
    cfg = TrainConfig.default_values()
    cfg.continue_last_backup = True  # asked to continue, but...
    cfg.workspace_dir = str(tmp_path / "workspace")  # ...no backup subdir exists
    service = RunpodSetupService()

    service._mutate_config_for_runpod(cfg, 200, {})

    assert cfg.sourceless_training is True
    assert cfg.latent_caching is True
    assert cfg.continue_last_backup is False


def test_mutate_config_respects_continue_last_backup_opt_out(monkeypatch, tmp_path):
    # user did NOT choose to continue: even though a backup exists, do not resume or upload it
    _force_native_rsync(monkeypatch)
    cfg = TrainConfig.default_values()
    cfg.continue_last_backup = False
    cfg.workspace_dir = str(tmp_path / "workspace")
    (Path(cfg.workspace_dir) / "backup" / "2026-06-19_22-21-20-backup-1-0-1").mkdir(parents=True)
    service = RunpodSetupService()

    service._mutate_config_for_runpod(cfg, 200, {})

    assert cfg.continue_last_backup is False


def test_ssh_auth_error_requires_existing_key_file(tmp_path):
    assert RunpodSetupService._ssh_auth_error("") is not None
    assert RunpodSetupService._ssh_auth_error("   ") is not None
    assert RunpodSetupService._ssh_auth_error(str(tmp_path / "missing")) is not None
    key = tmp_path / "id_ed25519"
    key.write_text("private-key", encoding="utf-8")
    assert RunpodSetupService._ssh_auth_error(str(key)) is None


def test_select_file_sync_prefers_rsync_then_wsl_then_sftp(monkeypatch):
    import web.backend.services.runpod_setup_service as svc

    # rsync on PATH -> native rsync
    monkeypatch.setattr(svc.shutil, "which", lambda name: "/usr/bin/rsync" if name == "rsync" else None)
    assert RunpodSetupService._select_file_sync() == CloudFileSync.NATIVE_RSYNC

    # no rsync, Windows with WSL-rsync available -> WSL rsync
    monkeypatch.setattr(svc.shutil, "which", lambda name: None)
    monkeypatch.setattr(svc.os, "name", "nt")
    monkeypatch.setattr(RunpodSetupService, "_wsl_rsync_available", staticmethod(lambda: True))
    assert RunpodSetupService._select_file_sync() == CloudFileSync.WSL_RSYNC

    # no rsync, no WSL -> pure-Python SFTP fallback (never a hard failure)
    monkeypatch.setattr(RunpodSetupService, "_wsl_rsync_available", staticmethod(lambda: False))
    assert RunpodSetupService._select_file_sync() == CloudFileSync.FABRIC_SFTP


def test_size_entries_skips_backup_when_not_continuing(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.pt").write_bytes(b"x")

    cfg = TrainConfig.default_values()
    cfg.cache_dir = str(cache)
    cfg.workspace_dir = str(tmp_path / "workspace")
    (Path(cfg.workspace_dir) / "backup" / "2026-06-19_22-21-20-backup-1-0-1").mkdir(parents=True)
    cfg.continue_last_backup = False  # not continuing -> backup must not be sized/uploaded
    cfg.transformer.model_name = "SomeOrg/HF-Repo-Id"  # remote id, not a local path
    cfg.base_model_name = ""

    entries = {entry.label: entry for entry in RunpodSetupService()._size_entries(cfg)}

    assert entries["cache"].required is True and entries["cache"].exists is True
    assert "latest_backup" not in entries  # opted out of continuation
    assert "transformer_override" not in entries  # HF id / empty override is not a local upload
    assert "base_model" not in entries

    cfg.continue_last_backup = True  # opting in surfaces the backup for sizing
    entries = {entry.label: entry for entry in RunpodSetupService()._size_entries(cfg)}
    assert entries["latest_backup"].exists is True


def test_runpod_create_uses_5090_secure_europe(monkeypatch):
    calls = []

    def fake_create_pod(**kwargs):
        calls.append(kwargs)
        return {"id": "pod-123"}

    monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.create_pod", fake_create_pod)

    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "NVIDIA GeForce RTX 5090"
    cfg.cloud.sub_type = "SECURE"
    cfg.cloud.volume_size = 300
    cfg.cloud.min_download = 250
    cloud = RunpodCloud(cfg)
    cloud._create()

    assert cfg.secrets.cloud.id == "pod-123"
    assert calls[0]["template_id"] == "1a33vbssq9"
    assert calls[0]["gpu_type_id"] == "NVIDIA GeForce RTX 5090"
    assert calls[0]["cloud_type"] == "SECURE"
    assert calls[0]["support_public_ip"] is True
    assert calls[0]["start_ssh"] is True
    assert calls[0]["data_center_id"] == "EU-NL-1"  # a real RunPod DC id, not a country code
    assert calls[0]["volume_in_gb"] == 300  # the persistent volume holds the data
    assert calls[0]["container_disk_in_gb"] == 30  # small ephemeral container disk
    assert "13.0" in calls[0]["allowed_cuda_versions"]  # cu130 torch needs a CUDA 13 host
    assert not any(v.startswith("12.") for v in calls[0]["allowed_cuda_versions"])


def test_runpod_data_centers_are_certified_only():
    # ISO/IEC 27001 or SOC 2 Type II European data centers only (the latent cache decodes back to the images)
    certified = {"EU-NL-1", "EUR-IS-1", "EUR-IS-3", "EUR-IS-4", "EUR-IS-5", "EU-SE-1"}
    assert set(RUNPOD_DATA_CENTERS) <= certified
    assert not {"EU-RO-1", "EU-CZ-1", "EU-FR-1", "EUR-IS-2", "EUR-NO-1", "EUR-NO-2"} & set(RUNPOD_DATA_CENTERS)


def test_runpod_create_falls_back_between_data_centers(monkeypatch):
    calls = []

    def fake_create_pod(**kwargs):
        calls.append(kwargs)
        if kwargs["data_center_id"] == RUNPOD_DATA_CENTERS[0]:
            raise RuntimeError("no capacity")
        return {"id": "pod-456"}

    monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.create_pod", fake_create_pod)

    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "NVIDIA GeForce RTX 5090"
    cfg.cloud.sub_type = "SECURE"
    cloud = RunpodCloud(cfg)
    cloud._create()

    assert cfg.secrets.cloud.id == "pod-456"
    assert [call["data_center_id"] for call in calls[:2]] == RUNPOD_DATA_CENTERS[:2]


def test_runpod_create_tries_all_data_centers_when_capacity_returns_none(monkeypatch):
    calls = []

    def fake_create_pod(**kwargs):
        calls.append(kwargs["data_center_id"])
        # simulate no-capacity: runpod returns None instead of raising for the first two DCs
        if kwargs["data_center_id"] in RUNPOD_DATA_CENTERS[:2]:
            return None
        return {"id": "pod-789"}

    monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.create_pod", fake_create_pod)

    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "NVIDIA GeForce RTX 5090"
    cfg.cloud.sub_type = "SECURE"
    cloud = RunpodCloud(cfg)
    cloud._create()

    assert cfg.secrets.cloud.id == "pod-789"
    assert calls[:3] == RUNPOD_DATA_CENTERS[:3]


def test_runpod_create_raises_when_no_region_has_capacity(monkeypatch):
    monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.create_pod", lambda **kwargs: None)

    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "NVIDIA GeForce RTX 5090"
    cfg.cloud.sub_type = "SECURE"
    cloud = RunpodCloud(cfg)

    import pytest

    with pytest.raises(RuntimeError, match="no capacity"):
        cloud._create()


def test_runpod_setup_installs_rsync_on_pod(monkeypatch):
    cfg = TrainConfig.default_values()
    cfg.cloud.file_sync = CloudFileSync.NATIVE_RSYNC
    cloud = RunpodCloud(cfg)

    commands = []

    class DummyConnection:
        def run(self, cmd, **kwargs):
            commands.append(cmd)

    cloud.connection = DummyConnection()
    # bypass the SSH/install machinery in BaseCloud.setup; we only test the rsync guarantee
    monkeypatch.setattr(RunpodCloud, "_connect", lambda self: None)
    cloud._ensure_remote_rsync()

    assert any("rsync" in cmd and "command -v rsync" in cmd for cmd in commands)


def test_runpod_setup_skips_rsync_install_for_non_rsync_sync():
    cfg = TrainConfig.default_values()
    cfg.cloud.file_sync = CloudFileSync.FABRIC_SFTP
    cloud = RunpodCloud(cfg)

    class ExplodingConnection:
        def run(self, cmd, **kwargs):
            raise AssertionError("should not install rsync for non-rsync sync")

    cloud.connection = ExplodingConnection()
    cloud._ensure_remote_rsync()  # must be a no-op


def test_runpod_detached_watchdog_stops_on_dead_process_or_missing_gpu():
    cfg = TrainConfig.default_values()
    cfg.cloud.remote_dir = "/workspace"
    cfg.cloud.run_id = "job1"
    cfg.cloud.detach_trainer = True
    cloud = RunpodCloud(cfg)

    cmd = cloud._get_detached_watchdog_cmd()

    assert "runpodctl stop pod $RUNPOD_POD_ID" in cmd
    assert "kill -0" in cmd
    assert "nvidia-smi -L" in cmd
    assert "1800" in cmd
    assert "job1.pid" in cmd


def test_sourceless_upload_skips_concepts_and_uploads_cache_backup(tmp_path):
    from modules.cloud.BaseCloud import BaseCloud

    class DummySync:
        def __init__(self):
            self.files = []
            self.dirs = []

        def sync_up_file(self, local, remote):
            self.files.append((Path(local), Path(remote)))

        def sync_up(self, local, remote):
            self.files.append((Path(local), Path(remote)))

        def sync_up_dir(self, local, remote, recursive, sync_info=None, skip_hidden=False, allowed_extensions=None):
            self.dirs.append((Path(local), Path(remote), recursive))

    class DummyCloud(BaseCloud):
        def run_trainer(self): ...
        def close(self): ...
        def exec_callback(self, callbacks): ...
        def send_commands(self, commands): ...
        def sync_workspace(self): ...
        def can_reattach(self): ...
        def _install_onetrainer(self, update=False): ...
        def _make_tensorboard_tunnel(self): ...
        def _upload_config_file(self, local):
            self.file_sync.sync_up_file(local, Path("/workspace/job.json"))

        def delete_workspace(self): ...

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "entry.pt").write_bytes(b"cache")
    backup = tmp_path / "workspace" / "backup" / "2026-06-19_22-21-20-backup-3928-0-3928"
    backup.mkdir(parents=True)
    (backup / "meta.json").write_text("{}", encoding="utf-8")

    cfg = TrainConfig.default_values()
    cfg.sourceless_training = True
    cfg.latent_caching = True
    cfg.continue_last_backup = True  # resuming -> backup should be uploaded
    cfg.local_cache_dir = str(cache)
    cfg.cache_dir = "/workspace/remote/F/workspace/cache"
    cfg.local_workspace_dir = str(tmp_path / "workspace")
    cfg.workspace_dir = "/workspace/remote/F/workspace/Base"
    cfg.concepts = []

    cloud = DummyCloud(cfg)
    cloud.file_sync = DummySync()
    cloud.upload_config()

    uploaded_dirs = [(local.name, remote.as_posix()) for local, remote, _recursive in cloud.file_sync.dirs]
    assert ("cache", "/workspace/remote/F/workspace/cache") in uploaded_dirs
    assert (
        "2026-06-19_22-21-20-backup-3928-0-3928",
        "/workspace/remote/F/workspace/Base/backup/2026-06-19_22-21-20-backup-3928-0-3928",
    ) in uploaded_dirs


def test_sourceless_upload_skips_backup_when_not_continuing(tmp_path):
    from modules.cloud.BaseCloud import BaseCloud

    class DummySync:
        def __init__(self):
            self.dirs = []

        def sync_up_file(self, local, remote): ...
        def sync_up(self, local, remote): ...

        def sync_up_dir(self, local, remote, recursive, sync_info=None, skip_hidden=False, allowed_extensions=None):
            self.dirs.append(Path(remote).as_posix())

    class DummyCloud(BaseCloud):
        def run_trainer(self): ...
        def close(self): ...
        def exec_callback(self, callbacks): ...
        def send_commands(self, commands): ...
        def sync_workspace(self): ...
        def can_reattach(self): ...
        def _install_onetrainer(self, update=False): ...
        def _make_tensorboard_tunnel(self): ...
        def _upload_config_file(self, local): ...
        def delete_workspace(self): ...

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "entry.pt").write_bytes(b"cache")
    backup = tmp_path / "workspace" / "backup" / "2026-06-19_22-21-20-backup-3928-0-3928"
    backup.mkdir(parents=True)
    (backup / "meta.json").write_text("{}", encoding="utf-8")

    cfg = TrainConfig.default_values()
    cfg.sourceless_training = True
    cfg.latent_caching = True
    cfg.continue_last_backup = False  # fresh run -> backup must be skipped even though it exists
    cfg.local_cache_dir = str(cache)
    cfg.cache_dir = "/workspace/remote/F/workspace/cache"
    cfg.local_workspace_dir = str(tmp_path / "workspace")
    cfg.workspace_dir = "/workspace/remote/F/workspace/Base"
    cfg.concepts = []

    cloud = DummyCloud(cfg)
    cloud.file_sync = DummySync()
    cloud.upload_config()

    assert "/workspace/remote/F/workspace/cache" in cloud.file_sync.dirs
    assert not any("backup" in remote for remote in cloud.file_sync.dirs)


def test_native_rsync_syncs_directories_with_delete(monkeypatch, tmp_path):
    from modules.cloud.NativeRsyncFileSync import NativeRsyncFileSync

    commands = []

    class DummyResult:
        def check_returncode(self): ...

    class DummyConnection:
        def open(self): ...
        def run(self, *args, **kwargs): ...
        def close(self): ...

    monkeypatch.setattr("modules.cloud.NativeRsyncFileSync.shutil.which", lambda name: "rsync")
    monkeypatch.setattr("modules.cloud.BaseSSHFileSync.fabric.Connection", lambda *args, **kwargs: DummyConnection())
    monkeypatch.setattr(
        "modules.cloud.NativeRsyncFileSync.subprocess.run", lambda args: commands.append(args) or DummyResult()
    )

    local = tmp_path / "cache"
    local.mkdir()
    (local / "entry.pt").write_bytes(b"cache")

    cfg = TrainConfig.default_values()
    secrets = cfg.secrets.cloud
    secrets.host = "1.2.3.4"
    secrets.port = 22
    secrets.user = "root"
    secrets.key_file = str(tmp_path / "id_ed25519")
    sync = NativeRsyncFileSync(cfg.cloud, secrets)

    sync.sync_up_dir(local=local, remote=Path("/workspace/cache"), recursive=True)

    assert commands
    assert commands[0][0] == "rsync"
    assert "--delete" in commands[0]
    assert "-e" in commands[0]
    assert commands[0][-1] == "root@1.2.3.4:/workspace/cache"


def test_native_rsync_recursive_creates_remote_parent(monkeypatch, tmp_path):
    # Regression: rsync only creates the final dest component, so a fresh-pod cache/backup upload
    # to a deep remote path must `mkdir -p` the target first or rsync aborts with "No such file".
    from modules.cloud.NativeRsyncFileSync import NativeRsyncFileSync

    runs = []

    class DummyResult:
        def check_returncode(self): ...

    class DummyConnection:
        def open(self): ...
        def run(self, cmd, *args, **kwargs):
            runs.append(cmd)

        def close(self): ...

    monkeypatch.setattr("modules.cloud.NativeRsyncFileSync.shutil.which", lambda name: "rsync")
    monkeypatch.setattr("modules.cloud.BaseSSHFileSync.fabric.Connection", lambda *args, **kwargs: DummyConnection())
    monkeypatch.setattr("modules.cloud.NativeRsyncFileSync.subprocess.run", lambda args: DummyResult())

    local = tmp_path / "backup"
    local.mkdir()
    (local / "optimizer.pt").write_bytes(b"x")

    cfg = TrainConfig.default_values()
    secrets = cfg.secrets.cloud
    secrets.host = "1.2.3.4"
    secrets.port = 22
    secrets.user = "root"
    secrets.key_file = str(tmp_path / "id_ed25519")
    sync = NativeRsyncFileSync(cfg.cloud, secrets)

    remote = "/workspace/remote/F/workspace/backup/2026-06-19_22-21-20-backup-1-0-1"
    sync.sync_up_dir(local=local, remote=Path(remote), recursive=True)

    assert any(cmd.startswith("mkdir -p") and remote in cmd for cmd in runs)


def test_build_live_test_cache_selects_matching_image_caption_pair(tmp_path):
    cache = tmp_path / "cache"
    image_dir = cache / "image"
    text_dir = cache / "text"
    image_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    image_path = r"F:\Datasets\SoReal!\image001.jpg"
    text_path = r"F:\Datasets\SoReal!\image001.txt"
    image_entry = {
        "filename": "image001.jpg",
        "hash": "imagehash",
        "modeltype": "Z_IMAGE",
        "cache_version": 3,
        "variants": {"640x448": {"cache_file": "image001_640x448"}},
    }
    text_entry = {
        "filename": "image001.txt",
        "hash": "texthash",
        "modeltype": "Z_IMAGE",
        "cache_version": 3,
        "variants": {"_": {"cache_file": "text001"}},
    }
    (image_dir / "image001_640x448_1.pt").write_bytes(b"image")
    (image_dir / "image001_640x448_2.pt").write_bytes(b"image2")
    (text_dir / "text001_1.pt").write_bytes(b"text")
    (text_dir / "text001_2.pt").write_bytes(b"text2")
    (image_dir / "cache.json").write_text(
        json.dumps({"version": 3, "entries": {image_path: image_entry}, "hash_index": {"imagehash": [image_path]}}),
        encoding="utf-8",
    )
    (text_dir / "cache.json").write_text(
        json.dumps({"version": 3, "entries": {text_path: text_entry}, "hash_index": {"texthash": [text_path]}}),
        encoding="utf-8",
    )

    cfg = TrainConfig.default_values()
    cfg.cache_dir = str(cache)
    cfg.workspace_dir = str(tmp_path / "workspace" / "Base")
    cfg.output_model_destination = str(tmp_path / "models" / "full.safetensors")
    cfg.cloud.run_id = "job1"

    smoke = RunpodSetupService()._build_live_test_cache(cfg, epochs=3)

    smoke_cache = Path(smoke["cache_dir"])
    assert smoke["image_path"] == image_path
    assert smoke["text_path"] == text_path
    assert (smoke_cache / "image" / "image001_640x448_1.pt").is_file()
    assert (smoke_cache / "image" / "image001_640x448_2.pt").is_file()
    assert (smoke_cache / "text" / "text001_1.pt").is_file()
    assert (smoke_cache / "text" / "text001_2.pt").is_file()
    assert list(json.loads((smoke_cache / "image" / "cache.json").read_text(encoding="utf-8"))["entries"]) == [
        image_path
    ]
    assert list(json.loads((smoke_cache / "text" / "cache.json").read_text(encoding="utf-8"))["entries"]) == [text_path]
    assert cfg.cache_dir == str(smoke_cache)
    assert cfg.continue_last_backup is False
    assert cfg.epochs == 3
    assert cfg.cloud.run_id == "job1-live-test"


def _write_cache_json(cache_subdir: Path, entries: dict) -> None:
    cache_subdir.mkdir(parents=True, exist_ok=True)
    (cache_subdir / "cache.json").write_text(json.dumps({"version": 3, "entries": entries}), encoding="utf-8")


def test_validation_errors_block_sourceless_cache_missing_metadata(tmp_path):
    cache = tmp_path / "cache"
    # image entry WITHOUT sourceless metadata; text entry WITH it
    _write_cache_json(cache / "image", {r"F:\ds\a.jpg": {"variants": {"640x448": {"cache_file": "a"}}}})
    _write_cache_json(
        cache / "text",
        {r"F:\ds\a.txt": {"variants": {"_": {"cache_file": "t"}}, "sourceless_rows": {"0": {"metadata": {"x": 1}}}}},
    )

    cfg = TrainConfig.default_values()
    cfg.cache_dir = str(cache)
    entries = RunpodSetupService()._size_entries(cfg)
    errors = RunpodSetupService._validation_errors(cfg, entries)

    assert any("missing sourceless metadata" in e for e in errors)

    # Stamp the image entry -> the metadata problem clears.
    _write_cache_json(
        cache / "image",
        {r"F:\ds\a.jpg": {"variants": {"640x448": {"cache_file": "a"}}, "sourceless": {"source_index": 0}}},
    )
    errors = RunpodSetupService._validation_errors(cfg, RunpodSetupService()._size_entries(cfg))
    assert not any("sourceless metadata" in e for e in errors)


def test_validation_errors_block_when_cache_has_no_index(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "stray.pt").write_bytes(b"x")  # dir exists but no cache.json anywhere

    cfg = TrainConfig.default_values()
    cfg.cache_dir = str(cache)
    errors = RunpodSetupService._validation_errors(cfg, RunpodSetupService()._size_entries(cfg))

    assert any("cache.json" in e for e in errors)


def test_sourceless_cache_upload_is_not_extension_filtered(tmp_path):
    # Invariant lock: the sourceless cache sync must mirror the WHOLE cache dir,
    # including cache.json (which carries the baked sourceless metadata). A
    # future *.pt-only filter would strand the metadata on the remote and break
    # sourceless training after a multi-hundred-GB upload.
    from modules.cloud.BaseCloud import BaseCloud

    class RecordingSync:
        def __init__(self):
            self.dir_calls = []

        def sync_up_file(self, local, remote): ...
        def sync_up(self, local, remote): ...

        def sync_up_dir(self, local, remote, recursive, sync_info=None, skip_hidden=False, allowed_extensions=None):
            self.dir_calls.append((Path(remote).as_posix(), allowed_extensions))

    class DummyCloud(BaseCloud):
        def run_trainer(self): ...
        def close(self): ...
        def exec_callback(self, callbacks): ...
        def send_commands(self, commands): ...
        def sync_workspace(self): ...
        def can_reattach(self): ...
        def _install_onetrainer(self, update=False): ...
        def _make_tensorboard_tunnel(self): ...
        def _upload_config_file(self, local): ...
        def delete_workspace(self): ...

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "entry.pt").write_bytes(b"cache")
    (cache / "cache.json").write_text("{}", encoding="utf-8")
    (tmp_path / "workspace").mkdir()  # upload_config writes a remote_config-*.json here

    cfg = TrainConfig.default_values()
    cfg.sourceless_training = True
    cfg.latent_caching = True
    cfg.continue_last_backup = False
    cfg.local_cache_dir = str(cache)
    cfg.cache_dir = "/workspace/remote/F/workspace/cache"
    cfg.local_workspace_dir = str(tmp_path / "workspace")
    cfg.workspace_dir = "/workspace/remote/F/workspace/Base"
    cfg.concepts = []

    cloud = DummyCloud(cfg)
    cloud.file_sync = RecordingSync()
    cloud.upload_config()

    cache_calls = [ext for remote, ext in cloud.file_sync.dir_calls if remote == "/workspace/remote/F/workspace/cache"]
    assert cache_calls == [None]  # exactly one cache sync, with NO extension filter


def test_wsl_rsync_translates_windows_drive_paths():
    from modules.cloud.WslRsyncFileSync import WslRsyncFileSync

    t = WslRsyncFileSync._to_wsl_path
    assert t(r"F:\workspace\SoReal!\cache") == "/mnt/f/workspace/SoReal!/cache"
    assert t("C:/Users/calla/.ssh/id_ed25519") == "/mnt/c/Users/calla/.ssh/id_ed25519"
    assert t("F:\\workspace\\backup\\") == "/mnt/f/workspace/backup/"  # trailing sep preserved
    # remote specs and already-POSIX args must pass through untouched
    assert t("root@1.2.3.4:/workspace/remote/cache") == "root@1.2.3.4:/workspace/remote/cache"
    assert t("/usr/bin/ssh -p 22 -i /home/u/.ssh/k") == "/usr/bin/ssh -p 22 -i /home/u/.ssh/k"
    assert t("rsync") == "rsync"


def test_wsl_rsync_run_delegates_to_wsl_with_translated_paths(monkeypatch):
    from modules.cloud.WslRsyncFileSync import WslRsyncFileSync

    captured = {}

    class DummyResult:
        def check_returncode(self): ...

    def fake_run(args, *a, **k):
        captured["args"] = args
        return DummyResult()

    monkeypatch.setattr("modules.cloud.WslRsyncFileSync.subprocess.run", fake_run)

    # build an instance without running __init__ (which would shell out to WSL)
    sync = WslRsyncFileSync.__new__(WslRsyncFileSync)
    sync.wsl = "wsl.exe"
    sync.wsl_rsync = "/usr/bin/rsync"

    sync._run(
        [
            "rsync",
            "-av",
            "--delete",
            "-e",
            "/usr/bin/ssh -i /home/u/.ssh/k",
            "F:\\workspace\\cache\\",
            "root@1.2.3.4:/workspace/remote/cache",
        ]
    )

    args = captured["args"]
    assert args[:3] == ["wsl.exe", "-e", "/usr/bin/rsync"]
    assert "/mnt/f/workspace/cache/" in args  # drive path translated for WSL
    assert "root@1.2.3.4:/workspace/remote/cache" in args  # remote spec untouched
    assert "F:" not in " ".join(args)  # no Windows drive-colon reaches rsync


def test_mutate_config_uses_requested_gpu_and_cloud_type(monkeypatch, tmp_path):
    _force_native_rsync(monkeypatch)
    cfg = TrainConfig.default_values()
    cfg.workspace_dir = str(tmp_path / "workspace")

    RunpodSetupService()._mutate_config_for_runpod(
        cfg, 100, {"gpu_type": "  NVIDIA H100 80GB HBM3 ", "cloud_type": "community"}
    )

    assert cfg.cloud.gpu_type == "NVIDIA H100 80GB HBM3"
    assert cfg.cloud.sub_type == "COMMUNITY"


def test_mutate_config_defaults_to_5090_secure_when_unset_or_invalid(monkeypatch, tmp_path):
    _force_native_rsync(monkeypatch)
    cfg = TrainConfig.default_values()
    cfg.workspace_dir = str(tmp_path / "workspace")

    RunpodSetupService()._mutate_config_for_runpod(cfg, 100, {"gpu_type": "", "cloud_type": "SPOT"})

    assert cfg.cloud.gpu_type == "NVIDIA GeForce RTX 5090"
    assert cfg.cloud.sub_type == "SECURE"


def test_list_gpu_types_sorts_filters_and_flags(monkeypatch):
    import web.backend.services.runpod_setup_service as svc

    seen_keys = []

    def fake_query(query):
        seen_keys.append(svc.runpod.api_key)
        return {
            "data": {
                "gpuTypes": [
                    {"id": "unknown", "displayName": "unknown", "memoryInGb": 0},
                    {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24},
                    {
                        "id": "NVIDIA GeForce RTX 5090",
                        "displayName": "RTX 5090",
                        "memoryInGb": 32,
                        "secureCloud": True,
                        "communityCloud": True,
                        "securePrice": 0.89,
                        "communityPrice": 0.69,
                    },
                    {"id": "Tesla V100-SXM2-16GB", "displayName": "V100 SXM2", "memoryInGb": 16},
                    {"id": "AMD Instinct MI300X OAM", "displayName": "MI300X", "memoryInGb": 192},
                ]
            }
        }

    monkeypatch.setattr(svc, "run_graphql_query", fake_query)
    monkeypatch.setattr(svc.runpod, "api_key", "previous-key")

    result = RunpodSetupService().list_gpu_types("typed-key")

    assert result["ok"] is True
    assert result["default"] == "NVIDIA GeForce RTX 5090"
    assert seen_keys == ["typed-key"]
    assert svc.runpod.api_key == "previous-key"  # the typed key does not leak into the SDK global
    ids = [gpu["id"] for gpu in result["gpus"]]
    assert ids == [
        "AMD Instinct MI300X OAM",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 4090",
        "Tesla V100-SXM2-16GB",
    ]
    by_id = {gpu["id"]: gpu for gpu in result["gpus"]}
    assert by_id["NVIDIA GeForce RTX 5090"]["secure_price"] == 0.89
    assert by_id["NVIDIA GeForce RTX 5090"]["compat_warning"] is None
    assert by_id["NVIDIA GeForce RTX 4090"]["compat_warning"] is None
    assert by_id["Tesla V100-SXM2-16GB"]["compat_warning"]
    assert by_id["AMD Instinct MI300X OAM"]["compat_warning"]


def test_list_gpu_types_falls_back_to_get_gpus_on_query_error(monkeypatch):
    import web.backend.services.runpod_setup_service as svc

    def failing_query(query):
        raise svc.runpod_error.QueryError("Cannot query field", query)

    monkeypatch.setattr(svc, "run_graphql_query", failing_query)
    monkeypatch.setattr(svc.runpod, "get_gpus", lambda: [{"id": "NVIDIA A40", "displayName": "A40", "memoryInGb": 48}])

    result = RunpodSetupService().list_gpu_types("key")

    assert result["ok"] is True
    assert [gpu["id"] for gpu in result["gpus"]] == ["NVIDIA A40"]
    assert result["gpus"][0]["secure_cloud"] is None


def test_list_gpu_types_reports_errors_instead_of_raising(monkeypatch):
    import web.backend.services.runpod_setup_service as svc

    def unauthorized(query):
        raise svc.runpod_error.AuthenticationError("Unauthorized request, please check your API key.")

    monkeypatch.setattr(svc, "run_graphql_query", unauthorized)

    result = RunpodSetupService().list_gpu_types("bad-key")

    assert result["ok"] is False
    assert result["gpus"] == []
    assert "Unauthorized" in result["error"]
    assert result["default"] == "NVIDIA GeForce RTX 5090"


def test_list_gpu_types_without_any_key(monkeypatch):
    import web.backend.services.runpod_setup_service as svc

    class FakeConfigService:
        @staticmethod
        def get_config_for_training():
            return TrainConfig.default_values()

    monkeypatch.setattr(svc.ConfigService, "get_instance", staticmethod(lambda: FakeConfigService()))

    result = RunpodSetupService().list_gpu_types("")

    assert result["ok"] is False
    assert "API key" in result["error"]


def test_runpod_gpu_compat_warning_flags_pre_turing_and_amd_only():
    from modules.cloud.RunpodCloud import runpod_gpu_compat_warning

    for supported in (
        "NVIDIA GeForce RTX 5090",
        "NVIDIA H100 80GB HBM3",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA RTX A4000",
        "NVIDIA L4",
        "Tesla T4",
        "NVIDIA RTX 4000 Ada Generation",
        "NVIDIA B200",
    ):
        assert runpod_gpu_compat_warning(supported) is None, supported
    for unsupported in (
        "Tesla V100-SXM2-16GB",
        "Tesla P100-PCIE-16GB",
        "Tesla P4",
        "NVIDIA GeForce GTX 1080 Ti",
        "AMD Instinct MI300X OAM",
    ):
        assert runpod_gpu_compat_warning(unsupported), unsupported


def test_runpod_create_passes_selected_gpu_and_warns_on_incompatible(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "modules.cloud.RunpodCloud.runpod.create_pod", lambda **kwargs: calls.append(kwargs) or {"id": "p"}
    )

    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "Tesla V100-SXM2-16GB"
    cfg.cloud.sub_type = "COMMUNITY"
    RunpodCloud(cfg)._create()

    assert calls[0]["gpu_type_id"] == "Tesla V100-SXM2-16GB"
    assert calls[0]["cloud_type"] == "COMMUNITY"
    assert "warning:" in capsys.readouterr().out


def test_mutate_config_passes_upload_without_gpu(monkeypatch, tmp_path):
    _force_native_rsync(monkeypatch)
    cfg = TrainConfig.default_values()
    cfg.workspace_dir = str(tmp_path / "workspace")

    RunpodSetupService()._mutate_config_for_runpod(cfg, 100, {"upload_without_gpu": True})
    assert cfg.cloud.upload_without_gpu is True

    RunpodSetupService()._mutate_config_for_runpod(cfg, 100, {})
    assert cfg.cloud.upload_without_gpu is False


class _FakeRunpodLifecycle:
    """create/stop/get/resume fakes; resume_pod fails for the gpu counts in `reject` (a list of counts, consumed)."""

    def __init__(self, monkeypatch, reject=()):
        self.calls = []
        self.reject = list(reject)
        self.status = "RUNNING"
        monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.create_pod", self.create_pod)
        monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.stop_pod", self.stop_pod)
        monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.get_pod", self.get_pod)
        monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.resume_pod", self.resume_pod)
        monkeypatch.setattr("modules.cloud.RunpodCloud.time.sleep", lambda seconds: None)

    def create_pod(self, **kwargs):
        self.calls.append(("create", kwargs["gpu_type_id"]))
        return {"id": "pod-zero"}

    def stop_pod(self, pod_id):
        self.calls.append(("stop", pod_id))
        self.status = "EXITED"

    def get_pod(self, pod_id):
        return {"id": pod_id, "desiredStatus": self.status}

    def resume_pod(self, pod_id, gpu_count):
        self.calls.append(("resume", gpu_count))
        if self.reject and self.reject[0] == gpu_count:
            self.reject.pop(0)
            raise RuntimeError("There are not enough free GPUs on the host machine to start this pod.")
        self.status = "RUNNING"


def _upload_without_gpu_cloud(tensorboard_tunnel=False):
    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "NVIDIA H100 80GB HBM3"
    cfg.cloud.upload_without_gpu = True
    cfg.cloud.tensorboard_tunnel = tensorboard_tunnel
    return RunpodCloud(cfg)


def test_upload_without_gpu_parks_new_pod_and_resumes_it_with_zero_gpus(monkeypatch):
    fake = _FakeRunpodLifecycle(monkeypatch)
    cloud = _upload_without_gpu_cloud()

    cloud._create()
    cloud._start()  # what _connect() does when it finds the parked pod EXITED

    assert fake.calls == [("create", "NVIDIA H100 80GB HBM3"), ("stop", "pod-zero"), ("resume", 0)]


def test_upload_without_gpu_falls_back_to_gpu_when_zero_gpu_resume_is_rejected(monkeypatch):
    fake = _FakeRunpodLifecycle(monkeypatch, reject=[0])
    cloud = _upload_without_gpu_cloud()

    cloud._create()
    cloud._start()
    cloud.attach_gpu()  # already has its GPU: nothing more to do

    assert fake.calls[-2:] == [("resume", 0), ("resume", 1)]
    assert fake.calls[1] == ("stop", "pod-zero") and fake.calls.count(("stop", "pod-zero")) == 1


def _stub_reconnect(monkeypatch, cloud, commands):
    class Result:
        exited = 0

    class DummyConnection:
        def run(self, cmd, **kwargs):
            commands.append(cmd)
            return Result()

    cloud.connection = DummyConnection()  # the zero-GPU connection the prefetch runs on

    def fake_connect(self):
        if runpod_status(self) == "EXITED":
            self._start()
        self.connection = DummyConnection()

    def runpod_status(self):
        import modules.cloud.RunpodCloud as rc

        return rc.runpod.get_pod(self.config.secrets.cloud.id)["desiredStatus"]

    monkeypatch.setattr(RunpodCloud, "_connect", fake_connect)
    monkeypatch.setattr(RunpodCloud, "close", lambda self: None)


def test_attach_gpu_stops_and_resumes_with_gpu_retrying_until_free(monkeypatch):
    fake = _FakeRunpodLifecycle(monkeypatch, reject=[1, 1])
    cloud = _upload_without_gpu_cloud()
    cloud.config.cloud.file_sync = CloudFileSync.NATIVE_RSYNC
    commands = []
    _stub_reconnect(monkeypatch, cloud, commands)

    cloud._create()
    cloud._start()
    cloud.attach_gpu()

    resumes = [count for kind, count in fake.calls if kind == "resume"]
    assert resumes == [0, 1, 1, 1]  # zero-GPU upload, then two "GPU taken" retries before it attaches
    assert fake.calls.count(("stop", "pod-zero")) == 2
    # the container disk was wiped by the stop: rsync and the command pipe are set up again
    assert any("command -v rsync" in cmd for cmd in commands)
    assert any(cmd.startswith("mkfifo ") for cmd in commands)
    # the base model was downloaded first, while the pod still had no GPU
    prefetch = [i for i, cmd in enumerate(commands) if "run-cmd.sh prefetch_model" in cmd]
    rsync = [i for i, cmd in enumerate(commands) if "command -v rsync" in cmd]
    assert prefetch and prefetch[0] < rsync[0]
    assert f"--config-path={cloud.config_file}" in commands[prefetch[0]]
    assert "OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt" in commands[prefetch[0]]


def test_attach_gpu_gives_up_with_pod_intact_after_timeout(monkeypatch):
    import modules.cloud.RunpodCloud as rc

    fake = _FakeRunpodLifecycle(monkeypatch, reject=[1] * 100)
    monkeypatch.setattr(rc, "RUNPOD_GPU_ATTACH_TIMEOUT_SECONDS", 0)
    cloud = _upload_without_gpu_cloud()
    _stub_reconnect(monkeypatch, cloud, [])

    cloud._create()
    cloud._start()
    try:
        cloud.attach_gpu()
    except rc.RunpodGpuUnavailableError as exc:
        assert "left stopped" in str(exc)
    else:
        raise AssertionError("expected RunpodGpuUnavailableError")
    assert not any(kind in ("terminate", "delete") for kind, _ in fake.calls)


def test_runpod_install_forces_cuda_requirements():
    cfg = TrainConfig.default_values()
    cloud = RunpodCloud(cfg)
    commands = []

    class Result:
        exited = 1  # no venv yet: fresh install

    class DummyConnection:
        def run(self, cmd, **kwargs):
            commands.append(cmd)
            return Result()

    cloud.connection = DummyConnection()
    cloud._install_onetrainer()

    install = next(cmd for cmd in commands if "./install.sh" in cmd)
    assert "OT_PLATFORM_REQUIREMENTS=requirements-cuda.txt" in install


def test_rsync_install_reuses_debs_kept_on_the_volume():
    cfg = TrainConfig.default_values()
    cfg.cloud.file_sync = CloudFileSync.NATIVE_RSYNC
    cloud = RunpodCloud(cfg)
    commands = []

    class DummyConnection:
        def run(self, cmd, **kwargs):
            commands.append(cmd)

    cloud.connection = DummyConnection()
    cloud._ensure_remote_rsync()

    cmd = commands[0]
    assert "dpkg -i /workspace/.ot-apt-cache/*.deb" in cmd  # fast path after a stop, no apt-get update
    assert "Dir::Cache::archives=/workspace/.ot-apt-cache" in cmd  # first install keeps its .debs there
    assert cmd.index("dpkg -i") < cmd.index("apt-get update")


def test_hf_prefetch_skips_overridden_parts_and_local_models():
    from modules.util.hf_prefetch_util import hf_prefetch_patterns, hf_prefetch_repo

    cfg = TrainConfig.default_values()
    cfg.base_model_name = "Tongyi-MAI/Z-Image"
    cfg.transformer.model_name = "/workspace/remote/SoReal!_Base_V1.5.safetensors"
    assert hf_prefetch_repo(cfg) == "Tongyi-MAI/Z-Image"
    patterns = hf_prefetch_patterns(cfg)
    assert "text_encoder*/*" in patterns and "vae/*" in patterns and "tokenizer*/*" in patterns
    assert "transformer/*" not in patterns  # overridden by the uploaded file

    cfg.transformer.model_name = ""
    assert "transformer/*" in hf_prefetch_patterns(cfg)

    for local in ("/workspace/models/base", "E:\\models\base", "models/base.safetensors", ""):
        cfg.base_model_name = local
        assert hf_prefetch_repo(cfg) is None, local
