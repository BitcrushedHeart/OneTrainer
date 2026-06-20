import json
from pathlib import Path

from modules.cloud.RunpodCloud import RunpodCloud
from modules.trainer.CloudTrainer import CloudTrainer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType
from web.backend.services.runpod_setup_service import RunpodSetupService


def test_storage_rounding():
    assert RunpodSetupService._round_storage_gb(284.61) == 300
    assert RunpodSetupService._round_storage_gb(300.01) == 350


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
    assert calls[0]["country_code"] == "EU"
    assert calls[0]["volume_in_gb"] == 300
    assert "container_disk_in_gb" not in calls[0]


def test_runpod_create_falls_back_between_europe_codes(monkeypatch):
    calls = []

    def fake_create_pod(**kwargs):
        calls.append(kwargs)
        if kwargs["country_code"] == "EU":
            raise RuntimeError("bad country")
        return {"id": "pod-456"}

    monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.create_pod", fake_create_pod)

    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "NVIDIA GeForce RTX 5090"
    cfg.cloud.sub_type = "SECURE"
    cloud = RunpodCloud(cfg)
    cloud._create()

    assert cfg.secrets.cloud.id == "pod-456"
    assert [call["country_code"] for call in calls[:2]] == ["EU", "GB"]


def test_runpod_create_tries_all_regions_when_capacity_returns_none(monkeypatch):
    calls = []

    def fake_create_pod(**kwargs):
        calls.append(kwargs["country_code"])
        # simulate no-capacity: runpod returns None instead of raising for the first two regions
        if kwargs["country_code"] in ("EU", "GB"):
            return None
        return {"id": "pod-789"}

    monkeypatch.setattr("modules.cloud.RunpodCloud.runpod.create_pod", fake_create_pod)

    cfg = TrainConfig.default_values()
    cfg.cloud.gpu_type = "NVIDIA GeForce RTX 5090"
    cfg.cloud.sub_type = "SECURE"
    cloud = RunpodCloud(cfg)
    cloud._create()

    assert cfg.secrets.cloud.id == "pod-789"
    assert calls[:3] == ["EU", "GB", "NL"]


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
