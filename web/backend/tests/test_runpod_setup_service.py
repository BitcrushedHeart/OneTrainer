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


def test_mutate_config_for_runpod_preserves_resume_and_sets_sourceless():
    cfg = TrainConfig.default_values()
    cfg.only_cache = True
    cfg.continue_last_backup = False
    cfg.clear_cache_before_training = True
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
    assert ("2026-06-19_22-21-20-backup-3928-0-3928", "/workspace/remote/F/workspace/Base/backup/2026-06-19_22-21-20-backup-3928-0-3928") in uploaded_dirs


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
    monkeypatch.setattr("modules.cloud.NativeRsyncFileSync.subprocess.run", lambda args: commands.append(args) or DummyResult())

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
    assert list(json.loads((smoke_cache / "text" / "cache.json").read_text(encoding="utf-8"))["entries"]) == [
        text_path
    ]
    assert cfg.cache_dir == str(smoke_cache)
    assert cfg.continue_last_backup is False
    assert cfg.epochs == 3
    assert cfg.cloud.run_id == "job1-live-test"
