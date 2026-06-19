import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.trainer.CloudTrainer import CloudTrainer
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType
from modules.util.enum.ConfigPart import ConfigPart
from modules.util.enum.TimeUnit import TimeUnit
from web.backend.paths import PROJECT_ROOT
from web.backend.services._singleton import SingletonMixin
from web.backend.services.config_service import ConfigService

BUFFER_GIB = 50
ROUND_GB = 50
RUNPOD_TEMPLATE_ID = "1a33vbssq9"
RUNPOD_GPU_TYPE = "NVIDIA GeForce RTX 5090"
RUNPOD_INSTALL_CMD = "git clone -b bitcrushed-blend https://github.com/BitcrushedHeart/OneTrainer.git"
MGDS_REPO = "https://github.com/BitcrushedHeart/mgds.git"
MGDS_SOURCE_DIR = Path(PROJECT_ROOT) / "venv" / "src" / "mgds"
REQUIREMENTS_GLOBAL = Path(PROJECT_ROOT) / "requirements-global.txt"


@dataclass
class SizeEntry:
    label: str
    path: str
    exists: bool
    bytes: int
    files: int
    required: bool = True

    @property
    def gib(self) -> float:
        return round(self.bytes / (1024**3), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "exists": self.exists,
            "bytes": self.bytes,
            "gib": self.gib,
            "files": self.files,
            "required": self.required,
        }


class RunpodSetupService(SingletonMixin):
    def preview(self) -> dict[str, Any]:
        config = ConfigService.get_instance().get_config_for_training()
        entries = self._size_entries(config)
        errors = self._validation_errors(config, entries)
        total_bytes = sum(entry.bytes for entry in entries if entry.exists)
        total_gib = total_bytes / (1024**3)
        required_gb = self._round_storage_gb(total_gib + BUFFER_GIB)

        return {
            "entries": [entry.to_dict() for entry in entries],
            "buffer_gib": BUFFER_GIB,
            "total_gib": round(total_gib, 2),
            "required_gb": required_gb,
            "latest_backup": self._latest_backup(config),
            "remote_paths": self._remote_paths(config),
            "errors": errors,
            "warnings": self._warnings(config),
            "git": self._git_summary(),
            "defaults": {
                "template_id": RUNPOD_TEMPLATE_ID,
                "gpu_type": RUNPOD_GPU_TYPE,
                "cloud_type": "SECURE",
                "install_cmd": RUNPOD_INSTALL_CMD,
            },
        }

    def run_git_preflight(self) -> dict[str, Any]:
        mgds = self._commit_and_push_mgds()
        requirements_changed = self._pin_mgds_requirement(mgds["commit"])
        main = self._commit_and_push_main()
        return {
            "ok": True,
            "mgds": mgds,
            "requirements_changed": requirements_changed,
            "main": main,
        }

    def prepare(self, body: dict[str, Any]) -> dict[str, Any]:
        service = ConfigService.get_instance()
        config = service.get_config_for_training()
        preview = self.preview()
        if preview["errors"]:
            return {"ok": False, "error": "; ".join(preview["errors"]), "preview": preview}

        secrets = config.secrets.cloud
        api_key = str(body.get("api_key") or secrets.api_key or "").strip()
        if not api_key:
            return {"ok": False, "error": "RunPod API key is required", "preview": preview}

        secrets.api_key = api_key
        secrets.user = str(body.get("ssh_user") or secrets.user or "root").strip() or "root"
        secrets.key_file = str(body.get("ssh_key_file") or secrets.key_file or "").strip()
        secrets.password = str(body.get("ssh_password") or secrets.password or "").strip()
        secrets.host = ""
        secrets.port = 0
        secrets.id = ""

        self._mutate_config_for_runpod(config, preview["required_gb"], body)
        reconciled = service.update_config(config.to_dict())
        service.update_cloud_secrets(secrets.to_dict())
        return {"ok": True, "config": reconciled, "preview": self.preview()}

    def start_live_test(self, body: dict[str, Any]) -> dict[str, Any]:
        from web.backend.services.trainer_service import TrainerService

        service = ConfigService.get_instance()
        base_config = service.get_config_for_training()
        preview = self.preview()
        base_errors = [
            error
            for error in preview["errors"]
            if not error.startswith("Missing required latest_backup")
        ]
        if base_errors:
            return {"ok": False, "error": "; ".join(base_errors), "preview": preview}

        api_key = str(body.get("api_key") or base_config.secrets.cloud.api_key or "").strip()
        if not api_key:
            return {"ok": False, "error": "RunPod API key is required", "preview": preview}

        smoke_config = TrainConfig.default_values().from_dict(base_config.to_dict())
        smoke_config.secrets.cloud.api_key = api_key
        smoke_config.secrets.cloud.user = str(body.get("ssh_user") or smoke_config.secrets.cloud.user or "root").strip() or "root"
        smoke_config.secrets.cloud.key_file = str(body.get("ssh_key_file") or smoke_config.secrets.cloud.key_file or "").strip()
        smoke_config.secrets.cloud.password = str(body.get("ssh_password") or smoke_config.secrets.cloud.password or "").strip()
        smoke_config.secrets.cloud.host = ""
        smoke_config.secrets.cloud.port = 0
        smoke_config.secrets.cloud.id = ""

        self._mutate_config_for_runpod(smoke_config, preview["required_gb"], body)
        smoke = self._build_live_test_cache(smoke_config, epochs=int(body.get("epochs") or 3))
        start_result = TrainerService.get_instance().start_training(reattach=False, train_config=smoke_config)
        if not start_result.get("ok"):
            return {"ok": False, "error": start_result.get("error", "Training start failed"), "preview": preview}

        return {"ok": True, "preview": preview, "smoke": smoke, "start": start_result}

    def _mutate_config_for_runpod(self, config: TrainConfig, volume_size: int, body: dict[str, Any]) -> None:
        config.cloud.enabled = True
        config.cloud.type = CloudType.RUNPOD
        config.cloud.file_sync = CloudFileSync.FABRIC_SFTP
        config.cloud.create = True
        config.cloud.name = str(body.get("pod_name") or config.cloud.name or "OneTrainer").strip() or "OneTrainer"
        config.cloud.sub_type = "SECURE"
        config.cloud.gpu_type = RUNPOD_GPU_TYPE
        config.cloud.volume_size = int(volume_size)
        config.cloud.min_download = int(body.get("min_download") or config.cloud.min_download or 0)
        config.cloud.remote_dir = config.cloud.remote_dir or "/workspace"
        config.cloud.huggingface_cache_dir = config.cloud.huggingface_cache_dir or "/workspace/huggingface_cache"
        config.cloud.onetrainer_dir = config.cloud.onetrainer_dir or "/workspace/OneTrainer"
        config.cloud.install_cmd = RUNPOD_INSTALL_CMD
        config.cloud.install_onetrainer = True
        config.cloud.update_onetrainer = False
        config.cloud.detach_trainer = True
        config.cloud.run_id = str(body.get("run_id") or config.cloud.run_id or "job1").strip() or "job1"

        config.latent_caching = True
        config.sourceless_training = True
        config.only_cache = False
        config.continue_last_backup = True
        config.clear_cache_before_training = False

    def _build_live_test_cache(self, config: TrainConfig, epochs: int) -> dict[str, Any]:
        source_cache = Path(config.cache_dir)
        image_index_path = source_cache / "image" / "cache.json"
        text_index_path = source_cache / "text" / "cache.json"
        if not image_index_path.is_file() or not text_index_path.is_file():
            raise RuntimeError("Live test requires image/cache.json and text/cache.json in the current cache directory.")

        image_index = json.loads(image_index_path.read_text(encoding="utf-8"))
        text_index = json.loads(text_index_path.read_text(encoding="utf-8"))
        image_entries = image_index.get("entries", {})
        text_entries = text_index.get("entries", {})

        candidates = list(image_entries.items())
        random.SystemRandom().shuffle(candidates)
        selected: tuple[str, dict, str, dict] | None = None
        for image_path, image_entry in candidates:
            text_path = str(Path(image_path).with_suffix(".txt"))
            text_entry = text_entries.get(text_path)
            if text_entry and self._first_cache_file(image_entry) and self._first_cache_file(text_entry):
                selected = (image_path, image_entry, text_path, text_entry)
                break
        if selected is None:
            raise RuntimeError("Could not find a cached image entry with a matching cached caption entry.")

        image_path, image_entry, text_path, text_entry = selected
        smoke_root = Path(config.cache_dir).parent / "runpod-live-test-cache"
        if smoke_root.exists():
            shutil.rmtree(smoke_root)
        (smoke_root / "image").mkdir(parents=True)
        (smoke_root / "text").mkdir(parents=True)

        self._write_cache_subset(source_cache / "image", smoke_root / "image", image_index, image_path, image_entry)
        self._write_cache_subset(source_cache / "text", smoke_root / "text", text_index, text_path, text_entry)

        config.cache_dir = str(smoke_root)
        config.workspace_dir = str(Path(config.workspace_dir).parent / "runpod-live-test-workspace")
        config.output_model_destination = str(Path(config.output_model_destination).with_name("runpod-live-test.safetensors"))
        config.continue_last_backup = False
        config.epochs = max(1, epochs)
        config.backup_after_unit = TimeUnit.NEVER
        config.save_every_unit = TimeUnit.NEVER
        config.sample_after_unit = TimeUnit.NEVER
        config.include_train_config = ConfigPart.NONE
        config.cloud.run_id = f"{config.cloud.run_id}-live-test"

        return {
            "cache_dir": str(smoke_root),
            "workspace_dir": config.workspace_dir,
            "image_path": image_path,
            "text_path": text_path,
            "epochs": config.epochs,
        }

    def _write_cache_subset(
        self,
        source_dir: Path,
        target_dir: Path,
        source_index: dict[str, Any],
        entry_path: str,
        entry: dict[str, Any],
    ) -> None:
        cache_file = self._first_cache_file(entry)
        if cache_file is None:
            raise RuntimeError(f"Cache entry for {entry_path} has no cache file")

        pt_name = f"{cache_file}_1.pt"
        shutil.copy2(source_dir / pt_name, target_dir / pt_name)
        subset = {
            key: value
            for key, value in source_index.items()
            if key not in {"entries", "hash_index", "last_validated", "watched_fingerprints"}
        }
        subset["entries"] = {entry_path: entry}
        if "hash" in entry:
            subset["hash_index"] = {entry["hash"]: [entry_path]}
        (target_dir / "cache.json").write_text(json.dumps(subset, indent=2), encoding="utf-8")

    @staticmethod
    def _first_cache_file(entry: dict[str, Any]) -> str | None:
        for variant in (entry.get("variants") or {}).values():
            cache_file = variant.get("cache_file")
            if cache_file:
                return str(cache_file)
        return None

    def _size_entries(self, config: TrainConfig) -> list[SizeEntry]:
        entries = [
            self._entry("cache", config.cache_dir, required=True),
            self._entry("latest_backup", self._latest_backup(config), required=True),
            self._entry("transformer_override", config.transformer.model_name, required=True),
        ]

        base_model = self._entry("base_model", config.base_model_name, required=False)
        if base_model.exists:
            entries.append(base_model)
        return entries

    def _entry(self, label: str, path: str | None, required: bool) -> SizeEntry:
        if not path:
            return SizeEntry(label=label, path="", exists=False, bytes=0, files=0, required=required)

        candidate = Path(path)
        if not candidate.exists():
            return SizeEntry(label=label, path=str(path), exists=False, bytes=0, files=0, required=required)

        if candidate.is_file():
            return SizeEntry(label=label, path=str(candidate), exists=True, bytes=candidate.stat().st_size, files=1, required=required)

        total = 0
        count = 0
        for file in candidate.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
                count += 1
        return SizeEntry(label=label, path=str(candidate), exists=True, bytes=total, files=count, required=required)

    @staticmethod
    def _round_storage_gb(required_gib: float) -> int:
        return int(math.ceil(required_gib / ROUND_GB) * ROUND_GB)

    @staticmethod
    def _latest_backup(config: TrainConfig) -> str:
        backups = Path(config.workspace_dir) / "backup"
        if not backups.is_dir():
            return ""
        dirs = [entry for entry in backups.iterdir() if entry.is_dir()]
        if not dirs:
            return ""
        return str(sorted(dirs, key=lambda p: p.name, reverse=True)[0])

    @staticmethod
    def _remote_paths(config: TrainConfig) -> dict[str, str]:
        remote_dir = config.cloud.remote_dir or "/workspace"
        result: dict[str, str] = {}
        for key, path in {
            "workspace_dir": config.workspace_dir,
            "cache_dir": config.cache_dir,
            "transformer_override": config.transformer.model_name,
        }.items():
            if path:
                result[key] = CloudTrainer.adjust_path_to_remote_dir(path, remote_dir)
        latest = RunpodSetupService._latest_backup(config)
        if latest:
            backup_parent = CloudTrainer.adjust_path_to_remote_dir(str(Path(config.workspace_dir) / "backup"), remote_dir)
            result["latest_backup"] = f"{backup_parent}/{Path(latest).name}"
        return result

    @staticmethod
    def _validation_errors(config: TrainConfig, entries: list[SizeEntry]) -> list[str]:
        errors = [f"Missing required {entry.label}: {entry.path or '(empty)'}" for entry in entries if entry.required and not entry.exists]
        if config.train_text_encoder_or_embedding():
            errors.append("Sourceless training cannot be used while text encoder or embedding training is enabled.")
        return errors

    @staticmethod
    def _warnings(config: TrainConfig) -> list[str]:
        warnings: list[str] = []
        if config.cloud.file_sync.name != "FABRIC_SFTP":
            warnings.append("RunPod setup will switch file sync to FABRIC_SFTP for streamed uploads.")
        if config.only_cache:
            warnings.append("Only Cache is enabled locally; RunPod setup will disable it before starting training.")
        return warnings

    def _git_summary(self) -> dict[str, Any]:
        return {
            "main_dirty": self._git_dirty(Path(PROJECT_ROOT)),
            "mgds_dirty": self._git_dirty(MGDS_SOURCE_DIR) if MGDS_SOURCE_DIR.exists() else False,
            "mgds_path": str(MGDS_SOURCE_DIR),
            "requirements_path": str(REQUIREMENTS_GLOBAL),
        }

    @staticmethod
    def _git_dirty(repo: Path) -> bool:
        if not (repo / ".git").exists():
            return False
        result = RunpodSetupService._run_git(repo, "status", "--porcelain")
        return bool(result["stdout"].strip())

    @staticmethod
    def _run_git(repo: Path, *args: str) -> dict[str, Any]:
        cmd = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "git command failed").strip())
        return {"stdout": proc.stdout, "stderr": proc.stderr}

    def _commit_and_push_mgds(self) -> dict[str, Any]:
        if not (MGDS_SOURCE_DIR / ".git").exists():
            raise RuntimeError(f"mgds repository not found at {MGDS_SOURCE_DIR}")

        dirty = self._git_dirty(MGDS_SOURCE_DIR)
        if dirty:
            self._run_git(MGDS_SOURCE_DIR, "add", "-A")
            self._run_git(MGDS_SOURCE_DIR, "commit", "-m", "chore: update mgds for runpod sourceless continuation")

        commit = self._run_git(MGDS_SOURCE_DIR, "rev-parse", "HEAD")["stdout"].strip()
        branch = self._run_git(MGDS_SOURCE_DIR, "branch", "--show-current")["stdout"].strip() or "SmartCache"
        self._run_git(MGDS_SOURCE_DIR, "push", "origin", f"HEAD:{branch}")
        return {"dirty": dirty, "commit": commit, "branch": branch}

    @staticmethod
    def _pin_mgds_requirement(commit: str) -> bool:
        text = REQUIREMENTS_GLOBAL.read_text(encoding="utf-8")
        pattern = r"-e git\+https://github\.com/BitcrushedHeart/mgds\.git@[^\s#]+#egg=mgds"
        replacement = f"-e git+{MGDS_REPO}@{commit}#egg=mgds"
        updated = re.sub(pattern, replacement, text)
        if updated == text:
            return False
        REQUIREMENTS_GLOBAL.write_text(updated, encoding="utf-8")
        return True

    def _commit_and_push_main(self) -> dict[str, Any]:
        repo = Path(PROJECT_ROOT)
        self._run_git(repo, "add", "-A")
        dirty = self._git_dirty(repo)
        if dirty:
            self._run_git(repo, "commit", "-m", "chore: prepare runpod sourceless continuation")
        commit = self._run_git(repo, "rev-parse", "HEAD")["stdout"].strip()
        self._run_git(repo, "push", "fork", "HEAD:bitcrushed-blend")
        return {"dirty": dirty, "commit": commit, "branch": "bitcrushed-blend", "remote": "fork"}
