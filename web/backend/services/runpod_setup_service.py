import json
import math
import os
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
from modules.util.sourceless_cache_util import sourceless_cache_problems
from web.backend.paths import PROJECT_ROOT
from web.backend.services._singleton import SingletonMixin
from web.backend.services.config_service import ConfigService

# Headroom on top of the uploaded total, for things that land on the volume but
# aren't in the upload size: the HF base-model download, the OneTrainer install,
# and training outputs/backups/samples. Rounded up to ROUND_GB. Keep some margin
# — a pod that runs out of disk mid-run wastes the GPU time, which costs more
# than a few spare GB of volume.
BUFFER_GIB = 30
ROUND_GB = 5
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

        ssh_error = self._ssh_auth_error(secrets.key_file)
        if ssh_error:
            return {"ok": False, "error": ssh_error, "preview": preview}

        self._mutate_config_for_runpod(config, preview["required_gb"], body)
        reconciled = service.update_config(config.to_dict())
        service.update_cloud_secrets(secrets.to_dict())
        return {"ok": True, "config": reconciled, "preview": self.preview()}

    def start_live_test(self, body: dict[str, Any]) -> dict[str, Any]:
        from web.backend.services.trainer_service import TrainerService

        service = ConfigService.get_instance()
        base_config = service.get_config_for_training()
        preview = self.preview()
        base_errors = [error for error in preview["errors"] if not error.startswith("Missing required latest_backup")]
        if base_errors:
            return {"ok": False, "error": "; ".join(base_errors), "preview": preview}

        api_key = str(body.get("api_key") or base_config.secrets.cloud.api_key or "").strip()
        if not api_key:
            return {"ok": False, "error": "RunPod API key is required", "preview": preview}

        smoke_config = TrainConfig.default_values().from_dict(base_config.to_dict())
        smoke_config.secrets.cloud.api_key = api_key
        smoke_config.secrets.cloud.user = (
            str(body.get("ssh_user") or smoke_config.secrets.cloud.user or "root").strip() or "root"
        )
        smoke_config.secrets.cloud.key_file = str(
            body.get("ssh_key_file") or smoke_config.secrets.cloud.key_file or ""
        ).strip()
        smoke_config.secrets.cloud.password = str(
            body.get("ssh_password") or smoke_config.secrets.cloud.password or ""
        ).strip()
        smoke_config.secrets.cloud.host = ""
        smoke_config.secrets.cloud.port = 0
        smoke_config.secrets.cloud.id = ""

        ssh_error = self._ssh_auth_error(smoke_config.secrets.cloud.key_file)
        if ssh_error:
            return {"ok": False, "error": ssh_error, "preview": preview}

        self._mutate_config_for_runpod(smoke_config, preview["required_gb"], body)
        smoke = self._build_live_test_cache(smoke_config, epochs=int(body.get("epochs") or 3))
        start_result = TrainerService.get_instance().start_training(reattach=False, train_config=smoke_config)
        if not start_result.get("ok"):
            return {"ok": False, "error": start_result.get("error", "Training start failed"), "preview": preview}

        return {"ok": True, "preview": preview, "smoke": smoke, "start": start_result}

    def _mutate_config_for_runpod(self, config: TrainConfig, volume_size: int, body: dict[str, Any]) -> None:
        config.cloud.enabled = True
        config.cloud.type = CloudType.RUNPOD
        config.cloud.file_sync = self._select_file_sync()
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
        # Respect the user's "Continue From Backup" choice: only resume when they asked for it AND a
        # backup actually exists locally to upload. If they did not ask, start a fresh sourceless run
        # and skip the backup upload entirely; if they asked but there is no backup, fall back to fresh
        # (continue_last_backup=True with no uploaded checkpoint would make the remote trainer fail).
        config.continue_last_backup = config.continue_last_backup and bool(self._latest_backup(config))
        config.clear_cache_before_training = False

    def _build_live_test_cache(self, config: TrainConfig, epochs: int) -> dict[str, Any]:
        source_cache = Path(config.cache_dir)
        image_index_path = source_cache / "image" / "cache.json"
        text_index_path = source_cache / "text" / "cache.json"
        if not image_index_path.is_file() or not text_index_path.is_file():
            raise RuntimeError(
                "Live test requires image/cache.json and text/cache.json in the current cache directory."
            )

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
        config.output_model_destination = str(
            Path(config.output_model_destination).with_name("runpod-live-test.safetensors")
        )
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

        copied = False
        variation = 1
        while True:
            pt_name = f"{cache_file}_{variation}.pt"
            source_pt = source_dir / pt_name
            if not source_pt.is_file():
                break
            shutil.copy2(source_pt, target_dir / pt_name)
            copied = True
            variation += 1
        if not copied:
            raise RuntimeError(f"Cache entry for {entry_path} has no cache tensors for {cache_file}")
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
        # The cache is the only hard requirement for sourceless training. The latest backup is
        # optional - its absence just means a fresh run instead of a continuation. The transformer
        # and base-model overrides are only uploaded when they are local files that exist; an empty
        # value or a Hugging Face repo id is downloaded on the pod instead, so they must not block.
        entries = [
            self._entry("cache", config.cache_dir, required=True),
        ]
        # the latest backup is only uploaded (and therefore only counts toward the volume) when the
        # user actually wants to continue from it; a fresh run skips it entirely.
        if config.continue_last_backup:
            entries.append(self._entry("latest_backup", self._latest_backup(config), required=False))

        for label, path in (
            ("transformer_override", config.transformer.model_name),
            ("base_model", config.base_model_name),
        ):
            entry = self._entry(label, path, required=False)
            if entry.exists:
                entries.append(entry)
        return entries

    def _entry(self, label: str, path: str | None, required: bool) -> SizeEntry:
        if not path:
            return SizeEntry(label=label, path="", exists=False, bytes=0, files=0, required=required)

        candidate = Path(path)
        if not candidate.exists():
            return SizeEntry(label=label, path=str(path), exists=False, bytes=0, files=0, required=required)

        if candidate.is_file():
            return SizeEntry(
                label=label,
                path=str(candidate),
                exists=True,
                bytes=candidate.stat().st_size,
                files=1,
                required=required,
            )

        total = 0
        count = 0
        for file in candidate.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
                count += 1
        return SizeEntry(label=label, path=str(candidate), exists=True, bytes=total, files=count, required=required)

    @staticmethod
    def _ssh_auth_error(key_file: str) -> str | None:
        # RunPod pod SSH is key-based, and NATIVE_RSYNC refuses password-only auth, so a usable
        # private key file is mandatory. Catch it here, before a billed pod is created.
        key_file = (key_file or "").strip()
        if not key_file:
            return (
                "RunPod requires SSH key authentication. Add your public key to your RunPod account "
                "and set 'SSH Key File' to the matching private key."
            )
        if not Path(key_file).expanduser().is_file():
            return f"SSH key file not found: {key_file}"
        return None

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
            backup_parent = CloudTrainer.adjust_path_to_remote_dir(
                str(Path(config.workspace_dir) / "backup"), remote_dir
            )
            result["latest_backup"] = f"{backup_parent}/{Path(latest).name}"
        return result

    @staticmethod
    def _validation_errors(config: TrainConfig, entries: list[SizeEntry]) -> list[str]:
        errors = [
            f"Missing required {entry.label}: {entry.path or '(empty)'}"
            for entry in entries
            if entry.required and not entry.exists
        ]
        if config.train_text_encoder_or_embedding():
            errors.append("Sourceless training cannot be used while text encoder or embedding training is enabled.")
        # The pod runs sourceless, so the cache must be metadata-complete before
        # we ship it. Catch an un-baked cache here — fail fast locally rather
        # than after a multi-hundred-GB upload and a billed pod start. Only
        # checked when the cache dir exists (its absence is already flagged
        # above).
        if any(entry.label == "cache" and entry.exists for entry in entries):
            errors.extend(sourceless_cache_problems(config.cache_dir))
        return errors

    @staticmethod
    def _select_file_sync() -> CloudFileSync:
        # Prefer native rsync (fast, resumable). On Windows, where rsync is absent and mis-parses
        # drive paths, delegate rsync to WSL if available; otherwise fall back to pure-Python SFTP,
        # which always works. There is therefore never a hard "no transfer available" failure.
        if shutil.which("rsync") is not None:
            return CloudFileSync.NATIVE_RSYNC
        if os.name == "nt" and RunpodSetupService._wsl_rsync_available():
            return CloudFileSync.WSL_RSYNC
        return CloudFileSync.FABRIC_SFTP

    @staticmethod
    def _wsl_rsync_available() -> bool:
        wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if wsl is None:
            return False
        try:
            result = subprocess.run(
                [wsl, "-e", "bash", "-lc", "command -v rsync"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    @staticmethod
    def _transfer_label(file_sync: CloudFileSync) -> str:
        return {
            CloudFileSync.NATIVE_RSYNC: "native rsync",
            CloudFileSync.WSL_RSYNC: "rsync via WSL (Windows backend)",
            CloudFileSync.FABRIC_SFTP: "SFTP (rsync not found)",
        }.get(file_sync, file_sync.name)

    @staticmethod
    def _warnings(config: TrainConfig) -> list[str]:
        warnings: list[str] = []
        chosen = RunpodSetupService._select_file_sync()
        warnings.append(f"RunPod uploads will use {RunpodSetupService._transfer_label(chosen)}.")
        if config.only_cache:
            warnings.append("Only Cache is enabled locally; RunPod setup will disable it before starting training.")
        if config.continue_last_backup and not RunpodSetupService._latest_backup(config):
            warnings.append(
                "No local backup found - RunPod will start a fresh sourceless run from the base model instead of continuing."
            )
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
