import os
import shutil
import subprocess
from pathlib import Path

from modules.cloud.BaseSSHFileSync import BaseSSHFileSync
from modules.util.config.CloudConfig import CloudConfig, CloudSecretsConfig


class NativeRsyncFileSync(BaseSSHFileSync):
    def __init__(self, config: CloudConfig, secrets: CloudSecretsConfig):
        super().__init__(config, secrets)
        if shutil.which("rsync") is None:
            raise RuntimeError("NATIVE_RSYNC selected, but rsync was not found on PATH")
        if getattr(secrets, "password", "").strip() and not secrets.expanded_key_file():
            raise RuntimeError("NATIVE_RSYNC requires SSH key authentication, or an ssh-agent/interactive SSH setup.")
        ssh_parts = ["ssh", "-p", str(secrets.port), "-o", "StrictHostKeyChecking=no"]
        key_file = secrets.expanded_key_file()
        if key_file:
            ssh_parts.extend(["-i", key_file])
        self.ssh_cmd = " ".join(subprocess.list2cmdline([part]) for part in ssh_parts)

    def _remote(self, path: Path) -> str:
        return f"{self.secrets.user}@{self.secrets.host}:{path.as_posix()}"

    def _run(self, args: list[str]) -> None:
        subprocess.run(args).check_returncode()

    def upload_files(self, local_files, remote_dir: Path):
        if not local_files:
            return
        self.sync_connection.open()
        self.sync_connection.run(f"mkdir -p {self._quote_remote(remote_dir)}", in_stream=False)
        self._run(["rsync", "-av", "-e", self.ssh_cmd, *map(str, local_files), self._remote(remote_dir)])

    def download_files(self, local_dir: Path, remote_files):
        if not remote_files:
            return
        local_dir.mkdir(parents=True, exist_ok=True)
        self._run(["rsync", "-av", "-e", self.ssh_cmd, *[self._remote(path) for path in remote_files], str(local_dir)])

    def upload_file(self, local_file: Path, remote_file: Path):
        self.sync_up_file(local=local_file, remote=remote_file)

    def download_file(self, local_file: Path, remote_file: Path):
        self.sync_down_file(local=local_file, remote=remote_file)

    def sync_up_file(self, local: Path, remote: Path):
        self.sync_connection.open()
        self.sync_connection.run(f"mkdir -p {self._quote_remote(remote.parent)}", in_stream=False)
        self._run(["rsync", "-av", "-e", self.ssh_cmd, str(local), self._remote(remote)])

    def sync_up_dir(
        self,
        local: Path,
        remote: Path,
        recursive: bool,
        sync_info=None,
        skip_hidden: bool = False,
        allowed_extensions: set[str] | None = None,
    ):
        if not recursive:
            self.sync_connection.open()
            self.sync_connection.run(f"mkdir -p {self._quote_remote(remote)}", in_stream=False)
            files = [
                path
                for path in local.iterdir()
                if path.is_file()
                and (allowed_extensions is None or path.suffix.lower() in allowed_extensions)
                and (not skip_hidden or not path.name.startswith("."))
            ]
            if files:
                self._run(["rsync", "-av", "-e", self.ssh_cmd, *map(str, files), self._remote(remote)])
            return

        args = ["rsync", "-av", "--delete", "-e", self.ssh_cmd]
        if skip_hidden:
            args.extend(["--exclude", ".*"])
        if allowed_extensions is not None:
            args.extend(self._include_extension_filter(allowed_extensions))
        args.extend([self._dir_source(local), self._remote(remote)])
        self._run(args)

    def sync_up_dir_stream(self, local: Path, remote: Path) -> bool:
        self.sync_up_dir(local=local, remote=remote, recursive=True)
        return True

    def sync_down_file(self, local: Path, remote: Path):
        local.parent.mkdir(parents=True, exist_ok=True)
        self._run(["rsync", "-av", "-e", self.ssh_cmd, self._remote(remote), str(local)])

    def sync_down_dir(self, local: Path, remote: Path, filter=None):
        if filter is None:
            local.mkdir(parents=True, exist_ok=True)
            self._run(["rsync", "-av", "--delete", "-e", self.ssh_cmd, self._remote(remote) + "/", str(local)])
            return

        super().sync_down_dir(local=local, remote=remote, filter=filter)

    @staticmethod
    def _dir_source(path: Path) -> str:
        text = str(path)
        return text if text.endswith(os.sep) else text + os.sep

    @staticmethod
    def _include_extension_filter(allowed_extensions: set[str]) -> list[str]:
        args = ["--include", "*/"]
        for ext in sorted(allowed_extensions):
            args.extend(["--include", f"*{ext}"])
        args.extend(["--exclude", "*"])
        return args

    @staticmethod
    def _quote_remote(path: Path) -> str:
        import shlex

        return shlex.quote(path.as_posix())
