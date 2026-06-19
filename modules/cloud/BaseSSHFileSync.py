import shlex
import shutil
import subprocess
from abc import abstractmethod
from pathlib import Path

from modules.cloud.BaseFileSync import BaseFileSync
from modules.util.config.CloudConfig import CloudConfig, CloudSecretsConfig

import fabric


class BaseSSHFileSync(BaseFileSync):
    def __init__(self, config: CloudConfig, secrets: CloudSecretsConfig):
        super().__init__(config, secrets)
        self.sync_connection = fabric.Connection(
            host=secrets.host, port=secrets.port, user=secrets.user, connect_kwargs=secrets.connect_kwargs()
        )

    def close(self):
        if self.sync_connection:
            self.sync_connection.close()

    @abstractmethod
    def upload_files(self, local_files, remote_dir: Path):
        pass

    @abstractmethod
    def download_files(self, local_dir: Path, remote_files):
        pass

    @abstractmethod
    def upload_file(self, local_file: Path, remote_file: Path):
        pass

    @abstractmethod
    def download_file(self, local_file: Path, remote_file: Path):
        pass

    def sync_up_file(self, local: Path, remote: Path):
        sync_info = self.__get_sync_info(remote)
        if not self.__needs_upload(local=local, remote=remote, sync_info=sync_info):
            return

        self.sync_connection.open()
        self.sync_connection.run(f"mkdir -p {shlex.quote(remote.parent.as_posix())}", in_stream=False)
        self.upload_file(local_file=local, remote_file=remote)

    def sync_up_dir(
        self,
        local: Path,
        remote: Path,
        recursive: bool,
        sync_info=None,
        skip_hidden: bool = False,
        allowed_extensions: set[str] | None = None,
    ):
        if sync_info is None:
            sync_info = self.__get_sync_info(remote)
        self.sync_connection.open()
        self.sync_connection.run(f"mkdir -p {shlex.quote(remote.as_posix())}", in_stream=False)
        files = []
        for local_entry in local.iterdir():
            if local_entry.is_file():
                if allowed_extensions is not None and local_entry.suffix.lower() not in allowed_extensions:
                    continue
                remote_entry = remote / local_entry.name
                if self.__needs_upload(local=local_entry, remote=remote_entry, sync_info=sync_info):
                    files.append(local_entry)
            elif recursive and local_entry.is_dir():
                if skip_hidden and local_entry.name.startswith("."):
                    continue
                self.sync_up_dir(
                    local=local_entry,
                    remote=remote / local_entry.name,
                    recursive=True,
                    sync_info=sync_info,
                    skip_hidden=skip_hidden,
                    allowed_extensions=allowed_extensions,
                )

        self.upload_files(local_files=files, remote_dir=remote)

    def sync_up_dir_stream(self, local: Path, remote: Path) -> bool:
        if not local.is_dir() or shutil.which("tar") is None:
            return False

        self.sync_connection.open()
        has_remote_tar = self.sync_connection.run("command -v tar", warn=True, hide=True, in_stream=False)
        if has_remote_tar.exited != 0:
            return False

        local_size = self.__local_file_size_sum(local)
        remote_posix = remote.as_posix()
        command = (
            f"rm -rf -- {shlex.quote(remote_posix)} "
            f"&& mkdir -p {shlex.quote(remote_posix)} "
            f"&& tar -xf - -C {shlex.quote(remote_posix)} "
            f"&& find {shlex.quote(remote_posix)} -type f -exec stat --printf '%s\\n' {{}} \\; "
            "| awk '{s+=$1} END {print s+0}'"
        )

        transport = self.sync_connection.client.get_transport()
        channel = transport.open_session()
        channel.exec_command(command)

        proc = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(local), "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        try:
            for chunk in iter(lambda: proc.stdout.read(1024 * 1024), b""):
                channel.sendall(chunk)
            channel.shutdown_write()
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr is not None else ""
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(stderr or "local tar failed")

            remote_output = b""
            remote_error = b""
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    remote_output += channel.recv(65536)
                if channel.recv_stderr_ready():
                    remote_error += channel.recv_stderr(65536)
            while channel.recv_ready():
                remote_output += channel.recv(65536)
            while channel.recv_stderr_ready():
                remote_error += channel.recv_stderr(65536)

            exit_status = channel.recv_exit_status()
            if exit_status != 0:
                raise RuntimeError(remote_error.decode(errors="replace") or "remote tar failed")
            remote_size = int(remote_output.decode(errors="replace").strip().splitlines()[-1])
            if remote_size != local_size:
                raise RuntimeError(
                    f"streamed upload verification failed for {local}: local={local_size} remote={remote_size}"
                )
            return True
        finally:
            channel.close()

    def sync_down_file(self, local: Path, remote: Path):
        sync_info = self.__get_sync_info(remote)
        if not self.__needs_download(local=local, remote=remote, sync_info=sync_info):
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        self.download_file(local_file=local, remote_file=remote)

    def sync_down_dir(self, local: Path, remote: Path, filter=None):
        sync_info = self.__get_sync_info(remote)
        dirs = {}
        for remote_entry in sync_info:
            local_entry = local / remote_entry.relative_to(remote)
            if (filter is not None and not filter(remote_entry)) or not self.__needs_download(
                local=local_entry, remote=remote_entry, sync_info=sync_info
            ):
                continue

            if local_entry.parent not in dirs:
                dirs[local_entry.parent] = []
            dirs[local_entry.parent].append(remote_entry)

        for dir, files in dirs.items():
            dir.mkdir(parents=True, exist_ok=True)
            self.download_files(local_dir=dir, remote_files=files)

    def __get_sync_info(self, remote: Path):
        cmd = f'find {shlex.quote(remote.as_posix())} -type f -exec stat --printf "%n\\t%s\\t%Y\\n"' + " {} \\;"
        self.sync_connection.open()
        result = self.sync_connection.run(cmd, warn=True, hide=True, in_stream=False)
        info = {}
        for line in result.stdout.splitlines():
            sp = line.split("\t")
            info[Path(sp[0])] = {"size": int(sp[1]), "mtime": int(sp[2])}
        return info

    @staticmethod
    def __needs_upload(local: Path, remote: Path, sync_info):
        return (
            remote not in sync_info
            or local.stat().st_size != sync_info[remote]["size"]
            or int(local.stat().st_mtime) > sync_info[remote]["mtime"]
        )

    @staticmethod
    def __needs_download(local: Path, remote: Path, sync_info):
        return (
            not local.exists()
            or remote not in sync_info
            or local.stat().st_size != sync_info[remote]["size"]
            or local.stat().st_mtime < sync_info[remote]["mtime"]
        )

    @staticmethod
    def __local_file_size_sum(local: Path) -> int:
        total = 0
        for path in local.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total
