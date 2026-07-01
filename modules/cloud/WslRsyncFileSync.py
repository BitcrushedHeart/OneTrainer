import re
import shlex
import shutil
import subprocess

from modules.cloud.BaseSSHFileSync import BaseSSHFileSync
from modules.cloud.NativeRsyncFileSync import NativeRsyncFileSync
from modules.util.config.CloudConfig import CloudConfig, CloudSecretsConfig

# matches a Windows drive path like  F:\foo\bar  or  C:/foo  (but not a remote spec like root@host:/p)
_WINDOWS_DRIVE_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


class WslRsyncFileSync(NativeRsyncFileSync):
    """rsync transfer that runs the rsync (and its `-e ssh`) inside WSL via `wsl.exe`.

    This lets the OneTrainer backend stay on native Windows while still using rsync: the only
    Windows-incompatible parts of NATIVE_RSYNC are that rsync is not on the Windows PATH and that
    rsync mis-parses `F:\\...` drive paths as remote specs. Both are solved by delegating the rsync
    invocation to WSL and translating local drive paths to `/mnt/<drive>/...` first. Control ops
    (mkdir -p, sync-info) keep using the Windows-side paramiko connection from BaseSSHFileSync.
    """

    def __init__(self, config: CloudConfig, secrets: CloudSecretsConfig):
        # deliberately skip NativeRsyncFileSync.__init__ (it requires rsync on the Windows PATH and
        # builds a Windows-quoted ssh command); set up the WSL-side equivalents instead.
        BaseSSHFileSync.__init__(self, config, secrets)

        self.wsl = shutil.which("wsl.exe") or shutil.which("wsl")
        if self.wsl is None:
            raise RuntimeError("WSL_RSYNC selected, but wsl.exe was not found on PATH")

        key_file = secrets.expanded_key_file()
        if not key_file:
            raise RuntimeError("WSL_RSYNC requires SSH key authentication (set an SSH key file)")

        self.wsl_rsync, self.wsl_ssh = self._resolve_wsl_tools()
        self.wsl_key = self._stage_key_in_wsl(self._to_wsl_path(key_file))

        self.ssh_cmd = (
            f"{self.wsl_ssh} -p {int(secrets.port)} "
            f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {self.wsl_key}"
        )

    def _resolve_wsl_tools(self) -> tuple[str, str]:
        result = subprocess.run(
            [self.wsl, "-e", "bash", "-lc", "command -v rsync && command -v ssh"],
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or len(lines) < 2:
            raise RuntimeError(
                "WSL_RSYNC selected, but rsync/ssh are not installed in WSL "
                "(run: sudo apt-get install -y rsync openssh-client)"
            )
        return lines[0], lines[1]

    def _stage_key_in_wsl(self, wsl_src_key: str) -> str:
        # ssh refuses private keys looser than 600, and keys on /mnt/* show as 0777, so copy the key
        # into WSL's native ~/.ssh with strict perms and use that path.
        script = (
            'd="$HOME/.ssh"; mkdir -p "$d"; chmod 700 "$d"; '
            f'install -m 600 {shlex.quote(wsl_src_key)} "$d/onetrainer_runpod_key" '
            '&& printf "%s" "$d/onetrainer_runpod_key"'
        )
        result = subprocess.run([self.wsl, "-e", "bash", "-lc", script], capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"WSL_RSYNC could not stage the SSH key in WSL: {result.stderr.strip()}")
        return result.stdout.strip()

    @staticmethod
    def _to_wsl_path(arg: str) -> str:
        match = _WINDOWS_DRIVE_PATH.match(arg)
        if not match:
            return arg
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    def _run(self, args: list[str]) -> None:
        # args[0] is always "rsync"; replace it with the resolved WSL binary and translate any local
        # Windows drive paths in the remaining arguments to /mnt/<drive>/... form.
        translated = [self._to_wsl_path(arg) for arg in args[1:]]
        # Capture stderr so a non-zero exit surfaces rsync's actual "rsync: ..." diagnostic line
        # (e.g. a failed chown/chmod/utime on a restrictive volume) instead of a bare exit code;
        # stdout keeps streaming so -v progress still shows live.
        result = subprocess.run([self.wsl, "-e", self.wsl_rsync, *translated], stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or "<no stderr captured>"
            raise RuntimeError(f"rsync failed (exit {result.returncode}):\n{detail}")
