from pathlib import Path

from modules.cloud.BaseCloud import UPLOAD_EXCLUDE
from modules.cloud.FabricFileSync import FabricFileSync
from modules.cloud.NativeRsyncFileSync import NativeRsyncFileSync


class _Result:
    exited = 0
    stdout = ""


class _FakeConn:
    def open(self):
        pass

    def run(self, cmd, **kwargs):
        return _Result()


def _make_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    (cache / "text").mkdir(parents=True)
    for name in ("cache.json", "cache.json.bak", "cache.json.pre-scrub-1.bak", "a_1.pt", "a_1.pt.123.456.tmp"):
        (cache / "text" / name).write_bytes(b"x")
    return cache


def test_upload_exclude_covers_index_backups_and_partial_writes():
    assert "*.bak" in UPLOAD_EXCLUDE
    assert "*.tmp" in UPLOAD_EXCLUDE


def test_rsync_upload_passes_excludes(tmp_path):
    sync = NativeRsyncFileSync.__new__(NativeRsyncFileSync)
    sync.sync_connection = _FakeConn()
    sync.ssh_cmd = "ssh"
    sync._remote = lambda path: f"root@host:{path.as_posix()}"
    runs = []
    sync._run = runs.append

    sync.sync_up_dir_stream(local=_make_cache(tmp_path), remote=Path("/workspace/cache"), exclude=UPLOAD_EXCLUDE)

    args = runs[0]
    for pattern in UPLOAD_EXCLUDE:
        assert ["--exclude", pattern] == args[args.index(pattern) - 1 : args.index(pattern) + 1]


def test_ssh_upload_skips_excluded_files(tmp_path):
    sync = FabricFileSync.__new__(FabricFileSync)
    sync.sync_connection = _FakeConn()
    uploaded = []
    sync.upload_files = lambda local_files, remote_dir: uploaded.extend(path.name for path in local_files)

    sync.sync_up_dir(
        local=_make_cache(tmp_path), remote=Path("/workspace/cache"), recursive=True, exclude=UPLOAD_EXCLUDE
    )

    assert sorted(uploaded) == ["a_1.pt", "cache.json"]
