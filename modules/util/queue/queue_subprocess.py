"""Run a single queue entry in an isolated subprocess.

Each entry trains in a brand-new Python interpreter so the OS reclaims *all* of
its memory (model weights, optimizer state, dataloader caches, native CUDA /
safetensors allocations) the instant the process exits. This is what keeps RAM
from spiralling across a chain of runs in the long-lived backend/queue process,
and it contains hard native crashes (e.g. 0xC0000005) to the child instead of
taking the whole backend down with them.

Progress / status events flow child -> parent and stop commands flow
parent -> child over a localhost TCP socket using newline-delimited JSON.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from modules.util.TrainProgress import TrainProgress

# modules/util/queue/queue_subprocess.py -> parents[3] == project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WORKER_SCRIPT = _PROJECT_ROOT / "scripts" / "train_queue_worker.py"

# How long to wait for the child to connect back before giving up on it.
_ACCEPT_TIMEOUT_S = 120.0
# Grace period after a stop request before we forcibly terminate the child.
_STOP_GRACE_S = 60.0


class SubprocessResult:
    """Outcome of an isolated run, in the vocabulary the QueueExecutor expects."""

    def __init__(
        self,
        *,
        completed: bool,
        stopped: bool,
        error_message: str | None = None,
        error_category: str | None = None,
        returncode: int | None = None,
    ) -> None:
        self.completed = completed
        self.stopped = stopped
        self.error_message = error_message
        self.error_category = error_category
        self.returncode = returncode


class QueueSubprocessRunner:
    """Launches the worker for one entry and bridges its IPC to callbacks.

    ``run()`` blocks on the calling (queue) thread until the child finishes.
    ``stop()`` is thread-safe and may be called from another thread to ask the
    child to stop gracefully (it will still save / back up as configured).
    """

    def __init__(
        self,
        config_dict: dict,
        on_progress: Callable[[TrainProgress, int, int], None],
        on_status: Callable[[str], None],
        on_output: Callable[[str], None] | None = None,
    ) -> None:
        self._config_dict = config_dict
        self._on_progress = on_progress
        self._on_status = on_status
        # Default: echo child stdout through our own stdout so it still reaches
        # the console and (in the web backend) the captured terminal/log stream.
        self._on_output = on_output or self._default_output
        self._process: subprocess.Popen | None = None
        self._conn: socket.socket | None = None
        self._conn_lock = threading.Lock()
        self._stop_requested = False

    @staticmethod
    def _default_output(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def run(self) -> SubprocessResult:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        config_path: str | None = None
        try:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(_ACCEPT_TIMEOUT_S)
            port = server.getsockname()[1]

            config_path = self._write_config()
            self._process = self._spawn(config_path, port)

            output_thread = threading.Thread(target=self._pump_output, daemon=True, name="queue-child-output")
            output_thread.start()

            try:
                conn, _ = server.accept()
            except TimeoutError:
                return self._handle_no_connection()
            with self._conn_lock:
                self._conn = conn
                # A stop arriving between spawn and connect still needs delivering.
                if self._stop_requested:
                    self._send_stop_locked()

            result = self._event_loop(conn)
            self._process.wait()
            output_thread.join(timeout=5.0)
            return self._finalize(result)
        finally:
            self._cleanup(server, config_path)

    def stop(self) -> None:
        with self._conn_lock:
            self._stop_requested = True
            if self._conn is not None:
                self._send_stop_locked()

    # --- internals ---

    def _write_config(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="ot_queue_entry_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._config_dict, fh)
        return path

    def _spawn(self, config_path: str, port: int) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-u", str(_WORKER_SCRIPT), "--config-path", config_path, "--ipc-port", str(port)],
            cwd=str(_PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def _send_stop_locked(self) -> None:
        with contextlib.suppress(OSError):
            self._conn.sendall(b'{"cmd": "stop"}\n')

    def _pump_output(self) -> None:
        assert self._process is not None
        stream = self._process.stdout
        if stream is None:
            return
        for line in stream:
            self._on_output(line)

    def _event_loop(self, conn: socket.socket) -> SubprocessResult | None:
        """Read child events until the terminal result/error frame or EOF."""
        buffer = b""
        result: SubprocessResult | None = None
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    terminal = self._dispatch(line)
                    if terminal is not None:
                        result = terminal
        except OSError:
            pass
        return result

    def _dispatch(self, line: bytes) -> SubprocessResult | None:
        try:
            event = json.loads(line.decode("utf-8"))
        except ValueError:
            return None
        kind = event.get("type")
        if kind == "progress":
            progress = TrainProgress(
                epoch=int(event.get("epoch", 0)),
                global_step=int(event.get("global_step", 0)),
            )
            self._on_progress(progress, int(event.get("max_step", 0)), int(event.get("max_epoch", 0)))
            return None
        if kind == "status":
            self._on_status(str(event.get("message", "")))
            return None
        if kind == "result":
            return SubprocessResult(completed=True, stopped=bool(event.get("stopped")))
        if kind == "error":
            return SubprocessResult(
                completed=False,
                stopped=bool(event.get("stopped")),
                error_message=str(event.get("message", "Unknown error")),
                error_category=str(event.get("category", "other")),
            )
        return None

    def _handle_no_connection(self) -> SubprocessResult:
        # Child never connected within the timeout — kill it and report failure.
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        rc = self._process.wait() if self._process is not None else None
        return SubprocessResult(
            completed=False,
            stopped=self._stop_requested,
            error_message=f"Training subprocess never connected back (exit code {rc}).",
            error_category="other",
            returncode=rc,
        )

    def _finalize(self, result: SubprocessResult | None) -> SubprocessResult:
        rc = self._process.returncode if self._process is not None else None
        if result is not None:
            result.returncode = rc
            return result
        # Socket closed with no terminal frame: the child died mid-run (native
        # crash / OOM access violation / killed). Isolation means this no longer
        # crashes the backend — we record it and let the queue's retry logic run.
        if self._stop_requested:
            return SubprocessResult(completed=False, stopped=True, returncode=rc)
        return SubprocessResult(
            completed=False,
            stopped=False,
            error_message=(
                f"Training subprocess exited unexpectedly (exit code {rc}). "
                "This usually means the run crashed natively (e.g. out-of-memory / access violation)."
            ),
            error_category="other",
            returncode=rc,
        )

    def _cleanup(self, server: socket.socket, config_path: str | None) -> None:
        with self._conn_lock:
            if self._conn is not None:
                with contextlib.suppress(OSError):
                    self._conn.shutdown(socket.SHUT_RDWR)
                self._conn.close()
                self._conn = None
        with contextlib.suppress(OSError):
            server.close()
        # If the child is somehow still alive (e.g. ignored a stop), don't leak it.
        if self._process is not None and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=_STOP_GRACE_S)
            except (OSError, subprocess.TimeoutExpired):
                with contextlib.suppress(OSError):
                    self._process.kill()
        if config_path is not None:
            with contextlib.suppress(OSError):
                os.remove(config_path)
