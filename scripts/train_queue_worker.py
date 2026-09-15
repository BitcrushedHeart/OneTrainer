from util.import_util import script_imports

script_imports()

import argparse
import atexit
import contextlib
import json
import logging
import os
import socket
import sys
import threading
import traceback

from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig

from tqdm import tqdm

# tqdm's monitor thread is joined during interpreter teardown. If its write lock is
# ever left held (a bar closed from another thread, a worker killed mid-write) that
# join never returns and this process outlives its parent as a zombie that still
# owns every byte of RAM the run allocated. Disable the monitor outright; we also
# hard-exit below so no other teardown hang can keep the process alive.
tqdm.monitor_interval = 0

# How long the orderly exit path (atexit handlers, log/stream flush) gets before
# the process is terminated regardless.
_EXIT_GRACE_S = 30.0


def _categorize(error: BaseException) -> str:
    """Classify a training failure so the parent queue can apply its retry/skip rules.

    Mirrors QueueExecutor.is_*_error so the same exception buckets survive the
    process boundary without shipping a pickled exception back.
    """
    message = str(error).lower()
    if isinstance(error, (FileNotFoundError, IsADirectoryError)):
        return "file_not_found"
    if isinstance(error, RuntimeError) and "out of memory" in message:
        return "oom"
    if isinstance(error, RuntimeError) and "nan" in message:
        return "nan"
    return "other"


class _EventChannel:
    """Newline-delimited JSON over the IPC socket. Sends are serialized so the
    training thread and any callback thread can't interleave a frame."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._lock = threading.Lock()

    def send(self, obj: dict) -> None:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        # Parent went away; nothing we can do but keep training/teardown going.
        with self._lock, contextlib.suppress(OSError):
            self._sock.sendall(data)


def _start_control_reader(
    sock: socket.socket,
    commands: TrainCommands,
    stop_flag: threading.Event,
    parent_gone: threading.Event,
    run_finished: threading.Event,
) -> None:
    """Read control frames ({"cmd": "stop"}) from the parent on a daemon thread.

    EOF on this socket before ``run_finished`` is set means the parent is gone
    (the backend was restarted or crashed). An orphan that keeps training has no
    UI, no stop button, and fights the next run for RAM and VRAM, so we treat it
    as a stop request: the trainer finishes the current step, saves / backs up
    as configured, and the process exits.
    """

    def _run() -> None:
        buffer = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line.decode("utf-8"))
                    except ValueError:
                        continue
                    if message.get("cmd") == "stop":
                        stop_flag.set()
                        commands.stop()
        except OSError:
            pass
        finally:
            if not run_finished.is_set() and not stop_flag.is_set():
                parent_gone.set()
                stop_flag.set()
                commands.stop()
                print("Queue parent disconnected; stopping this run gracefully.", flush=True)

    threading.Thread(target=_run, daemon=True, name="queue-ipc-control").start()


def _exit_now(code: int) -> None:
    """Leave the process without relying on interpreter teardown finishing.

    The run is already reported to the parent by the time this is called; the only
    remaining work is best-effort flushing. A watchdog guarantees the exit even if
    an atexit handler or a stuck lock would otherwise hang finalization.
    """

    def _force() -> None:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(code)

    watchdog = threading.Timer(_EXIT_GRACE_S, _force)
    watchdog.daemon = True
    watchdog.start()
    with contextlib.suppress(Exception):
        atexit._run_exitfuncs()
    with contextlib.suppress(Exception):
        logging.shutdown()
    _force()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single OneTrainer queue entry in an isolated process.")
    parser.add_argument("--config-path", required=True, help="Path to the merged TrainConfig JSON for this entry.")
    parser.add_argument("--ipc-port", type=int, required=True, help="Localhost TCP port the parent is listening on.")
    args = parser.parse_args()

    sock = socket.create_connection(("127.0.0.1", args.ipc_port))
    channel = _EventChannel(sock)

    commands = TrainCommands()
    stop_flag = threading.Event()
    parent_gone = threading.Event()
    run_finished = threading.Event()
    _start_control_reader(sock, commands, stop_flag, parent_gone, run_finished)

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r", encoding="utf-8") as fh:
        train_config.from_dict(json.load(fh))

    if train_config.skip_cache_validation:
        os.environ["OT_SKIP_CACHE_VALIDATION"] = "1"

    def on_progress(train_progress, max_step: int, max_epoch: int) -> None:
        channel.send(
            {
                "type": "progress",
                "global_step": getattr(train_progress, "global_step", -1),
                "epoch": getattr(train_progress, "epoch", -1),
                "max_step": max_step,
                "max_epoch": max_epoch,
            }
        )

    def on_status(message: str) -> None:
        channel.send({"type": "status", "message": message})

    callbacks = TrainCallbacks(
        on_update_train_progress=on_progress,
        on_update_status=on_status,
    )

    error: BaseException | None = None
    trainer = create.create_trainer(train_config, callbacks, commands)
    try:
        try:
            trainer.start()
            trainer.train()
        except Exception as train_err:  # noqa: BLE001 — reported to parent, re-raised semantics preserved below
            error = train_err
        try:
            trainer.end()
        except Exception as end_err:  # noqa: BLE001
            if error is None:
                error = end_err
            else:
                traceback.print_exc()
    finally:
        if error is not None:
            traceback.print_exception(type(error), error, error.__traceback__)
            channel.send(
                {
                    "type": "error",
                    "message": str(error),
                    "category": _categorize(error),
                    "stopped": stop_flag.is_set(),
                }
            )
        else:
            channel.send({"type": "result", "status": "completed", "stopped": stop_flag.is_set()})
        run_finished.set()
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        sock.close()
    return 0 if error is None else 1


if __name__ == "__main__":
    _exit_now(main())
