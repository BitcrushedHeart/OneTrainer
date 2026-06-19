from util.import_util import script_imports

script_imports()

import argparse
import contextlib
import json
import os
import socket
import threading
import traceback

from modules.util import create
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import TrainConfig


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


def _start_control_reader(sock: socket.socket, commands: TrainCommands, stop_flag: threading.Event) -> None:
    """Read control frames ({"cmd": "stop"}) from the parent on a daemon thread."""

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

    threading.Thread(target=_run, daemon=True, name="queue-ipc-control").start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single OneTrainer queue entry in an isolated process.")
    parser.add_argument("--config-path", required=True, help="Path to the merged TrainConfig JSON for this entry.")
    parser.add_argument("--ipc-port", type=int, required=True, help="Localhost TCP port the parent is listening on.")
    args = parser.parse_args()

    sock = socket.create_connection(("127.0.0.1", args.ipc_port))
    channel = _EventChannel(sock)

    commands = TrainCommands()
    stop_flag = threading.Event()
    _start_control_reader(sock, commands, stop_flag)

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
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
        sock.close()


if __name__ == "__main__":
    main()
