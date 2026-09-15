"""Regression tests for the queue worker's exit path.

A worker that finished its run once hung forever in interpreter teardown (tqdm's
monitor-thread join on a stuck write lock). The parent blocked on an unbounded
wait, and a backend restart left the child orphaned with the whole run's RAM
paged out, starving the next run. These tests pin the three fixes: the worker
hard-exits after reporting, it stops when its parent disappears, and the parent
bounds its wait on the child.
"""

import importlib.util
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

from modules.util.commands.TrainCommands import TrainCommands
from modules.util.queue import queue_subprocess

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
_WORKER = _SCRIPTS / "train_queue_worker.py"


def _load_worker():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("train_queue_worker_under_test", _WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_worker_disables_tqdm_monitor_thread():
    from tqdm import tqdm

    _load_worker()
    assert tqdm.monitor_interval == 0


def test_parent_disconnect_stops_run():
    worker = _load_worker()
    parent_end, child_end = socket.socketpair()
    commands = TrainCommands()
    stop_flag, parent_gone, run_finished = threading.Event(), threading.Event(), threading.Event()
    worker._start_control_reader(child_end, commands, stop_flag, parent_gone, run_finished)

    parent_end.close()

    assert _wait_for(stop_flag.is_set)
    assert parent_gone.is_set()
    assert commands.get_stop_command()
    child_end.close()


def test_socket_close_after_finish_is_not_a_parent_loss():
    worker = _load_worker()
    parent_end, child_end = socket.socketpair()
    commands = TrainCommands()
    stop_flag, parent_gone, run_finished = threading.Event(), threading.Event(), threading.Event()
    worker._start_control_reader(child_end, commands, stop_flag, parent_gone, run_finished)

    run_finished.set()
    child_end.close()
    parent_end.close()

    assert not _wait_for(stop_flag.is_set, timeout=0.5)
    assert not parent_gone.is_set()
    assert not commands.get_stop_command()


def test_explicit_stop_is_not_reported_as_parent_loss():
    worker = _load_worker()
    parent_end, child_end = socket.socketpair()
    commands = TrainCommands()
    stop_flag, parent_gone, run_finished = threading.Event(), threading.Event(), threading.Event()
    worker._start_control_reader(child_end, commands, stop_flag, parent_gone, run_finished)

    parent_end.sendall(b'{"cmd": "stop"}\n')
    assert _wait_for(stop_flag.is_set)
    parent_end.close()

    assert not _wait_for(parent_gone.is_set, timeout=0.5)
    child_end.close()


def test_exit_now_terminates_despite_hanging_teardown():
    """A wedged atexit handler must not keep the worker alive past its grace period."""
    code = textwrap.dedent(
        f"""
        import atexit, importlib.util, sys, threading, time
        sys.path.insert(0, {str(_SCRIPTS)!r})
        spec = importlib.util.spec_from_file_location("w", {str(_WORKER)!r})
        w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(w)
        w._EXIT_GRACE_S = 2.0
        atexit.register(lambda: time.sleep(600))
        # A non-daemon thread that never ends would also block normal shutdown.
        threading.Thread(target=lambda: time.sleep(600)).start()
        w._exit_now(7)
        """
    )
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(_ROOT), timeout=150)
    assert proc.returncode == 7
    # Generous bound: import of the worker (torch) dominates, the grace itself is 2s.
    assert time.monotonic() - started < 150


def test_parent_kills_child_that_never_exits(monkeypatch):
    monkeypatch.setattr(queue_subprocess, "_EXIT_GRACE_S", 1.0)
    monkeypatch.setattr(queue_subprocess, "_STOP_GRACE_S", 10.0)
    statuses: list[str] = []
    runner = queue_subprocess.QueueSubprocessRunner(
        config_dict={},
        on_progress=lambda *_: None,
        on_status=statuses.append,
    )
    runner._process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        started = time.monotonic()
        runner._wait_for_exit()
        assert runner._process.poll() is not None
        assert time.monotonic() - started < 15
        assert any("did not exit" in s for s in statuses)
    finally:
        if runner._process.poll() is None:
            runner._process.kill()
