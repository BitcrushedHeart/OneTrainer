"""The Dynamo precompile cache flag has to reach every compiling thread.

Since torch 2.12 the dynamo config is a ContextVar, so a value set on the
training thread is invisible to the autograd worker threads that recompute
checkpointed blocks. init_compile() is what those threads call, so it must
re-apply the flag; and the queue parent must hand the matching process default
to its child before torch is imported there.
"""

import threading

from modules.util import compile_util
from modules.util.queue.queue_subprocess import QueueSubprocessRunner

import torch


def _flag_seen_in_new_thread() -> bool:
    seen = {}

    def _run():
        compile_util.init_compile()
        seen["value"] = torch._dynamo.config.caching_precompile

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()
    return seen["value"]


def test_init_compile_applies_precompile_flag_per_thread(monkeypatch):
    monkeypatch.setattr(compile_util, "_precompile_cache_enabled", False)
    compile_util.set_precompile_cache(True)
    assert compile_util.precompile_cache_enabled()
    assert _flag_seen_in_new_thread() is True

    compile_util.set_precompile_cache(False)
    assert _flag_seen_in_new_thread() is False


def test_save_precompile_cache_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(compile_util, "_precompile_cache_enabled", False)
    called = []
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: called.append(True))
    compile_util.save_precompile_cache("test")
    assert called == []


def test_save_precompile_cache_survives_torch_errors(monkeypatch):
    monkeypatch.setattr(compile_util, "_precompile_cache_enabled", True)

    def _boom():
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", _boom)
    compile_util.save_precompile_cache("test")  # must not raise: the run goes on without the cache


def _child_env(config: dict) -> str:
    runner = QueueSubprocessRunner(config, on_progress=lambda *_: None, on_status=lambda _s: None)
    return runner._child_env()["TORCH_CACHING_PRECOMPILE"]


def test_queue_child_env_follows_compile_and_cache_flags(monkeypatch):
    monkeypatch.delenv("TORCH_CACHING_PRECOMPILE", raising=False)
    assert _child_env({"compile": True, "compile_cache": True}) == "1"
    assert _child_env({"compile": True}) == "1"  # cache defaults on
    assert _child_env({"compile": True, "compile_cache": False}) == "0"
    assert _child_env({"compile": False, "compile_cache": True}) == "0"
    assert _child_env({}) == "0"
