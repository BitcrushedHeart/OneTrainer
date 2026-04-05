from __future__ import annotations

import copy
import datetime
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

from modules.util.config.QueueConfig import QueueEntry
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.QueueEntryStatus import QueueEntryStatus
from modules.util.queue.QueueManager import QueueManager

if TYPE_CHECKING:
    from modules.util.commands.TrainCommands import TrainCommands
    from modules.util.TrainProgress import TrainProgress


class QueueExecutor:
    def __init__(
        self,
        queue_manager: QueueManager,
        global_config: TrainConfig,
        on_entry_start: Callable[[QueueEntry, int, int], None] = lambda _e, _i, _t: None,
        on_entry_complete: Callable[[QueueEntry], None] = lambda _e: None,
        on_entry_failed: Callable[[QueueEntry, str], None] = lambda _e, _m: None,
        on_entry_skipped: Callable[[QueueEntry], None] = lambda _e: None,
        on_progress: Callable[[TrainProgress, int, int], None] = lambda _p, _s, _ep: None,
        on_status: Callable[[str], None] = lambda _s: None,
        on_queue_complete: Callable[[], None] = lambda: None,
    ):
        self.queue_manager = queue_manager
        self.global_config = global_config
        self._on_entry_start = on_entry_start
        self._on_entry_complete = on_entry_complete
        self._on_entry_failed = on_entry_failed
        self._on_entry_skipped = on_entry_skipped
        self._on_progress = on_progress
        self._on_status = on_status
        self._on_queue_complete = on_queue_complete
        self._stop_queue = False
        self._stop_current_run = False
        self._current_commands: TrainCommands | None = None
        self._current_entry: QueueEntry | None = None
        self._last_global_step: int = -1

    def run(self):
        try:
            self._run_loop()
        finally:
            self._current_commands = None
            self._current_entry = None
            self._on_queue_complete()

    def _run_loop(self):
        while True:
            if self._stop_queue:
                break
            entry = self.queue_manager.get_next_runnable()
            if entry is None:
                break
            is_resume = entry.status == QueueEntryStatus.RUNNING
            self.queue_manager.set_status(entry.id, QueueEntryStatus.RUNNING)
            self._current_entry = entry
            run_idx = (self.queue_manager.get_run_index(entry.id) or 0) + 1
            total = self.queue_manager.total_runnable()
            self._on_entry_start(entry, run_idx, total)
            result = self._execute_with_retry(entry, is_resume)
            if self._stop_queue and result != QueueEntryStatus.COMPLETED:
                break
            self.queue_manager.set_status(entry.id, result)
            if result == QueueEntryStatus.COMPLETED:
                self._on_entry_complete(entry)
            elif result == QueueEntryStatus.FAILED:
                self._on_entry_failed(entry, self._last_error_message(entry))
            elif result == QueueEntryStatus.SKIPPED:
                self._on_entry_skipped(entry)

    def stop_current_run(self):
        self._stop_current_run = True
        if self._current_commands:
            self._current_commands.stop()

    def stop_queue(self):
        self._stop_queue = True

    def stop_queue_immediate(self):
        self._stop_queue = True
        if self._current_commands:
            self._current_commands.stop()

    @staticmethod
    def merge_config(global_config: TrainConfig, overrides: dict) -> TrainConfig:
        base_dict = global_config.to_dict()
        merged_dict = _deep_merge(base_dict, overrides)
        return TrainConfig.default_values().from_dict(merged_dict)

    @staticmethod
    def is_nan_error(error: Exception) -> bool:
        return isinstance(error, RuntimeError) and "nan" in str(error).lower()

    @staticmethod
    def is_file_not_found_error(error: Exception) -> bool:
        return isinstance(error, (FileNotFoundError, IsADirectoryError))

    @staticmethod
    def is_oom_error(error: Exception) -> bool:
        return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()

    def should_skip_oom(self, entry: QueueEntry) -> bool:
        settings = self.queue_manager.settings
        oom_failures = [
            f for f in entry.failure_history
            if "out of memory" in f.get("error", "").lower()
        ]
        if len(oom_failures) < settings.oom_skip_threshold:
            return False
        recent = oom_failures[-settings.oom_skip_threshold:]
        steps = [f.get("step", -1) for f in recent if f.get("step", -1) >= 0]
        if len(steps) < settings.oom_skip_threshold:
            return False
        min_step = min(steps)
        max_step = max(steps)
        return (max_step - min_step) <= settings.oom_step_window

    def _execute_with_retry(self, entry: QueueEntry, is_resume: bool) -> QueueEntryStatus:
        settings = self.queue_manager.settings
        max_attempts = (settings.max_retries + 1) if settings.retry_on_error else 1
        for attempt in range(max_attempts):
            status = self._try_execute(entry, is_resume, attempt, settings, max_attempts)
            if status is not None:
                return status
        return QueueEntryStatus.FAILED

    def _try_execute(self, entry, is_resume, attempt, settings, max_attempts):
        try:
            self._stop_current_run = False
            self._execute_entry(entry, is_resume or (attempt > 0 and settings.retry_from_backup))
            return QueueEntryStatus.COMPLETED
        except Exception as e:
            step = self._get_current_step()
            entry.failure_history.append({
                "step": step,
                "error": str(e),
                "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            })
            self.queue_manager.save()
            traceback.print_exc()
            if self._stop_current_run:
                return QueueEntryStatus.COMPLETED
            if self.is_nan_error(e) or self.is_file_not_found_error(e):
                return QueueEntryStatus.FAILED
            if self.is_oom_error(e) and self.should_skip_oom(entry):
                return QueueEntryStatus.SKIPPED
            if self._stop_queue:
                return QueueEntryStatus.RUNNING
            if attempt >= max_attempts - 1:
                return QueueEntryStatus.FAILED
            self._on_status(f"Retrying ({attempt + 1}/{settings.max_retries})...")
            return None

    def _execute_entry(self, entry: QueueEntry, resume_from_backup: bool):
        from modules.util import create
        from modules.util.callbacks.TrainCallbacks import TrainCallbacks
        from modules.util.commands.TrainCommands import TrainCommands
        from modules.util.torch_util import torch_gc

        import torch

        merged_config = self.merge_config(self.global_config, entry.overrides)
        if resume_from_backup:
            merged_config.continue_last_backup = True
        commands = TrainCommands()
        self._current_commands = commands
        def _track_progress(p, s, ep):
            self._last_global_step = getattr(p, "global_step", -1)
            self._on_progress(p, s, ep)

        callbacks = TrainCallbacks(
            on_update_train_progress=_track_progress,
            on_update_status=self._on_status,
        )
        self._on_status(f"Starting: {entry.name}")
        trainer = create.create_trainer(merged_config, callbacks, commands)
        error = None
        try:
            trainer.start()
            trainer.train()
        except Exception as e:
            error = e
        trainer.end()
        del trainer
        torch.clear_autocast_cache()
        torch_gc()
        if error is not None:
            raise error
        if self._stop_current_run:
            self._stop_current_run = False

    def _get_current_step(self) -> int:
        return self._last_global_step

    @staticmethod
    def _last_error_message(entry: QueueEntry) -> str:
        if entry.failure_history:
            return entry.failure_history[-1].get("error", "Unknown error")
        return "Unknown error"


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
