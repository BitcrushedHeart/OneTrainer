import logging
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from web.backend.services._singleton import SingletonMixin
from web.backend.services.config_service import ConfigService

from modules.util.config.QueueConfig import QueueEntry, QueueSettings
from modules.util.enum.QueueEntryStatus import QueueEntryStatus
from modules.util.queue.QueueExecutor import QueueExecutor
from modules.util.queue.QueueManager import QueueManager
from modules.util.queue.QueueValidator import QueueValidator

logger = logging.getLogger(__name__)

QueueStatus = Literal["idle", "running", "stopping"]


class QueueService(SingletonMixin):

    def __init__(self) -> None:
        self._manager = QueueManager("queue.json")
        self._executor: QueueExecutor | None = None
        self._thread: threading.Thread | None = None
        self._status: QueueStatus = "idle"
        self._current_entry_id: str | None = None
        self._run_index: int = 0
        self._total_entries: int = 0
        self._lock = threading.Lock()
        self._ws_broadcast: Callable[[dict], None] | None = None

    def set_ws_broadcast(self, fn: Callable[[dict], None]) -> None:
        self._ws_broadcast = fn

    def _broadcast(self, message: dict) -> None:
        if self._ws_broadcast is not None:
            with suppress(Exception):
                self._ws_broadcast(message)

    # --- State queries ---

    def get_state(self) -> dict:
        return {
            "status": self._status,
            "current_entry_id": self._current_entry_id,
            "run_index": self._run_index,
            "total_entries": self._total_entries,
            "settings": self._manager.settings.to_dict(),
            "entries": [e.to_dict() for e in self._manager.entries],
        }

    # --- Entry CRUD ---

    def add_entry(self, name: str = "", overrides: dict | None = None) -> dict:
        entry = self._manager.add_entry(name=name, overrides=overrides or {})
        return entry.to_dict()

    def update_entry(self, entry_id: str, **kwargs: Any) -> dict:
        self._manager.update_entry(entry_id, **kwargs)
        entry = self._manager.get_entry(entry_id)
        return entry.to_dict() if entry else {}

    def remove_entry(self, entry_id: str) -> dict:
        self._manager.remove_entry(entry_id)
        return {"ok": True}

    def duplicate_entry(self, entry_id: str) -> dict:
        new_entry = self._manager.duplicate_entry(entry_id)
        if new_entry is None:
            return {"ok": False, "error": "Entry not found"}
        return {"ok": True, "entry": new_entry.to_dict()}

    def reorder(self, entry_id: str, direction: str) -> dict:
        if direction == "up":
            self._manager.move_up(entry_id)
        elif direction == "down":
            self._manager.move_down(entry_id)
        return {"ok": True}

    # --- Diff & full-config import ---

    def entry_diff(self, entry_id: str) -> dict:
        """Return overrides grouped by SECTION_MAP for display in the queue UI."""
        from web.backend.services._queue_diff import diff_section_grouped

        entry = self._manager.get_entry(entry_id)
        if entry is None:
            return {"ok": False, "error": "Entry not found"}

        config_service = ConfigService.get_instance()
        defaults = config_service.get_defaults()
        grouped = diff_section_grouped(entry.overrides or {}, defaults)
        return {"ok": True, "sections": grouped, "name": entry.name}

    def add_entry_from_full_config(self, full_config: dict, name: str = "") -> dict:
        """Diff a full training-config dict against defaults and create an entry."""
        from web.backend.services._queue_diff import diff_full_config

        config_service = ConfigService.get_instance()
        defaults = config_service.get_defaults()
        overrides = diff_full_config(full_config, defaults)
        entry = self._manager.add_entry(name=name, overrides=overrides)
        return {"ok": True, "entry": entry.to_dict()}

    # --- Settings ---

    def update_settings(self, settings_data: dict) -> dict:
        self._manager.settings = QueueSettings.from_dict(settings_data)
        self._manager.save()
        return self._manager.settings.to_dict()

    # --- Validation ---

    def validate(self) -> dict:
        config_service = ConfigService.get_instance()
        global_config = config_service.get_config_for_training()
        results = QueueValidator.validate_queue(self._manager, global_config)
        return {
            entry_id: {"errors": errors, "warnings": warnings}
            for entry_id, (errors, warnings) in results.items()
        }

    # --- Execution ---

    def execute(self) -> dict:
        with self._lock:
            if self._status != "idle":
                return {"ok": False, "error": f"Queue is already {self._status}"}
            self._status = "running"

        config_service = ConfigService.get_instance()
        global_config = config_service.get_config_for_training()

        self._manager.reset_all_to_pending()
        self._executor = QueueExecutor(
            queue_manager=self._manager,
            global_config=global_config,
            on_entry_start=self._on_entry_start,
            on_entry_complete=self._on_entry_complete,
            on_entry_failed=self._on_entry_failed,
            on_entry_skipped=self._on_entry_skipped,
            on_progress=self._on_progress,
            on_status=self._on_status_msg,
            on_queue_complete=self._on_queue_complete,
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return {"ok": True}

    def stop_current(self) -> dict:
        if self._executor:
            self._executor.stop_current_run()
            return {"ok": True}
        return {"ok": False, "error": "No executor running"}

    def stop_all(self) -> dict:
        with self._lock:
            self._status = "stopping"
        if self._executor:
            self._executor.stop_queue_immediate()
            return {"ok": True}
        return {"ok": False, "error": "No executor running"}

    # --- Import/Export ---

    def export_queue(self) -> dict:
        return {
            "settings": self._manager.settings.to_dict(),
            "entries": [e.to_dict() for e in self._manager.entries],
        }

    def import_queue(self, data: dict) -> dict:
        # Write to temp file and import
        import json
        import tempfile
        import os
        fd, tmp_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            warnings = self._manager.import_from_file(tmp_path)
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        return {"ok": True, "warnings": warnings}

    # --- Internal callbacks ---

    def _run(self) -> None:
        try:
            self._executor.run()
        except Exception:
            logger.exception("Queue execution failed")
        finally:
            with self._lock:
                self._status = "idle"
                self._current_entry_id = None
                self._executor = None

    def _on_entry_start(self, entry: QueueEntry, run_idx: int, total: int) -> None:
        self._current_entry_id = entry.id
        self._run_index = run_idx
        self._total_entries = total
        self._broadcast({
            "type": "queue:entry_started",
            "entry_id": entry.id,
            "name": entry.name,
            "run_index": run_idx,
            "total": total,
        })

    def _on_entry_complete(self, entry: QueueEntry) -> None:
        self._broadcast({
            "type": "queue:entry_completed",
            "entry_id": entry.id,
        })

    def _on_entry_failed(self, entry: QueueEntry, message: str) -> None:
        self._broadcast({
            "type": "queue:entry_failed",
            "entry_id": entry.id,
            "error": message,
        })

    def _on_entry_skipped(self, entry: QueueEntry) -> None:
        self._broadcast({
            "type": "queue:entry_skipped",
            "entry_id": entry.id,
        })

    def _on_progress(self, progress: Any, step: int, epoch: int) -> None:
        self._broadcast({
            "type": "queue:progress",
            "entry_id": self._current_entry_id,
            "step": step,
            "epoch": epoch,
            "global_step": getattr(progress, "global_step", -1),
        })

    def _on_status_msg(self, msg: str) -> None:
        self._broadcast({
            "type": "queue:status",
            "message": msg,
        })

    def _on_queue_complete(self) -> None:
        self._broadcast({"type": "queue:complete"})
