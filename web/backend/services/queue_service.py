import logging
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from modules.util.auto_batch import compute_auto_batch
from modules.util.config.QueueConfig import AutoBatchSettings, QueueEntry, QueueSettings
from modules.util.queue.QueueExecutor import QueueExecutor
from modules.util.queue.QueueManager import QueueManager
from modules.util.queue.QueueValidator import QueueValidator
from web.backend.services._singleton import SingletonMixin
from web.backend.services.config_service import ConfigService

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

    # --- Auto-Batch ---

    @staticmethod
    def _input_fields_changed(new: AutoBatchSettings, old: AutoBatchSettings) -> bool:
        return (
            new.min_batch_size != old.min_batch_size
            or new.max_batch_size != old.max_batch_size
            or new.target_pct != old.target_pct
            or new.max_drop_pct != old.max_drop_pct
        )

    def update_auto_batch(self, entry_id: str, settings: dict) -> dict:
        entry = self._manager.get_entry(entry_id)
        if entry is None:
            return {"ok": False, "error": "Entry not found"}
        merged = {**entry.auto_batch.to_dict(), **(settings or {})}
        new_settings = AutoBatchSettings.from_dict(merged)
        if new_settings.min_batch_size > new_settings.max_batch_size:
            return {"ok": False, "error": "min_batch_size must be <= max_batch_size"}
        if self._input_fields_changed(new_settings, entry.auto_batch):
            new_settings.last_batch_size = None
            new_settings.last_accum = None
            new_settings.last_effective_samples = None
            new_settings.last_dropped = None
        entry.auto_batch = new_settings
        self._manager.save()
        return {"ok": True, "entry": entry.to_dict()}

    def auto_batch_calculate(self, entry_id: str) -> dict:
        entry = self._manager.get_entry(entry_id)
        if entry is None:
            return {"ok": False, "error": "Entry not found"}
        config_service = ConfigService.get_instance()
        global_config = config_service.get_config_for_training()
        try:
            merged = QueueExecutor.merge_config(global_config, entry.overrides or {})
        except Exception as exc:
            logger.exception("Auto-batch: failed to merge config for entry %s", entry_id)
            return {"ok": False, "error": f"Failed to merge config: {exc}"}
        try:
            result = compute_auto_batch(
                merged_config=merged,
                min_batch_size=entry.auto_batch.min_batch_size,
                max_batch_size=entry.auto_batch.max_batch_size,
                target_pct=entry.auto_batch.target_pct,
                max_drop_pct=entry.auto_batch.max_drop_pct,
            )
        except Exception as exc:
            logger.exception("Auto-batch: compute failed for entry %s", entry_id)
            return {"ok": False, "error": str(exc)}
        entry.auto_batch.last_batch_size = result.batch_size
        entry.auto_batch.last_accum = result.accum
        entry.auto_batch.last_effective_samples = result.effective_sample_count
        entry.auto_batch.last_dropped = result.dropped
        self._manager.save()
        return {"ok": True, "result": result.to_dict(), "entry": entry.to_dict()}

    def auto_batch_calculate_all(self) -> dict:
        results: list[dict] = []
        for entry in list(self._manager.entries):
            if not entry.included or not entry.auto_batch.enabled:
                continue
            res = self.auto_batch_calculate(entry.id)
            results.append({"entry_id": entry.id, **res})
        return {"ok": True, "results": results}

    def bulk_set_auto_batch(self, settings: dict) -> dict:
        if not isinstance(settings, dict):
            return {"ok": False, "error": "Invalid payload"}
        min_bs = int(settings.get("min_batch_size", 1))
        max_bs = int(settings.get("max_batch_size", 8))
        if min_bs < 1 or max_bs < min_bs:
            return {"ok": False, "error": "min_batch_size must be >= 1 and <= max_batch_size"}
        for entry in self._manager.entries:
            entry.auto_batch = AutoBatchSettings(
                enabled=True,
                min_batch_size=min_bs,
                max_batch_size=max_bs,
                target_pct=float(settings.get("target_pct", 100.0)),
                max_drop_pct=float(settings.get("max_drop_pct", 5.0)),
            )
        self._manager.save()
        return {"ok": True, "count": len(self._manager.entries)}

    # --- Validation ---

    def validate(self) -> dict:
        config_service = ConfigService.get_instance()
        global_config = config_service.get_config_for_training()
        results = QueueValidator.validate_queue(self._manager, global_config)
        return {entry_id: {"errors": errors, "warnings": warnings} for entry_id, (errors, warnings) in results.items()}

    # --- Execution ---

    def execute(self) -> dict:
        with self._lock:
            if self._status != "idle":
                return {"ok": False, "error": f"Queue is already {self._status}"}
            self._status = "running"

        config_service = ConfigService.get_instance()
        global_config = config_service.get_config_for_training()

        auto_batch_events = self._resolve_auto_batch_for_run()
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
        return {"ok": True, "auto_batch_events": auto_batch_events}

    def _resolve_auto_batch_for_run(self) -> list[dict]:
        """Compute Auto-Batch overrides for every enabled entry just before launch.

        On success, writes batch_size + gradient_accumulation_steps into the entry's
        overrides dict. On failure for an individual entry, records the event and
        continues with the rest of the queue. Returns a list of events that the
        caller surfaces to the UI in the /queue/execute response.
        """
        events: list[dict] = []
        for entry in list(self._manager.entries):
            if not entry.included:
                continue
            if not entry.auto_batch.enabled:
                continue
            res = self.auto_batch_calculate(entry.id)
            if not res.get("ok"):
                events.append(
                    {
                        "type": "queue:auto_batch_failed",
                        "entry_id": entry.id,
                        "error": res.get("error", "Auto-Batch calculation failed"),
                    }
                )
                continue
            result = res.get("result") or {}
            bs = result.get("batch_size")
            accum = result.get("accum")
            warning = result.get("warning")
            if not isinstance(bs, int) or not isinstance(accum, int):
                events.append(
                    {
                        "type": "queue:auto_batch_failed",
                        "entry_id": entry.id,
                        "error": "Auto-Batch returned no batch size",
                    }
                )
                continue
            entry.overrides["batch_size"] = bs
            entry.overrides["gradient_accumulation_steps"] = accum
            logger.info(
                "Auto-Batch resolved entry %s: batch_size=%d accum=%d (effective=%s, drops=%s)%s",
                entry.id,
                bs,
                accum,
                result.get("effective_sample_count"),
                result.get("dropped"),
                f" — warning: {warning}" if warning else "",
            )
            if warning:
                events.append(
                    {
                        "type": "queue:auto_batch_warning",
                        "entry_id": entry.id,
                        "warning": warning,
                    }
                )
        self._manager.save()
        return events

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
        import os
        import tempfile

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
        self._broadcast(
            {
                "type": "queue:entry_started",
                "entry_id": entry.id,
                "name": entry.name,
                "run_index": run_idx,
                "total": total,
            }
        )

    def _on_entry_complete(self, entry: QueueEntry) -> None:
        self._broadcast(
            {
                "type": "queue:entry_completed",
                "entry_id": entry.id,
            }
        )

    def _on_entry_failed(self, entry: QueueEntry, message: str) -> None:
        self._broadcast(
            {
                "type": "queue:entry_failed",
                "entry_id": entry.id,
                "error": message,
            }
        )

    def _on_entry_skipped(self, entry: QueueEntry) -> None:
        self._broadcast(
            {
                "type": "queue:entry_skipped",
                "entry_id": entry.id,
            }
        )

    def _on_progress(self, progress: Any, step: int, epoch: int) -> None:
        self._broadcast(
            {
                "type": "queue:progress",
                "entry_id": self._current_entry_id,
                "step": step,
                "epoch": epoch,
                "global_step": getattr(progress, "global_step", -1),
            }
        )

    def _on_status_msg(self, msg: str) -> None:
        self._broadcast(
            {
                "type": "queue:status",
                "message": msg,
            }
        )

    def _on_queue_complete(self) -> None:
        self._broadcast({"type": "queue:complete"})
