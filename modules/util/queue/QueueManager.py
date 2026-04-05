import copy
import json
import os
import tempfile
from pathlib import Path

from modules.util.config.QueueConfig import QueueEntry, QueueSettings
from modules.util.enum.QueueEntryStatus import QueueEntryStatus


class QueueManager:
    def __init__(self, queue_file_path: str = "queue.json"):
        self.queue_file_path = queue_file_path
        self.entries: list[QueueEntry] = []
        self.settings: QueueSettings = QueueSettings()
        self.load()

    def load(self):
        if not os.path.isfile(self.queue_file_path):
            return
        try:
            with open(self.queue_file_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        self.settings = QueueSettings.from_dict(data.get("settings", {}))
        self.entries = [QueueEntry.from_dict(e) for e in data.get("entries", [])]

    def save(self):
        data = {
            "__version": 1,
            "settings": self.settings.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
        }
        dir_path = os.path.dirname(os.path.abspath(self.queue_file_path))
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.queue_file_path)
        except Exception:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            raise

    def add_entry(self, name: str = "", overrides: dict | None = None) -> QueueEntry:
        entry = QueueEntry(name=name, overrides=overrides or {})
        self.entries.append(entry)
        self.save()
        return entry

    def remove_entry(self, entry_id: str):
        self.entries = [e for e in self.entries if e.id != entry_id]
        self.save()

    def duplicate_entry(self, entry_id: str) -> QueueEntry | None:
        source = self.get_entry(entry_id)
        if source is None:
            return None
        new_entry = QueueEntry(
            name=f"{source.name} (copy)",
            overrides=copy.deepcopy(source.overrides),
        )
        idx = next((i for i, e in enumerate(self.entries) if e.id == entry_id), len(self.entries))
        self.entries.insert(idx + 1, new_entry)
        self.save()
        return new_entry

    def get_entry(self, entry_id: str) -> QueueEntry | None:
        for e in self.entries:
            if e.id == entry_id:
                return e
        return None

    def update_entry(self, entry_id: str, **kwargs):
        entry = self.get_entry(entry_id)
        if entry is None:
            return
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        self.save()

    def move_up(self, entry_id: str):
        idx = next((i for i, e in enumerate(self.entries) if e.id == entry_id), -1)
        if idx > 0:
            self.entries[idx - 1], self.entries[idx] = self.entries[idx], self.entries[idx - 1]
            self.save()

    def move_down(self, entry_id: str):
        idx = next((i for i, e in enumerate(self.entries) if e.id == entry_id), -1)
        if 0 <= idx < len(self.entries) - 1:
            self.entries[idx], self.entries[idx + 1] = self.entries[idx + 1], self.entries[idx]
            self.save()

    def set_status(self, entry_id: str, status: QueueEntryStatus):
        entry = self.get_entry(entry_id)
        if entry is not None:
            entry.status = status
            self.save()

    def reset_all_to_pending(self):
        for e in self.entries:
            if e.status in (QueueEntryStatus.COMPLETED, QueueEntryStatus.FAILED, QueueEntryStatus.SKIPPED):
                e.status = QueueEntryStatus.PENDING
                e.failure_history = []
        self.save()

    def get_next_runnable(self) -> QueueEntry | None:
        for e in self.entries:
            if e.status == QueueEntryStatus.RUNNING:
                return e
        for e in self.entries:
            if e.status == QueueEntryStatus.PENDING:
                return e
        return None

    def get_run_index(self, entry_id: str) -> int | None:
        runnable = [e for e in self.entries if e.status in (
            QueueEntryStatus.PENDING, QueueEntryStatus.RUNNING, QueueEntryStatus.COMPLETED,
        )]
        for i, e in enumerate(runnable):
            if e.id == entry_id:
                return i
        return None

    def total_runnable(self) -> int:
        return sum(1 for e in self.entries if e.status in (
            QueueEntryStatus.PENDING, QueueEntryStatus.RUNNING, QueueEntryStatus.COMPLETED,
        ))

    def export_to_file(self, file_path: str):
        data = {
            "__version": 1,
            "settings": self.settings.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def import_from_file(self, file_path: str) -> list[str]:
        with open(file_path) as f:
            data = json.load(f)
        warnings = []
        imported_settings = data.get("settings")
        if imported_settings:
            self.settings = QueueSettings.from_dict(imported_settings)
        for entry_data in data.get("entries", []):
            entry = QueueEntry.from_dict(entry_data)
            entry.status = QueueEntryStatus.PENDING
            entry.failure_history = []
            path_keys = _find_path_keys(entry.overrides)
            for key, path_val in path_keys:
                if not os.path.exists(path_val):
                    warnings.append(f"Run '{entry.name}': {key} path does not exist: {path_val}")
            self.entries.append(entry)
        self.save()
        return warnings


def _find_path_keys(overrides: dict, prefix: str = "") -> list[tuple[str, str]]:
    path_hints = {"dir", "path", "destination", "name", "file"}
    results = []
    for key, value in overrides.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            results.extend(_find_path_keys(value, full_key))
        elif isinstance(value, str) and any(h in key.lower() for h in path_hints):
            if os.sep in value or "/" in value or Path(value).suffix:
                results.append((full_key, value))
    return results
