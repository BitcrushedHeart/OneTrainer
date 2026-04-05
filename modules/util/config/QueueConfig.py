import uuid
from dataclasses import dataclass, field

from modules.util.enum.QueueEntryStatus import QueueEntryStatus


@dataclass
class QueueEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    overrides: dict = field(default_factory=dict)
    status: QueueEntryStatus = QueueEntryStatus.PENDING
    failure_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "overrides": self.overrides,
            "status": str(self.status),
            "failure_history": self.failure_history,
        }

    @staticmethod
    def from_dict(data: dict) -> "QueueEntry":
        return QueueEntry(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            overrides=data.get("overrides", {}),
            status=QueueEntryStatus(data.get("status", "PENDING")),
            failure_history=data.get("failure_history", []),
        )


@dataclass
class QueueSettings:
    retry_on_error: bool = True
    retry_from_backup: bool = True
    max_retries: int = 2
    oom_skip_threshold: int = 3
    oom_step_window: int = 10

    def to_dict(self) -> dict:
        return {
            "retry_on_error": self.retry_on_error,
            "retry_from_backup": self.retry_from_backup,
            "max_retries": self.max_retries,
            "oom_skip_threshold": self.oom_skip_threshold,
            "oom_step_window": self.oom_step_window,
        }

    @staticmethod
    def from_dict(data: dict) -> "QueueSettings":
        return QueueSettings(
            retry_on_error=data.get("retry_on_error", True),
            retry_from_backup=data.get("retry_from_backup", True),
            max_retries=data.get("max_retries", 2),
            oom_skip_threshold=data.get("oom_skip_threshold", 3),
            oom_step_window=data.get("oom_step_window", 10),
        )
