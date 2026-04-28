import uuid
from dataclasses import dataclass, field

from modules.util.enum.QueueEntryStatus import QueueEntryStatus


@dataclass
class AutoBatchSettings:
    enabled: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 8
    target_pct: float = 100.0
    max_drop_pct: float = 5.0
    last_batch_size: int | None = None
    last_accum: int | None = None
    last_effective_samples: int | None = None
    last_dropped: int | None = None

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "target_pct": self.target_pct,
            "max_drop_pct": self.max_drop_pct,
            "last_batch_size": self.last_batch_size,
            "last_accum": self.last_accum,
            "last_effective_samples": self.last_effective_samples,
            "last_dropped": self.last_dropped,
        }

    @staticmethod
    def from_dict(data: dict) -> "AutoBatchSettings":
        if not isinstance(data, dict):
            return AutoBatchSettings()
        return AutoBatchSettings(
            enabled=bool(data.get("enabled", False)),
            min_batch_size=int(data.get("min_batch_size", 1)),
            max_batch_size=int(data.get("max_batch_size", 8)),
            target_pct=float(data.get("target_pct", 100.0)),
            max_drop_pct=float(data.get("max_drop_pct", 5.0)),
            last_batch_size=data.get("last_batch_size"),
            last_accum=data.get("last_accum"),
            last_effective_samples=data.get("last_effective_samples"),
            last_dropped=data.get("last_dropped"),
        )


@dataclass
class QueueEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    overrides: dict = field(default_factory=dict)
    status: QueueEntryStatus = QueueEntryStatus.PENDING
    failure_history: list[dict] = field(default_factory=list)
    auto_batch: AutoBatchSettings = field(default_factory=AutoBatchSettings)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "overrides": self.overrides,
            "status": str(self.status),
            "failure_history": self.failure_history,
            "auto_batch": self.auto_batch.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict) -> "QueueEntry":
        return QueueEntry(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            overrides=data.get("overrides", {}),
            status=QueueEntryStatus(data.get("status", "PENDING")),
            failure_history=data.get("failure_history", []),
            auto_batch=AutoBatchSettings.from_dict(data.get("auto_batch", {})),
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
