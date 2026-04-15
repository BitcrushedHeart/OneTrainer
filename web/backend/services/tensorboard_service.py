import logging
import os
import threading
import time
from pathlib import Path

from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False
    logger.warning(
        "tensorboard package not found. TensorBoard tab will return empty data. "
        "Install with: pip install tensorboard"
    )


MAX_CACHED_ACCUMULATORS = 10


class TensorboardService(SingletonMixin):

    def __init__(self) -> None:
        self._accumulators: dict[str, EventAccumulator] = {}
        self._access_times: dict[str, float] = {}
        self._accumulator_lock = threading.Lock()

    @staticmethod
    def _resolve_log_dir(log_dir: str | None = None) -> str:
        if log_dir:
            return log_dir

        from web.backend.services.config_service import ConfigService

        config_service = ConfigService.get_instance()
        return config_service.config.workspace_dir or "workspace"

    @staticmethod
    def _is_tfevents_dir(directory: str) -> bool:
        try:
            for entry in os.scandir(directory):
                if entry.is_file() and entry.name.startswith("events.out.tfevents"):
                    return True
        except OSError:
            pass
        return False

    def list_runs(self, log_dir: str | None = None) -> list[str]:
        resolved = self._resolve_log_dir(log_dir)
        if not os.path.isdir(resolved):
            return []

        base = Path(resolved)
        runs: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(resolved):
            if any(f.startswith("events.out.tfevents") for f in filenames):
                rel = Path(dirpath).relative_to(base).as_posix()
                runs.append(rel)

        runs.sort(reverse=True)
        return runs

    def _maybe_evict(self) -> None:
        while len(self._accumulators) > MAX_CACHED_ACCUMULATORS:
            oldest_key = min(self._access_times, key=self._access_times.get)  # type: ignore[arg-type]
            del self._accumulators[oldest_key]
            del self._access_times[oldest_key]
            logger.debug("Evicted accumulator cache for: %s", oldest_key)

    def _get_accumulator(self, run_dir: str) -> "EventAccumulator | None":
        if not _HAS_TENSORBOARD:
            return None

        with self._accumulator_lock:
            if run_dir not in self._accumulators:
                self._maybe_evict()
                acc = EventAccumulator(run_dir)
                acc.Reload()
                self._accumulators[run_dir] = acc
            else:
                self._accumulators[run_dir].Reload()

            self._access_times[run_dir] = time.time()
            return self._accumulators[run_dir]

    def _run_path(self, run_name: str, log_dir: str | None = None) -> str:
        resolved = self._resolve_log_dir(log_dir)
        return os.path.join(resolved, run_name)

    def list_tags(self, run_name: str, log_dir: str | None = None) -> list[str]:
        run_path = self._run_path(run_name, log_dir)
        if not os.path.isdir(run_path):
            return []

        acc = self._get_accumulator(run_path)
        if acc is None:
            return []

        tags = acc.Tags()
        return sorted(tags.get("scalars", []))

    def get_scalars(
        self,
        run_name: str,
        tag: str,
        after_step: int = 0,
        log_dir: str | None = None,
    ) -> list[dict]:
        run_path = self._run_path(run_name, log_dir)
        if not os.path.isdir(run_path):
            return []

        acc = self._get_accumulator(run_path)
        if acc is None:
            return []

        try:
            events = acc.Scalars(tag)
        except KeyError:
            return []

        return [
            {
                "wall_time": event.wall_time,
                "step": event.step,
                "value": event.value,
            }
            for event in events
            if event.step > after_step
        ]

    def clear_cache(self, run_name: str | None = None, log_dir: str | None = None) -> None:
        with self._accumulator_lock:
            if run_name:
                run_path = self._run_path(run_name, log_dir)
                self._accumulators.pop(run_path, None)
                self._access_times.pop(run_path, None)
            else:
                self._accumulators.clear()
                self._access_times.clear()
