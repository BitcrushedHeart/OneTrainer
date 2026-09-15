"""Text Alignment Dataset tool: scan a SwarmUI output folder for generations
whose on-image text is rendered but wrong, and package them for the offline
text reward (see ``modules/util/text_alignment_dataset_util``).

Session shape: ``scan`` runs the classification in a background thread
(the OCR is the slow part) and ``status`` reports progress and per-reason
counts; ``build`` copies the kept candidates of the last scan; ``wildcard``
writes reward prompts for SwarmUI's wildcard system.
"""

from __future__ import annotations

import logging
import os
import threading

from modules.util.text_alignment_dataset_util import (
    ScanPlan,
    apply_plan,
    make_count_boxed,
    make_read_text,
    scan_dataset,
    write_wildcard_file,
)
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

DEFAULT_WILDCARDS_DIR = r"E:\AI\Data\Packages\SwarmUI\Data\Wildcards"
DEFAULT_WILDCARD_NAME = "text_alignment"
_PREVIEW_LIMIT = 500


class TextAlignmentDatasetService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._phase = "idle"  # idle | scanning | scanned | building | built | error | cancelled
        self._error: str | None = None
        self._progress_done = 0
        self._progress_total = 0
        self._current = ""
        self._plan: ScanPlan | None = None
        self._build_result: dict | None = None
        self._read_text = None
        self._count_boxed = None
        # Injection points for tests.
        self.read_text_factory = make_read_text
        self.count_boxed_factory = make_count_boxed

    # -- helpers -----------------------------------------------------------------

    def _busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _progress(self, done: int, total: int, current: str) -> None:
        with self._lock:
            self._progress_done = done
            self._progress_total = total
            self._current = current

    def _models(self, ocr_device: str, detector_device: str):
        if self._read_text is None:
            self._read_text = self.read_text_factory(ocr_device)
        if self._count_boxed is None:
            self._count_boxed = self.count_boxed_factory(detector_device)
        return self._read_text, self._count_boxed

    # -- API ---------------------------------------------------------------------

    def scan(
        self,
        source_folder: str,
        output_folder: str,
        *,
        recursive: bool = True,
        keep_absent: bool = False,
        keep_verbatim: bool = False,
        min_boxed_words: int = 1,
        min_agreement: float = 0.0,
        limit: int = 0,
        ocr_device: str = "auto",
        detector_device: str = "auto",
    ) -> dict:
        if not source_folder or not os.path.isdir(source_folder):
            return {"ok": False, "error": "Source folder does not exist."}
        if not output_folder:
            return {"ok": False, "error": "Output folder is required."}
        with self._lock:
            if self._busy():
                return {"ok": False, "error": "A scan or build is already running."}
            self._stop = threading.Event()
            self._phase = "scanning"
            self._error = None
            self._plan = None
            self._build_result = None
            self._progress_done = 0
            self._progress_total = 0
            stop = self._stop

        def run() -> None:
            try:
                read_text, count_boxed = self._models(ocr_device, detector_device)
                plan = scan_dataset(
                    source_folder,
                    output_folder,
                    read_text=read_text,
                    count_boxed=count_boxed,
                    recursive=recursive,
                    keep_absent=keep_absent,
                    keep_verbatim=keep_verbatim,
                    min_boxed_words=max(0, int(min_boxed_words)),
                    min_agreement=min(1.0, max(0.0, float(min_agreement))),
                    limit=max(0, int(limit)),
                    progress=self._progress,
                    should_stop=stop.is_set,
                )
                with self._lock:
                    self._plan = plan
                    self._phase = "cancelled" if plan.cancelled else "scanned"
            except Exception as ex:  # noqa: BLE001 - surfaced to the UI
                logger.exception("text alignment scan failed")
                with self._lock:
                    self._error = str(ex)
                    self._phase = "error"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return {"ok": True}

    def build(self) -> dict:
        with self._lock:
            if self._busy():
                return {"ok": False, "error": "A scan or build is already running."}
            if self._plan is None:
                return {"ok": False, "error": "Scan first."}
            plan = self._plan
            self._stop = threading.Event()
            self._phase = "building"
            self._error = None
            self._build_result = None
            self._progress_done = 0
            self._progress_total = len(plan.kept)
            stop = self._stop

        def run() -> None:
            try:
                result = apply_plan(plan, progress=self._progress, should_stop=stop.is_set)
                with self._lock:
                    self._build_result = result
                    self._phase = "cancelled" if result.get("cancelled") else "built"
            except Exception as ex:  # noqa: BLE001
                logger.exception("text alignment build failed")
                with self._lock:
                    self._error = str(ex)
                    self._phase = "error"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return {"ok": True}

    def cancel(self) -> dict:
        self._stop.set()
        return {"ok": True}

    def status(self) -> dict:
        with self._lock:
            plan = self._plan
            out = {
                "ok": True,
                "phase": self._phase,
                "running": self._busy(),
                "error": self._error,
                "progress_done": self._progress_done,
                "progress_total": self._progress_total,
                "current": self._current,
                "counts": plan.counts if plan is not None else {},
                "kept": len(plan.kept) if plan is not None else 0,
                "scanned": len(plan.candidates) if plan is not None else 0,
                "build": self._build_result,
                "output_folder": plan.output_folder if plan is not None else None,
            }
        return out

    def results(self, offset: int = 0, limit: int = _PREVIEW_LIMIT, only_kept: bool = True) -> dict:
        with self._lock:
            plan = self._plan
        if plan is None:
            return {"ok": True, "total": 0, "items": []}
        items = plan.kept if only_kept else plan.candidates
        page = items[offset : offset + max(0, limit)]
        return {"ok": True, "total": len(items), "items": [c.to_record() for c in page]}

    def serve_image(self, path: str) -> str | None:
        """Only images the last scan classified are served (no arbitrary reads)."""
        with self._lock:
            plan = self._plan
        if plan is None:
            return None
        wanted = os.path.normcase(os.path.abspath(path))
        for cand in plan.candidates:
            if os.path.normcase(os.path.abspath(cand.source_image)) == wanted and os.path.isfile(cand.source_image):
                return cand.source_image
        return None

    @staticmethod
    def wildcard(
        count: int = 1000,
        seed: int = 0,
        name: str = DEFAULT_WILDCARD_NAME,
        directory: str = DEFAULT_WILDCARDS_DIR,
        min_words: int = 1,
        max_words: int = 8,
    ) -> dict:
        from modules.util.text_reward_util import DEFAULT_VOCAB, generate_text_specs

        count = max(1, min(int(count), 100_000))
        if not name.strip():
            return {"ok": False, "error": "Wildcard name is required."}
        specs = generate_text_specs(
            count,
            seed=int(seed),
            vocab=DEFAULT_VOCAB,
            min_words=max(1, int(min_words)),
            max_words=max(int(min_words), int(max_words)),
        )
        try:
            path = write_wildcard_file(directory, name.strip(), [s.prompt for s in specs])
        except OSError as ex:
            return {"ok": False, "error": str(ex)}
        return {"ok": True, "path": path, "count": len(specs), "sample": [s.prompt for s in specs[:5]]}
