"""Night Darken: review candidates one at a time and adjust the accepted
chosen images in place (see ``modules/util/dpo_night_darken_util.py``).

One concept is open at a time. The scan and the commit run on a background
thread and report through :meth:`status`; everything else is in-process and
quick. Previews are rendered on demand from a cached downscaled original, so
dragging a slider never touches the dataset.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict

from modules.util import dpo_night_darken_util as nd
from web.backend.services._singleton import SingletonMixin

import numpy as np

logger = logging.getLogger(__name__)

_PREVIEW_CACHE_SIZE = 6


class DPONightDarkenService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._concept_dir: str | None = None
        self._ledger: dict | None = None
        self._job: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previews: OrderedDict[str, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------ #
    # Session
    # ------------------------------------------------------------------ #

    def open(self, concept_dir: str) -> dict:
        concept_dir = os.path.abspath(concept_dir or "")
        if not os.path.isdir(os.path.join(concept_dir, "chosen")):
            return {"ok": False, "error": "Not a DPO concept folder (no chosen/ subfolder)"}
        reason = nd.source_exclusion_reason(concept_dir)
        if reason:
            return {"ok": False, "error": f"Refused as a source: {reason}"}
        with self._lock:
            if self._running():
                return {"ok": False, "error": "A scan or commit is still running"}
            try:
                ledger = nd.load_ledger(concept_dir)
            except (OSError, ValueError) as ex:
                return {"ok": False, "error": f"The ledger is unreadable: {ex}"}
            self._concept_dir = concept_dir
            self._ledger = ledger or nd.new_ledger(concept_dir)
            self._previews.clear()
            self._job = None
        return self.state()

    def _require(self) -> tuple[str, dict] | None:
        if self._concept_dir is None or self._ledger is None:
            return None
        return self._concept_dir, self._ledger

    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def defaults(self) -> dict:
        return {
            "phrases": list(nd.NIGHT_PHRASES),
            "thresholds": {"median": nd.DEFAULT_DARK_MEDIAN, "mean": nd.DEFAULT_DARK_MEAN},
            "fraction": nd.DEFAULT_MINOR_EDITS_FRACTION,
            "adjust": nd.Adjust(strength=nd.DEFAULT_STRENGTH).to_dict(),
            "hue_range": nd.HUE_RANGE,
            "saturation_max": nd.SATURATION_MAX,
            "night_edits_dir_name": nd.NIGHT_EDITS_DIR_NAME,
            "curve": {
                "exposure_stops": nd.EXPOSURE_STOPS,
                "gamma_lift": nd.GAMMA_LIFT,
                "highlight_knee": nd.HIGHLIGHT_KNEE,
                "highlight_keep": nd.HIGHLIGHT_KEEP,
                "noise_max_sigma": nd.NOISE_MAX_SIGMA,
                "noise_floor": nd.NOISE_FLOOR,
                "noise_chroma": nd.NOISE_CHROMA,
            },
            "bright_pixel_luma": nd.BRIGHT_PIXEL_LUMA,
        }

    def state(self) -> dict:
        with self._lock:
            opened = self._require()
            out = {"ok": True, "defaults": self.defaults(), "job": self._job_view()}
            if opened is None:
                out.update(concept_dir=None, scan=None, counts=None, commits=[], default_night_edits_dir=None)
                return out
            concept_dir, ledger = opened
            decisions = ledger.get("decisions", {})
            candidates = ledger.get("candidates", [])
            accepted = [d for d in decisions.values() if d.get("state") == "accepted"]
            out.update(
                concept_dir=concept_dir,
                ledger_dir=nd.ledger_dir(concept_dir),
                backup_dir=nd.backup_dir(concept_dir),
                default_night_edits_dir=nd.default_night_edits_dir(concept_dir),
                scan=ledger.get("scan"),
                counts={
                    "candidates": len(candidates),
                    "accepted": len(accepted),
                    "accepted_uncommitted": sum(1 for d in accepted if not d.get("committed")),
                    "skipped": sum(1 for d in decisions.values() if d.get("state") == "skipped"),
                    "undecided": sum(1 for c in candidates if c["rel"] not in decisions),
                },
                commits=[self._commit_view(c) for c in ledger.get("commits", [])],
            )
            return out

    @staticmethod
    def _commit_view(commit: dict) -> dict:
        items = commit.get("items", [])
        return {
            "id": commit.get("id"),
            "at": commit.get("at"),
            "items": len(items),
            "pairs": sum(1 for it in items if it.get("night_pair")),
            "night_edits_dir": commit.get("night_edits_dir"),
            "fraction": commit.get("fraction"),
            "restored": bool(commit.get("restored")),
            "restored_at": commit.get("restored_at"),
            "restore_missing": commit.get("restore_missing") or [],
            "error": commit.get("error"),
            "mean_delta_e": (round(float(np.mean([it.get("delta_e") or 0.0 for it in items])), 2) if items else None),
        }

    def _job_view(self) -> dict | None:
        job = self._job
        if job is None:
            return None
        view = dict(job)
        view["running"] = self._running()
        view["elapsed"] = round(time.time() - job.get("started_at", time.time()), 1)
        return view

    # ------------------------------------------------------------------ #
    # Scan
    # ------------------------------------------------------------------ #

    def scan(self, thresholds: dict | None = None, phrases: list[str] | None = None) -> dict:
        with self._lock:
            opened = self._require()
            if opened is None:
                return {"ok": False, "error": "Open a concept folder first"}
            if self._running():
                return {"ok": False, "error": "A scan or commit is still running"}
            concept_dir, ledger = opened
            thresholds_obj = nd.DarkThresholds.from_dict(thresholds)
            phrase_list = [p for p in (phrases or []) if p.strip()] or list(nd.NIGHT_PHRASES)
            self._stop = threading.Event()
            self._job = {
                "kind": "scan",
                "phase": "captions",
                "done": 0,
                "total": 0,
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
            }
            self._thread = threading.Thread(
                target=self._scan_worker,
                args=(concept_dir, ledger, thresholds_obj, phrase_list, self._stop),
                name="night-darken-scan",
                daemon=True,
            )
            self._thread.start()
        return {"ok": True}

    def _progress(self, done: int, total: int, phase: str | None = None) -> None:
        with self._lock:
            if self._job is not None:
                self._job["done"] = done
                self._job["total"] = total
                if phase:
                    self._job["phase"] = phase

    def _scan_worker(self, concept_dir, ledger, thresholds, phrases, stop) -> None:
        try:
            self._progress(0, 0, "brightness")
            result = nd.scan_candidates(
                concept_dir,
                phrases=phrases,
                thresholds=thresholds,
                progress=lambda d, t: self._progress(d, t),
                stop=stop,
            )
            if stop.is_set():
                with self._lock:
                    self._job["phase"] = "cancelled"
                return
            with self._lock:
                old = ledger.get("decisions", {})
                # Keep decisions on candidates that are still candidates; an
                # image that is now dark (already committed, say) drops out.
                keep = {c["rel"] for c in result["candidates"]}
                ledger["candidates"] = sorted(result["candidates"], key=lambda c: c["rel"])
                ledger["decisions"] = {r: d for r, d in old.items() if r in keep or d.get("committed")}
                ledger["scan"] = {
                    "at": time.time(),
                    "phrases": phrases,
                    "thresholds": {"median": thresholds.median, "mean": thresholds.mean},
                    "counts": result["counts"],
                    "already_dark": result["already_dark"],
                    "unreadable": result["unreadable"],
                }
                nd.save_ledger(concept_dir, ledger)
                self._previews.clear()
                self._job["phase"] = "done"
            logger.info(
                "night darken scan of %s: %d images, %d caption matches, %d already dark, %d candidates",
                concept_dir,
                result["counts"]["images"],
                result["counts"]["caption_matches"],
                result["counts"]["already_dark"],
                result["counts"]["candidates"],
            )
        except Exception as ex:
            logger.exception("night darken scan failed")
            with self._lock:
                if self._job is not None:
                    self._job["error"] = str(ex)
                    self._job["phase"] = "error"
        finally:
            with self._lock:
                if self._job is not None:
                    self._job["finished_at"] = time.time()

    def status(self) -> dict:
        with self._lock:
            return {"ok": True, "job": self._job_view()}

    def cancel(self) -> dict:
        with self._lock:
            if not self._running():
                return {"ok": True, "cancelled": False}
            self._stop.set()
            return {"ok": True, "cancelled": True}

    # ------------------------------------------------------------------ #
    # Review
    # ------------------------------------------------------------------ #

    def candidates(self) -> dict:
        with self._lock:
            opened = self._require()
            if opened is None:
                return {"ok": False, "error": "Open a concept folder first", "candidates": []}
            _, ledger = opened
            decisions = ledger.get("decisions", {})
            out = []
            for candidate in ledger.get("candidates", []):
                row = dict(candidate)
                row["decision"] = decisions.get(candidate["rel"])
                out.append(row)
            return {"ok": True, "candidates": out}

    def caption(self, rel: str) -> dict:
        """The caption text for the UI only; never logged."""
        opened = self._require()
        if opened is None:
            return {"ok": False, "error": "Open a concept folder first"}
        path = self._candidate_path(rel)
        if path is None:
            return {"ok": False, "error": "Not a candidate"}
        caption_path = os.path.splitext(path)[0] + ".txt"
        return {"ok": True, "caption": nd.read_caption(caption_path if os.path.isfile(caption_path) else None)}

    def _candidate_path(self, rel: str) -> str | None:
        opened = self._require()
        if opened is None:
            return None
        concept_dir, ledger = opened
        if not any(c["rel"] == rel for c in ledger.get("candidates", [])):
            return None
        path = nd.absolute(concept_dir, rel)
        return path if path and os.path.isfile(path) else None

    def _preview_source(self, rel: str) -> np.ndarray | None:
        with self._lock:
            cached = self._previews.get(rel)
            if cached is not None:
                self._previews.move_to_end(rel)
                return cached
        path = self._candidate_path(rel)
        if path is None:
            return None
        pixels = nd.load_rgb(path, nd.PREVIEW_MAX_SIDE)
        with self._lock:
            self._previews[rel] = pixels
            while len(self._previews) > _PREVIEW_CACHE_SIZE:
                self._previews.popitem(last=False)
        return pixels

    def preview(self, rel: str, adjust: dict | None) -> bytes | None:
        """JPEG of the candidate at preview size with ``adjust`` applied
        (neutral settings = the original at the same size, so the two halves
        of the wipe line up exactly). The noise seed is the file stem, as on
        commit."""
        source = self._preview_source(rel)
        if source is None:
            return None
        return nd.encode_jpeg(nd.apply_adjustments(source, adjust, rel))

    def decide(self, rel: str, state: str, adjust: dict | None = None) -> dict:
        if state not in ("accepted", "skipped"):
            return {"ok": False, "error": "state must be accepted or skipped"}
        with self._lock:
            opened = self._require()
            if opened is None:
                return {"ok": False, "error": "Open a concept folder first"}
            concept_dir, ledger = opened
            if self._running() and self._job and self._job.get("kind") == "commit":
                return {"ok": False, "error": "A commit is running"}
            path = self._candidate_path(rel)
            if path is None:
                return {"ok": False, "error": "Not a candidate"}
            existing = ledger.setdefault("decisions", {}).get(rel)
            if existing and existing.get("committed"):
                return {"ok": False, "error": "Already committed; restore that commit first"}
            decision = {"state": state, "at": time.time()}
            if state == "accepted":
                settings = nd.Adjust.from_dict(adjust if adjust is not None else {"strength": nd.DEFAULT_STRENGTH})
                source = self._preview_source(rel)
                delta_e = delta_luma = 0.0
                if source is not None:
                    adjusted = nd.apply_adjustments(source, settings, rel)
                    delta_e = nd.mean_delta_e(source, adjusted)
                    delta_luma = nd.mean_abs_luma_delta(source, adjusted)
                decision.update(
                    adjust=settings.to_dict(),
                    strength=settings.strength,
                    delta_e=round(delta_e, 4),
                    delta_luma=round(delta_luma, 4),
                )
            ledger["decisions"][rel] = decision
            nd.save_ledger(concept_dir, ledger)
            return {"ok": True, "decision": decision}

    def undo(self, rel: str) -> dict:
        with self._lock:
            opened = self._require()
            if opened is None:
                return {"ok": False, "error": "Open a concept folder first"}
            concept_dir, ledger = opened
            existing = ledger.get("decisions", {}).get(rel)
            if existing is None:
                return {"ok": True, "removed": False}
            if existing.get("committed"):
                return {"ok": False, "error": "Already committed; restore that commit first"}
            del ledger["decisions"][rel]
            nd.save_ledger(concept_dir, ledger)
            return {"ok": True, "removed": True}

    # ------------------------------------------------------------------ #
    # Commit / restore
    # ------------------------------------------------------------------ #

    def commit(self, night_edits_dir: str | None, fraction: float) -> dict:
        with self._lock:
            opened = self._require()
            if opened is None:
                return {"ok": False, "error": "Open a concept folder first"}
            if self._running():
                return {"ok": False, "error": "A scan or commit is still running"}
            concept_dir, ledger = opened
            pending = sum(
                1
                for d in ledger.get("decisions", {}).values()
                if d.get("state") == "accepted" and not d.get("committed")
            )
            if pending == 0:
                return {"ok": False, "error": "Nothing accepted to commit"}
            fraction = float(min(max(fraction, 0.0), 1.0))
            count = nd.night_edits_count(pending, fraction)
            night = os.path.abspath(night_edits_dir) if night_edits_dir else nd.default_night_edits_dir(concept_dir)
            if count:
                if os.path.normcase(night) == os.path.normcase(concept_dir):
                    return {"ok": False, "error": "The Night Edits folder must differ from the source concept"}
                problem = nd.night_edits_preflight(night, count)
                if problem:
                    return {"ok": False, "error": f"Night Edits: {problem}"}
            else:
                night = None
            self._stop = threading.Event()
            self._job = {
                "kind": "commit",
                "phase": "adjust",
                "done": 0,
                "total": pending,
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
                "pairs_planned": count,
                "night_edits_dir": night,
                "commit_id": None,
            }
            self._thread = threading.Thread(
                target=self._commit_worker,
                args=(concept_dir, ledger, night, fraction, self._stop),
                name="night-darken-commit",
                daemon=True,
            )
            self._thread.start()
            return {"ok": True, "pending": pending, "pairs_planned": count, "night_edits_dir": night}

    def _commit_worker(self, concept_dir, ledger, night, fraction, stop) -> None:
        try:
            commit = nd.commit_accepted(
                concept_dir,
                ledger,
                night_edits_dir=night,
                fraction=fraction,
                progress=self._progress,
                stop=stop,
            )
            with self._lock:
                self._job["commit_id"] = commit["id"]
                self._job["phase"] = "cancelled" if stop.is_set() else "done"
                self._previews.clear()
            logger.info(
                "night darken commit %s: %d images adjusted in place, %d Night Edits pairs",
                commit["id"],
                len(commit["items"]),
                sum(1 for it in commit["items"] if it.get("night_pair")),
            )
        except Exception as ex:
            logger.exception("night darken commit failed")
            with self._lock:
                if self._job is not None:
                    self._job["error"] = str(ex)
                    self._job["phase"] = "error"
        finally:
            with self._lock:
                if self._job is not None:
                    self._job["finished_at"] = time.time()

    def restore(self, commit_id: str) -> dict:
        with self._lock:
            opened = self._require()
            if opened is None:
                return {"ok": False, "error": "Open a concept folder first"}
            if self._running():
                return {"ok": False, "error": "A scan or commit is still running"}
            concept_dir, ledger = opened
            try:
                result = nd.restore_commit(concept_dir, ledger, commit_id)
            except KeyError:
                return {"ok": False, "error": "Unknown commit"}
            self._previews.clear()
            logger.info(
                "night darken restore %s: %d originals back, %d pairs removed, %d missing",
                commit_id,
                result["restored"],
                result["pairs_removed"],
                len(result["missing"]),
            )
            return {"ok": True, **result}
