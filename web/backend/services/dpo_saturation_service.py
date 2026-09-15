"""DPO Saturation: score how much preference the lineage has already absorbed
per DPO pair, then retire the pairs that are dead weight.

The scoring pass loads a full diffusion checkpoint (two of them, one at a
time), so unlike the other DPO tools it does not run in a backend thread: it
is the CLI ``scripts/util/dpo_saturation_score.py score`` launched as a child
process with ``--progress-json``, so the OS reclaims every byte when it exits
and a native crash cannot take the backend down. This service only pumps the
child's stdout (progress lines + a log tail for the GUI) and holds the phase.

Everything after scoring is cheap and in-process over the concept's table
(``<concept>/.dpo_saturation/scores.json``): the summary at a chosen beta and
dead-weight budget, the histogram and utility curve, both tails for the
gallery, and retire/restore, all from ``modules.util.dpo_saturation_util``.

Errors are returned as ``{"ok": False, "error": ...}``; HTTP stays 200.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from modules.util import dpo_saturation_util as sat
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCORE_SCRIPT = _PROJECT_ROOT / "scripts" / "util" / "dpo_saturation_score.py"
_LOG_TAIL = 200
_MAX_ROWS = 400


class DPOSaturationService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._pump_thread: threading.Thread | None = None
        self._phase = "idle"  # idle | running | done | error | cancelled
        self._progress_phase = ""
        self._done = 0
        self._total = 0
        self._error: str | None = None
        self._log: deque[str] = deque(maxlen=_LOG_TAIL)
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._last_request: dict = {}
        # Test hook: replace the child launcher.
        self.popen_factory = subprocess.Popen

    # ------------------------------------------------------------------ #
    # Concepts and tables
    # ------------------------------------------------------------------ #

    def _train_config(self):
        from web.backend.services.config_service import ConfigService

        return ConfigService.get_instance().get_config_for_training()

    def _dpo_concepts(self) -> list:
        from web.backend.services.dpo_service import DPOService

        concepts = DPOService.get_instance()._resolve_concepts()
        return [c for c in concepts if c.enabled and c.is_dpo()]

    def concepts(self) -> dict:
        """Configured DPO concepts with the state of their saturation table."""
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        try:
            config = self._train_config()
            concepts = self._dpo_concepts()
            dirs = dpo_concept_pattern_dirs(concepts)
        except Exception as ex:  # a missing config is not an error here
            logger.warning("could not list configured DPO concepts: %s", ex)
            return {"ok": True, "concepts": [], "warning": str(ex)}
        out = []
        for concept, (chosen_dir, rejected_dir) in zip(concepts, dirs, strict=True):
            table = None
            try:
                table = sat.load_table(concept.path)
            except (OSError, ValueError) as ex:
                logger.warning("unreadable saturation table for %s: %s", concept.path, ex)
            entry = {
                "name": concept.name,
                "path": concept.path,
                "chosen_dir": chosen_dir,
                "rejected_dir": rejected_dir,
                "beta_override": float(concept.dpo_beta_override or 0.0),
                "retired": len(sat.retired_keys(concept.path)),
                "table": None,
            }
            if table is not None:
                scored = sum(1 for p in table.get("pairs", {}).values() if (p.get("origin") and p.get("current")))
                entry["table"] = {
                    "pairs": len(table.get("pairs", {})),
                    "scored": scored,
                    "draws": len(table.get("draws", [])),
                    "origin": (table.get("origin") or {}).get("path"),
                    "current": (table.get("current") or {}).get("path"),
                    "current_scored_at": table.get("current_scored_at"),
                    "num_train_timesteps": table.get("num_train_timesteps"),
                }
            out.append(entry)
        return {
            "ok": True,
            "concepts": out,
            "global_beta": float(config.rlhf_dpo_beta),
            "default_origin": config.transformer.model_name,
            "adapter": config.lora_model_name or "",
            "cache_dir": config.cache_dir,
        }

    def _concept_by_path(self, concept_path: str):
        wanted = os.path.normcase(os.path.abspath(concept_path))
        for concept in self._dpo_concepts():
            if os.path.normcase(os.path.abspath(concept.path)) == wanted:
                return concept
        return None

    # ------------------------------------------------------------------ #
    # Scoring subprocess
    # ------------------------------------------------------------------ #

    def _running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(
        self,
        *,
        current: str = "",
        origin: str = "",
        adapter: str = "",
        draws: int = 8,
        timesteps: str = "",
        seeds: int = 2,
        seed_base: int = 1_000_000,
        batch_size: int = 1,
        cache_dir: str = "",
        force_origin: bool = False,
        concepts: list[str] | None = None,
    ) -> dict:
        with self._lock:
            if self._running():
                return {"ok": False, "error": "A scoring run is already in progress."}
            if not adapter and not current:
                return {"ok": False, "error": "Pick a current checkpoint, or an adapter to score against the base."}
            if adapter and (current or origin):
                return {"ok": False, "error": "Adapter mode cannot be combined with origin/current checkpoints."}
            for label, path in (("current", current), ("origin", origin), ("adapter", adapter)):
                if path and not os.path.isfile(path):
                    return {"ok": False, "error": f"{label} checkpoint not found: {path}"}
            if not timesteps and draws <= 0:
                return {"ok": False, "error": "Draws must be at least 1 unless fixed timesteps are given."}
            if not _SCORE_SCRIPT.is_file():
                return {"ok": False, "error": f"scorer script missing: {_SCORE_SCRIPT}"}

            try:
                config_path, concepts_path = self._write_config_files()
            except Exception as ex:
                logger.exception("DPO saturation: could not write config")
                return {"ok": False, "error": f"Could not prepare the training config: {ex}"}

            cmd = [
                sys.executable,
                "-u",
                str(_SCORE_SCRIPT),
                "score",
                "--config",
                config_path,
                "--progress-json",
                "--seeds",
                str(int(seeds)),
                "--seed-base",
                str(int(seed_base)),
                "--batch-size",
                str(max(1, int(batch_size))),
            ]
            if concepts_path:
                cmd += ["--concepts", concepts_path]
            if adapter:
                cmd += ["--adapter", adapter]
            else:
                cmd += ["--current", current]
                if origin:
                    cmd += ["--origin", origin]
            if timesteps:
                cmd += ["--timesteps", timesteps]
            else:
                cmd += ["--draws", str(int(draws))]
            if cache_dir:
                cmd += ["--cache-dir", cache_dir]
            if force_origin:
                cmd.append("--force-origin")
            for token in concepts or []:
                cmd += ["--concept", token]

            self._phase = "running"
            self._progress_phase = "starting"
            self._done = 0
            self._total = 0
            self._error = None
            self._log.clear()
            self._started_at = time.time()
            self._finished_at = None
            self._last_request = {
                "current": current,
                "origin": origin,
                "adapter": adapter,
                "draws": draws,
                "timesteps": timesteps,
                "seeds": seeds,
                "concepts": concepts or [],
            }
            try:
                self._process = self.popen_factory(
                    cmd,
                    cwd=str(_PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as ex:
                self._phase = "error"
                self._error = f"Could not launch the scorer: {ex}"
                return {"ok": False, "error": self._error}
            self._pump_thread = threading.Thread(
                target=self._pump, args=(self._process,), daemon=True, name="dpo-saturation-pump"
            )
            self._pump_thread.start()
        return {"ok": True}

    def _write_config_files(self) -> tuple[str, str | None]:
        """The active training config (and inline concepts, if any) as files
        the CLI can read. Written next to the workspace so they survive the
        run for inspection; replaced on every start."""
        config = self._train_config()
        temp_dir = os.path.join(tempfile.gettempdir(), "onetrainer_dpo_saturation")
        os.makedirs(temp_dir, exist_ok=True)
        config_path = os.path.join(temp_dir, "train_config.json")
        concepts_path = None
        payload = config.to_dict()
        if config.concepts:
            concepts_path = os.path.join(temp_dir, "concepts.json")
            with open(concepts_path, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in config.concepts], f)
            payload["concept_file_name"] = concepts_path
        payload["concepts"] = None
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return config_path, concepts_path

    def _pump(self, process: subprocess.Popen) -> None:
        try:
            assert process.stdout is not None
            for raw in process.stdout:
                line = raw.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("{") and '"progress"' in line:
                    try:
                        event = json.loads(line)["progress"]
                        with self._lock:
                            self._progress_phase = str(event.get("phase", ""))
                            self._done = int(event.get("done", 0))
                            self._total = int(event.get("total", 0))
                        continue
                    except (ValueError, KeyError, TypeError):
                        pass
                # tqdm rewrites its line with \r; keep only the last segment.
                line = line.split("\r")[-1].strip()
                if line:
                    with self._lock:
                        self._log.append(line)
        except Exception as ex:  # pragma: no cover - defensive
            logger.exception("DPO saturation: output pump failed")
            with self._lock:
                self._log.append(f"[pump error] {ex}")
        finally:
            self._finish(process, process.wait())

    def _finish(self, process: subprocess.Popen, code: int) -> None:
        with self._lock:
            if process is not self._process:
                return
            self._finished_at = time.time()
            if self._phase == "cancelled":
                return
            if code == 0:
                self._phase = "done"
                self._progress_phase = "done"
            else:
                self._phase = "error"
                tail = [entry for entry in list(self._log)[-6:] if entry]
                self._error = f"scorer exited with code {code}" + (": " + tail[-1] if tail else "")

    def status(self) -> dict:
        with self._lock:
            running = self._running()
            return {
                "ok": True,
                "running": running,
                "phase": self._phase,
                "progress_phase": self._progress_phase,
                "done": self._done,
                "total": self._total,
                "error": self._error,
                "log": list(self._log)[-40:],
                "started_at": self._started_at,
                "elapsed": (
                    ((self._finished_at or time.time()) - self._started_at) if self._started_at is not None else 0.0
                ),
                "request": dict(self._last_request),
            }

    def cancel(self) -> dict:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return {"ok": True, "cancelled": False}
            self._phase = "cancelled"
            self._progress_phase = "cancelled"
        with contextlib.suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
        return {"ok": True, "cancelled": True}

    # ------------------------------------------------------------------ #
    # Analysis over the table
    # ------------------------------------------------------------------ #

    def _load(self, concept_path: str) -> tuple[dict | None, str | None]:
        if self._concept_by_path(concept_path) is None:
            return None, "That folder is not an enabled DPO concept of the current config."
        try:
            table = sat.load_table(concept_path)
        except (OSError, ValueError) as ex:
            return None, f"Could not read the saturation table: {ex}"
        if table is None:
            return None, "No saturation table yet: run a scoring pass first."
        return table, None

    def summary(
        self,
        concept_path: str,
        *,
        beta: float | None = None,
        dead_weight: float = sat.DEFAULT_DEAD_WEIGHT,
        require_all_buckets: bool = False,
        min_draws: int = 1,
        tails: int = 24,
        bins: int = 40,
    ) -> dict:
        table, error = self._load(concept_path)
        if error:
            return {"ok": False, "error": error}
        retired = sat.retired_keys(concept_path)
        try:
            summary = sat.summarize(
                table,
                beta=beta,
                dead_weight=dead_weight,
                require_all_buckets=require_all_buckets,
                min_draws=min_draws,
                tails=tails,
                exclude_keys=retired,
            )
        except ValueError as ex:
            return {"ok": False, "error": str(ex)}
        rows = summary.pop("rows")
        logits = [r["mean_logit"] for r in rows]
        # Utility curve as points: pairs kept (highest weight first) -> share of total weight.
        weights = sorted((r["mean_weight"] for r in rows), reverse=True)
        total_w = sum(weights) or 1.0
        curve = []
        running = 0.0
        for i, w in enumerate(weights, start=1):
            running += w
            if i == len(weights) or i % max(1, len(weights) // 100) == 0:
                curve.append({"pairs": i, "share": running / total_w})
        saturated_rows = [r for r in rows if r["saturated"]]
        saturated_rows.sort(key=lambda r: r["mean_logit"], reverse=True)
        summary.update(
            {
                "ok": True,
                "concept_path": concept_path,
                "retired": len(retired),
                "origin": (table.get("origin") or {}).get("path"),
                "current": (table.get("current") or {}).get("path"),
                "current_scored_at": table.get("current_scored_at"),
                "logit_histogram": sat.histogram(logits, bins=bins),
                "utility_curve": curve,
                "saturated_rows": saturated_rows[:_MAX_ROWS],
                "saturated_truncated": max(0, len(saturated_rows) - _MAX_ROWS),
            }
        )
        return summary

    def pair(self, concept_path: str, key: str, *, beta: float | None = None) -> dict:
        """One pair's per-draw margins for the detail view."""
        table, error = self._load(concept_path)
        if error:
            return {"ok": False, "error": error}
        entry = table.get("pairs", {}).get(key)
        if entry is None:
            return {"ok": False, "error": "Unknown pair."}
        beta_used = sat.resolve_beta(table, beta)
        T = int(table.get("num_train_timesteps") or 1000)
        draws = [
            {"t": t, "t_frac": t / T, "margin": m, "logit": beta_used * m, "weight": sat.gradient_weight(m, beta_used)}
            for t, m in sat.pair_draws(entry)
        ]
        return {
            "ok": True,
            "key": key,
            "chosen": entry.get("chosen"),
            "rejected": entry.get("rejected"),
            "beta": beta_used,
            "draws": draws,
            "retired": key in sat.retired_keys(concept_path),
        }

    def retire(
        self,
        concept_path: str,
        *,
        keys: list[str] | None = None,
        beta: float | None = None,
        dead_weight: float = sat.DEFAULT_DEAD_WEIGHT,
        require_all_buckets: bool = False,
        min_draws: int = 1,
        reason: str = "",
        dry_run: bool = False,
    ) -> dict:
        """Retire the given keys, or every saturated pair under the given rule."""
        if self._running():
            return {"ok": False, "error": "Wait for the scoring run to finish before moving pairs."}
        table, error = self._load(concept_path)
        if error:
            return {"ok": False, "error": error}
        if keys is None:
            try:
                summary = sat.summarize(
                    table,
                    beta=beta,
                    dead_weight=dead_weight,
                    require_all_buckets=require_all_buckets,
                    min_draws=min_draws,
                    tails=0,
                    exclude_keys=sat.retired_keys(concept_path),
                )
            except ValueError as ex:
                return {"ok": False, "error": str(ex)}
            keys = summary["saturated_keys"]
            reason = reason or (
                f"saturated at beta {summary['beta']:g}, weight < {dead_weight:g}"
                + (" in every bucket" if require_all_buckets else "")
            )
        try:
            moved = sat.retire_pairs(concept_path, table, keys, reason=reason or "manual", dry_run=dry_run)
        except OSError as ex:
            return {"ok": False, "error": f"Move failed, nothing changed for that pair: {ex}"}
        return {
            "ok": True,
            "moved": [m["key"] for m in moved],
            "dry_run": dry_run,
            "retired": len(sat.retired_keys(concept_path)),
        }

    def restore(self, concept_path: str, keys: list[str] | None = None) -> dict:
        if self._running():
            return {"ok": False, "error": "Wait for the scoring run to finish before moving pairs."}
        if self._concept_by_path(concept_path) is None:
            return {"ok": False, "error": "That folder is not an enabled DPO concept of the current config."}
        try:
            restored = sat.restore_pairs(concept_path, keys)
        except OSError as ex:
            return {"ok": False, "error": f"Restore failed: {ex}"}
        return {"ok": True, "restored": [r["key"] for r in restored], "retired": len(sat.retired_keys(concept_path))}

    def retired(self, concept_path: str) -> dict:
        if self._concept_by_path(concept_path) is None:
            return {"ok": False, "error": "That folder is not an enabled DPO concept of the current config."}
        records = [r for r in sat.load_retired_log(concept_path) if not r.get("restored_at")]
        table = None
        with contextlib.suppress(OSError, ValueError):
            table = sat.load_table(concept_path)
        out = []
        for record in records:
            moves = record.get("moves") or []
            chosen = next((m["to"] for m in moves if _is_image(m["to"]) and "chosen" in _side_of(m, table)), None)
            rejected = next((m["to"] for m in moves if _is_image(m["to"]) and "rejected" in _side_of(m, table)), None)
            out.append(
                {
                    "key": record["key"],
                    "reason": record.get("reason", ""),
                    "retired_at": record.get("retired_at"),
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
        return {"ok": True, "pairs": out}

    def serve_image(self, path: str) -> str | None:
        """Only files the tables or the retired logs name are served."""
        if not path or not os.path.isfile(path):
            return None
        wanted = os.path.normcase(os.path.abspath(path))
        for concept in self._dpo_concepts():
            try:
                table = sat.load_table(concept.path)
            except (OSError, ValueError):
                table = None
            for entry in (table or {}).get("pairs", {}).values():
                for candidate in (entry.get("chosen"), entry.get("rejected")):
                    if candidate and os.path.normcase(os.path.abspath(candidate)) == wanted:
                        return os.path.abspath(path)
            for record in sat.load_retired_log(concept.path):
                for move in record.get("moves") or []:
                    for candidate in (move.get("from"), move.get("to")):
                        if candidate and os.path.normcase(os.path.abspath(candidate)) == wanted:
                            return os.path.abspath(path)
        return None


def _is_image(path: str) -> bool:
    from modules.util.path_util import is_supported_image_extension

    return is_supported_image_extension(os.path.splitext(path)[1])


def _side_of(move: dict, table: dict | None) -> str:
    """'chosen' / 'rejected' by which side folder the original file lived in."""
    concept = (table or {}).get("concept") or {}
    src = os.path.normcase(os.path.abspath(move.get("from", "")))
    for side in ("chosen", "rejected"):
        side_dir = concept.get(f"{side}_dir")
        if side_dir and src.startswith(os.path.normcase(os.path.abspath(side_dir)) + os.sep):
            return side
    # Fallback for a table-less concept: the folder name.
    return "chosen" if f"{os.sep}chosen{os.sep}" in src else "rejected"
