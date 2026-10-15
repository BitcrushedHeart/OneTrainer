"""Live DPO pair saturation, recorded while a run trains.

The offline scorer (``dpo_saturation_util`` and ``scripts/util/
dpo_saturation_score.py``) measures how much preference the *lineage* has
absorbed, against a fixed origin checkpoint. This module answers the in-run
question instead: which pairs still put gradient into *this* run, and whether
each concept's beta suits where its pairs currently sit.

Every DPO micro-step already computes each pair's raw margin against the run's
frozen reference; the trainer hands those per-sample values here. Two consumers:

* ``DPOConceptWindow`` - per-concept sums over one gradient-accumulation
  window, logged to TensorBoard under ``dpo/concept/<concept>/...``.
* ``LiveSaturationLedger`` - a per-pair visit history under
  ``<workspace>/dpo_live/``, rewritten as ``ledger.json`` (state, survives a
  resume), ``report.md`` (per-concept verdicts and beta advice) and
  ``pairs.csv`` (one row per pair).

Margins here are relative to the run's DPO reference. A rebase or a fresh
capture restarts every margin at zero, so the ledger fingerprints the reference
and archives its history when that changes; a resume that restores the same
reference keeps it. The numbers are not comparable with the offline
origin-relative scores and are deliberately stored apart from them.

The gradient factor is the sigmoid objective's (``gradient_weight``); label
smoothing is not modelled, as in the offline tool.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import statistics
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime

from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
from modules.util.dpo_saturation_util import (
    DEFAULT_DEAD_WEIGHT,
    DEFAULT_QUARTILES,
    concept_for_chosen_path,
    gradient_weight,
    is_saturated,
    sigmoid,
    timestep_bucket,
)

LIVE_DIRNAME = "dpo_live"
LEDGER_FILENAME = "ledger.json"
REPORT_FILENAME = "report.md"
PAIRS_FILENAME = "pairs.csv"
ARCHIVE_DIRNAME = "archive"
LEDGER_VERSION = 1

DEFAULT_MAX_VISITS = 16
DEFAULT_MIN_VISITS = 3
DEFAULT_FLUSH_SECONDS = 60.0
# A concept counts as "near the peak" while its typical pair gets at least this
# fraction of the push the best beta would give it.
NEAR_PEAK_EFFICIENCY = 0.8
TOP_PAIRS = 10

# One visit in ledger.json: [global_step, t (0..1, or None), margin, beta, chosen_reward, rejected_reward].
_STEP, _T, _MARGIN, _BETA, _CHOSEN, _REJECTED = range(6)

VERDICT_VIOLATING = "violating"
VERDICT_LEARNING = "learning"
VERDICT_DONE = "done"
VERDICT_NEW = "new"
VERDICT_ORDER = (VERDICT_VIOLATING, VERDICT_LEARNING, VERDICT_DONE, VERDICT_NEW)


# --------------------------------------------------------------------------- #
# Push math
# --------------------------------------------------------------------------- #


def push(beta: float, margin: float) -> float:
    """Gradient magnitude a pair lands on each log-prob: ``beta * sigmoid(-beta * margin)``.

    ``gradient_weight`` is the sigmoid factor alone, the offline tool's
    saturation measure; this is the force itself, the quantity to compare across
    concepts with different betas."""
    return beta * gradient_weight(margin, beta)


def _solve_push_peak_logit() -> float:
    # For a fixed margin m, d/dbeta [beta * sigmoid(-beta*m)] = 0 exactly where
    # x * sigmoid(x) = 1 with x = beta*m. The left side is increasing for x > 0.
    lo, hi = 0.5, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid * sigmoid(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# beta * margin at which a pair's push peaks (~1.2785): the best beta for a
# typical margin m is PUSH_PEAK_LOGIT / m. Past it, raising beta shrinks the push
# on pairs already on the right side; below it, raising beta grows it.
PUSH_PEAK_LOGIT = _solve_push_peak_logit()


def beta_advice(
    typical_margin: float | None,
    applied_beta: float | None,
    configured_beta: float,
    violating_margin: float | None = None,
) -> dict:
    """Where a concept's beta sits relative to the push peak at its typical margin.

    ``applied_beta`` is what the loss used (after the adaptive controller's
    ratio); ``configured_beta`` is the concept's override, or the global beta.
    The suggestion is converted back into the configured value so it can be
    typed into the concept as-is."""
    if typical_margin is None or applied_beta is None or applied_beta <= 0:
        return {"kind": "no-data"}
    if typical_margin <= 0:
        return {"kind": "unlearned"}
    peak_beta = PUSH_PEAK_LOGIT / typical_margin
    efficiency = push(applied_beta, typical_margin) / push(peak_beta, typical_margin)
    if efficiency >= NEAR_PEAK_EFFICIENCY:
        kind = "near-peak"
    elif applied_beta > peak_beta:
        kind = "high"
    else:
        kind = "low"
    adaptive_scale = applied_beta / configured_beta if configured_beta > 0 else 1.0
    violator_gain = None
    if violating_margin is not None and violating_margin < 0:
        violator_gain = push(peak_beta, violating_margin) / push(applied_beta, violating_margin)
    return {
        "kind": kind,
        "efficiency": efficiency,
        "peak_beta_applied": peak_beta,
        "suggested_beta": peak_beta / adaptive_scale,
        "typical_gain": 1.0 / efficiency,
        "violator_gain": violator_gain,
    }


# --------------------------------------------------------------------------- #
# Concepts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DPOConceptInfo:
    label: str  # readable, from the concept folder
    tag: str  # TensorBoard-safe
    path: str
    chosen_dir: str
    beta_override: float  # <= 0: inherits the global beta

    def configured_beta(self, global_beta: float) -> float:
        return self.beta_override if self.beta_override > 0 else global_beta


def _tag(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_") or "concept"


def dpo_concept_infos(concepts: Iterable) -> list[DPOConceptInfo]:
    """One entry per enabled DPO pattern concept, labelled by its folder name.

    Concept names are not unique (two concepts can both be called "DPO
    Off-Policy"); folder names nearly always are, and a clash gets a suffix."""
    infos = []
    tags_seen: dict[str, int] = {}
    for concept in concepts:
        if not concept.enabled or not concept.is_dpo():
            continue
        dirs = dpo_concept_pattern_dirs([concept])
        if not dirs:
            continue
        label = os.path.basename(os.path.normpath(concept.path)) or concept.name or "concept"
        tag = _tag(label)
        seen = tags_seen.get(tag, 0) + 1
        tags_seen[tag] = seen
        if seen > 1:
            label, tag = f"{label} ({seen})", f"{tag}_{seen}"
        infos.append(
            DPOConceptInfo(
                label=label,
                tag=tag,
                path=os.path.normpath(concept.path),
                chosen_dir=dirs[0][0],
                beta_override=float(concept.dpo_beta_override or 0.0),
            )
        )
    return infos


_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _path_parts(path: str) -> list[str]:
    return [part.casefold() for part in re.split(r"[\\/]+", path) if part]


class DPOConceptResolver:
    """Maps a chosen image path to its concept (longest chosen-dir prefix)."""

    def __init__(self, infos: Iterable[DPOConceptInfo]):
        self.infos = list(infos)
        self._dirs = [(info.chosen_dir, "") for info in self.infos]
        self._dir_parts = [_path_parts(info.chosen_dir) for info in self.infos]
        self._cache: dict[str, DPOConceptInfo | None] = {}

    def resolve(self, image_path: str | None) -> DPOConceptInfo | None:
        if not image_path:
            return None
        if image_path not in self._cache:
            index = concept_for_chosen_path(image_path, self._dirs)
            if index is None and _DRIVE_PATH.match(image_path):
                index = self._resolve_remapped(image_path)
            self._cache[image_path] = None if index is None else self.infos[index]
        return self._cache[image_path]

    def _resolve_remapped(self, image_path: str) -> int | None:
        """A sourceless cloud run keeps the local machine's pair paths (``F:\\...``) in
        its cache, while CloudTrainer moved every concept to ``<remote_dir>/remote/``
        plus the path without its drive. Match the drive-less pair path against the
        tail of each chosen dir; the longest matching tail wins."""
        parts = _path_parts(image_path[2:])
        best, best_len = None, 1  # a bare "chosen" tail is too weak to count
        for index, dir_parts in enumerate(self._dir_parts):
            for start in range(len(dir_parts)):
                tail = dir_parts[start:]
                if len(tail) > best_len and len(tail) < len(parts) and parts[: len(tail)] == tail:
                    best, best_len = index, len(tail)
                    break
        return best


# --------------------------------------------------------------------------- #
# Per-concept TensorBoard window
# --------------------------------------------------------------------------- #


class DPOConceptWindow:
    """Per-concept sums over one gradient-accumulation window."""

    _FIELDS = ("count", "margin", "correct", "chosen", "rejected", "beta", "weight")

    def __init__(self, *, with_weight: bool = True):
        # The gradient factor only describes the sigmoid objective; IPO has no beta.
        self.with_weight = with_weight
        self._sums: dict[str, dict[str, float]] = {}

    def add(self, info: DPOConceptInfo, record: dict) -> None:
        margin, beta = record["margin"], record["beta"]
        sums = self._sums.setdefault(info.tag, dict.fromkeys(self._FIELDS, 0.0))
        sums["count"] += 1.0
        sums["margin"] += margin
        sums["correct"] += 1.0 if margin > 0 else 0.0
        sums["chosen"] += record["chosen"]
        sums["rejected"] += record["rejected"]
        sums["beta"] += beta
        if self.with_weight:
            sums["weight"] += gradient_weight(margin, beta)

    def scalars(self) -> dict[str, float]:
        out = {}
        for tag, sums in self._sums.items():
            n = sums["count"]
            if n <= 0:
                continue
            prefix = f"dpo/concept/{tag}/"
            out[prefix + "pairs"] = n
            out[prefix + "margin"] = sums["margin"] / n
            out[prefix + "accuracy"] = sums["correct"] / n
            out[prefix + "chosen_reward"] = sums["chosen"] / n
            out[prefix + "rejected_reward"] = sums["rejected"] / n
            out[prefix + "beta"] = sums["beta"] / n
            if self.with_weight:
                out[prefix + "grad_weight"] = sums["weight"] / n
        return out

    def reset(self) -> None:
        self._sums.clear()


# --------------------------------------------------------------------------- #
# Per-pair rows
# --------------------------------------------------------------------------- #


def _round(x: float | None) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.6g}")


def pair_row(
    key: str, entry: dict, min_visits: int, buckets: int = DEFAULT_QUARTILES, beta: float | None = None
) -> dict:
    """A pair's retained visits in the offline ``score_pairs`` row schema, so
    ``is_saturated`` applies unchanged, plus the live fields and a verdict.

    ``beta`` scores every visit at the beta the concept applies *now* rather
    than the one stored with each visit. A pair's history outlives a retune (and
    the adaptive controller moves beta constantly), and the question a verdict
    answers - does this pair still move the run? - is about the current beta."""
    visits = entry["visits"]
    n = len(visits)
    margins = [v[_MARGIN] for v in visits]
    weights = [gradient_weight(v[_MARGIN], beta if beta is not None else v[_BETA]) for v in visits]
    per_bucket_w: list[list[float]] = [[] for _ in range(buckets)]
    per_bucket_m: list[list[float]] = [[] for _ in range(buckets)]
    for visit, weight in zip(visits, weights, strict=True):
        t = visit[_T]
        if t is None:
            continue
        # t is a fraction of the schedule; bucket it on a 1000-step grid like the offline tool.
        bucket = timestep_bucket(min(999, max(0, int(t * 1000))), 1000, buckets)
        per_bucket_w[bucket].append(weight)
        per_bucket_m[bucket].append(visit[_MARGIN])
    row = {
        "key": key,
        "concept": entry.get("concept"),
        "chosen": key,
        "draws": n,
        "total_visits": n + int(entry.get("dropped", 0)),
        "mean_margin": sum(margins) / n,
        "mean_logit": sum((beta if beta is not None else v[_BETA]) * v[_MARGIN] for v in visits) / n,
        "mean_weight": sum(weights) / n,
        "min_weight": min(weights),
        "max_weight": max(weights),
        "bucket_weight": [sum(b) / len(b) if b else None for b in per_bucket_w],
        "bucket_margin": [sum(b) / len(b) if b else None for b in per_bucket_m],
        "wins": sum(1 for m in margins if m > 0) / n,
        "mean_chosen": sum(v[_CHOSEN] for v in visits) / n,
        "mean_rejected": sum(v[_REJECTED] for v in visits) / n,
        "last_step": visits[-1][_STEP],
    }
    row["verdict"] = pair_verdict(row, min_visits)
    return row


def pair_verdict(row: dict, min_visits: int) -> str:
    if row["draws"] < min_visits:
        return VERDICT_NEW
    if is_saturated(row, dead_weight=DEFAULT_DEAD_WEIGHT, require_all_buckets=False, min_draws=min_visits):
        return VERDICT_DONE
    if row["wins"] < 0.5:
        return VERDICT_VIOLATING
    return VERDICT_LEARNING


def concept_summary(
    info: DPOConceptInfo,
    entries: list[dict],
    rows: list[dict],
    global_beta: float,
    applied_beta: float | None = None,
) -> dict:
    """Verdict counts plus pooled visit statistics and beta advice for one concept.

    ``applied_beta`` is the beta the loss uses now (see ``pair_row``); without it
    the pooled median over the stored visits stands in."""
    visits = [v for entry in entries for v in entry["visits"]]
    counts = dict.fromkeys(VERDICT_ORDER, 0)
    for row in rows:
        counts[row["verdict"]] += 1
    summary = {
        "label": info.label,
        "pairs": len(rows),
        "visits": len(visits),
        "verdicts": counts,
        "configured_beta": info.configured_beta(global_beta),
        "median_margin": None,
        "applied_beta": None,
        "median_logit": None,
        "win_rate": None,
        "violating_margin": None,
        "retuned_from": None,  # a beta the stored history still carries, if it differs
        "advice": {"kind": "no-data"},
    }
    if not visits:
        return summary
    margins = [v[_MARGIN] for v in visits]
    negative = [m for m in margins if m < 0]
    beta = applied_beta if applied_beta is not None else statistics.median(v[_BETA] for v in visits)
    stored = {v[_BETA] for v in visits if v[_BETA] is not None}
    summary.update(
        median_margin=statistics.median(margins),
        applied_beta=beta,
        median_logit=statistics.median(beta * m for m in margins),
        win_rate=sum(1 for m in margins if m > 0) / len(margins),
        violating_margin=statistics.median(negative) if negative else None,
        retuned_from=min(stored) if stored and any(abs(b - beta) > 1e-9 for b in stored) else None,
    )
    summary["advice"] = beta_advice(
        summary["median_margin"], summary["applied_beta"], summary["configured_beta"], summary["violating_margin"]
    )
    return summary


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


class LiveSaturationLedger:
    """Per-pair visit history for the run's current DPO reference, flushed to
    ``out_dir`` as ledger.json, report.md and pairs.csv.

    ``record`` is in-memory only. ``flush`` writes at most every
    ``flush_seconds`` unless forced (backups, the end of training), always
    atomically, and never raises: a locked or full disk costs the report, not
    the run.
    """

    def __init__(
        self,
        out_dir: str,
        concepts: Iterable[DPOConceptInfo],
        *,
        global_beta: float,
        start_step: int,
        max_visits: int = DEFAULT_MAX_VISITS,
        min_visits: int = DEFAULT_MIN_VISITS,
        flush_seconds: float = DEFAULT_FLUSH_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.out_dir = out_dir
        self.concepts = {info.label: info for info in concepts}
        self.global_beta = float(global_beta)
        self.max_visits = max(1, int(max_visits))
        self.min_visits = max(1, int(min_visits))
        self.flush_seconds = float(flush_seconds)
        self._clock = clock
        self.reference: str | None = None
        self.reference_step: int | None = None
        self.pairs: dict[str, dict] = {}
        self._dirty = False
        self._last_flush: float | None = None
        self._warned: set[str] = set()
        # Adopted or archived at the first record, once the reference is known.
        self._resumed = self._read_existing(int(start_step))

    # ----------------------------------------------------------------- state

    def _read_existing(self, start_step: int) -> dict | None:
        path = self._path(LEDGER_FILENAME)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            self._warn("read", f"DPO live saturation: ignoring unreadable {path}: {e}")
            return {"reference": None, "pairs": {}}
        if not isinstance(data, dict) or data.get("version") != LEDGER_VERSION:
            return {"reference": None, "pairs": {}}
        # A resume replays from the backup's step: visits at or past it belong
        # to the abandoned stretch of the previous attempt.
        pairs = data.get("pairs") or {}
        for key in list(pairs):
            entry = pairs[key]
            entry["visits"] = [v for v in entry.get("visits", []) if v[_STEP] < start_step]
            if not entry["visits"]:
                del pairs[key]
        data["pairs"] = pairs
        return data

    def record(self, step: int, info: DPOConceptInfo, record: dict, reference: str) -> None:
        """Adds one pair visit. ``reference`` fingerprints the DPO reference the
        margin was measured against; a new one archives the history first."""
        if reference != self.reference:
            self._switch_reference(reference, step)
        key = record["path"]
        entry = self.pairs.get(key)
        if entry is None:
            entry = self.pairs[key] = {"concept": info.label, "visits": [], "dropped": 0}
        entry["concept"] = info.label
        entry["visits"].append(
            [
                int(step),
                _round(record.get("t")),
                _round(record["margin"]),
                _round(record["beta"]),
                _round(record["chosen"]),
                _round(record["rejected"]),
            ]
        )
        overflow = len(entry["visits"]) - self.max_visits
        if overflow > 0:
            del entry["visits"][:overflow]
            entry["dropped"] = int(entry.get("dropped", 0)) + overflow
        self._dirty = True

    def _switch_reference(self, reference: str, step: int) -> None:
        resumed, self._resumed = self._resumed, None
        if self.reference is None and resumed is not None and resumed.get("reference") == reference:
            # The resume restored the reference this history was measured against.
            self.reference = reference
            self.reference_step = resumed.get("reference_step")
            self.pairs = resumed.get("pairs") or {}
            return
        if self.reference is not None:
            # Rebased mid-run: write out the outgoing history, then set it aside.
            self.flush(step, force=True)
            self._archive(self.reference)
        elif resumed is not None:
            self._archive(resumed.get("reference"))
        self.reference = reference
        self.reference_step = int(step)
        self.pairs = {}
        self._dirty = True

    def _archive(self, reference: str | None) -> None:
        names = [n for n in (LEDGER_FILENAME, REPORT_FILENAME, PAIRS_FILENAME) if os.path.isfile(self._path(n))]
        if not names:
            return
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = os.path.join(self.out_dir, ARCHIVE_DIRNAME, f"{stamp}_ref-{(reference or 'unknown')[:8]}")
        dest, suffix = base, 1
        while os.path.exists(dest):
            suffix += 1
            dest = f"{base}-{suffix}"
        try:
            os.makedirs(dest)
            for name in names:
                os.replace(self._path(name), os.path.join(dest, name))
            print(f"DPO live saturation: the DPO reference changed; previous history archived to {dest}")
        except OSError as e:
            self._warn("archive", f"DPO live saturation: could not archive the previous history to {dest}: {e}")

    # ---------------------------------------------------------------- output

    def current_betas(self) -> dict[str, float]:
        """The beta each concept's loss applies now: the median over its pairs'
        most recent visits. Betas are retuned between restarts and the adaptive
        controller moves them, while a pair's stored visits can span the change."""
        latest: dict[str, list[float]] = {}
        for entry in self.pairs.values():
            if entry["visits"]:
                newest = max(entry["visits"], key=lambda v: v[_STEP])
                latest.setdefault(entry.get("concept"), []).append(newest[_BETA])
        return {label: statistics.median(betas) for label, betas in latest.items() if betas}

    def rows(self) -> list[dict]:
        betas = self.current_betas()
        return [
            pair_row(key, entry, self.min_visits, beta=betas.get(entry.get("concept")))
            for key, entry in self.pairs.items()
            if entry["visits"]
        ]

    def summaries(self, rows: list[dict] | None = None) -> list[dict]:
        rows = self.rows() if rows is None else rows
        betas = self.current_betas()
        out = []
        for label, info in self.concepts.items():
            entries = [e for e in self.pairs.values() if e.get("concept") == label]
            out.append(
                concept_summary(
                    info,
                    entries,
                    [r for r in rows if r["concept"] == label],
                    self.global_beta,
                    applied_beta=betas.get(label),
                )
            )
        return out

    def flush(self, step: int, *, force: bool = False) -> bool:
        """Rewrites the three files if anything changed. Throttled to
        ``flush_seconds`` unless ``force``. Returns whether it wrote."""
        if not self._dirty or self.reference is None:
            return False
        now = self._clock()
        if not force and self._last_flush is not None and now - self._last_flush < self.flush_seconds:
            return False
        rows = self.rows()
        summaries = self.summaries(rows)
        self._write(LEDGER_FILENAME, json.dumps(self._state(step), separators=(",", ":")))
        self._write(REPORT_FILENAME, format_live_report(self, summaries, rows, step))
        self._write(PAIRS_FILENAME, pairs_csv(rows))
        self._last_flush = now
        self._dirty = False
        return True

    def _state(self, step: int) -> dict:
        return {
            "version": LEDGER_VERSION,
            "updated_step": int(step),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "reference": self.reference,
            "reference_step": self.reference_step,
            "global_beta": self.global_beta,
            "max_visits": self.max_visits,
            "concepts": {
                label: {"path": info.path, "chosen_dir": info.chosen_dir, "beta_override": info.beta_override}
                for label, info in self.concepts.items()
            },
            "pairs": self.pairs,
        }

    def _path(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    def _write(self, name: str, text: str) -> None:
        path = self._path(name)
        tmp = path + ".tmp"
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, path)
            self._warned.discard(name)
        except OSError as e:
            # Typically the file is open in a spreadsheet; the next flush retries.
            self._warn(name, f"DPO live saturation: could not write {path}: {e}")
            with suppress(OSError):
                os.remove(tmp)

    def _warn(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(message)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

_CSV_COLUMNS = (
    "concept",
    "verdict",
    "visits",
    "total_visits",
    "mean_margin",
    "mean_logit",
    "mean_weight",
    "min_weight",
    "wins",
    "mean_chosen_reward",
    "mean_rejected_reward",
    "weight_q1",
    "weight_q2",
    "weight_q3",
    "weight_q4",
    "last_step",
    "chosen",
)


def _sort_key(row: dict):
    return (row["concept"] or "", VERDICT_ORDER.index(row["verdict"]), row["mean_logit"])


def pairs_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for row in sorted(rows, key=_sort_key):
        quartiles = (list(row["bucket_weight"]) + [None] * 4)[:4]
        writer.writerow(
            [
                row["concept"],
                row["verdict"],
                row["draws"],
                row["total_visits"],
                _num(row["mean_margin"]),
                _num(row["mean_logit"]),
                _num(row["mean_weight"]),
                _num(row["min_weight"]),
                _num(row["wins"]),
                _num(row["mean_chosen"]),
                _num(row["mean_rejected"]),
                *(_num(w) for w in quartiles),
                row["last_step"],
                row["chosen"],
            ]
        )
    return buffer.getvalue()


def _num(x, digits: int = 6) -> str:
    return "" if x is None else f"{x:.{digits}g}"


def _fmt(x, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def _relpath(path: str, root: str) -> str:
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # another drive
        return path
    return path if rel.startswith("..") else rel


def _advice_line(summary: dict) -> str:
    advice = summary["advice"]
    kind = advice["kind"]
    label = f"**{summary['label']}**"
    if kind == "no-data":
        return f"- {label}: no visits yet."
    if kind == "unlearned":
        return (
            f"- {label}: the median visit is on the wrong side of zero, so most pairs are still unlearned "
            "against this reference. Beta here mostly sets how hard those are pushed: raising it pushes harder."
        )
    efficiency = advice["efficiency"]
    if kind == "near-peak":
        return f"- {label}: near the push peak; the typical pair gets {efficiency:.0%} of the most it could."
    direction = "past" if kind == "high" else "short of"
    line = (
        f"- {label}: {direction} the push peak; the typical pair gets {efficiency:.0%} of the most it could. "
        f"Setting beta to ~{advice['suggested_beta']:.0f} puts it at the peak "
        f"(×{advice['typical_gain']:.1f} on the typical pair"
    )
    if advice["violator_gain"] is not None:
        line += f", ×{advice['violator_gain']:.2f} on violating pairs"
    return line + ")."


def format_live_report(ledger: LiveSaturationLedger, summaries: list[dict], rows: list[dict], step: int) -> str:
    lines = [
        "# DPO live saturation",
        "",
        f"Step {step} · updated {datetime.now():%Y-%m-%d %H:%M} · "
        f"reference `{(ledger.reference or '?')[:8]}` since step {ledger.reference_step}",
        "",
        "Margins are measured against this run's DPO reference. They restart at zero when the reference is "
        "rebased or re-captured, so they are not comparable with the offline origin-relative scores in "
        "`.dpo_saturation/`.",
        "",
        f"A pair needs {ledger.min_visits} visits for a verdict. **done**: mean gradient weight below "
        f"{DEFAULT_DEAD_WEIGHT:g}, so it no longer moves this run. **violating**: on the wrong side of zero on "
        f"most visits. Only a pair's last {ledger.max_visits} visits count.",
        "",
        "| concept | pairs | done | learning | violating | new | median margin | beta set | beta applied "
        "| median beta·m | typical push vs peak |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        v = s["verdicts"]
        lines.append(
            f"| {s['label']} | {s['pairs']} | {v[VERDICT_DONE]} | {v[VERDICT_LEARNING]} | {v[VERDICT_VIOLATING]} "
            f"| {v[VERDICT_NEW]} | {_fmt(s['median_margin'], '.4f')} | {_fmt(s['configured_beta'], '.0f')} "
            f"| {_fmt(s['applied_beta'], '.0f')} | {_fmt(s['median_logit'], '.2f')} "
            f"| {_fmt(s['advice'].get('efficiency'), '.0%')} |"
        )
    lines += ["", "## Beta", ""]
    lines += [_advice_line(s) for s in summaries]
    lines += ["", "A high beta can be deliberate: it ignores pairs that hold and hits hard on any that slip back."]
    retuned = [s for s in summaries if s["retuned_from"] is not None]
    if retuned:
        lines += [
            "",
            "Scored at the beta in force now. These carry visits from before a retune, so their margins were "
            "earned under a different beta: "
            + ", ".join(f"{s['label']} (was {s['retuned_from']:.0f}, now {s['applied_beta']:.0f})" for s in retuned)
            + ".",
        ]
    for s in summaries:
        info = ledger.concepts[s["label"]]
        concept_rows = [r for r in rows if r["concept"] == s["label"] and r["verdict"] != VERDICT_NEW]
        if not concept_rows:
            continue
        lines += ["", f"## {s['label']}", ""]
        by_logit = sorted(concept_rows, key=lambda r: r["mean_logit"])
        for title, picked in (
            ("Most saturated", list(reversed(by_logit))[:TOP_PAIRS]),
            ("Most violating", [r for r in by_logit if r["mean_logit"] < 0][:TOP_PAIRS]),
        ):
            if not picked:
                continue
            lines += [f"{title}:", "", "| mean beta·m | weight | wins | visits | pair |", "|---|---|---|---|---|"]
            for r in picked:
                lines.append(
                    f"| {r['mean_logit']:.2f} | {r['mean_weight']:.3f} | {r['wins']:.0%} | {r['draws']} "
                    f"| {_relpath(r['chosen'], info.path)} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
