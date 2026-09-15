"""DPO pair saturation: which pairs the lineage has already absorbed.

For the sigmoid DPO objective the gradient a pair contributes is the gradient
of its raw margin scaled by ``beta * sigmoid(-beta * margin)``. Once the policy
already prefers chosen over rejected by a margin where ``beta * margin`` is a few
nats, that factor is close to zero and the pair is dead weight in the batch.

DPO only ever measures the change since its *reference*, so within a run the
saturation is relative to that run's frozen reference. Across a merge the
reference is re-snapshotted from the merged base and every margin restarts at
zero: the model remembers the preference, the loss does not, and the next run
pushes already-learned pairs again at full strength. This module therefore
scores pairs against a fixed *origin* checkpoint (the pre-DPO base of the
lineage) so the number is stable across generations: it measures how much total
preference the lineage has absorbed for the pair, in the same units the loss
sees. Deciding how much is enough is a curation budget, not a runtime rule, so
the threshold and the per-timestep strictness live here as options.

The expensive part is beta-independent. A scorer (``scripts/util/
dpo_saturation_score.py``) records the per-sample denoising MSE of the chosen
and rejected image under the origin and the current checkpoint for a fixed list
of (noise seed, timestep) draws, exactly as the trainer's DPO path computes
them. Everything else - beta sweeps, thresholds, timestep buckets, the utility
curve, both tails - is post-processing over that table, done here.

Table layout (``<concept root>/.dpo_saturation/scores.json``)::

    {
      "version": 1,
      "concept": {"name", "path", "chosen_dir", "rejected_dir", "beta_override"},
      "global_beta": float,
      "num_train_timesteps": int,
      "origin": checkpoint identity,
      "current": checkpoint identity or null,
      "current_scored_at": iso timestamp or null,
      "draws": [{"seed": int, "t": int or null}, ...],
      "pairs": {
        "<pair key>": {
          "chosen": path, "rejected": path,
          "chosen_mtime": float, "rejected_mtime": float,
          "origin":  {"<draw id>": [t_used, mse_chosen, mse_rejected], ...},
          "current": {"<draw id>": [t_used, mse_chosen, mse_rejected], ...}
        }
      }
    }

The origin block of a pair never changes for a given origin checkpoint and draw
list, so a rescore against a new current checkpoint only runs the current pass.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import time
from datetime import datetime, timezone

TABLE_VERSION = 1
TABLE_DIRNAME = ".dpo_saturation"
TABLE_FILENAME = "scores.json"
RETIRED_LOG_FILENAME = "retired.json"
RETIRED_DIRNAME = ".retired"

# Sidecars that travel with an image when a pair is retired or restored.
_SIDECAR_SUFFIXES = (".txt", "-masklabel.png", "-condlabel.png")

DEFAULT_DEAD_WEIGHT = 0.05
DEFAULT_BETA_SWEEP = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_QUARTILES = 4


# --------------------------------------------------------------------------- #
# Paths and identities
# --------------------------------------------------------------------------- #


def table_dir(concept_root: str) -> str:
    return os.path.join(concept_root, TABLE_DIRNAME)


def table_path(concept_root: str) -> str:
    return os.path.join(table_dir(concept_root), TABLE_FILENAME)


def retired_log_path(concept_root: str) -> str:
    return os.path.join(table_dir(concept_root), RETIRED_LOG_FILENAME)


def _safetensors_metadata(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 256 * 1024 * 1024:
                # Not a safetensors file: eight arbitrary bytes read as a length.
                return {}
            header = json.loads(f.read(header_len))
        meta = header.get("__metadata__") or {}
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError, struct.error, MemoryError):
        return {}


def checkpoint_identity(path: str) -> dict:
    """Cheap, stable identity of a checkpoint file: the safetensors model-spec
    hash when the file carries one (OneTrainer writes it on save), else size
    and mtime. Hashing a 12 GB file on every rescore is not worth it."""
    path = os.path.abspath(path)
    try:
        stat = os.stat(path)
        size, mtime = stat.st_size, stat.st_mtime
    except OSError:
        size, mtime = None, None
    meta = _safetensors_metadata(path)
    return {
        "path": path,
        "size": size,
        "mtime": mtime,
        "hash": meta.get("modelspec.hash_sha256") or None,
        "title": meta.get("modelspec.title") or None,
    }


def identity_matches(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return False
    if a.get("hash") and b.get("hash"):
        return a["hash"] == b["hash"]
    return (
        os.path.normcase(os.path.abspath(a.get("path", ""))) == os.path.normcase(os.path.abspath(b.get("path", "")))
        and a.get("size") == b.get("size")
        and a.get("mtime") == b.get("mtime")
    )


def file_mtime(path: str | None) -> float | None:
    if not path:
        return None
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Draws
# --------------------------------------------------------------------------- #


def make_draws(
    *,
    draws: int = 0,
    timesteps: list[int] | None = None,
    seeds: int = 1,
    seed_base: int = 1_000_000,
) -> list[dict]:
    """Draw list for a scoring run.

    ``draws`` > 0 with no ``timesteps`` samples ``draws`` (noise, timestep)
    pairs from the trainer's own timestep distribution (the seed *is* the
    trainer's global step, so the draw reproduces a training step exactly).
    With ``timesteps`` (integer scheduler indices) every timestep gets ``seeds``
    noise draws at a forced timestep, for controlled per-bucket coverage.
    """
    if timesteps:
        out = [{"seed": seed_base + s, "t": int(t)} for t in timesteps for s in range(seeds)]
    else:
        out = [{"seed": seed_base + i, "t": None} for i in range(draws)]
    if not out:
        raise ValueError("no draws: pass draws > 0 or a non-empty timestep list")
    return out


def draw_id(draw: dict) -> str:
    t = draw.get("t")
    return f"{int(draw['seed'])}:{'train' if t is None else int(t)}"


def draws_match(a: list[dict], b: list[dict]) -> bool:
    return [draw_id(d) for d in a] == [draw_id(d) for d in b]


def parse_timestep_fractions(spec: str, num_train_timesteps: int) -> list[int]:
    """'0.1,0.5,0.9' -> scheduler indices. Fractions are the noise level
    (1.0 = pure noise); the index is ``int(frac * T)`` clamped to [0, T-1]."""
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        frac = float(token)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"timestep fraction {frac} outside [0, 1]")
        out.append(max(0, min(num_train_timesteps - 1, int(frac * num_train_timesteps))))
    return out


# --------------------------------------------------------------------------- #
# Table IO
# --------------------------------------------------------------------------- #


def new_table(
    *,
    concept: dict,
    global_beta: float,
    num_train_timesteps: int,
    origin: dict,
    draws: list[dict],
) -> dict:
    return {
        "version": TABLE_VERSION,
        "concept": dict(concept),
        "global_beta": float(global_beta),
        "num_train_timesteps": int(num_train_timesteps),
        "origin": dict(origin),
        "current": None,
        "current_scored_at": None,
        "draws": [dict(d) for d in draws],
        "pairs": {},
    }


def load_table(concept_root: str) -> dict | None:
    path = table_path(concept_root)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        table = json.load(f)
    if table.get("version") != TABLE_VERSION:
        raise ValueError(f"{path}: unsupported table version {table.get('version')}")
    return table


def save_table(concept_root: str, table: dict) -> str:
    path = table_path(concept_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1)
    os.replace(tmp, path)
    return path


def reusable_origin(existing: dict | None, *, origin: dict, draws: list[dict]) -> dict:
    """Origin scalars of an existing table that are still valid for this run:
    same origin checkpoint, every wanted draw already scored (a subset of the
    old draw list is fine), pair files untouched since."""
    if not existing or not identity_matches(existing.get("origin"), origin):
        return {}
    wanted = {draw_id(d) for d in draws}
    if not wanted.issubset({draw_id(d) for d in existing.get("draws", [])}):
        return {}
    out = {}
    for key, entry in existing.get("pairs", {}).items():
        origin_block = entry.get("origin") or {}
        if not wanted.issubset(origin_block):
            continue
        if entry.get("chosen_mtime") != file_mtime(entry.get("chosen")):
            continue
        if entry.get("rejected_mtime") != file_mtime(entry.get("rejected")):
            continue
        out[key] = {
            "chosen": entry["chosen"],
            "rejected": entry["rejected"],
            "chosen_mtime": entry["chosen_mtime"],
            "rejected_mtime": entry["rejected_mtime"],
            "origin": {k: list(v) for k, v in origin_block.items() if k in wanted},
            "current": {},
        }
    return out


def record_draw(table: dict, key: str, side: str, draw: dict, t_used: int, mse_chosen: float, mse_rejected: float):
    entry = table["pairs"][key]
    entry.setdefault(side, {})[draw_id(draw)] = [int(t_used), float(mse_chosen), float(mse_rejected)]


def ensure_pair(table: dict, key: str, chosen: str, rejected: str) -> dict:
    entry = table["pairs"].get(key)
    if entry is None:
        entry = {
            "chosen": chosen,
            "rejected": rejected,
            "chosen_mtime": file_mtime(chosen),
            "rejected_mtime": file_mtime(rejected),
            "origin": {},
            "current": {},
        }
        table["pairs"][key] = entry
    return entry


def missing_draws(entry: dict, side: str, draws: list[dict]) -> list[dict]:
    have = entry.get(side) or {}
    return [d for d in draws if draw_id(d) not in have]


def mark_current(table: dict, current: dict):
    table["current"] = dict(current)
    table["current_scored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Concept bookkeeping
# --------------------------------------------------------------------------- #


def concept_for_chosen_path(image_path: str, concept_dirs: list[tuple[str, str]]) -> int | None:
    """Index of the (chosen_dir, rejected_dir) pair whose chosen_dir contains
    ``image_path`` - the longest matching prefix wins."""
    norm = os.path.normcase(os.path.normpath(os.path.abspath(image_path)))
    best, best_len = None, -1
    for i, (chosen_dir, _) in enumerate(concept_dirs):
        prefix = os.path.normcase(os.path.normpath(os.path.abspath(chosen_dir)))
        if norm.startswith(prefix + os.sep) and len(prefix) > best_len:
            best, best_len = i, len(prefix)
    return best


def resolve_beta(table: dict, override: float | None = None) -> float:
    """Beta the training run would apply to this concept: the concept's own
    override when set, else the global beta; a CLI override beats both."""
    if override is not None and override > 0:
        return float(override)
    concept_beta = float((table.get("concept") or {}).get("beta_override") or 0.0)
    if concept_beta > 0:
        return concept_beta
    return float(table.get("global_beta") or 0.0)


# --------------------------------------------------------------------------- #
# Math
# --------------------------------------------------------------------------- #


def margin_from_mse(origin_chosen: float, origin_rejected: float, current_chosen: float, current_rejected: float):
    """Raw DPO margin in the trainer's units. ``logp = -mse`` per sample, so
    ``margin = (logp_cur_c - logp_org_c) - (logp_cur_r - logp_org_r)``."""
    return (origin_chosen - current_chosen) - (origin_rejected - current_rejected)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def gradient_weight(margin: float, beta: float) -> float:
    """``sigmoid(-beta * margin)``: the factor the sigmoid DPO gradient carries
    for this pair. 0.5 at margin 0, ~0 once beta*margin is a few nats."""
    return sigmoid(-beta * margin)


def pair_draws(entry: dict, draw_ids: list[str] | None = None) -> list[tuple[int, float]]:
    """(timestep index, margin) for every draw scored on both sides."""
    origin = entry.get("origin") or {}
    current = entry.get("current") or {}
    ids = draw_ids if draw_ids is not None else sorted(set(origin) & set(current))
    out = []
    for did in ids:
        o = origin.get(did)
        c = current.get(did)
        if o is None or c is None:
            continue
        t = int(c[0])
        out.append((t, margin_from_mse(o[1], o[2], c[1], c[2])))
    return out


def timestep_bucket(t_index: int, num_train_timesteps: int, buckets: int) -> int:
    frac = t_index / max(1, num_train_timesteps)
    return max(0, min(buckets - 1, int(frac * buckets)))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def score_pairs(
    table: dict,
    *,
    beta: float,
    buckets: int = DEFAULT_QUARTILES,
    exclude_keys: set[str] | None = None,
) -> list[dict]:
    """Per-pair statistics at one beta. Pairs without a completed draw on both
    sides, and pairs in ``exclude_keys`` (retired ones), are skipped."""
    T = int(table.get("num_train_timesteps") or 1000)
    draw_ids = [draw_id(d) for d in table.get("draws", [])]
    rows = []
    for key, entry in table.get("pairs", {}).items():
        if exclude_keys and key in exclude_keys:
            continue
        draws = pair_draws(entry, draw_ids)
        if not draws:
            continue
        margins = [m for _, m in draws]
        weights = [gradient_weight(m, beta) for m in margins]
        per_bucket_w: list[list[float]] = [[] for _ in range(buckets)]
        per_bucket_m: list[list[float]] = [[] for _ in range(buckets)]
        for (t, m), w in zip(draws, weights, strict=True):
            b = timestep_bucket(t, T, buckets)
            per_bucket_w[b].append(w)
            per_bucket_m[b].append(m)
        mean_margin = sum(margins) / len(margins)
        rows.append(
            {
                "key": key,
                "chosen": entry.get("chosen"),
                "rejected": entry.get("rejected"),
                "draws": len(draws),
                "mean_margin": mean_margin,
                "mean_logit": beta * mean_margin,
                "mean_weight": sum(weights) / len(weights),
                "min_weight": min(weights),
                "max_weight": max(weights),
                "bucket_weight": [sum(b) / len(b) if b else None for b in per_bucket_w],
                "bucket_margin": [sum(b) / len(b) if b else None for b in per_bucket_m],
                "wins": sum(1 for m in margins if m > 0) / len(margins),
            }
        )
    return rows


def is_saturated(row: dict, *, dead_weight: float, require_all_buckets: bool, min_draws: int) -> bool:
    if row["draws"] < min_draws:
        return False
    if row["mean_weight"] >= dead_weight:
        return False
    if require_all_buckets:
        populated = [w for w in row["bucket_weight"] if w is not None]
        if not populated or any(w >= dead_weight for w in populated):
            return False
    return True


def utility_curve(rows: list[dict], fractions=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """How many pairs (sorted by expected gradient weight, highest first) carry
    each fraction of the total weight."""
    weights = sorted((r["mean_weight"] for r in rows), reverse=True)
    total = sum(weights)
    out = {"total_weight": total, "pairs": len(rows), "share": {}}
    if total <= 0 or not weights:
        return out
    running = 0.0
    idx = 0
    for frac in fractions:
        target = frac * total
        while idx < len(weights) and running < target:
            running += weights[idx]
            idx += 1
        out["share"][f"{frac:g}"] = idx
    return out


def summarize(
    table: dict,
    *,
    beta: float | None = None,
    dead_weight: float = DEFAULT_DEAD_WEIGHT,
    require_all_buckets: bool = False,
    min_draws: int = 1,
    buckets: int = DEFAULT_QUARTILES,
    tails: int = 10,
    beta_sweep=DEFAULT_BETA_SWEEP,
    exclude_keys: set[str] | None = None,
) -> dict:
    beta_used = resolve_beta(table, beta)
    if beta_used <= 0:
        raise ValueError("beta must be positive (no concept override, no global beta, no --beta)")
    rows = score_pairs(table, beta=beta_used, buckets=buckets, exclude_keys=exclude_keys)
    for row in rows:
        row["saturated"] = is_saturated(
            row, dead_weight=dead_weight, require_all_buckets=require_all_buckets, min_draws=min_draws
        )
    saturated = [r for r in rows if r["saturated"]]

    bucket_counts = [0] * buckets
    bucket_dead = [0] * buckets
    for row in rows:
        for b, w in enumerate(row["bucket_weight"]):
            if w is None:
                continue
            bucket_counts[b] += 1
            if w < dead_weight:
                bucket_dead[b] += 1

    sweep = []
    for mult in beta_sweep:
        b = beta_used * mult
        sweep_rows = score_pairs(table, beta=b, buckets=buckets, exclude_keys=exclude_keys)
        dead = sum(
            1
            for r in sweep_rows
            if is_saturated(r, dead_weight=dead_weight, require_all_buckets=require_all_buckets, min_draws=min_draws)
        )
        sweep.append({"multiplier": mult, "beta": b, "dead": dead, "pairs": len(sweep_rows)})

    by_logit = sorted(rows, key=lambda r: r["mean_logit"], reverse=True)
    most_saturated = by_logit[:tails]
    most_negative = [r for r in reversed(by_logit) if r["mean_logit"] < 0][:tails]

    logits = sorted(r["mean_logit"] for r in rows)
    weights = sorted(r["mean_weight"] for r in rows)

    return {
        "beta": beta_used,
        "dead_weight": dead_weight,
        "require_all_buckets": require_all_buckets,
        "min_draws": min_draws,
        "buckets": buckets,
        "pairs": len(rows),
        "pairs_in_table": len(table.get("pairs", {})),
        "draws_per_pair": len(table.get("draws", [])),
        "saturated": len(saturated),
        "saturated_keys": [r["key"] for r in saturated],
        "wins_mean": (sum(r["wins"] for r in rows) / len(rows)) if rows else None,
        "logit_percentiles": _percentiles(logits),
        "weight_percentiles": _percentiles(weights),
        "bucket_counts": bucket_counts,
        "bucket_dead": bucket_dead,
        "utility": utility_curve(rows),
        "beta_sweep": sweep,
        "most_saturated": most_saturated,
        "most_negative": most_negative,
        "rows": rows,
    }


def histogram(values: list[float], *, bins: int = 30, lo: float | None = None, hi: float | None = None) -> dict:
    """Equal-width histogram for a GUI: ``{"edges": [bins+1], "counts": [bins]}``.
    Values outside [lo, hi] land in the first/last bin."""
    if not values:
        return {"edges": [], "counts": []}
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / width)
        counts[max(0, min(bins - 1, idx))] += 1
    return {"edges": [lo + i * width for i in range(bins + 1)], "counts": counts}


def _percentiles(sorted_values: list[float], points=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    if not sorted_values:
        return {}
    n = len(sorted_values)
    out = {}
    for p in points:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        out[f"p{int(p * 100)}"] = sorted_values[idx]
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if abs(x) >= 1000 or (abs(x) < 1e-3 and x != 0):
            return f"{x:.{digits}e}"
        return f"{x:.{digits}f}"
    return str(x)


def format_report(summary: dict, *, concept_label: str = "", show_keys: bool = False) -> str:
    s = summary
    lines = []
    head = f"DPO saturation report{(' - ' + concept_label) if concept_label else ''}"
    lines.append(head)
    lines.append("=" * len(head))
    lines.append(
        f"pairs scored: {s['pairs']} / {s['pairs_in_table']} in table, "
        f"{s['draws_per_pair']} draws per pair, beta = {s['beta']:g}"
    )
    rule = "mean weight" + (" and every populated timestep bucket" if s["require_all_buckets"] else "")
    lines.append(f"dead rule: {rule} < {s['dead_weight']:g}, at least {s['min_draws']} draw(s)")
    pct = (100.0 * s["saturated"] / s["pairs"]) if s["pairs"] else 0.0
    lines.append(f"saturated: {s['saturated']} pairs ({pct:.1f}%)")
    if s["wins_mean"] is not None:
        lines.append(f"chosen preferred on {100.0 * s['wins_mean']:.1f}% of draws (current vs origin)")
    lines.append("")

    lines.append("beta * margin (nats already absorbed), percentiles:")
    lp = s["logit_percentiles"]
    lines.append("  " + "  ".join(f"{k}={_fmt(v, 2)}" for k, v in lp.items()))
    lines.append("expected gradient weight sigmoid(-beta*margin), percentiles:")
    wp = s["weight_percentiles"]
    lines.append("  " + "  ".join(f"{k}={_fmt(v, 3)}" for k, v in wp.items()))
    lines.append("")

    lines.append("timestep buckets (low noise -> high noise): dead / scored")
    lines.append(
        "  "
        + "  ".join(
            f"q{i + 1}: {d}/{c}" for i, (d, c) in enumerate(zip(s["bucket_dead"], s["bucket_counts"], strict=True))
        )
    )
    lines.append("")

    u = s["utility"]
    if u.get("share"):
        lines.append("utility curve: pairs carrying a share of total gradient weight")
        lines.append("  " + "  ".join(f"{float(k) * 100:.0f}%: {v} pairs" for k, v in u["share"].items()))
        lines.append("")

    lines.append("beta sweep: saturated pairs if the run used")
    lines.extend(
        f"  {row['multiplier']:>5g} x beta = {row['beta']:>8g}: {row['dead']:>5} / {row['pairs']}"
        for row in s["beta_sweep"]
    )
    lines.append("")

    def _tail(title, rows):
        lines.append(title)
        if not rows:
            lines.append("  (none)")
            return
        for r in rows:
            label = r["key"] if show_keys else "pair"
            buckets = ",".join(_fmt(w, 2) for w in r["bucket_weight"])
            lines.append(
                f"  {label}: beta*margin={_fmt(r['mean_logit'], 2)} weight={_fmt(r['mean_weight'], 3)} "
                f"wins={100 * r['wins']:.0f}% buckets=[{buckets}]"
            )

    _tail("most saturated (highest absorbed preference):", s["most_saturated"])
    lines.append("")
    _tail("most negative (current prefers rejected - check the label):", s["most_negative"])
    return "\n".join(lines)


def rows_to_csv(rows: list[dict], path: str):
    import csv

    if not rows:
        return
    buckets = len(rows[0]["bucket_weight"])
    fields = [
        "key",
        "chosen",
        "rejected",
        "draws",
        "mean_margin",
        "mean_logit",
        "mean_weight",
        "min_weight",
        "max_weight",
        "wins",
        "saturated",
    ] + [f"weight_q{i + 1}" for i in range(buckets)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in rows:
            writer.writerow(
                [
                    r["key"],
                    r.get("chosen"),
                    r.get("rejected"),
                    r["draws"],
                    r["mean_margin"],
                    r["mean_logit"],
                    r["mean_weight"],
                    r["min_weight"],
                    r["max_weight"],
                    r["wins"],
                    int(bool(r.get("saturated"))),
                ]
                + ["" if w is None else w for w in r["bucket_weight"]]
            )


# --------------------------------------------------------------------------- #
# Retire / restore
# --------------------------------------------------------------------------- #


def _sidecars(image_path: str) -> list[str]:
    stem = os.path.splitext(image_path)[0]
    return [stem + suffix for suffix in _SIDECAR_SUFFIXES if os.path.isfile(stem + suffix)]


def _retired_target(side_dir: str, path: str) -> str:
    """Mirror ``path``'s position under ``side_dir`` into ``side_dir/.retired``.
    Dot-prefixed folders are pruned by the trainer's path collector and by
    the curation tools' walkers, so a retired pair is invisible to both but
    keeps its relative layout for restore."""
    rel = os.path.relpath(path, side_dir)
    if rel.startswith(".."):
        rel = os.path.basename(path)
    return os.path.join(side_dir, RETIRED_DIRNAME, rel)


def _move(src: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        raise FileExistsError(dest)
    shutil.move(src, dest)


def load_retired_log(concept_root: str) -> list[dict]:
    path = retired_log_path(concept_root)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_retired_log(concept_root: str, log: list[dict]):
    path = retired_log_path(concept_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    os.replace(tmp, path)


def retire_pairs(
    concept_root: str, table: dict, keys: list[str], *, reason: str = "", dry_run: bool = False
) -> list[dict]:
    """Move the chosen and rejected file of each pair (with caption and mask
    sidecars) into the ``.retired`` mirror under their own side folder and log
    the move so :func:`restore_pairs` can undo it. The table entry is kept so
    a later rescore can report the pair; its files are simply not in the
    training tree any more."""
    concept = table.get("concept") or {}
    chosen_dir = concept.get("chosen_dir")
    rejected_dir = concept.get("rejected_dir")
    if not chosen_dir or not rejected_dir:
        raise ValueError("table has no chosen/rejected directories")
    log = load_retired_log(concept_root)
    already = {rec["key"] for rec in log if not rec.get("restored_at")}
    moved = []
    for key in keys:
        entry = table.get("pairs", {}).get(key)
        if entry is None or key in already:
            continue
        plan = []
        for side_dir, path in ((chosen_dir, entry.get("chosen")), (rejected_dir, entry.get("rejected"))):
            if not path or not os.path.isfile(path):
                continue
            plan.extend((src, _retired_target(side_dir, src)) for src in [path, *_sidecars(path)])
        if not plan:
            continue
        record = {
            "key": key,
            "reason": reason,
            "retired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "moves": [{"from": src, "to": dest} for src, dest in plan],
            "restored_at": None,
        }
        if not dry_run:
            done = []
            try:
                for src, dest in plan:
                    _move(src, dest)
                    done.append((src, dest))
            except OSError:
                for src, dest in reversed(done):
                    shutil.move(dest, src)
                raise
            log.append(record)
        moved.append(record)
    if not dry_run and moved:
        save_retired_log(concept_root, log)
    return moved


def restore_pairs(concept_root: str, keys: list[str] | None = None) -> list[dict]:
    """Undo :func:`retire_pairs` for ``keys`` (all retired pairs when None)."""
    log = load_retired_log(concept_root)
    wanted = None if keys is None else set(keys)
    restored = []
    for record in log:
        if record.get("restored_at"):
            continue
        if wanted is not None and record["key"] not in wanted:
            continue
        for move in record["moves"]:
            if os.path.isfile(move["to"]):
                _move(move["to"], move["from"])
        record["restored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        restored.append(record)
    if restored:
        save_retired_log(concept_root, log)
    return restored


def retired_keys(concept_root: str) -> set[str]:
    return {rec["key"] for rec in load_retired_log(concept_root) if not rec.get("restored_at")}


def now_stamp() -> float:
    return time.time()
