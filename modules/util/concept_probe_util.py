"""Per-image learning signals for the gradient diagnostics (config
``log_grad_components``): the visit ledger and the fixed probe set.

The per-concept window loss averages different images every update, so its
trend mostly measures which images happened to come up. Two signals that
compare like with like replace it as "is this concept learning":

* :class:`ConceptVisitLedger` keeps each image's timestep-normalised loss per
  visit (keyed by the sample's own source path, the last few visits only) and
  reports, per concept over images seen at least twice, the median change
  between the latest two visits, the median change since the first visit and
  the fraction of images whose latest loss is below their first.

* :class:`ProbeSet` holds a fixed set of K samples per concept, captured from
  the first batches the training loader yields (their tensors sliced out and
  kept on the CPU, so the probe reads exactly what training reads: the same
  cache, masks and conditioning) and re-run every N updates with fixed
  timesteps and fixed noise under no_grad: a clean learning curve per concept.

Both persist next to the backups and are restored on resume.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import deque
from collections.abc import Sequence

import torch

DEFAULT_VISITS_KEPT = 8
DEFAULT_PROBE_IMAGES = 4
DEFAULT_PROBE_TIMESTEPS = (0.2, 0.5, 0.8)
DIAGNOSTICS_DIRNAME = "diagnostics"
LEDGER_FILENAME = "visit_ledger.json"
PROBES_FILENAME = "probes.pt"


class ConceptVisitLedger:
    """Per-image loss history and the per-concept paired deltas."""

    def __init__(self, visits_kept: int = DEFAULT_VISITS_KEPT):
        self.visits_kept = max(2, int(visits_kept))
        self._images: dict[str, dict] = {}  # path -> {"tag", "kind", "visits": deque[float]}

    def __len__(self) -> int:
        return len(self._images)

    def record(self, tag: str, path: str, value: float, kind: str = "sft") -> None:
        if path is None or value is None or value != value:
            return
        image = self._images.get(path)
        if image is None:
            image = self._images[path] = {
                "tag": tag,
                "kind": kind,
                "visits": deque(maxlen=self.visits_kept),
                "first": None,
            }
        if image["first"] is None:
            image["first"] = float(value)
        image["tag"] = tag
        image["visits"].append(float(value))

    def visits(self, path: str) -> list[float]:
        image = self._images.get(path)
        return list(image["visits"]) if image else []

    def scalars(self) -> dict[str, float]:
        """``<tag>/visit_delta`` (median latest - previous), ``<tag>/since_first``
        (median latest - first), ``<tag>/frac_improving`` (latest < first) and
        ``<tag>/revisited`` (images with at least two visits), per concept."""
        per_tag: dict[str, list[tuple[float, float, bool]]] = {}
        for image in self._images.values():
            visits = image["visits"]
            if len(visits) < 2:
                continue
            latest, previous = visits[-1], visits[-2]
            first = image["first"] if image["first"] is not None else visits[0]
            per_tag.setdefault(image["tag"], []).append((latest - previous, latest - first, latest < first))
        out: dict[str, float] = {}
        for tag, rows in per_tag.items():
            out[f"{tag}/visit_delta"] = statistics.median(r[0] for r in rows)
            out[f"{tag}/since_first"] = statistics.median(r[1] for r in rows)
            out[f"{tag}/frac_improving"] = sum(1 for r in rows if r[2]) / len(rows)
            out[f"{tag}/revisited"] = float(len(rows))
        return out

    # -- persistence ---------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "visits_kept": self.visits_kept,
            "images": {
                path: {
                    "tag": image["tag"],
                    "kind": image["kind"],
                    "first": image["first"],
                    "visits": list(image["visits"]),
                }
                for path, image in self._images.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self._images = {}
        for path, image in (state.get("images") or {}).items():
            visits = deque((float(v) for v in image.get("visits", [])), maxlen=self.visits_kept)
            first = image.get("first")
            self._images[path] = {
                "tag": image.get("tag", "concept"),
                "kind": image.get("kind", "sft"),
                "visits": visits,
                "first": float(first) if first is not None else (visits[0] if visits else None),
            }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, separators=(",", ":"))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, visits_kept: int = DEFAULT_VISITS_KEPT) -> ConceptVisitLedger:
        """A ledger from ``path``; empty when the file is missing or unreadable."""
        ledger = cls(visits_kept)
        try:
            with open(path, encoding="utf-8") as f:
                ledger.load_state_dict(json.load(f))
        except (OSError, ValueError, TypeError, AttributeError):
            ledger._images = {}
        return ledger


def slice_sample(batch: dict, index: int, size: int) -> dict:
    """One sample of a collated batch as a batch of one, on the CPU: tensors
    with a leading batch dim are sliced, lists of ``size`` entries indexed,
    nested dicts (``concept``) recursed, everything else passed through."""
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == size:
                out[key] = value[index : index + 1].detach().to("cpu")
            else:
                out[key] = value.detach().to("cpu")
        elif isinstance(value, dict):
            out[key] = slice_sample(value, index, size)
        elif isinstance(value, (list, tuple)) and len(value) == size and not isinstance(value, str):
            element = value[index]
            if isinstance(element, torch.Tensor):
                element = element.detach().to("cpu")
                out[key] = element.unsqueeze(0) if element.ndim == 0 else element.unsqueeze(0)
            else:
                out[key] = [element]
        else:
            out[key] = value
    return out


def move_sample(sample: dict, device: torch.device) -> dict:
    out = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = move_sample(value, device)
        else:
            out[key] = value
    return out


class ProbeSet:
    """K fixed samples per concept, captured from the training stream."""

    def __init__(self, images_per_concept: int = DEFAULT_PROBE_IMAGES):
        self.images_per_concept = max(1, int(images_per_concept))
        self._samples: dict[str, dict[str, dict]] = {}  # tag -> path -> sample (CPU batch of one)
        self._kinds: dict[str, str] = {}
        # first probe value per (tag, set size): the delta is only ever taken
        # against a probe of the same images
        self._first: dict[str, tuple[int, float]] = {}
        self.dirty = False

    def wants(self, tag: str, path: str) -> bool:
        samples = self._samples.get(tag, {})
        return path not in samples and len(samples) < self.images_per_concept

    def capture(self, tag: str, kind: str, batch: dict, paths: Sequence[str | None], size: int) -> int:
        """Keep every sample of ``batch`` this concept still has room for.
        Returns the number captured."""
        captured = 0
        for index in range(size):
            path = paths[index] if index < len(paths) else None
            if path is None or not self.wants(tag, path):
                continue
            self._samples.setdefault(tag, {})[path] = slice_sample(batch, index, size)
            self._kinds[tag] = kind
            captured += 1
        if captured:
            self.dirty = True
        return captured

    def tags(self) -> list[str]:
        return sorted(self._samples)

    def kind(self, tag: str) -> str:
        return self._kinds.get(tag, "sft")

    def samples(self, tag: str) -> list[tuple[str, dict]]:
        return sorted(self._samples.get(tag, {}).items())

    def count(self, tag: str) -> int:
        return len(self._samples.get(tag, {}))

    def total(self) -> int:
        return sum(len(s) for s in self._samples.values())

    def delta_since_first(self, tag: str, value: float) -> float | None:
        """``value`` minus the first probe taken over this exact set size; a
        grown set starts a new baseline (None on that probe)."""
        size = self.count(tag)
        first = self._first.get(tag)
        if first is None or first[0] != size:
            self._first[tag] = (size, value)
            self.dirty = True
            return None
        return value - first[1]

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            {
                "images_per_concept": self.images_per_concept,
                "samples": self._samples,
                "kinds": self._kinds,
                "first": self._first,
            },
            tmp,
        )
        os.replace(tmp, path)
        self.dirty = False

    @classmethod
    def load(cls, path: str, images_per_concept: int = DEFAULT_PROBE_IMAGES) -> ProbeSet:
        """A probe set from ``path``; empty when the file is missing or unreadable."""
        probes = cls(images_per_concept)
        try:
            state = torch.load(path, weights_only=False, map_location="cpu")
            probes._samples = {tag: dict(samples) for tag, samples in state["samples"].items()}
            probes._kinds = dict(state.get("kinds", {}))
            probes._first = {tag: tuple(v) for tag, v in state.get("first", {}).items()}
        except Exception:  # noqa: BLE001 - a missing or stale file starts the set empty
            probes._samples, probes._kinds, probes._first = {}, {}, {}
        return probes
