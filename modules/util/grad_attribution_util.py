"""Gradient attribution diagnostics (config ``log_grad_components``).

Three questions about a run that mixes objectives (standard batches, DPO
pairs, KTO batches, text reward rollouts) and concepts:

1. **Which objective moves the parameters, and do they agree?**
   :class:`GradAttribution` snapshots the trainable parameters' ``.grad``
   before a micro-step's backward(s) and adds ``grad - snapshot`` to an
   accumulator for that micro-step's kind. At the update step the per-kind
   norms, the pairwise cosines and the norm of the sum are read off the
   accumulated vectors themselves -- never derived from norms of sums, which
   say nothing about the components. The same object, keyed by concept tag,
   serves the per-concept conflict probe (split-by-concept backwards).

2. **How hard does each concept pull, and is it learning?**
   Per-sample output-gradient norms ``||dL/d(prediction)||`` and per-sample
   losses are dominated by the timestep, so :class:`TimestepBinNormalizer`
   divides each sample by a running mean for its timestep bin (per kind)
   before :class:`ConceptWindowStats` averages them per concept over a
   gradient-accumulation window.

3. **Is a concept's gradient cancelled by the others?** Every N updates the
   trainer backwards each concept group of a micro-batch separately (same
   scaling, so the summed gradient is unchanged) and reads each concept's
   norm and its cosine against the sum of everything else in that update.

Everything on the device, in float32 buffers allocated once and reused; the
only host syncs are one ``.tolist()`` per micro-step for the per-sample
stats and one per update for the scalars.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor


def concept_tag(label: str) -> str:
    """TensorBoard-safe tag for a concept folder name (the DPO scheme)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_") or "concept"


class GradAttribution:
    """Snapshot-diff attribution of the trainable parameters' gradients.

    ``begin()`` copies the current ``.grad`` of every parameter into the
    snapshot; ``end(key)`` adds ``grad - snapshot`` to ``key``'s accumulator
    and re-snapshots, so consecutive terms within one micro-step chain.
    Buffers are float32 (an exact record of what landed in ``.grad``,
    whatever its dtype), allocated on first use and zeroed, never freed.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], order: Sequence[str] = ()):
        self.parameters = [p for p in parameters if p.requires_grad]
        # Keys named in ``order`` are reported in that order (so a pair is
        # always ``cos_sft_dpo``, whichever came first in the window); the
        # rest follow in first-seen order.
        self._order = {key: i for i, key in enumerate(order)}
        self._snapshot: list[Tensor] | None = None
        self._buffers: dict[str, list[Tensor]] = {}
        self._active: list[str] = []

    # -- buffers -----------------------------------------------------------

    def _new_buffer(self) -> list[Tensor]:
        return [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in self.parameters]

    def _buffer(self, key: str) -> list[Tensor]:
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = self._buffers[key] = self._new_buffer()
        return buffer

    def _grads(self) -> tuple[list[int], list[Tensor]]:
        indices, grads = [], []
        for i, p in enumerate(self.parameters):
            if p.grad is not None:
                indices.append(i)
                grads.append(p.grad)
        return indices, grads

    def _copy_grads_into(self, target: list[Tensor]) -> None:
        indices, grads = self._grads()
        if grads:
            torch._foreach_copy_([target[i] for i in indices], grads)
        present = set(indices)
        missing = [target[i] for i in range(len(target)) if i not in present]
        if missing:
            torch._foreach_zero_(missing)

    @property
    def keys(self) -> list[str]:
        """Keys that received a delta since the last reset, in report order."""
        fallback = len(self._order)
        return sorted(self._active, key=lambda key: (self._order.get(key, fallback), self._active.index(key)))

    def buffer_bytes(self) -> int:
        per_copy = sum(p.numel() for p in self.parameters) * 4
        return per_copy * (len(self._buffers) + (1 if self._snapshot is not None else 0))

    # -- the two calls per segment -------------------------------------------

    @torch.no_grad()
    def begin(self) -> None:
        if self._snapshot is None:
            self._snapshot = self._new_buffer()
        self._copy_grads_into(self._snapshot)

    @torch.no_grad()
    def end(self, key: str) -> None:
        if self._snapshot is None:
            self.begin()
        acc = self._buffer(key)
        indices, grads = self._grads()
        if grads:
            torch._foreach_add_([acc[i] for i in indices], grads)
        torch._foreach_sub_(acc, self._snapshot)
        if key not in self._active:
            self._active.append(key)
        self._copy_grads_into(self._snapshot)

    @torch.no_grad()
    def reset(self) -> None:
        for buffer in self._buffers.values():
            torch._foreach_zero_(buffer)
        self._active = []

    # -- readings --------------------------------------------------------------

    @staticmethod
    def _norm(buffer: list[Tensor]) -> Tensor:
        return torch.stack(torch._foreach_norm(buffer)).square().sum().sqrt()

    @staticmethod
    def _dot(a: list[Tensor], b: list[Tensor]) -> Tensor:
        return torch.stack([(x * y).sum() for x, y in zip(a, b, strict=True)]).sum()

    def vector(self, key: str) -> list[Tensor]:
        return self._buffers[key]

    @torch.no_grad()
    def scalars(self, scale: float = 1.0) -> dict[str, float]:
        """``norm_<key>`` for every active key, ``cos_<a>_<b>`` for every pair
        and ``norm_total`` (the norm of the sum), one host sync. ``scale``
        divides the norms (an AMP scaler's scale; cosines are unaffected)."""
        keys = self.keys
        if not keys:
            return {}
        names: list[str] = []
        values: list[Tensor] = []
        norms = {key: self._norm(self._buffers[key]) for key in keys}
        for key in keys:
            names.append(f"norm_{key}")
            values.append(norms[key] / scale)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                names.append(f"cos_{a}_{b}")
                values.append(self._dot(self._buffers[a], self._buffers[b]) / (norms[a] * norms[b]).clamp_min(1e-30))
        names.append("norm_total")
        values.append(self._norm(self._sum(keys)) / scale)
        return dict(zip(names, torch.stack(values).tolist(), strict=True))

    @torch.no_grad()
    def rest_cosines(self, scale: float = 1.0) -> dict[str, float]:
        """``norm_<key>`` and ``cos_rest_<key>``: each key's vector against the
        sum of all the other keys' vectors (the concept conflict probe)."""
        keys = self.keys
        if not keys:
            return {}
        total = self._sum(keys)
        names, values = [], []
        for key in keys:
            own = self._buffers[key]
            norm = self._norm(own)
            # rest = total - own, evaluated without a third buffer:
            # <own, rest> = <own, total> - ||own||^2, ||rest||^2 = ||total||^2 - 2<own,total> + ||own||^2
            own_total = self._dot(own, total)
            total_norm_sq = self._norm(total).square()
            rest_norm = (total_norm_sq - 2 * own_total + norm.square()).clamp_min(0).sqrt()
            names.append(f"norm_{key}")
            values.append(norm / scale)
            names.append(f"cos_rest_{key}")
            values.append((own_total - norm.square()) / (norm * rest_norm).clamp_min(1e-30))
        return dict(zip(names, torch.stack(values).tolist(), strict=True))

    def _sum(self, keys: Sequence[str]) -> list[Tensor]:
        # The snapshot is free between a micro-step's end() and the next
        # begin(), which is where the readings are taken: use it as scratch.
        scratch = self._snapshot if self._snapshot is not None else self._new_buffer()
        torch._foreach_zero_(scratch)
        for key in keys:
            torch._foreach_add_(scratch, self._buffers[key])
        return scratch


def cosine(a: Sequence[Tensor], b: Sequence[Tensor]) -> float:
    """Cosine between two parameter-space vectors given as tensor lists."""
    dot = sum(float((x.float() * y.float()).sum()) for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(float(x.float().square().sum()) for x in a))
    nb = math.sqrt(sum(float(y.float().square().sum()) for y in b))
    return dot / max(na * nb, 1e-30)


class TimestepBinNormalizer:
    """Running (EMA) mean of a per-sample quantity per timestep bin, per key.

    ``normalize(key, values, ts)`` divides each value by the mean of its bin
    as it stood before this batch (the batch's own bin mean the first time a
    bin is seen), then folds the batch into the means. Values whose bin mean
    is not positive are returned as NaN and should be skipped."""

    def __init__(self, bins: int = 10, decay: float = 0.99):
        self.bins = max(1, int(bins))
        self.decay = float(decay)
        self.means: dict[str, list[float | None]] = {}

    def bin_of(self, t: float | None) -> int:
        if t is None or t != t:
            return 0
        return min(max(int(t * self.bins), 0), self.bins - 1)

    def normalize(self, key: str, values: Sequence[float], ts: Sequence[float | None]) -> list[float]:
        means = self.means.setdefault(key, [None] * self.bins)
        bins = [self.bin_of(t) for t in ts]
        batch_sum: dict[int, float] = {}
        batch_count: dict[int, int] = {}
        for value, b in zip(values, bins, strict=True):
            if value != value:
                continue
            batch_sum[b] = batch_sum.get(b, 0.0) + float(value)
            batch_count[b] = batch_count.get(b, 0) + 1
        out = []
        for value, b in zip(values, bins, strict=True):
            ref = means[b]
            if ref is None and batch_count.get(b):
                ref = batch_sum[b] / batch_count[b]
            out.append(float(value) / ref if ref is not None and ref > 0 and value == value else float("nan"))
        for b, count in batch_count.items():
            mean = batch_sum[b] / count
            means[b] = mean if means[b] is None else means[b] * self.decay + mean * (1.0 - self.decay)
        return out


class ConceptWindowStats:
    """Per-concept sums of the per-sample diagnostics over one update window."""

    def __init__(self):
        self._sums: dict[str, dict[str, float]] = {}

    def add(self, tag: str, name: str, value: float) -> None:
        if value != value:
            return
        sums = self._sums.setdefault(tag, {})
        sums[name] = sums.get(name, 0.0) + float(value)
        sums[f"{name}__n"] = sums.get(f"{name}__n", 0.0) + 1.0

    def scalars(self) -> dict[str, float]:
        """``<tag>/<name>`` means for every recorded quantity."""
        out = {}
        for tag, sums in self._sums.items():
            for name, total in sums.items():
                if name.endswith("__n"):
                    continue
                count = sums.get(f"{name}__n", 0.0)
                if count > 0:
                    out[f"{tag}/{name}"] = total / count
        return out

    def reset(self) -> None:
        self._sums = {}

    def __bool__(self) -> bool:
        return bool(self._sums)


def concept_groups(tags: Sequence[str | None]) -> dict[str, list[int]]:
    """Sample indices per concept tag, in first-seen order; untagged samples
    are grouped under ``"other"``."""
    groups: dict[str, list[int]] = {}
    for i, tag in enumerate(tags):
        groups.setdefault(tag or "other", []).append(i)
    return groups


def sample_mask(indices: Sequence[int], like: Tensor) -> Tensor:
    """A ``[B, 1, ..., 1]`` float mask selecting ``indices`` along the batch
    dim of ``like``."""
    mask = torch.zeros(like.shape[0], device=like.device, dtype=like.dtype)
    mask[list(indices)] = 1.0
    return mask.view(-1, *([1] * (like.ndim - 1)))
