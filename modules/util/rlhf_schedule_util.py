"""Independent micro-step schedule for SFT + add-on runs.

With ``rlhf_addon_schedule = INDEPENDENT`` an epoch is no longer "one micro-step
per standard batch, add-ons riding along": the standard batches, the DPO pair
batches, the KTO batches and the text reward rollouts each get their own
micro-step, so every source is consumed at its full configured length (its
own epoch length and repeats) and no add-on is ever starved by another.

The interleave is a pure function of the per-kind counts and therefore
deterministic and resumable: the ``j``-th of ``n`` micro-steps of a kind sits at
fractional epoch position ``(j + 0.5) / n`` and the kinds are merged in order of
that position (ties broken by the fixed kind order below). Each kind is thus
spread as evenly as its share allows -- the gap between two of its micro-steps
is always ``total / n`` rounded one way or the other, so no kind clumps and
every gradient-accumulation window mixes kinds as far as the proportions
permit (a kind holding 38 % of the micro-steps can be in at most 76 % of the
GA=2 windows, and the schedule gets there).
"""

from __future__ import annotations

import heapq
from array import array
from collections.abc import Iterator, Mapping
from enum import Enum


class MicroStepKind(Enum):
    SFT = "sft"  # one standard batch from the main loader
    DPO = "dpo"  # one chosen/rejected pair batch from the DPO add-on loader
    KTO = "kto"  # one Liked/Disliked batch from the KTO loader
    TEXT = "text"  # one on-image text reward rollout

    def __str__(self):
        return self.value


#: Merge order for ties and the order counts are reported in.
KIND_ORDER: tuple[MicroStepKind, ...] = (MicroStepKind.SFT, MicroStepKind.DPO, MicroStepKind.KTO, MicroStepKind.TEXT)

_KIND_INDEX = {kind: index for index, kind in enumerate(KIND_ORDER)}


def interleave(counts: Mapping[MicroStepKind, int]) -> array:
    """The epoch's micro-step kinds as an ``array('B')`` of ``KIND_ORDER``
    indices, each kind appearing exactly ``counts[kind]`` times, spread evenly
    (see the module docstring). Kinds with a count of 0 or less are left out."""

    def positions(kind: MicroStepKind, count: int):
        order = _KIND_INDEX[kind]
        for j in range(count):
            yield ((j + 0.5) / count, order)

    streams = [positions(kind, int(counts[kind])) for kind in KIND_ORDER if int(counts.get(kind, 0)) > 0]
    return array("B", (order for _, order in heapq.merge(*streams)))


class IndependentSchedule:
    """One epoch's micro-step sequence and the bookkeeping the trainer needs to
    walk it from any position (fresh epoch or a mid-epoch resume)."""

    def __init__(self, counts: Mapping[MicroStepKind, int]):
        self.counts: dict[MicroStepKind, int] = {
            kind: int(counts.get(kind, 0)) for kind in KIND_ORDER if int(counts.get(kind, 0)) > 0
        }
        self._kinds = interleave(self.counts)

    def __len__(self) -> int:
        return len(self._kinds)

    def kind_at(self, position: int) -> MicroStepKind:
        return KIND_ORDER[self._kinds[position]]

    def kinds_from(self, position: int) -> Iterator[MicroStepKind]:
        """The kinds of micro-steps ``position .. len - 1``; empty when
        ``position`` is at or past the end."""
        for index in range(max(0, int(position)), len(self._kinds)):
            yield KIND_ORDER[self._kinds[index]]

    def clamp(self, position: int) -> int:
        """A resume position inside this schedule. A backup written under a
        different schedule (or dataset) may carry an epoch_step past this
        epoch's end; that resumes at the end, i.e. moves straight on to the
        next epoch rather than training a phantom tail."""
        return min(max(0, int(position)), len(self._kinds))

    def consumed_before(self, position: int) -> dict[MicroStepKind, int]:
        """How many micro-steps of each kind lie in ``[0, position)``: the
        number of batches each loader must be wound forward by on a resume."""
        position = self.clamp(position)
        consumed = dict.fromkeys(self.counts, 0)
        for index in range(position):
            consumed[KIND_ORDER[self._kinds[index]]] += 1
        return consumed

    def optimizer_steps(self, gradient_accumulation_steps: int) -> int:
        return len(self._kinds) // max(1, int(gradient_accumulation_steps))

    def describe(self, gradient_accumulation_steps: int) -> str:
        """``4857 standard + 5306 DPO + 2565 text reward = 12728 micro-steps
        per epoch, 6364 optimizer steps (gradient accumulation 2)``."""
        names = {
            MicroStepKind.SFT: "standard",
            MicroStepKind.DPO: "DPO",
            MicroStepKind.KTO: "KTO",
            MicroStepKind.TEXT: "text reward",
        }
        parts = " + ".join(f"{count} {names[kind]}" for kind, count in self.counts.items())
        return (
            f"{parts} = {len(self)} micro-steps per epoch, "
            f"{self.optimizer_steps(gradient_accumulation_steps)} optimizer steps "
            f"(gradient accumulation {gradient_accumulation_steps})"
        )


def wind_data_loader(data_loader, batches: int, batch_size: int) -> bool:
    """Make ``data_loader``'s *current* epoch start ``batches`` batches in.

    mgds only honours its start offset on the epoch the pipeline was created
    for (``initial_epoch``), and re-starts its serial modules from that offset
    on the first ``next()`` after ``start_next_epoch()``; this rewrites the
    offset between the two. The offset the loader was built with is the
    trainer's ``epoch_sample``, which counts standard samples -- right for the
    main loader, wrong for a secondary loader whose own batches were consumed
    at a different rate. Returns False when the loader is not an mgds one
    (tests), in which case nothing is wound.
    """
    data_set = data_loader.get_data_set()
    pipeline = getattr(data_set, "loading_pipeline", None)
    if pipeline is None or not hasattr(pipeline, "_LoadingPipeline__initial_index"):
        return False
    pipeline._LoadingPipeline__initial_index = max(0, int(batches)) * int(batch_size)
    pipeline._LoadingPipeline__mark_reset_serial_modules = True
    return True
