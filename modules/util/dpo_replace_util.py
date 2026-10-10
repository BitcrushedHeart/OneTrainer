"""Replace Pairs: refresh the sides of existing DPO pairs against the current
checkpoint.

An off-policy rejected sample decays as the model improves — a negative made
by an old checkpoint is an easy negative for the current policy, so the pair
teaches nothing new. This module holds the pure pieces of the tool that
regenerates a pair's caption with whatever the user is generating with right
now and swaps the result in:

* reading a pair's state from disk (the caption the trainer will actually
  read, who generated each side, when),
* ranking pairs by how stale they are,
* deciding what a fresh generation should become (a harder negative, or —
  when the chosen side is itself a generation — the new chosen, demoting the
  old one),
* applying and undoing the swap with a backup of every file it displaces.

Nothing here talks to SwarmUI or the realism model; the service does that.
"""

import os
import shutil
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

from modules.util.dpo_curation_util import (
    _normalize_source_path,
    compute_file_sha256,
    find_exported_file,
    save_manifest,
)
from modules.util.dpo_offpolicy_util import caption_match_key, image_size
from modules.util.image_metadata_util import strip_angle_bracket_segments
from modules.util.swarm_generate_util import _params_from_image

#: Where displaced files go, under the concept folder. Dot-prefixed so the
#: trainer's path collector and every scan in this tool family skip it.
REPLACED_DIR = ".replaced"

#: Presentation orders for the ranked pair list.
ORDER_STALE_FIRST = "stale_first"
ORDER_OLDEST = "oldest"
ORDER_LEAST_REALISTIC = "least_realistic"
ORDER_PATH = "path"
ORDERS = (ORDER_STALE_FIRST, ORDER_OLDEST, ORDER_LEAST_REALISTIC, ORDER_PATH)

ACTION_PROMOTE = "promote"
ACTION_REPLACE_REJECTED = "replace_rejected"
ACTION_KEEP = "keep"

#: Caption sources, in the order they are trusted.
CAPTION_FROM_PAIR = "pair"  # chosen/<pair>.txt on disk (what the trainer reads)
CAPTION_FROM_MANIFEST = "manifest"  # nothing on disk to prefer


@dataclass
class Provenance:
    model: str = ""
    loras: list[str] = field(default_factory=list)
    loraweights: list[float] = field(default_factory=list)
    generated_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "loras": list(self.loras),
            "loraweights": list(self.loraweights),
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "Provenance":
        if not isinstance(data, dict):
            return cls()
        return cls(
            model=str(data.get("model") or ""),
            loras=[str(x) for x in (data.get("loras") or [])],
            loraweights=[float(x) for x in (data.get("loraweights") or []) if _is_number(x)],
            generated_at=data.get("generated_at") if _is_number(data.get("generated_at")) else None,
        )

    @property
    def label(self) -> str:
        """Model plus adapters, the way the UI groups pairs."""
        parts = [os.path.basename(self.model) if self.model else "unknown model"]
        for index, lora in enumerate(self.loras):
            weight = self.loraweights[index] if index < len(self.loraweights) else 1.0
            parts.append(f"{os.path.basename(lora)}@{weight:g}")
        return " + ".join(parts)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    # SwarmUI writes comma-separated strings in some versions.
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _weight(raw) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def provenance_from_params(params: dict, generated_at: float | None = None) -> Provenance:
    """Who generated an image, from SwarmUI's embedded parameter block."""
    if not isinstance(params, dict):
        return Provenance(generated_at=generated_at)
    weights = [_weight(raw) for raw in _as_list(params.get("loraweights"))]
    return Provenance(
        model=str(params.get("model") or ""),
        loras=[str(x) for x in _as_list(params.get("loras"))],
        loraweights=weights,
        generated_at=generated_at,
    )


def generation_params(path: str) -> dict:
    """SwarmUI's parameter block from a generated image, ``{}`` for a photo.
    The block survives ``export_single_pair`` because it copies bytes."""
    return _params_from_image(path)


def provenance_of_file(path: str) -> Provenance:
    try:
        mtime: float | None = os.path.getmtime(path)
    except OSError:
        mtime = None
    return provenance_from_params(generation_params(path), mtime)


def file_is_generation(path: str) -> bool:
    """True when the file carries a generator parameter block."""
    return bool(generation_params(path))


# ---------------------------------------------------------------------------
# Reading a pair from disk
# ---------------------------------------------------------------------------


@dataclass
class PairState:
    pair_id: int
    entry: dict
    chosen_path: str
    rejected_path: str
    #: The caption the trainer reads: ``chosen/<pair>.txt`` on disk.
    caption: str
    caption_source: str
    #: True when the on-disk caption differs from what the manifest recorded
    #: at export time — the pair was corrected after it was made.
    caption_changed: bool
    chosen_size: tuple[int, int] | None
    rejected_size: tuple[int, int] | None
    chosen_provenance: Provenance
    rejected_provenance: Provenance
    rejected_mtime: float | None
    chosen_score: float | None = None
    rejected_score: float | None = None
    replacements: int = 0

    @property
    def stale_caption(self) -> bool:
        return self.caption_changed

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "chosen_path": self.chosen_path,
            "rejected_path": self.rejected_path,
            "chosen_file": self.entry.get("chosen_file"),
            "rejected_file": self.entry.get("rejected_file"),
            "caption": self.caption,
            "caption_source": self.caption_source,
            "caption_changed": self.caption_changed,
            "manifest_prompt": self.entry.get("prompt", ""),
            "chosen_size": list(self.chosen_size) if self.chosen_size else None,
            "rejected_size": list(self.rejected_size) if self.rejected_size else None,
            "chosen_provenance": self.chosen_provenance.to_dict(),
            "rejected_provenance": self.rejected_provenance.to_dict(),
            "rejected_model": self.rejected_provenance.label,
            "rejected_mtime": self.rejected_mtime,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "replacements": self.replacements,
            "stale_caption": self.stale_caption,
        }


def pair_caption_path(concept_dir: str, entry: dict) -> str | None:
    """The chosen-side ``.txt`` for a manifest entry, if it exists."""
    stem = os.path.splitext(entry.get("chosen_file", ""))[0]
    if not stem:
        return None
    return find_exported_file(os.path.join(concept_dir, "chosen"), stem + ".txt")


def read_pair_caption(concept_dir: str, entry: dict) -> str | None:
    path = pair_caption_path(concept_dir, entry)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def read_pair_state(concept_dir: str, entry: dict, score_of=None) -> PairState | None:
    """Everything the ranking and the proposal need about one pair, read
    from disk. ``None`` when either image is gone (the manifest pruner owns
    that case). ``score_of(path)`` returns a cached P(AI) or None."""
    chosen_dir = os.path.join(concept_dir, "chosen")
    rejected_dir = os.path.join(concept_dir, "rejected")
    chosen_path = find_exported_file(chosen_dir, entry.get("chosen_file", ""))
    rejected_path = find_exported_file(rejected_dir, entry.get("rejected_file", ""))
    if not chosen_path or not rejected_path:
        return None

    manifest_prompt = strip_angle_bracket_segments(str(entry.get("prompt", "")))
    disk_caption = read_pair_caption(concept_dir, entry)
    if disk_caption is not None and disk_caption != "":
        caption, source = disk_caption, CAPTION_FROM_PAIR
    else:
        caption, source = manifest_prompt, CAPTION_FROM_MANIFEST
    caption_changed = caption_match_key(caption) != caption_match_key(manifest_prompt)

    chosen_params = generation_params(chosen_path)

    stored_rejected = Provenance.from_dict(entry.get("rejected_provenance"))
    try:
        rejected_mtime: float | None = os.path.getmtime(rejected_path)
    except OSError:
        rejected_mtime = None
    if stored_rejected.model or stored_rejected.loras:
        rejected_prov = stored_rejected
        if rejected_prov.generated_at is None:
            rejected_prov.generated_at = rejected_mtime
    else:
        rejected_prov = provenance_from_params(generation_params(rejected_path), rejected_mtime)

    stored_chosen = Provenance.from_dict(entry.get("chosen_provenance"))
    if stored_chosen.model or stored_chosen.loras:
        chosen_prov = stored_chosen
    else:
        try:
            chosen_mtime: float | None = os.path.getmtime(chosen_path)
        except OSError:
            chosen_mtime = None
        chosen_prov = provenance_from_params(chosen_params, chosen_mtime)

    return PairState(
        pair_id=int(entry.get("pair_id", -1)),
        entry=entry,
        chosen_path=chosen_path,
        rejected_path=rejected_path,
        caption=caption,
        caption_source=source,
        caption_changed=caption_changed,
        chosen_size=image_size(chosen_path),
        rejected_size=image_size(rejected_path),
        chosen_provenance=chosen_prov,
        rejected_provenance=rejected_prov,
        rejected_mtime=rejected_prov.generated_at if rejected_prov.generated_at is not None else rejected_mtime,
        chosen_score=score_of(chosen_path) if score_of else None,
        rejected_score=score_of(rejected_path) if score_of else None,
        replacements=len(entry.get("history") or []),
    )


def backfill_provenance(entry: dict, state: PairState) -> bool:
    """Record what the files say about themselves into the manifest entry, so
    later scans and other tools need not open the images. Returns True when
    the entry changed."""
    changed = False
    if not isinstance(entry.get("rejected_provenance"), dict) and (
        state.rejected_provenance.model or state.rejected_provenance.loras
    ):
        entry["rejected_provenance"] = state.rejected_provenance.to_dict()
        changed = True
    if not isinstance(entry.get("chosen_provenance"), dict) and (
        state.chosen_provenance.model or state.chosen_provenance.loras
    ):
        entry["chosen_provenance"] = state.chosen_provenance.to_dict()
        changed = True
    return changed


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_pairs(states: list[PairState], order: str = ORDER_STALE_FIRST) -> list[PairState]:
    """Most worth replacing first.

    ``stale_first``: pairs whose caption changed (both sides answer an old
    prompt) come first, then by the age of the rejected generation, oldest
    first. ``oldest``: purely by that age. ``least_realistic``: the rejected
    images the head already finds most obviously fake — the easiest
    negatives — first; unscored last. ``path``: by pair id."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    far_future = float("inf")

    def age(state: PairState) -> float:
        return state.rejected_mtime if state.rejected_mtime is not None else far_future

    if order == ORDER_PATH:
        return sorted(states, key=lambda s: s.pair_id)
    if order == ORDER_OLDEST:
        return sorted(states, key=lambda s: (age(s), s.pair_id))
    if order == ORDER_LEAST_REALISTIC:
        return sorted(
            states,
            key=lambda s: (s.rejected_score is None, -(s.rejected_score or 0.0), age(s), s.pair_id),
        )
    return sorted(states, key=lambda s: (not s.stale_caption, age(s), s.pair_id))


def model_groups(states: list[PairState]) -> list[dict]:
    """Pairs per generating model+adapters, oldest group first, for the
    summary line: "V39: 412 pairs (Aug 3 – Aug 12)"."""
    groups: dict[str, dict] = {}
    for state in states:
        label = state.rejected_provenance.label
        group = groups.setdefault(label, {"label": label, "pairs": 0, "oldest": None, "newest": None})
        group["pairs"] += 1
        stamp = state.rejected_mtime
        if stamp is not None:
            group["oldest"] = stamp if group["oldest"] is None else min(group["oldest"], stamp)
            group["newest"] = stamp if group["newest"] is None else max(group["newest"], stamp)
    return sorted(groups.values(), key=lambda g: (g["oldest"] is None, g["oldest"] or 0.0))


# ---------------------------------------------------------------------------
# The selection group: old chosen, old rejected, and the fresh generations
# ---------------------------------------------------------------------------

ROLE_CHOSEN = "chosen"
ROLE_REJECTED = "rejected"
ROLE_FRESH = "fresh"


@dataclass
class GroupImage:
    path: str
    #: Where the image currently sits in the pair: the old chosen, the old
    #: rejected, or a fresh generation not yet part of it.
    role: str
    size: tuple[int, int] | None
    score: float | None = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "role": self.role,
            "size": list(self.size) if self.size else None,
            "score": self.score,
        }


@dataclass
class Proposal:
    """One pair's selection group with a suggested pick.

    The person reviewing chooses the best image as the new chosen and the
    second-best (the hardest negative) as the new rejected from the whole
    group. The suggestion is only a pre-selection from the realism scores."""

    state: PairState
    caption: str
    images: list[GroupImage]
    suggested_chosen: str
    suggested_rejected: str
    scored: bool
    provenance: Provenance
    reason: str
    #: Promotion turned off for the session: only the rejected side is picked.
    chosen_locked: bool

    @property
    def old_chosen(self) -> str:
        return self.state.chosen_path

    @property
    def old_rejected(self) -> str:
        return self.state.rejected_path

    @property
    def fresh(self) -> list[GroupImage]:
        return [image for image in self.images if image.role == ROLE_FRESH]

    def image(self, path: str) -> GroupImage | None:
        return next((image for image in self.images if image.path == path), None)

    @property
    def suggestion_is_keep(self) -> bool:
        return self.suggested_chosen == self.old_chosen and self.suggested_rejected == self.old_rejected

    @property
    def suggestion_moves_chosen(self) -> bool:
        return self.suggested_chosen != self.old_chosen

    def to_dict(self) -> dict:
        data = self.state.to_dict()
        data.update(
            {
                "caption": self.caption,
                "images": [image.to_dict() for image in self.images],
                "fresh_count": len(self.fresh),
                "suggested_chosen": self.suggested_chosen,
                "suggested_rejected": self.suggested_rejected,
                "suggestion_is_keep": self.suggestion_is_keep,
                "scored": self.scored,
                "reason": self.reason,
                "chosen_locked": self.chosen_locked,
                "new_provenance": self.provenance.to_dict(),
                "new_model": self.provenance.label,
            }
        )
        return data


def build_proposal(
    state: PairState,
    caption: str,
    fresh: list[GroupImage],
    provenance: Provenance,
    *,
    allow_promote: bool = True,
) -> Proposal:
    """Assemble the group and suggest a pick from the realism scores.

    Scores are P(AI): lower is more photoreal. The chosen side may be beaten
    by any image in the group (the old rejected included), and the hardest
    negative is the most real of what remains. With promotion turned off the
    chosen side is locked, so only the rejected pick is suggested. Without
    scores the suggestion is "keep": the reviewer decides by eye."""
    chosen_locked = not allow_promote
    images = [
        GroupImage(state.chosen_path, ROLE_CHOSEN, state.chosen_size, state.chosen_score),
        GroupImage(state.rejected_path, ROLE_REJECTED, state.rejected_size, state.rejected_score),
        *fresh,
    ]
    scored = bool(fresh) and all(image.score is not None for image in images)
    if not fresh:
        return Proposal(
            state,
            caption,
            images,
            state.chosen_path,
            state.rejected_path,
            False,
            provenance,
            "no generation was produced",
            chosen_locked,
        )
    if not scored:
        return Proposal(
            state,
            caption,
            images,
            state.chosen_path,
            state.rejected_path,
            False,
            provenance,
            "no realism scores — pick by eye",
            chosen_locked,
        )

    def most_real(pool: list[GroupImage]) -> GroupImage:
        return min(pool, key=lambda image: (image.score, image.role != ROLE_CHOSEN, image.path))

    if chosen_locked:
        chosen = images[0]
        rejected = most_real([image for image in images if image.path != chosen.path])
    else:
        chosen = most_real(images)
        rejected = most_real([image for image in images if image.path != chosen.path])

    if chosen.path == state.chosen_path and rejected.path == state.rejected_path:
        best_fresh = most_real(fresh)
        reason = (
            f"nothing beats the current pair: best fresh P(AI) {best_fresh.score:.2f} vs "
            f"rejected {state.rejected_score:.2f}"
        )
    elif chosen.path == state.chosen_path:
        reason = (
            f"fresh P(AI) {rejected.score:.2f} beats the current rejected {state.rejected_score:.2f}: harder negative"
        )
    elif chosen.role == ROLE_FRESH:
        reason = (
            f"fresh P(AI) {chosen.score:.2f} beats the current chosen {state.chosen_score:.2f}: "
            f"it becomes chosen, {rejected.role} P(AI) {rejected.score:.2f} becomes rejected"
        )
    else:
        reason = f"the current rejected P(AI) {chosen.score:.2f} now looks more real than the chosen {state.chosen_score:.2f}"
    return Proposal(state, caption, images, chosen.path, rejected.path, True, provenance, reason, chosen_locked)


# ---------------------------------------------------------------------------
# Applying and undoing
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def _unique(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{stem}_{i}{ext}"):
        i += 1
    return f"{stem}_{i}{ext}"


def _side_fields(side: str, path: str, source: str | None, provenance: Provenance | None) -> dict:
    fields = {
        f"{side}_file": os.path.basename(path),
        f"{side}_source": _normalize_source_path(source) if source else None,
        f"{side}_sha256": compute_file_sha256(path),
    }
    try:
        fields[f"{side}_size"] = os.path.getsize(path)
    except OSError:
        fields[f"{side}_size"] = None
    fields[f"{side}_provenance"] = provenance.to_dict() if provenance is not None else None
    return fields


_SIDE_KEYS = (
    "prompt",
    "chosen_file",
    "chosen_source",
    "chosen_sha256",
    "chosen_size",
    "chosen_provenance",
    "rejected_file",
    "rejected_source",
    "rejected_sha256",
    "rejected_size",
    "rejected_provenance",
)


def _snapshot(entry: dict) -> dict:
    return {key: entry.get(key) for key in _SIDE_KEYS if key in entry}


def _restore(placed: list, backups: dict[str, str]) -> None:
    """Put a pair back the way it was: placed fresh copies are deleted,
    placed old files return to their backup slot, then every old side
    returns from its backup to its original path."""
    backups = dict(backups)
    for target, kind, origin, old_path in reversed(placed):
        if kind == "fresh":
            if os.path.isfile(target):
                os.remove(target)
        elif os.path.isfile(target):
            shutil.move(target, origin)
            backups[old_path] = origin
    for old_path, backup in backups.items():
        if os.path.isfile(backup):
            shutil.move(backup, old_path)


def describe_selection(proposal: Proposal, chosen_path: str, rejected_path: str) -> str:
    """A short label for the manifest history: what each side came from."""

    def origin(path: str) -> str:
        image = proposal.image(path)
        return image.role if image else "?"

    return f"chosen<-{origin(chosen_path)} rejected<-{origin(rejected_path)}"


def apply_selection(
    concept_dir: str,
    manifest: dict,
    proposal: Proposal,
    chosen_path: str,
    rejected_path: str,
) -> dict:
    """Make ``chosen_path`` and ``rejected_path`` (any two images of the
    group) the pair's sides on disk and in the manifest.

    Both old files are first moved to ``<concept>/.replaced/`` under the pair
    id and a timestamp; a side that keeps an old image moves it back, a side
    that takes a fresh image copies it in. What stays in ``.replaced`` is the
    image the pair no longer uses. The returned record lets
    :func:`undo_selection` reverse all of it."""
    if chosen_path == rejected_path:
        raise ValueError("chosen and rejected must be different images")
    for path in (chosen_path, rejected_path):
        if proposal.image(path) is None:
            raise ValueError(f"{path} is not part of this pair's group")
    if proposal.chosen_locked and chosen_path != proposal.old_chosen:
        raise ValueError("promotion is off for this session: the chosen side stays as it is")
    if chosen_path == proposal.old_chosen and rejected_path == proposal.old_rejected:
        raise ValueError("nothing to apply: the pair already is this selection")
    for image in proposal.fresh:
        if image.path in (chosen_path, rejected_path) and not os.path.isfile(image.path):
            raise FileNotFoundError(image.path)

    state = proposal.state
    entry = state.entry
    chosen_dir = os.path.join(concept_dir, "chosen")
    rejected_dir = os.path.join(concept_dir, "rejected")
    replaced_dir = os.path.join(concept_dir, REPLACED_DIR)
    os.makedirs(replaced_dir, exist_ok=True)
    stamp = _timestamp()
    stem = os.path.splitext(entry["chosen_file"])[0]
    before = _snapshot(entry)

    old_side = {
        proposal.old_chosen: {
            "source": before.get("chosen_source"),
            "provenance": state.chosen_provenance,
        },
        proposal.old_rejected: {"source": before.get("rejected_source"), "provenance": state.rejected_provenance},
    }
    backups: dict[str, str] = {}
    #: (placed_path, origin_kind, origin_path, old_path_or_None)
    placed: list[tuple[str, str, str, str | None]] = []
    try:
        for old_path, name in ((proposal.old_chosen, "chosen"), (proposal.old_rejected, "rejected")):
            target = _unique(os.path.join(replaced_dir, f"{stem}__{stamp}_{name}{os.path.splitext(old_path)[1]}"))
            shutil.move(old_path, target)
            backups[old_path] = target
        for path, side_dir in ((chosen_path, chosen_dir), (rejected_path, rejected_dir)):
            target = os.path.join(side_dir, stem + (os.path.splitext(path)[1].lower() or ".png"))
            if path in backups:
                backup = backups.pop(path)
                shutil.move(backup, target)
                placed.append((target, "backup", backup, path))
            else:
                shutil.copy2(path, target)
                placed.append((target, "fresh", path, None))
    except OSError:
        _restore(placed, backups)
        raise

    def fields(side: str, target: str, path: str) -> dict:
        if path in old_side:
            info = old_side[path]
            return _side_fields(side, target, info["source"], info["provenance"])
        return _side_fields(side, target, path, proposal.provenance)

    entry.update(fields("chosen", placed[0][0], chosen_path))
    entry.update(fields("rejected", placed[1][0], rejected_path))

    # A caption corrected on disk after export is now what the pair answers.
    if state.caption_changed:
        entry["prompt"] = proposal.caption

    action = describe_selection(proposal, chosen_path, rejected_path)
    record = {
        "at": time.time(),
        "action": action,
        "pair_id": state.pair_id,
        "caption": proposal.caption,
        "chosen_from": chosen_path,
        "rejected_from": rejected_path,
        "chosen_score": (proposal.image(chosen_path) or GroupImage("", "", None)).score,
        "rejected_score": (proposal.image(rejected_path) or GroupImage("", "", None)).score,
        "provenance": proposal.provenance.to_dict(),
        "placed": placed,
        "leftover_backups": dict(backups),
        "old_paths": {"chosen": proposal.old_chosen, "rejected": proposal.old_rejected},
        "before": before,
    }
    entry.setdefault("history", []).append(
        {
            "at": record["at"],
            "action": action,
            "caption": proposal.caption,
            "chosen_from": _normalize_source_path(chosen_path),
            "rejected_from": _normalize_source_path(rejected_path),
            "chosen_score": record["chosen_score"],
            "rejected_score": record["rejected_score"],
            "provenance": record["provenance"],
            "retired": [os.path.relpath(b, concept_dir) for b in backups.values()],
        }
    )
    save_manifest(concept_dir, manifest)
    return record


def undo_selection(concept_dir: str, manifest: dict, record: dict) -> None:
    """Reverse :func:`apply_selection` exactly."""
    pairs = manifest.get("pairs", [])
    entry = next((e for e in pairs if e.get("pair_id") == record["pair_id"]), None)
    if entry is None:
        raise KeyError(record["pair_id"])
    _restore(record["placed"], dict(record["leftover_backups"]))

    before = record["before"]
    for key in _SIDE_KEYS:
        if key in before:
            entry[key] = before[key]
        else:
            entry.pop(key, None)
    history = entry.get("history") or []
    if history:
        history.pop()
    if not history:
        entry.pop("history", None)
    save_manifest(concept_dir, manifest)


# ---------------------------------------------------------------------------
# Locating what SwarmUI wrote
# ---------------------------------------------------------------------------


def local_path_for_view(view_path: str, generated_folder: str) -> str | None:
    """Map a SwarmUI ``View/local/raw/<date>/<name>`` path to the file under
    the generated folder (SwarmUI's raw output root). Falls back to a
    basename search when the layout differs."""
    # A result the tool already stored itself (``save_to`` runs) is a local path.
    if os.path.isfile(view_path):
        return view_path
    # API results are URLs: query strings/fragments are not filename text,
    # and spaces or non-ASCII characters may be percent-encoded.
    path = unquote(urlsplit(view_path.replace("\\", "/")).path)
    parts = [p for p in path.split("/") if p]
    if "raw" in parts:
        rel = parts[parts.index("raw") + 1 :]
    elif parts and parts[0] == "View":
        rel = parts[1:]
    else:
        rel = parts
    if rel:
        direct = os.path.join(generated_folder, *rel)
        if os.path.isfile(direct):
            return direct
    name = parts[-1] if parts else ""
    if not name:
        return None
    for root, dirs, files in os.walk(generated_folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if name in files:
            return os.path.join(root, name)
    return None
