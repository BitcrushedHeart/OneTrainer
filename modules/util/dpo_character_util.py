"""Per-character DPO dataset tooling: the pure pieces behind the Character
DPO modal.

A character's dataset lives under ``<root>/<Name>``::

    train/, val/                 SFT images + .txt (+ -masklabel.png)
    dpo/generated/               SwarmUI renders
    dpo/offpolicy/, dpo/aivai/   pair folders (manifest.json, chosen/, rejected/)
    dpo/concepts.json            the four DPO concepts the trainer reads
    dpo/character.json           tool state (imports, last finalize)

This module holds what can be unit-tested without a server: LoRA resolution,
name / LoRA matching against a pair, the incremental train/val split, the
concept file writer and the queue entry builder. Nothing here talks to the
queue service, SwarmUI or a mask model; ``character_dpo_service`` does that.
"""

import json
import os
import re
import shutil
from contextlib import suppress

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.dpo_curation_util import find_exported_file, save_manifest
from modules.util.enum.ConceptType import ConceptType

PAIR_KINDS = ("offpolicy", "aivai")
SPLITS = ("train", "val")
MASK_SUFFIX = "-masklabel.png"
CHARACTER_STATE_FILENAME = "character.json"
CONCEPTS_FILENAME = "concepts.json"
PATIENCE_SECOND_SUFFIX = "-patience-second"

_COPY_SUFFIX_RE = re.compile(r"(\s*\(copy\))+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Names and paths
# ---------------------------------------------------------------------------


def name_no_spaces(name: str) -> str:
    return "".join(str(name or "").split())


def normalize_queue_name(name: str) -> str:
    """Queue entry name with any trailing `` (copy)`` suffixes removed, lowered."""
    return _COPY_SUFFIX_RE.sub("", str(name or "")).strip().lower()


def character_dir(root: str, name: str) -> str:
    return os.path.join(root, name)


def dpo_dir(root: str, name: str) -> str:
    return os.path.join(root, name, "dpo")


def pair_dir(root: str, name: str, kind: str) -> str:
    return os.path.join(dpo_dir(root, name), kind)


def mask_path_for(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + MASK_SUFFIX


#: Separator between the folders a LoRA may live in (``models_dir`` on every
#: character route): the trainer's own output folder and the folder finished
#: LoRAs are moved to, say.
MODELS_DIR_SEPARATOR = ";"


def split_models_dirs(models_dir: str | list[str] | None) -> list[str]:
    """The folders of a ``;``-separated ``models_dir`` (or a list), in order,
    blanks dropped and repeats removed."""
    raw = models_dir if isinstance(models_dir, list | tuple) else str(models_dir or "").split(MODELS_DIR_SEPARATOR)
    out: list[str] = []
    for folder in raw:
        folder = str(folder or "").strip()
        if folder and folder not in out:
            out.append(folder)
    return out


def resolve_lora(models_dir: str | list[str], name: str) -> tuple[str | None, str | None]:
    """``(path, kind)`` of the character's LoRA under ``models_dir`` — one
    folder or several (``;``-separated or a list), searched in order:
    ``<NameNoSpaces>.safetensors`` (or the exact name with spaces) is the
    patience best → ``"best"``, taken from the first folder that has one;
    only when no folder does, ``<...>-patience-second.safetensors`` →
    ``"second"`` from the first folder that has it; else ``(None, None)``.
    ``-patience-alt`` is never used."""
    stems = []
    for stem in (name_no_spaces(name), str(name or "").strip()):
        if stem and stem not in stems:
            stems.append(stem)
    folders = split_models_dirs(models_dir)
    for suffix, kind in (("", "best"), (PATIENCE_SECOND_SUFFIX, "second")):
        for folder in folders:
            for stem in stems:
                candidate = os.path.join(folder, stem + suffix + ".safetensors")
                if os.path.isfile(candidate):
                    return candidate, kind
    return None, None


def lora_stem(lora_path: str | None) -> str:
    return os.path.splitext(os.path.basename(lora_path or ""))[0]


# ---------------------------------------------------------------------------
# Tool state
# ---------------------------------------------------------------------------


def load_character_state(root: str, name: str) -> dict:
    path = os.path.join(dpo_dir(root, name), CHARACTER_STATE_FILENAME)
    state: dict = {"name": name, "lora_path": None, "imported": [], "last_finalize": None, "settings": {}}
    if os.path.isfile(path):
        with suppress(OSError, ValueError):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                state.update(loaded)
    if not isinstance(state.get("imported"), list):
        state["imported"] = []
    return state


def save_character_state(root: str, name: str, state: dict) -> None:
    folder = dpo_dir(root, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, CHARACTER_STATE_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Matching a pair to a character
# ---------------------------------------------------------------------------


def _squash(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def name_matches(name: str, *texts: str | None) -> bool:
    """Character name (case- and whitespace-insensitive) appears in any text."""
    needle = _squash(name)
    if not needle:
        return False
    return any(needle in _squash(text) for text in texts if text)


def lora_match_stems(name: str, lora_path: str | None) -> set[str]:
    """Stems a render's ``loras`` entry may carry for this character: the
    resolved file's stem, that stem without a ``-patience-*`` suffix, and the
    plain ``NameNoSpaces`` (an earlier LoRA version shares it)."""
    stems = {name_no_spaces(name).lower()}
    stem = lora_stem(lora_path).lower()
    if stem:
        stems.add(stem)
        stems.add(re.sub(r"-patience-[a-z0-9]+$", "", stem))
    return {s for s in stems if s}


def lora_matches(stems, *provenances: dict | None) -> bool:
    """A LoRA stem (case-insensitive, extension and folders ignored) is among
    the ``loras`` of any provenance dict. ``stems`` is one stem or a set."""
    wanted = {str(s).lower() for s in ([stems] if isinstance(stems, str) else stems or []) if s}
    if not wanted:
        return False
    for prov in provenances:
        if not isinstance(prov, dict):
            continue
        for lora in prov.get("loras") or []:
            base = os.path.splitext(os.path.basename(str(lora).replace("\\", "/")))[0].lower()
            if any(base == stem or stem in base for stem in wanted):
                return True
    return False


def match_kind(by_name: bool, by_lora: bool) -> str | None:
    if by_name and by_lora:
        return "both"
    if by_name:
        return "name"
    if by_lora:
        return "lora"
    return None


# ---------------------------------------------------------------------------
# Pair folder inspection
# ---------------------------------------------------------------------------


def locate_pair_files(pair_folder: str, entry: dict) -> tuple[str | None, str | None]:
    chosen = find_exported_file(os.path.join(pair_folder, "chosen"), entry.get("chosen_file", ""))
    rejected = find_exported_file(os.path.join(pair_folder, "rejected"), entry.get("rejected_file", ""))
    return chosen, rejected


def split_of_path(pair_folder: str, side: str, path: str | None) -> str | None:
    """``"train"`` / ``"val"`` when ``path`` sits under ``<side>/<split>/``,
    else None (flat, unsplit)."""
    if not path:
        return None
    base = os.path.normcase(os.path.abspath(os.path.join(pair_folder, side)))
    parent = os.path.normcase(os.path.abspath(os.path.dirname(path)))
    for split in SPLITS:
        if parent == os.path.normcase(os.path.join(base, split)):
            return split
    return None


def pair_folder_summary(pair_folder: str, manifest: dict) -> dict:
    """``{train, val, unsplit, missing_masks}`` for one pair folder."""
    summary = {"train": 0, "val": 0, "unsplit": 0, "missing_masks": 0}
    for entry in manifest.get("pairs", []):
        chosen, rejected = locate_pair_files(pair_folder, entry)
        if not chosen or not rejected:
            continue
        split = split_of_path(pair_folder, "chosen", chosen)
        if split is None:
            summary["unsplit"] += 1
        else:
            summary[split] += 1
        for path in (chosen, rejected):
            if not os.path.isfile(mask_path_for(path)):
                summary["missing_masks"] += 1
    return summary


def pairs_missing_masks(pair_folder: str, manifest: dict) -> list[tuple[dict, str, str]]:
    """``(entry, side, image_path)`` for every exported side without a mask."""
    missing = []
    for entry in manifest.get("pairs", []):
        chosen, rejected = locate_pair_files(pair_folder, entry)
        for side, path in (("chosen", chosen), ("rejected", rejected)):
            if path and not os.path.isfile(mask_path_for(path)):
                missing.append((entry, side, path))
    return missing


# ---------------------------------------------------------------------------
# Incremental train / val split
# ---------------------------------------------------------------------------


def _sidecar_paths(image_path: str) -> list[str]:
    stem = os.path.splitext(image_path)[0]
    return [stem + ".txt", stem + MASK_SUFFIX]


def _move_side(pair_folder: str, side: str, src: str, split: str) -> None:
    dest_dir = os.path.join(pair_folder, side, split)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    if os.path.normcase(os.path.abspath(src)) != os.path.normcase(os.path.abspath(dest)):
        shutil.move(src, dest)
    for sidecar in _sidecar_paths(src):
        if os.path.isfile(sidecar):
            target = os.path.join(dest_dir, os.path.basename(sidecar))
            if os.path.normcase(os.path.abspath(sidecar)) != os.path.normcase(os.path.abspath(target)):
                shutil.move(sidecar, target)


def split_new_pairs(pair_folder: str, manifest: dict, val_pct: float, rng=None) -> dict:
    """Assign every unsplit pair to ``train`` or ``val`` and move its files.

    Pairs already under ``chosen/<split>/`` (or with ``entry["split"]`` set
    and files in place) keep their assignment. New pairs are assigned so the
    val total approaches ``round(total * val_pct / 100)``, with at least one
    val pair when ``total >= 2`` and ``val_pct > 0``. Image, ``.txt`` and
    ``-masklabel.png`` move together for both sides. The manifest is saved
    through :func:`save_manifest` (pass the object ``load_manifest`` gave).
    Returns ``{"train", "val", "moved"}`` counts."""
    import random

    rng = rng or random.Random()
    pairs = [e for e in manifest.get("pairs", []) if isinstance(e, dict)]
    total = 0
    counts = {"train": 0, "val": 0}
    new_entries = []
    #: Flat pairs whose entry already names a split (files put back by hand,
    #: or a manifest restored from backup) go to that split, not the draw.
    recorded: list[tuple[str, str, str]] = []
    for entry in pairs:
        chosen, rejected = locate_pair_files(pair_folder, entry)
        if not chosen or not rejected:
            continue
        total += 1
        split = split_of_path(pair_folder, "chosen", chosen)
        if split is not None:
            counts[split] += 1
            if entry.get("split") != split:
                entry["split"] = split
        elif entry.get("split") in SPLITS:
            recorded.append((chosen, rejected, entry["split"]))
            counts[entry["split"]] += 1
        else:
            new_entries.append((entry, chosen, rejected))

    moved = 0
    for chosen, rejected, split in recorded:
        _move_side(pair_folder, "chosen", chosen, split)
        _move_side(pair_folder, "rejected", rejected, split)
        moved += 1

    target_val = round(total * max(0.0, float(val_pct)) / 100.0)
    if val_pct > 0 and total >= 2:
        target_val = max(1, target_val)
    target_val = min(target_val, total)
    need_val = max(0, target_val - counts["val"])

    order = list(range(len(new_entries)))
    rng.shuffle(order)
    val_indices = set(order[:need_val])
    for index, (entry, chosen, rejected) in enumerate(new_entries):
        split = "val" if index in val_indices else "train"
        _move_side(pair_folder, "chosen", chosen, split)
        _move_side(pair_folder, "rejected", rejected, split)
        entry["split"] = split
        counts[split] += 1
        moved += 1
    if pairs:
        save_manifest(pair_folder, manifest)
    return {"train": counts["train"], "val": counts["val"], "moved": moved}


# ---------------------------------------------------------------------------
# Concept file
# ---------------------------------------------------------------------------


#: Resolution the DPO pair concepts train at. The pairs are exported at
#: roughly this size, where the global run's own resolution is a lot lower.
DEFAULT_DPO_RESOLUTION = "1024"
DEFAULT_DPO_BETA = 200.0


def _concept(
    name: str,
    path: str,
    split: str,
    concept_type,
    enabled: bool,
    unmasked_probability: float,
    resolution_override: str,
) -> dict:
    cfg = ConceptConfig.default_values()
    cfg.name = name
    cfg.path = path
    cfg.type = concept_type
    cfg.enabled = enabled
    cfg.include_subdirectories = True
    cfg.dpo_chosen_pattern = f"chosen/{split}/{{}}"
    cfg.dpo_rejected_pattern = f"rejected/{split}/{{}}"
    cfg.unmasked_probability_override = unmasked_probability
    cfg.image.enable_resolution_override = True
    cfg.image.resolution_override = str(resolution_override)
    return cfg.to_dict()


def build_character_concepts(
    root: str,
    name: str,
    counts: dict[str, dict[str, int]],
    resolution_override: str = DEFAULT_DPO_RESOLUTION,
) -> list[dict]:
    """The four DPO concept dicts for a character. ``counts`` maps kind →
    ``{"train": n, "val": n}``; a concept whose folder has no pairs is
    written disabled. Every concept carries ``resolution_override`` (the
    pairs' own size, not the global run's). Paths use forward slashes."""
    concepts = []
    resolution = str(resolution_override or DEFAULT_DPO_RESOLUTION).strip() or DEFAULT_DPO_RESOLUTION
    for kind, label in (("offpolicy", "offpolicy"), ("aivai", "aivai")):
        path = os.path.abspath(pair_dir(root, name, kind)).replace("\\", "/")
        kind_counts = counts.get(kind) or {}
        unmasked = 0.0 if kind == "offpolicy" else -1.0
        concepts.append(
            _concept(
                f"DPO {label} - {name}",
                path,
                "train",
                ConceptType.STANDARD,
                bool(kind_counts.get("train", 0)),
                unmasked,
                resolution,
            )
        )
        concepts.append(
            _concept(
                f"DPO {label} val - {name}",
                path,
                "val",
                ConceptType.VALIDATION,
                bool(kind_counts.get("val", 0)),
                unmasked,
                resolution,
            )
        )
    return concepts


def write_character_concepts(
    root: str,
    name: str,
    counts: dict[str, dict[str, int]],
    resolution_override: str = DEFAULT_DPO_RESOLUTION,
) -> str:
    concepts = build_character_concepts(root, name, counts, resolution_override)
    folder = dpo_dir(root, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, CONCEPTS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2)
    return path


def load_concept_dicts(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a concept list")
    return [c for c in data if isinstance(c, dict)]


# ---------------------------------------------------------------------------
# Queue entry
# ---------------------------------------------------------------------------


def _is_sft_concept(concept: dict, name: str) -> bool:
    cname = str(concept.get("name") or "")
    lowered = cname.lower()
    if not lowered.startswith(("train", "validation")):
        return False
    return name_matches(name, cname)


def select_sft_concepts(concepts: list[dict], name: str) -> list[dict]:
    """Enabled ``Train - <Name>`` / ``Validation - <Name>`` concept dicts."""
    return [c for c in concepts if isinstance(c, dict) and c.get("enabled", True) and _is_sft_concept(c, name)]


def find_sft_queue_entry(entries, name: str):
    """First queue entry whose name (minus `` (copy)`` suffixes) equals the
    character name, case-insensitive. ``entries`` may be ``QueueEntry``
    objects or dicts; returns the entry or None."""
    wanted = normalize_queue_name(name)
    if not wanted:
        return None
    for entry in entries or []:
        entry_name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", "")
        if normalize_queue_name(entry_name) == wanted:
            return entry
    return None


def _entry_overrides(entry) -> dict:
    if entry is None:
        return {}
    overrides = entry.get("overrides") if isinstance(entry, dict) else getattr(entry, "overrides", None)
    return overrides if isinstance(overrides, dict) else {}


def _dpo_concept_has_pairs(concept: dict) -> bool:
    if not concept.get("enabled", True):
        return False
    path = str(concept.get("path") or "")
    chosen_dir = os.path.join(path, os.path.dirname(str(concept.get("dpo_chosen_pattern") or "")))
    if not os.path.isdir(chosen_dir):
        return False
    return any(name.startswith("pair_") and not name.endswith(MASK_SUFFIX) for name in os.listdir(chosen_dir))


def build_dpo_entry(
    name: str,
    root: str,
    models_dir: str,
    workspace_dir: str,
    sft_queue_entry=None,
    global_concepts: list[dict] | None = None,
    add_on_weight: float = 0.7,
    dpo_concepts: list[dict] | None = None,
    dpo_beta: float = DEFAULT_DPO_BETA,
) -> dict:
    """``{"name", "overrides", "warnings"}`` for the character's SFT+DPO queue
    entry. Raises ``ValueError`` with a user-facing message when the LoRA,
    the DPO concepts or the SFT concepts cannot be found.

    ``dpo_concepts`` defaults to the character's ``dpo/concepts.json``."""
    warnings: list[str] = []
    display = str(name or "").strip()
    if not display:
        raise ValueError("Character name is required")
    compact = name_no_spaces(display)

    lora_path, _kind = resolve_lora(models_dir, display)
    if not lora_path:
        raise ValueError(f"No LoRA for {display} in {models_dir} ({compact}.safetensors or {compact}-patience-second)")

    if dpo_concepts is None:
        concepts_path = os.path.join(dpo_dir(root, display), CONCEPTS_FILENAME)
        if not os.path.isfile(concepts_path):
            raise ValueError(f"No DPO concepts for {display}: finalize first")
        dpo_concepts = load_concept_dicts(concepts_path)
    dpo_concepts = [dict(c) for c in dpo_concepts]

    sft_overrides = _entry_overrides(sft_queue_entry)
    sft_source = sft_overrides.get("concepts") if sft_queue_entry is not None else None
    sft_concepts = select_sft_concepts(sft_source, display) if isinstance(sft_source, list) else []
    if not sft_concepts:
        if sft_queue_entry is None:
            warnings.append(f"No SFT queue entry named {display}; SFT concepts taken from the global concept file")
        sft_concepts = select_sft_concepts(global_concepts or [], display)
    if not sft_concepts:
        raise ValueError(f"No enabled 'Train - {display}' / 'Validation - {display}' concepts found")
    sft_concepts = [dict(c) for c in sft_concepts]

    val_concepts = [
        c
        for c in dpo_concepts
        if str(c.get("type") or "").upper().endswith("VALIDATION") or " val " in f" {c.get('name', '')} ".lower()
    ]
    dpo_validation = any(_dpo_concept_has_pairs(c) for c in val_concepts)
    if not dpo_validation:
        warnings.append("No DPO validation pairs: DPO validation disabled for this entry")

    missing_masks = 0
    for kind in PAIR_KINDS:
        folder = pair_dir(root, display, kind)
        if not os.path.isdir(folder):
            continue
        for walk_root, _dirs, files in os.walk(folder):
            for file in files:
                stem, ext = os.path.splitext(file)
                if not stem.startswith("pair_") or ext.lower() in (".txt", ".json") or file.endswith(MASK_SUFFIX):
                    continue
                if not os.path.isfile(os.path.join(walk_root, stem + MASK_SUFFIX)):
                    missing_masks += 1
    if missing_masks:
        warnings.append(f"{missing_masks} pair image(s) have no -masklabel.png; run finalize to generate masks")

    workspace = os.path.join(workspace_dir, "dpo", compact).replace("\\", "/")
    overrides = {
        "concepts": sft_concepts + dpo_concepts,
        "lora_model_name": lora_path,
        "rlhf_enabled": True,
        "rlhf_mode": "SFT_DPO",
        "rlhf_dpo_addon_weight": float(add_on_weight),
        "rlhf_dpo_addon_interval": 0,
        "rlhf_dpo_validation": dpo_validation,
        "rlhf_dpo_save_best": False,
        "rlhf_dpo_patience_enabled": False,
        # The Train concept already carries the supervised signal at full weight; a
        # chosen-image MSE inside the add-on would double-count the real photos and,
        # on AI-vs-AI pairs, pull the adapter toward the chosen render's look.
        "rlhf_supervised_mix": 0.0,
        # Warm start from the character's own LoRA: the margin begins at zero, so a
        # large fixed beta yanks the adapter on the first informative batches. Start
        # low and let beta-DPO scale it from the smoothed margin (bounded x1/4..x4).
        "rlhf_dpo_beta": float(dpo_beta),
        "rlhf_dpo_adaptive_beta": True,
        "validation": True,
        "patience_min_checks": 0,
        "continue_last_backup": False,
        # The DPO concepts carry a resolution cap; adaptive mode is what snaps each
        # pair to the ladder tier below its own size instead of upscaling to the cap.
        "adaptive_resolution": True,
        "workspace_dir": workspace,
        # Next to the LoRA it was trained from, whichever of the folders that is.
        "output_model_destination": os.path.join(os.path.dirname(lora_path), f"{compact}-dpo.safetensors").replace(
            "\\", "/"
        ),
        "save_filename_prefix": f"{compact}DPO",
    }
    for key in ("gradient_accumulation_steps", "patience_epochs"):
        if key in sft_overrides:
            overrides[key] = sft_overrides[key]
    return {"name": f"{display} DPO", "overrides": overrides, "warnings": warnings}
