"""
Cheap, stable fingerprint of the configured dataset.

Used by the accumulator-persistence path (Fix B): on resume, we compare
the saved fingerprint to the current one and warn if they differ.  The
fingerprint is purely informational -- partial accumulator/grad state
is restored regardless of mismatch (the user's call: losing hours of
gradient state is worse than an off-spec effective batch composition
on one window).

The fingerprint hashes a stable tuple of concept identifiers --
specifically the fields that determine *which samples will be drawn*:
name, path, seed, type, include_subdirectories, enabled.  It does NOT
enumerate sample files on disk: that would be expensive and only
partially more informative (a moved file with the same path-stem will
hash differently anyway via the dataloader's per-sample seeding).

Concepts in OneTrainer can come from either ``config.concepts`` (the
in-memory list used by API callers and tests) or, more commonly under
the GUI, ``config.concept_file_name`` (a JSON file whose contents
``DataLoaderMgdsMixin`` lazily loads at training start). The helper
accepts both and falls back to reading the JSON file when the
in-memory list is empty or None -- matching the pattern in
``TrainConfig.to_pack_dict`` at lines 1013-1018.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable

from modules.util.config.ConceptConfig import ConceptConfig


def _identifier_tuple(c) -> tuple:
    """Return the stable identifier tuple for one concept.

    Works on both ConceptConfig instances (attribute access) and the
    raw dicts that come out of concepts.json (item access). Falls back
    to safe defaults for any missing key.
    """

    def g(name, default):
        if hasattr(c, name):
            return getattr(c, name)
        if isinstance(c, dict):
            return c.get(name, default)
        return default

    raw_type = g("type", "")
    # ConceptType has a stable .value; raw JSON gives a string already.
    type_str = getattr(raw_type, "value", raw_type)

    return (
        str(g("name", "") or ""),
        str(g("path", "") or ""),
        int(g("seed", 0) or 0),
        str(type_str or ""),
        bool(g("include_subdirectories", False)),
        bool(g("enabled", True)),
    )


def compute_concept_fingerprint(
    concepts: Iterable[ConceptConfig] | Iterable[dict] | None,
    concept_file_name: str | None = None,
) -> tuple[str, int]:
    """Return (sha256_hex, concept_count) for the configured dataset.

    ``concepts``: the in-memory list (preferred when populated).
    ``concept_file_name``: fallback JSON path. Read only if ``concepts``
    is empty/None AND the file exists; mirrors
    ``TrainConfig.to_pack_dict``'s lazy-load pattern. Failure to read
    the file is non-fatal: returns the empty-array hash, count=0 (the
    caller's mismatch check is warn-only anyway).
    """
    items: list = []
    if concepts:
        items = list(concepts)
    elif concept_file_name and os.path.exists(concept_file_name):
        try:
            with open(concept_file_name, "r") as f:
                items = json.load(f) or []
        except (OSError, ValueError):
            items = []

    payload = [_identifier_tuple(c) for c in items]
    payload.sort(key=lambda t: t[1])  # stable order: by path
    blob = json.dumps(payload, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(blob).hexdigest(), len(payload)
