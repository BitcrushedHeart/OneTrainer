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
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from modules.util.config.ConceptConfig import ConceptConfig


def compute_concept_fingerprint(
        concepts: Iterable[ConceptConfig] | None,
) -> tuple[str, int]:
    """Return (sha256_hex, concept_count) for the given concept list.

    Empty/None input is allowed; returns the SHA-256 of the empty
    JSON array and count=0.  Mismatch handling is the caller's job.
    """
    payload = []
    if concepts:
        for c in concepts:
            payload.append((
                getattr(c, 'name', '') or '',
                getattr(c, 'path', '') or '',
                int(getattr(c, 'seed', 0) or 0),
                # ConceptType has a stable .value
                str(getattr(getattr(c, 'type', None), 'value', '') or ''),
                bool(getattr(c, 'include_subdirectories', False)),
                bool(getattr(c, 'enabled', True)),
            ))
    payload.sort(key=lambda t: t[1])   # stable order: by path
    blob = json.dumps(payload, separators=(',', ':'), sort_keys=False).encode()
    return hashlib.sha256(blob).hexdigest(), len(payload)
