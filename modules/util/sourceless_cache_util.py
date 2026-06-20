"""Shared helpers for checking whether a SmartDiskCache is ready for
sourceless training.

Sourceless training reads everything (latents/embeddings + per-sample
concept/prompt/pairing metadata) from the cache, with the source images and
captions stubbed out. That only works if the cache index (``cache.json``) has
the sourceless metadata baked in. These helpers read the index JSON only — no
``.pt`` tensors, no torch — so they are cheap enough to run as a pre-flight
guard before training or before a multi-hundred-GB upload to a cloud GPU.

The metadata-presence predicate here mirrors
``SmartDiskCache._entry_has_any_sourceless_metadata`` in the mgds library; keep
the two in sync.
"""

import json
import os
from dataclasses import dataclass, field

# Cache layout produced by DataLoaderText2ImageMixin._cache_modules_from_names:
# <cache_dir>/image/cache.json and <cache_dir>/text/cache.json. The VAE
# fine-tune loader uses <cache_dir>/cache.json directly.
_SUBDIRS = ("image", "text")


def entry_has_sourceless_metadata(entry: dict) -> bool:
    """True when a cache.json entry carries usable sourceless metadata.

    Matches the mgds-side presence check: either the legacy entry-level
    ``sourceless`` block, or any stamped per-row ``metadata``. Deliberately does
    NOT require ``runtime_values`` (legitimately empty for no-concept rows and
    optional at read time).
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("sourceless"):
        return True
    rows = entry.get("sourceless_rows") or {}
    if not isinstance(rows, dict):
        return False
    return any(isinstance(row, dict) and row.get("metadata") for row in rows.values())


@dataclass
class SubdirCoverage:
    """Sourceless-metadata coverage for a single cache.json directory."""

    path: str
    exists: bool
    total: int = 0
    missing: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def stamped(self) -> int:
        return self.total - len(self.missing)

    @property
    def complete(self) -> bool:
        # A directory with no cache.json (not part of this run) is not a
        # problem; a present index with any unstamped entry is.
        if not self.exists or self.error is not None:
            return self.error is None
        return self.total > 0 and not self.missing


def _coverage_for_cache_json_dir(cache_json_dir: str) -> SubdirCoverage:
    cache_json = os.path.join(cache_json_dir, "cache.json")
    if not os.path.isfile(cache_json):
        return SubdirCoverage(path=cache_json, exists=False)
    try:
        with open(cache_json, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return SubdirCoverage(path=cache_json, exists=True, error=str(e))

    entries = index.get("entries", {})
    if not isinstance(entries, dict):
        return SubdirCoverage(path=cache_json, exists=True, error="malformed 'entries'")

    missing = [fp for fp, entry in entries.items() if not entry_has_sourceless_metadata(entry)]
    return SubdirCoverage(path=cache_json, exists=True, total=len(entries), missing=missing)


def sourceless_cache_coverage(cache_dir: str) -> list[SubdirCoverage]:
    """Coverage for the cache subdirs that actually exist.

    Checks ``<cache_dir>/image`` and ``<cache_dir>/text`` (the text2image
    layout) and falls back to ``<cache_dir>`` itself (the VAE-finetune layout)
    when no subdir index is present.
    """
    coverages = [_coverage_for_cache_json_dir(os.path.join(cache_dir, sub)) for sub in _SUBDIRS]
    if not any(c.exists for c in coverages):
        coverages = [_coverage_for_cache_json_dir(cache_dir)]
    return coverages


def sourceless_cache_problems(cache_dir: str, *, sample: int = 3) -> list[str]:
    """Human-readable problems blocking sourceless training; empty when ready.

    Each string is a complete, user-facing sentence suitable for an exception
    message or a wizard validation error.
    """
    problems: list[str] = []
    coverages = sourceless_cache_coverage(cache_dir)

    if not any(c.exists for c in coverages):
        problems.append(
            f"No cache index (cache.json) found under '{cache_dir}'. Build the cache first "
            "(a normal 'only_cache' pass) before sourceless training."
        )
        return problems

    for cov in coverages:
        if not cov.exists:
            continue
        if cov.error is not None:
            problems.append(f"Cache index '{cov.path}' could not be read: {cov.error}.")
            continue
        if cov.total == 0:
            problems.append(f"Cache index '{cov.path}' has no entries.")
            continue
        if cov.missing:
            examples = ", ".join(os.path.basename(p) for p in cov.missing[:sample])
            problems.append(
                f"{len(cov.missing)} of {cov.total} entries in '{cov.path}' are missing sourceless "
                f"metadata (e.g. {examples}). Run a normal cache pass (only_cache, with source files "
                "present and 'skip cache validation' OFF) to bake the metadata into cache.json, then retry."
            )
    return problems
