"""Per-concept cache roots.

A concept's ``cache_dir`` moves its image/text caches out of the run's
``cache_dir`` (``<root>/image`` and ``<root>/text``), so a large dataset can
keep one cache that every run training on it shares. Resolution and identity
rules come from mgds' ConceptCacheRouter, which does the routing at train time.
"""

from modules.util.config.ConceptConfig import ConceptConfig

from mgds.pipelineModules.ConceptCacheRouter import normalize_cache_root, resolve_cache_root


def concept_cache_root(concept: ConceptConfig, default_root: str) -> str:
    return resolve_cache_root(concept.cache_dir, default_root)


def concept_cache_roots(concepts: list[ConceptConfig], default_root: str, *, enabled_only: bool = True) -> list[str]:
    """Distinct cache roots used by ``concepts``, in concept order."""
    roots: dict[str, str] = {}
    for concept in concepts:
        if enabled_only and not concept.enabled:
            continue
        root = concept_cache_root(concept, default_root)
        roots.setdefault(normalize_cache_root(root), root)
    return list(roots.values())


def uses_concept_cache_roots(concepts: list[ConceptConfig], default_root: str) -> bool:
    """True when any enabled concept caches somewhere other than ``default_root``."""
    default_key = normalize_cache_root(default_root)
    return any(normalize_cache_root(root) != default_key for root in concept_cache_roots(concepts, default_root))
