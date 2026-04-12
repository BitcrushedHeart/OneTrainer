import logging
import os
import string

from fastapi import HTTPException

logger = logging.getLogger(__name__)

_BLOCKED_NAMES = {
    ".git",
    ".env",
    "__pycache__",
    "secrets.json",
    "node_modules",
}

_BLOCKED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".key",
    ".pem",
}

_WIN_LONG_PATH_PREFIX = "\\\\?\\"

_MIN_ALLOWLIST_DEPTH = 2


def _strip_win_long_prefix(path: str) -> str:
    if path.startswith(_WIN_LONG_PATH_PREFIX):
        remainder = path[len(_WIN_LONG_PATH_PREFIX):]
        if remainder.upper().startswith("UNC\\") or remainder.upper().startswith("UNC/"):
            return "\\\\" + remainder[4:]
        return remainder
    return path


def _is_filesystem_root(path: str) -> bool:
    path = _strip_win_long_prefix(path)
    normed = os.path.normpath(path)

    if normed == "/" or normed == os.sep:
        return True

    if len(normed) <= 3 and len(normed) >= 2 and normed[0] in string.ascii_letters and normed[1] == ":":
        return True

    if normed.startswith("\\\\"):
        unc_parts = normed.split(os.sep)
        if len(unc_parts) <= 4:
            return True

    return False


def _path_depth(path: str) -> int:
    path = _strip_win_long_prefix(path)
    normed = os.path.normpath(path)
    _, tail = os.path.splitdrive(normed)
    tail = tail.lstrip(os.sep)
    if not tail:
        return 0
    return len(tail.split(os.sep))


def _is_safe_for_allowlist(path: str) -> bool:
    if _is_filesystem_root(path):
        logger.warning(
            "Rejecting allowlist path (filesystem root): %s",
            path,
        )
        return False

    depth = _path_depth(path)
    if depth < _MIN_ALLOWLIST_DEPTH:
        logger.warning(
            "Rejecting allowlist path (depth %d < minimum %d): %s",
            depth,
            _MIN_ALLOWLIST_DEPTH,
            path,
        )
        return False

    return True


def base_match(canonical: str, base: str) -> bool:
    canonical = os.path.normpath(canonical)
    base = os.path.normpath(base)

    if os.name == "nt":
        canonical = canonical.lower()
        base = base.lower()

    base_clean = base.rstrip(os.sep)
    canon_clean = canonical.rstrip(os.sep)

    if canon_clean == base_clean:
        return True

    base_with_sep = base_clean + os.sep
    return canonical.startswith(base_with_sep)


def validate_path(
    user_path: str,
    *,
    must_exist: bool = True,
    allow_file: bool = True,
    allow_dir: bool = True,
) -> str:
    if not user_path or not user_path.strip():
        raise HTTPException(status_code=400, detail="Empty path")

    try:
        canonical = os.path.realpath(user_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    if must_exist:
        if not os.path.exists(canonical):
            raise HTTPException(status_code=404, detail="Path not found")
        if not allow_file and os.path.isfile(canonical):
            raise HTTPException(status_code=400, detail="Expected directory, got file")
        if not allow_dir and os.path.isdir(canonical):
            raise HTTPException(status_code=400, detail="Expected file, got directory")

    parts = os.path.normpath(canonical).split(os.sep)
    for part in parts:
        if part.lower() in _BLOCKED_NAMES:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: path contains restricted component '{part}'",
            )

    _, ext = os.path.splitext(canonical)
    if ext.lower() in _BLOCKED_SUFFIXES:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: file type '{ext}' is restricted",
        )

    allowed_bases = _get_allowed_bases()
    if allowed_bases:
        if not any(base_match(canonical, base) for base in allowed_bases):
            raise HTTPException(
                status_code=403,
                detail="Access denied: path is outside allowed directories",
            )

    return canonical


def _get_allowed_bases() -> list[str]:
    from web.backend.paths import PROJECT_ROOT

    bases = [os.path.realpath(PROJECT_ROOT)]

    if _path_depth(PROJECT_ROOT) < _MIN_ALLOWLIST_DEPTH:
        logger.warning(
            "PROJECT_ROOT is at depth %d (%s) — allowlist may be overly broad",
            _path_depth(PROJECT_ROOT),
            PROJECT_ROOT,
        )

    try:
        from web.backend.services.config_service import ConfigService

        config = ConfigService.get_instance().config

        if config.workspace_dir:
            resolved = os.path.realpath(config.workspace_dir)
            if resolved not in bases and _is_safe_for_allowlist(resolved):
                bases.append(resolved)

        concepts = config.concepts
        if not concepts:
            import json
            try:
                concept_file = getattr(config, "concept_file_name", None)
                if concept_file and os.path.isfile(concept_file):
                    with open(concept_file, "r", encoding="utf-8") as fh:
                        concepts = json.load(fh)
            except Exception:
                concepts = None

        if concepts:
            for concept in concepts:
                cpath = concept.get("path") if isinstance(concept, dict) else getattr(concept, "path", None)
                if cpath:
                    resolved = os.path.realpath(cpath)
                    if resolved not in bases and _is_safe_for_allowlist(resolved):
                        bases.append(resolved)
    except Exception as exc:
        logger.warning("Could not read config for path allowlist: %s", exc)

    return bases
