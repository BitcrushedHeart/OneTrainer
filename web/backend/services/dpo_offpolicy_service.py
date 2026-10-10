"""Off-policy DPO builder: real photo (chosen) vs the model's own generation
of one of that photo's captions (rejected, hardest negative by realism score),
with a one-click confirmer that the caption actually describes the photo.

Runs as a background scan that streams scored candidates into a queue, the
same shape as the curation session, so the first pair is ready while the
rest are still being scored.
"""

import logging
import os
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress

from modules.util.dpo_offpolicy_util import (
    DEFAULT_ASPECT_TOLERANCE,
    DEFAULT_MAX_UPSCALE,
    REAL_INDEX_FILENAME,
    Candidate,
    GeneratedImage,
    RealPhoto,
    RealPhotoIndex,
    bucket_summary,
    build_candidates,
    caption_match_key,
    choose_negative,
    entry_stamp,
    filter_pool_by_keywords,
    find_photos_by_name,
    find_sidecar_entry,
    group_generated_by_key,
    is_within,
    load_permanent_skips,
    migrate_permanent_skips,
    photo_from_entry,
    photo_generation_brief,
    pick_hardest_negative,
    purge_trash,
    read_caption_lines,
    refilter_candidate,
    remove_caption_line,
    reroll_negative,
    restore_caption_file,
    retire_obsolete_skip_files,
    save_permanent_skips,
    sidecar_map,
    suggest_captions,
    suggestion_pool,
    used_caption_keys_from_chosen_dir,
    used_caption_keys_from_manifest,
    walk_entries_skipping_dotted,
)
from web.backend.services._dpo_session_lock import dpo_session_admission
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)


class DPOOffPolicyService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = False
        #: Bumped by every ``start``. The scan worker captures the value it was
        #: started under and stops writing session state once it changes, so
        #: a cancel-then-start while scoring is still running (the model call
        #: can take seconds) cannot leak a stale worker's candidates into the
        #: new session or mark it finished.
        self._generation = 0
        self._generated_folder: str | None = None
        self._real_folder: str | None = None
        self._output_dir: str | None = None
        self._aspect_tolerance = DEFAULT_ASPECT_TOLERANCE
        self._max_upscale = DEFAULT_MAX_UPSCALE
        self._base_resolution = 1024
        #: Candidate presentation order: "random" (shuffled) or "path" (sorted).
        self._order = "random"
        self._manifest: dict = {"pairs": []}
        #: Photos skipped (or caption-edited) this session — normalized source
        #: paths. Deliberately not persisted: a plain skip means "not now", and
        #: the photo comes back in the next session.
        self._session_skipped: set[str] = set()
        #: Photos the user never wants offered again, loaded from and written
        #: to the output folder. Always a subset of ``_session_skipped`` while
        #: the session runs.
        self._permanent_skipped: set[str] = set()
        self._cache = None
        self._used_fingerprints: dict[int, set[str]] = {}
        #: Source paths committed to *other* configured DPO concepts (the
        #: curated AI-vs-AI set, for one), so their images are never re-paired.
        self._foreign_used_sources: set[str] = set()
        self._too_small = 0
        #: Cache entries imported from other DPO concepts' scan caches at start.
        self._cache_imported = 0
        self._foreign_fingerprints: dict[int, set[str]] = {}
        #: Actions taken this session, newest last, for Ctrl+Z. Each entry
        #: holds what is needed to put things back exactly.
        self._undo_stack: list[dict] = []
        #: Files listed so far during indexing (the walk reports no other progress).
        self._listed = 0
        self._photos: list[RealPhoto] = []
        #: Photos indexed so far — live during the indexing pass, then == len(_photos).
        self._photo_count = 0
        self._index_hits = 0
        self._generated: list[GeneratedImage] = []
        self._generated_by_key: dict[str, list[GeneratedImage]] = {}
        self._excluded_keys: set[str] = set()
        self._candidates_total = 0
        self._ready: deque[Candidate] = deque()
        self._current: Candidate | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._worker_finished = False
        self._phase = "idle"
        self._progress_done = 0
        self._progress_total = 0
        self._error: str | None = None
        self._warning: str | None = None
        self._unscored = 0
        self._shown = 0
        self._confirmed = 0
        self._skipped = 0
        self._captions_removed = 0
        self._skipped_forever = 0
        self._rng = random.Random()

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    @staticmethod
    def _folder_layout_error(generated_folder: str, real_folder: str, output_dir: str) -> str | None:
        """The three trees must not nest. Output inside the generated tree
        would feed exported rejected copies back in as fresh generations;
        output inside the real tree would index exported chosen copies (with
        their sidecars) as new photos; real inside generated would run
        metadata extraction — a full read for files without prompt markers —
        over every photo in the dataset."""
        if is_within(generated_folder, real_folder) or is_within(real_folder, generated_folder):
            return "The real photo folder and the generated folder must not contain each other"
        for label, folder in (("generated", generated_folder), ("real photo", real_folder)):
            if is_within(output_dir, folder):
                return f"The output folder must not be inside the {label} folder"
            if is_within(folder, output_dir):
                return f"The {label} folder must not be inside the output folder"
        return None

    @staticmethod
    def _foreign_cache_paths(output_dir: str) -> list[str]:
        """``.dpo_hash_cache.json`` of every configured DPO concept other than
        ``output_dir`` — the curation tool writes one per output folder."""
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
        from web.backend.services.dpo_service import DPOService

        try:
            concepts = DPOService.get_instance()._resolve_concepts()
            roots = {os.path.dirname(chosen) for chosen, _rejected in dpo_concept_pattern_dirs(concepts)}
        except Exception as ex:  # advisory: a bad concept must not block a start
            logger.warning("could not enumerate DPO concepts for cache reuse: %s", ex)
            return []
        own = os.path.normcase(os.path.abspath(output_dir))
        paths = []
        for root in sorted(roots):
            if os.path.normcase(os.path.abspath(root)) == own:
                continue
            candidate = os.path.join(root, ".dpo_hash_cache.json")
            if os.path.isfile(candidate):
                paths.append(candidate)
        return paths

    @classmethod
    def _import_foreign_caches(cls, cache, output_dir: str) -> int:
        imported = 0
        for path in cls._foreign_cache_paths(output_dir):
            n = cache.merge_from(path)
            if n:
                logger.info("imported %d scan-cache entries from %s", n, path)
            imported += n
        return imported

    @dpo_session_admission
    def start(
        self,
        generated_folder: str,
        real_folder: str,
        output_dir: str,
        aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
        max_upscale: float = DEFAULT_MAX_UPSCALE,
        base_resolution: int = 1024,
        order: str = "random",
    ) -> dict:
        from modules.util.dpo_curation_util import (
            DpoScanCache,
            load_manifest,
            manifest_used_fingerprints,
            prune_orphaned_pairs,
        )

        if not os.path.isdir(generated_folder):
            return {"ok": False, "error": f"Generated folder not found: {generated_folder}"}
        if not os.path.isdir(real_folder):
            return {"ok": False, "error": f"Real photo folder not found: {real_folder}"}
        if not output_dir:
            return {"ok": False, "error": "Output folder is required"}
        if not (0.0 <= aspect_tolerance <= 1.0):
            return {"ok": False, "error": "Aspect tolerance must be between 0 and 1"}
        if not (1.0 <= max_upscale <= 4.0):
            return {"ok": False, "error": "Allowed upscale must be between 1.0 (none) and 4.0"}
        if order not in ("random", "path"):
            return {"ok": False, "error": "order must be 'random' or 'path'"}
        layout_error = self._folder_layout_error(generated_folder, real_folder, output_dir)
        if layout_error:
            return {"ok": False, "error": layout_error}
        with self._lock:
            if self._active:
                return {"ok": False, "error": "Session already active"}
            self._active = True
            self._stop.set()  # retire any worker still running from a previous session
            self._generation += 1
            generation = self._generation

        try:
            os.makedirs(output_dir, exist_ok=True)
            # Deletions from a previous session are final once it ended.
            purge_trash(real_folder)
            manifest = load_manifest(output_dir)
            pruned = prune_orphaned_pairs(output_dir, manifest)
            # Byte fingerprints of everything already exported, so a duplicate
            # real photo under another name (or a generation copied elsewhere)
            # can't be paired twice. Backfills old entries once and saves.
            fingerprints = manifest_used_fingerprints(manifest, output_dir)
        except (OSError, ValueError) as ex:
            with self._lock:
                self._active = False
            return {"ok": False, "error": f"Could not open output folder: {ex}"}

        with self._lock:
            self._generated_folder = generated_folder
            self._real_folder = real_folder
            self._output_dir = output_dir
            self._aspect_tolerance = aspect_tolerance
            self._max_upscale = max_upscale
            self._base_resolution = max(256, int(base_resolution))
            self._order = order
            self._manifest = manifest
            # A skip list left behind under an earlier build's filename is
            # adopted here (once) rather than silently read as an empty list —
            # that is what made "never again" look like it did nothing.
            retire_obsolete_skip_files(output_dir)
            adopted = migrate_permanent_skips(output_dir)
            if adopted:
                logger.info("adopted %d permanent skip(s) from a legacy skip file in %s", adopted, output_dir)
            self._permanent_skipped = load_permanent_skips(output_dir)
            self._session_skipped = set(self._permanent_skipped)
            self._used_fingerprints = fingerprints
            self._foreign_used_sources = set()
            self._foreign_fingerprints = {}
            self._undo_stack = []
            self._listed = 0
            self._too_small = 0
            # Shared with the curation tool's cache format so metadata, sizes
            # and realism scores are reused across both tools on the same
            # output dir.
            self._cache = DpoScanCache(os.path.join(output_dir, ".dpo_hash_cache.json"))
            # The curation tool has usually already scanned the same generated
            # tree into its own output folder's cache — metadata, sizes, pixel
            # hashes and realism scores. Import what it has rather than
            # recomputing it; every entry is still validated per file on use.
            self._cache_imported = self._import_foreign_caches(self._cache, output_dir)
            self._photos = []
            self._photo_count = 0
            self._index_hits = 0
            self._generated = []
            self._generated_by_key = {}
            self._excluded_keys = set()
            self._candidates_total = 0
            self._ready = deque()
            self._current = None
            self._worker_finished = False
            self._phase = "indexing"
            self._progress_done = 0
            self._progress_total = 0
            self._error = None
            self._warning = None
            self._unscored = 0
            self._shown = self._confirmed = self._skipped = self._captions_removed = 0
            self._skipped_forever = 0
            self._stop = threading.Event()
            self._worker_thread = threading.Thread(
                target=self._scan, args=(generation, self._stop, self._cache), daemon=True
            )
            self._worker_thread.start()
        return {
            "ok": True,
            "existing_pairs": len(self._manifest.get("pairs", [])),
            "pruned": pruned,
            "cache_imported": self._cache_imported,
            # Reported so a list that silently failed to load is visible in the
            # UI at start, rather than only as photos coming back.
            "permanently_skipped": len(self._permanent_skipped),
        }

    def cancel(self) -> dict:
        with self._lock:
            self._stop.set()
            was_active = self._active
            self._active = False
            self._current = None
            self._ready.clear()
            self._undo_stack = []
            self._phase = "idle"
            cache = self._cache
            real_folder = self._real_folder
        if cache is not None:
            cache.save()
        purged = 0
        if was_active and real_folder:
            # Finishing the session makes this session's deletions permanent.
            purged = purge_trash(real_folder)
        return {"ok": True, "purged": purged}

    def status(self) -> dict:
        return {
            "active": self._active,
            "phase": self._phase,
            "progress_done": self._progress_done,
            "progress_total": self._progress_total,
            "worker_finished": self._worker_finished,
            "error": self._error,
            "warning": self._warning,
            "unscored": self._unscored,
            "photos": self._photo_count,
            "index_hits": self._index_hits,
            "generated": len(self._generated),
            "excluded_captions": len(self._excluded_keys),
            "too_small": self._too_small,
            "listed": self._listed,
            "can_undo": bool(self._undo_stack),
            "undo_action": self._undo_stack[-1]["action"] if self._undo_stack else None,
            "cache_imported": self._cache_imported,
            "candidates": self._candidates_total,
            "ready": len(self._ready),
            "shown": self._shown,
            "confirmed": self._confirmed,
            "skipped": self._skipped,
            "captions_removed": self._captions_removed,
            "skipped_forever": self._skipped_forever,
            "permanently_skipped": len(self._permanent_skipped),
            "total_pairs": len(self._manifest.get("pairs", [])),
            "has_candidate": self._current is not None,
        }

    # ------------------------------------------------------------------ #
    # Scan worker
    # ------------------------------------------------------------------ #

    def _live(self, generation: int, stop: threading.Event) -> bool:
        return self._generation == generation and not stop.is_set()

    def _scan(self, generation: int, stop: threading.Event, cache) -> None:
        try:
            self._scan_inner(generation, stop, cache)
        except Exception as ex:  # surfaced to the UI, not swallowed
            logger.exception("off-policy DPO scan failed")
            if self._generation == generation:
                self._error = str(ex)
                self._phase = "error"
        finally:
            if self._generation == generation:
                self._worker_finished = True
            cache.save()

    def _scan_inner(self, generation: int, stop: threading.Event, cache) -> None:
        from modules.util import path_util
        from modules.util.dpo_curation_util import (
            normalize_prompt_for_grouping,
            resolve_aspect_ratio,
            walk_skipping_dotted,
        )
        from modules.util.dpo_offpolicy_util import image_size
        from modules.util.image_metadata_util import extract_metadata

        supported = path_util.supported_image_extensions()
        workers = min(16, (os.cpu_count() or 4) * 2)
        real_folder = self._real_folder
        generated_folder = self._generated_folder

        # --- real photos -------------------------------------------------
        # One directory listing per folder; each entry's stat() is served from
        # that listing on Windows. Photos whose image+sidecar stamps match the
        # persistent index are resolved inline with no file access; only new
        # or changed files go to the pool to read the sidecar and the image
        # header.
        index = RealPhotoIndex(os.path.join(self._output_dir, REAL_INDEX_FILENAME))
        photos: list[RealPhoto] = []
        seen: set[str] = set()
        misses: list[tuple[str, str, str, tuple[int, int], tuple[int, int]]] = []
        self._progress_total = 0
        self._progress_done = 0
        for _root, files in walk_entries_skipping_dotted(real_folder):
            if not self._live(generation, stop):
                return
            self._listed += len(files)
            sidecars = sidecar_map(files)
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() not in supported:
                    continue
                sidecar = find_sidecar_entry(files, name, sidecars)
                if sidecar is None:
                    continue
                image_entry = files[name]
                image_stamp = entry_stamp(image_entry)
                sidecar_stamp = entry_stamp(sidecar)
                if image_stamp is None or sidecar_stamp is None:
                    continue
                path = image_entry.path
                key = RealPhotoIndex.key(path)
                seen.add(key)
                self._progress_total += 1
                cached = index.get(key, image_stamp, sidecar_stamp)
                if cached is not None:
                    photo = photo_from_entry(path, sidecar.path, cached)
                    if photo is not None:
                        photos.append(photo)
                        self._photo_count = len(photos)
                    self._progress_done += 1
                else:
                    misses.append((path, sidecar.path, key, image_stamp, sidecar_stamp))
        self._index_hits = index.hits

        def index_photo(item) -> RealPhoto | None:
            path, caption_path, key, image_stamp, sidecar_stamp = item
            lines = read_caption_lines(caption_path)
            size = image_size(path) if lines else None
            entry = index.put(key, image_stamp, sidecar_stamp, size, lines)
            return photo_from_entry(path, caption_path, entry)

        if misses:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(index_photo, item): item for item in misses}
                for future in as_completed(futures):
                    if not self._live(generation, stop):
                        pool.shutdown(wait=False, cancel_futures=True)
                        index.save(seen)
                        return
                    self._progress_done += 1
                    with suppress(Exception):
                        photo = future.result()
                        if photo is not None:
                            photos.append(photo)
                            self._photo_count = len(photos)
        index.save(seen)
        photos.sort(key=lambda p: p.path)
        if not self._live(generation, stop):
            return
        self._photos = photos
        self._photo_count = len(photos)

        # --- generated images -------------------------------------------
        self._phase = "scanning"
        gen_paths: list[str] = []
        for root, files in walk_skipping_dotted(generated_folder):
            if not self._live(generation, stop):
                return
            gen_paths.extend(
                os.path.join(root, f) for f in sorted(files) if os.path.splitext(f)[1].lower() in supported
            )
        self._progress_total = len(gen_paths)
        self._progress_done = 0

        def extract_group_key(path: str) -> tuple[str, str]:
            meta = extract_metadata(path)
            ar = resolve_aspect_ratio(meta.get("aspectratio", ""), path)
            prompt = normalize_prompt_for_grouping(meta.get("prompt", ""))
            return prompt, ar

        def index_generated(path: str) -> GeneratedImage | None:
            key = cache.get_group_key(path, extract_group_key)
            if key is None:
                return None
            size = cache.get_image_size(path, image_size)
            if size is None:
                return None
            return GeneratedImage(path, caption_match_key(key[0]), size[0], size[1])

        generated: list[GeneratedImage] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(index_generated, p): p for p in gen_paths}
            for future in as_completed(futures):
                if not self._live(generation, stop):
                    pool.shutdown(wait=False, cancel_futures=True)
                    return
                self._progress_done += 1
                with suppress(Exception):
                    gen = future.result()
                    if gen is not None and gen.prompt_key:
                        generated.append(gen)
        generated.sort(key=lambda g: g.path)
        if not self._live(generation, stop):
            return
        self._generated = generated
        self._generated_by_key = group_generated_by_key(generated)

        # --- captions and images that already have a pair anywhere ----------
        self._phase = "matching"
        excluded = self._collect_excluded_keys()
        foreign_sources, foreign_fingerprints = self._collect_foreign_usage()
        if not self._live(generation, stop):
            return
        self._excluded_keys = excluded
        stats: dict[str, int] = {}
        with self._lock:
            self._foreign_used_sources = foreign_sources
            self._foreign_fingerprints = foreign_fingerprints
            for size, digests in foreign_fingerprints.items():
                self._used_fingerprints.setdefault(size, set()).update(digests)
            candidates = build_candidates(
                self._photos,
                self._generated,
                excluded_caption_keys=self._excluded_keys,
                used_sources=self._all_used_sources(),
                skipped_sources=self._session_skipped,
                aspect_tolerance=self._aspect_tolerance,
                max_upscale=self._max_upscale,
                stats=stats,
            )
        self._too_small = stats.get("too_small", 0)
        # Path order tracks file naming (usually generation time), so a session
        # would walk the newest shots first; shuffle unless asked not to.
        if self._order == "random":
            self._rng.shuffle(candidates)
        self._candidates_total = len(candidates)

        # --- hardest negative per candidate, from cached scores only --------
        # The realism model is never run by this tool: scores come from this
        # output dir's cache (including what was imported from the curation
        # tool's cache at start). A candidate with no cached score for any of
        # its generations gets a random one and is shown as UNSCORED.
        self._phase = "scoring"
        self._progress_total = len(candidates)
        self._progress_done = 0
        score_of = self._cached_score_fn(cache)
        for candidate in candidates:
            if not self._live(generation, stop):
                return
            pick_hardest_negative(candidate, score_of, self._rng)
            candidate.real_score = score_of(candidate.real.path)
            # The pick may have blocked (cache lookups can); a worker that was
            # retired meanwhile must not hand its candidate to the new session.
            if not self._live(generation, stop):
                return
            if not candidate.scored:
                self._unscored += 1
            self._ready.append(candidate)
            self._progress_done += 1
        self._phase = "ready"

    def _collect_excluded_keys(self) -> set[str]:
        """Caption keys that already have a DPO pair: everything in this output
        dir's manifest plus every configured DPO concept's chosen captions."""
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
        from web.backend.services.dpo_service import DPOService

        keys = used_caption_keys_from_manifest(self._manifest)
        try:
            concepts = DPOService.get_instance()._resolve_concepts()
            for chosen_dir, _rejected_dir in dpo_concept_pattern_dirs(concepts):
                keys |= used_caption_keys_from_chosen_dir(chosen_dir)
        except Exception as ex:
            # Not fatal — but a silently empty exclusion list looks the same as
            # "no concept has pairs yet", so say it where the user can see it.
            logger.warning("could not read configured DPO concepts for exclusion: %s", ex)
            self._warning = f"Configured DPO concepts could not be read for caption exclusion: {ex}"
        return keys

    def _collect_foreign_usage(self) -> tuple[set[str], dict[int, set[str]]]:
        """Images already committed to any *other* configured DPO concept —
        source paths and byte fingerprints — so a generation the curation tool
        paired (or a photo another off-policy concept used) is never picked
        here. Read errors degrade to "nothing foreign", with the same visible
        warning as the caption exclusion."""
        from modules.util.dpo_curation_util import concept_usage_for_exclusion
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
        from web.backend.services.dpo_service import DPOService

        try:
            concepts = DPOService.get_instance()._resolve_concepts()
            sources, fingerprints, _paths = concept_usage_for_exclusion(
                dpo_concept_pattern_dirs(concepts), exclude_dir=self._output_dir
            )
            return sources, fingerprints
        except Exception as ex:
            logger.warning("could not read configured DPO concepts for image exclusion: %s", ex)
            self._warning = f"Configured DPO concepts could not be read for exclusion: {ex}"
            return set(), {}

    def _all_used_sources(self) -> set[str]:
        from modules.util.dpo_curation_util import manifest_used_sources

        return manifest_used_sources(self._manifest) | self._foreign_used_sources

    @staticmethod
    def _cached_score_fn(cache):
        """``path -> cached P(AI) or None``. Reads the cache only; a missing
        model (no version to validate against) means nothing is scored."""
        from modules.util.realism_detect_util import realism_model_version

        try:
            version = realism_model_version()
        except Exception:
            version = None
        if cache is None or version is None:
            return lambda _path: None
        return lambda path: cache.peek_realism(path, version)

    # ------------------------------------------------------------------ #
    # Confirmer
    # ------------------------------------------------------------------ #

    def _is_content_used(self, path: str) -> bool:
        from modules.util.dpo_curation_util import is_byte_identical_used

        return is_byte_identical_used(self._used_fingerprints, path)

    def next(self) -> dict:
        from modules.util.dpo_curation_util import _normalize_source_path, is_source_used

        if not self._active:
            return {"ok": False, "error": "No active session"}
        with self._lock:
            while True:
                if not self._ready:
                    if self._worker_finished:
                        self._current = None
                        return {"ok": True, "done": True, "total_pairs": len(self._manifest.get("pairs", []))}
                    return {"ok": True, "waiting": True}
                candidate = self._ready.popleft()
                # Re-check at present time: the manifest and the exclusion set
                # have grown since the scan built this candidate. The photo may
                # have been paired (a duplicate under another name), its
                # captions may have been paired for another photo, and a
                # generation may have been committed elsewhere.
                used = self._all_used_sources()
                if is_source_used(used, candidate.real.path) or self._is_content_used(candidate.real.path):
                    continue
                # The queue was built before this session's skips existed (and
                # ``undo`` can push a candidate back into it), so a photo the
                # user skipped — "not now" or "never again" — could otherwise
                # still be served from it. ``_session_skipped`` is a superset
                # of ``_permanent_skipped`` for the life of the session.
                if _normalize_source_path(candidate.real.path) in self._session_skipped:
                    continue
                previous = candidate.rejected
                if not refilter_candidate(
                    candidate,
                    excluded_caption_keys=self._excluded_keys,
                    used_sources=used,
                    is_content_used=self._is_content_used,
                ):
                    continue
                if candidate.rejected is None or candidate.rejected is not previous:
                    pick_hardest_negative(candidate, self._cached_score, self._rng)
                if candidate.rejected is not None and candidate.rejected.path not in candidate.shown_paths:
                    candidate.shown_paths.append(candidate.rejected.path)
                self._current = candidate
                self._shown += 1
                return {"ok": True, "candidate": self._candidate_payload(candidate)}

    def reroll(self) -> dict:
        """Show a different generation for the photo on screen (see
        :func:`reroll_negative`). The photo and caption pool stay the same."""
        with self._lock:
            candidate = self._current
            if candidate is None:
                return {"ok": False, "error": "No candidate on screen"}
            if not reroll_negative(candidate, self._cached_score, self._rng):
                return {"ok": False, "error": "This photo has only one usable generation"}
            return {"ok": True, "candidate": self._candidate_payload(candidate)}

    def choose(self, path: str) -> dict:
        """Make one of the on-screen photo's generations (by path, from
        ``generated_paths``) the rejected image — the side-by-side picker's
        explicit choice, where :meth:`reroll` cycles."""
        with self._lock:
            candidate = self._current
            if candidate is None:
                return {"ok": False, "error": "No candidate on screen"}
            if not path or not choose_negative(candidate, str(path), self._cached_score):
                return {"ok": False, "error": "That image is not one of this photo's generations"}
            return {"ok": True, "candidate": self._candidate_payload(candidate)}

    def current(self) -> dict:
        """The candidate on screen, so a reloaded UI can pick the session back
        up instead of waiting for a ``next`` that never comes."""
        if not self._active:
            return {"ok": False, "error": "No active session"}
        with self._lock:
            if self._current is None:
                return {"ok": True, "candidate": None}
            return {"ok": True, "candidate": self._candidate_payload(self._current)}

    def _cached_score(self, path: str) -> float | None:
        from modules.util.realism_detect_util import realism_model_version

        if self._cache is None:
            return None
        return self._cache.peek_realism(path, realism_model_version())

    def _candidate_payload(self, candidate: Candidate) -> dict:
        from modules.util.dpo_curation_util import trainer_bucket_key

        data = candidate.to_dict()
        data["aspect_bucket"] = trainer_bucket_key(candidate.real.width, candidate.real.height)
        data["index"] = self._shown
        data["total"] = self._candidates_total
        return data

    def confirm(self) -> dict:
        """The caption fits the photo: export real-as-chosen / generated-as-rejected."""
        from modules.util.dpo_curation_util import export_single_pair, trainer_bucket_key

        with self._lock:
            candidate = self._current
            if candidate is None or candidate.rejected is None:
                return {"ok": False, "error": "No candidate to confirm"}
            aspect = trainer_bucket_key(candidate.real.width, candidate.real.height)
            try:
                export_single_pair(
                    self._output_dir,
                    self._manifest,
                    candidate.real.path,
                    candidate.rejected.path,
                    candidate.caption,
                    aspect,
                )
            except OSError as ex:
                return {"ok": False, "error": f"Export failed: {ex}"}
            entry = self._manifest["pairs"][-1]
            for hash_key, size_key in (("chosen_sha256", "chosen_size"), ("rejected_sha256", "rejected_size")):
                digest, size = entry.get(hash_key), entry.get(size_key)
                if digest and isinstance(size, int):
                    self._used_fingerprints.setdefault(size, set()).add(digest)
            key = caption_match_key(candidate.caption)
            key_added = key not in self._excluded_keys
            self._excluded_keys.add(key)
            self._confirmed += 1
            self._current = None
            self._undo_stack.append(
                {"action": "confirm", "candidate": candidate, "pair_id": entry["pair_id"], "key_added": key_added}
            )
            return {"ok": True, "total_pairs": len(self._manifest.get("pairs", []))}

    def skip(self) -> dict:
        """Not now: leave the photo alone for the rest of this session. Not
        persisted — it is offered again next session."""
        from modules.util.dpo_curation_util import _normalize_source_path

        with self._lock:
            candidate = self._current
            if candidate is None:
                return {"ok": False, "error": "No candidate to skip"}
            self._session_skipped.add(_normalize_source_path(candidate.real.path))
            self._skipped += 1
            self._current = None
            self._undo_stack.append({"action": "skip", "candidate": candidate})
            return {"ok": True}

    def skip_forever(self) -> dict:
        """Never offer this photo again for this output folder. Persisted, so
        it survives restarts; undo takes it back off the list."""
        from modules.util.dpo_curation_util import _normalize_source_path

        with self._lock:
            candidate = self._current
            if candidate is None:
                return {"ok": False, "error": "No candidate to skip"}
            normed = _normalize_source_path(candidate.real.path)
            was_listed = normed in self._permanent_skipped
            self._permanent_skipped.add(normed)
            try:
                save_permanent_skips(self._output_dir, self._permanent_skipped)
            except OSError as ex:
                if not was_listed:
                    self._permanent_skipped.discard(normed)
                return {"ok": False, "error": f"Could not save the skip list: {ex}"}
            self._session_skipped.add(normed)
            self._skipped_forever += 1
            self._current = None
            self._undo_stack.append({"action": "skip_forever", "candidate": candidate, "was_listed": was_listed})
            return {"ok": True, "permanently_skipped": len(self._permanent_skipped)}

    def remove_caption(self) -> dict:
        """The caption on screen is wrong for the photo: remove that line from
        the sidecar (the other lines and the photo stay). If it was the only
        usable line the sidecar is deleted. The original is copied to
        ``<real folder>/.offpolicy_trash`` and can be restored with undo; the
        trash is purged when the session finishes."""
        from modules.util.dpo_curation_util import _normalize_source_path

        with self._lock:
            candidate = self._current
            if candidate is None:
                return {"ok": False, "error": "No candidate on screen"}
            try:
                result = remove_caption_line(candidate.real.caption_path, candidate.caption, self._real_folder)
            except (OSError, ValueError) as ex:
                return {"ok": False, "error": f"Could not edit the caption: {ex}"}
            if result["removed"] == 0:
                return {"ok": False, "error": "That line is no longer in the caption file"}
            previous_lines = list(candidate.real.lines)
            candidate.real.lines = result["remaining"]
            # Its other lines may still have generations; leave that for the
            # next session's re-index rather than re-queueing it now.
            self._session_skipped.add(_normalize_source_path(candidate.real.path))
            self._captions_removed += 1
            self._current = None
            self._undo_stack.append(
                {
                    "action": "caption",
                    "candidate": candidate,
                    "original": result["original"],
                    "backup": result["backup"],
                    "previous_lines": previous_lines,
                }
            )
            return {
                "ok": True,
                "removed_lines": result["removed"],
                "removed_file": result["removed_file"],
                "remaining_lines": len(result["remaining"]),
            }

    def undo(self) -> dict:
        """Reverse the most recent confirm, skip or delete and put that
        candidate back on screen. A candidate currently on screen goes back
        to the front of the queue."""
        from modules.util.dpo_curation_util import manifest_used_fingerprints, save_manifest

        with self._lock:
            if not self._active:
                return {"ok": False, "error": "No active session"}
            if not self._undo_stack:
                return {"ok": False, "error": "Nothing to undo"}
            record = self._undo_stack.pop()
            candidate: Candidate = record["candidate"]
            action = record["action"]
            try:
                if action == "confirm":
                    pairs = self._manifest.get("pairs", [])
                    entry = next((e for e in pairs if e.get("pair_id") == record["pair_id"]), None)
                    if entry is not None:
                        for sub, name in (("chosen", entry["chosen_file"]), ("rejected", entry["rejected_file"])):
                            path = os.path.join(self._output_dir, sub, name)
                            if os.path.isfile(path):
                                os.remove(path)
                        txt = os.path.join(
                            self._output_dir, "chosen", os.path.splitext(entry["chosen_file"])[0] + ".txt"
                        )
                        if os.path.isfile(txt):
                            os.remove(txt)
                        pairs.remove(entry)
                        save_manifest(self._output_dir, self._manifest)
                    self._used_fingerprints = manifest_used_fingerprints(self._manifest, self._output_dir)
                    for size, digests in self._foreign_fingerprints.items():
                        self._used_fingerprints.setdefault(size, set()).update(digests)
                    if record.get("key_added"):
                        self._excluded_keys.discard(caption_match_key(candidate.caption))
                    self._confirmed = max(0, self._confirmed - 1)
                elif action == "skip":
                    from modules.util.dpo_curation_util import _normalize_source_path

                    self._session_skipped.discard(_normalize_source_path(candidate.real.path))
                    self._skipped = max(0, self._skipped - 1)
                elif action == "skip_forever":
                    from modules.util.dpo_curation_util import _normalize_source_path

                    normed = _normalize_source_path(candidate.real.path)
                    if not record.get("was_listed"):
                        self._permanent_skipped.discard(normed)
                        save_permanent_skips(self._output_dir, self._permanent_skipped)
                        self._session_skipped.discard(normed)
                    self._skipped_forever = max(0, self._skipped_forever - 1)
                elif action == "caption":
                    from modules.util.dpo_curation_util import _normalize_source_path

                    restore_caption_file(candidate.real.caption_path, record["original"])
                    # The backup exists only to undo this edit; once undone it
                    # would otherwise linger and be counted by the purge.
                    backup = record.get("backup")
                    if backup and os.path.isfile(backup):
                        with suppress(OSError):
                            os.remove(backup)
                    candidate.real.lines = list(record["previous_lines"])
                    self._session_skipped.discard(_normalize_source_path(candidate.real.path))
                    self._captions_removed = max(0, self._captions_removed - 1)
                else:
                    return {"ok": False, "error": f"Unknown action {action!r}"}
            except OSError as ex:
                return {"ok": False, "error": f"Undo failed: {ex}"}
            if self._current is not None:
                self._ready.appendleft(self._current)
            self._current = candidate
            return {"ok": True, "undone": action, "candidate": self._candidate_payload(candidate)}

    # ------------------------------------------------------------------ #
    # "What should I generate next?"
    # ------------------------------------------------------------------ #

    def _suggestion_pool(self, keywords: str = "") -> list:
        """Caller holds ``_lock``. ``keywords`` narrows to caption lines that
        contain every comma-separated term."""
        from modules.util.dpo_curation_util import parse_caption_keywords

        pool = suggestion_pool(
            self._photos,
            self._generated_by_key,
            excluded_caption_keys=self._excluded_keys,
            used_sources=self._all_used_sources(),
            skipped_sources=self._session_skipped,
            base_resolution=self._base_resolution,
            aspect_tolerance=self._aspect_tolerance,
            max_upscale=self._max_upscale,
        )
        return filter_pool_by_keywords(pool, parse_caption_keywords(keywords))

    def suggest(self, aspect_bucket: str | None = None, count: int = 1, keywords: str = "") -> dict:
        """Random captions worth generating for, optionally only from photos in
        one trainer bucket, with the size to generate each at. Also returns the
        per-bucket counts so a UI can offer the buckets that still have work.
        ``suggestion`` (first entry or None) is kept for single-pick callers."""
        if not self._active:
            return {"ok": False, "error": "No active session"}
        if self._phase in ("indexing", "scanning"):
            return {"ok": True, "waiting": True, "phase": self._phase}
        count = max(1, min(int(count), 100))
        with self._lock:
            pool = self._suggestion_pool(keywords)
            entries = suggest_captions(
                pool,
                self._rng,
                count,
                aspect_bucket=aspect_bucket or None,
                base_resolution=self._base_resolution,
                aspect_tolerance=self._aspect_tolerance,
            )
            buckets = bucket_summary(pool, self._base_resolution)
        first = dict(entries[0]) | {"remaining": len(pool)} if entries else None
        return {
            "ok": True,
            "suggestion": first,
            "suggestions": entries,
            "remaining": len(pool),
            "aspect_bucket": aspect_bucket or None,
            "keywords": keywords or "",
            "buckets": buckets,
        }

    def buckets(self, keywords: str = "") -> dict:
        """Trainer buckets that still have photos to generate for, with counts
        and the canonical size to generate each bucket at; ``keywords``
        narrows the count to photos with a matching caption line."""
        if not self._active:
            return {"ok": False, "error": "No active session"}
        if self._phase in ("indexing", "scanning"):
            return {"ok": True, "waiting": True, "phase": self._phase}
        with self._lock:
            pool = self._suggestion_pool(keywords)
            buckets = bucket_summary(pool, self._base_resolution)
        return {"ok": True, "buckets": buckets, "remaining": len(pool)}

    def lookup(self, filename: str) -> dict:
        """Find indexed photos by file name and report the size to generate
        at for each. String matching against the index only — the name is
        never resolved on disk. Available as soon as the real photos are
        indexed, even while generations are still being scanned (a photo's
        generation status is then reported against what has been read so far)."""
        if not self._active:
            return {"ok": False, "error": "No active session"}
        if self._phase == "indexing":
            return {"ok": True, "waiting": True, "phase": self._phase}
        query = (filename or "").strip()
        if not query:
            return {"ok": False, "error": "Type a file name"}
        with self._lock:
            matches = find_photos_by_name(self._photos, query)
            used = self._all_used_sources()
            skipped = self._session_skipped
            briefs = [
                photo_generation_brief(
                    photo,
                    self._generated_by_key,
                    excluded_caption_keys=self._excluded_keys,
                    used_sources=used,
                    skipped_sources=skipped,
                    base_resolution=self._base_resolution,
                    aspect_tolerance=self._aspect_tolerance,
                    max_upscale=self._max_upscale,
                )
                for photo in matches
            ]
        return {"ok": True, "matches": briefs, "generations_scanned": self._phase not in ("indexing", "scanning")}
