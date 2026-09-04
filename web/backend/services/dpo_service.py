import logging
import os
import random
import threading
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from queue import Empty, Full, Queue

from modules.util.dpo_swiss_service import SwissTournament, outermost_pairs
from web.backend.services._singleton import SingletonMixin
from web.backend.services.config_service import ConfigService

logger = logging.getLogger(__name__)


class DPOService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ws_broadcast: Callable[[dict], None] | None = None

        # Curation session state
        self._session_active = False
        self._source_folder: str | None = None
        self._output_dir: str | None = None
        self._manifest: dict = {"pairs": []}
        self._pairs_per_group = 1
        self._group_order = "most_realistic"
        self._caption_keywords: list[str] = []
        self._ready_queue: Queue = Queue(maxsize=10)
        self._worker_thread: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._worker_finished = False
        self._scan_count = 0
        self._scan_total = 0
        self._hash_count = 0
        # Persistent per-file scan cache (DpoScanCache, created per session) and
        # the set of pixel hashes already committed to the output dataset.
        # The set is read/updated from both the scan worker and API threads;
        # individual set add/contains ops are atomic under the GIL, matching
        # the lock-free style of the other counters here.
        self._hash_cache = None
        self._used_pixel_hashes: set[str] = set()
        # Source paths committed to *other* configured DPO concepts (e.g. the
        # off-policy builder's output): an image that is already a chosen or
        # rejected sample somewhere must not be offered again here. Filled by
        # the scan worker, read by next_group alongside this session's manifest.
        self._foreign_used_sources: set[str] = set()
        self._groups_queued = 0
        self._groups_shown = 0
        self._current_group: dict | None = None
        self._remaining_images: list[str] = []
        self._mode: str = "selection"  # "selection" | "swiss" | "triage"
        self._selection_phase = "best"  # "best" or "worst"
        self._selected_best: str | None = None
        self._pairs_created_in_group = 0
        # (chosen, rejected) source paths exported from the current group in
        # this session. Only used when the group is retired from the source
        # tree (discard group / discard rest) so the paired sources land in
        # .chosen/.rejected instead of being left behind in the scan tree.
        self._exported_in_group: list[tuple[str, str]] = []
        # Held when the user has picked both best and worst but has not yet
        # confirmed the pair. No file is written and no counters advance
        # until confirm_pair commits or cancel_pending_pair discards it —
        # matches the Ctk yes/no/cancel dialog semantics.
        self._pending_pair: dict | None = None

        # Re-pair-by-similarity background job state (separate from curation
        # sessions — it operates on already-exported concept folders).
        self._repair_lock = threading.Lock()
        self._repair_thread: threading.Thread | None = None
        self._repair_state: dict = {"running": False}

        # Realism backfill: score the whole source tree once, ahead of curation,
        # so Selection never waits on the model. Its own thread and state, kept
        # apart from the scan worker because it outlives any single group.
        self._backfill_thread: threading.Thread | None = None
        self._backfill_stop = threading.Event()
        self._backfill_state: dict = {"running": False}
        # Where the realism head runs: "auto" (CUDA when usable), "cpu", or
        # "off" for no scoring at all. Deliberately not persisted to the config
        # — it answers "what else is using this box right now", which is true of
        # a sitting, not a project.
        self._realism_device = "auto"

        # Swiss tournament state (per-group)
        self._swiss: SwissTournament | None = None
        # Server-authoritative best-to-worst ranking for the review step;
        # populated by swiss_ranking, mutated by swiss_reorder, consumed by
        # swiss_export so a stale client can't export mismatched pairs.
        self._ranked_order: list[str] | None = None

    def set_ws_broadcast(self, fn: Callable[[dict], None]) -> None:
        self._ws_broadcast = fn

    def _broadcast(self, message: dict) -> None:
        if self._ws_broadcast is not None:
            with suppress(Exception):
                self._ws_broadcast(message)

    def _resolve_concepts(self) -> list:
        """Concepts for the active config. The web stores concepts in a separate
        file (``concept_file_name``) rather than inline on the config, so the
        in-memory ``concepts`` list is usually empty — fall back to loading that
        file (mirroring TrainerService) so the DPO tools actually see them."""
        train_config = ConfigService.get_instance().get_config_for_training()
        concepts = train_config.concepts or []
        if not concepts and train_config.concept_file_name and os.path.isfile(train_config.concept_file_name):
            from modules.util.config.ConceptConfig import ConceptConfig
            from web.backend.services.concept_service import ConceptService

            concepts = [
                ConceptConfig.default_values().from_dict(entry)
                for entry in ConceptService().load_concepts(train_config.concept_file_name)
            ]
        return concepts

    def _no_pairs_reason(self, concepts: list) -> str:
        """A specific message for why no DPO concept pairs were found, so the user
        can tell 'no concepts at all' from 'concepts exist but lack DPO patterns'."""
        if not concepts:
            return "No concepts configured. Add concepts (with chosen/rejected DPO patterns) in the Concepts tab."
        enabled = [c for c in concepts if getattr(c, "enabled", False)]
        if not enabled:
            return f"Found {len(concepts)} concept(s) but none are enabled."
        return (
            f"Found {len(enabled)} enabled concept(s), but none have DPO patterns set. "
            "Set the chosen and rejected patterns on a concept (e.g. 'chosen/{}' and 'rejected/{}')."
        )

    def check_pairs(self) -> dict:
        from modules.util.dpo_curation_util import check_dpo_pairs
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        concepts = self._resolve_concepts()
        concept_pairs = dpo_concept_pattern_dirs(concepts)
        if not concept_pairs:
            return {"ok": False, "error": self._no_pairs_reason(concepts)}

        result = check_dpo_pairs(concept_pairs)
        return {"ok": True, "result": result}

    def remove_strays(self) -> dict:
        import os

        from modules.util.dpo_curation_util import (
            check_dpo_pairs,
            dpo_pair_key,
            remove_finalized_pair,
        )
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
        from modules.util.path_util import supported_image_extensions

        concepts = self._resolve_concepts()
        exts = supported_image_extensions()

        concept_pairs = dpo_concept_pattern_dirs(concepts)
        if not concept_pairs:
            return {"ok": False, "error": self._no_pairs_reason(concepts)}

        result = check_dpo_pairs(concept_pairs)
        removed = 0

        for i, (chosen_path, rejected_path) in enumerate(concept_pairs):
            pair_info = result["pairs"][i]
            if pair_info.get("chosen_stray", 0) == 0 and pair_info.get("rejected_stray", 0) == 0:
                continue

            chosen_keys: dict[str, str] = {}
            rejected_keys: dict[str, str] = {}

            for root, _dirs, files in os.walk(chosen_path):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in exts:
                        full = os.path.join(root, fname)
                        chosen_keys[dpo_pair_key(full, chosen_path)] = full

            for root, _dirs, files in os.walk(rejected_path):
                for fname in files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in exts:
                        full = os.path.join(root, fname)
                        rejected_keys[dpo_pair_key(full, rejected_path)] = full

            matched = set(chosen_keys) & set(rejected_keys)
            for key, path in chosen_keys.items():
                if key not in matched:
                    remove_finalized_pair(path, None)
                    removed += 1
            for key, path in rejected_keys.items():
                if key not in matched:
                    remove_finalized_pair(None, path)
                    removed += 1

        return {"ok": True, "removed": removed}

    def review_pairs(self) -> dict:
        from modules.util.dpo_curation_util import scan_finalized_pairs
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        concepts = self._resolve_concepts()
        all_pairs = dpo_concept_pattern_dirs(concepts)
        if not all_pairs:
            return {"ok": False, "error": self._no_pairs_reason(concepts)}

        pairs = scan_finalized_pairs(all_pairs)
        return {"ok": True, "pairs": pairs}

    def remove_pair(self, chosen_path: str | None, rejected_path: str | None) -> dict:
        from modules.util.dpo_curation_util import remove_finalized_pair

        remove_finalized_pair(chosen_path, rejected_path)
        return {"ok": True}

    def fix_multiline_captions(self) -> dict:
        from modules.util.dpo_curation_util import fix_multiline_captions
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        all_pairs = dpo_concept_pattern_dirs(self._resolve_concepts())
        fixed = fix_multiline_captions(all_pairs)
        return {"ok": True, "fixed": fixed}

    def _all_concept_pairs(self) -> list[tuple[str, str]]:
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        return dpo_concept_pattern_dirs(self._resolve_concepts())

    def _is_path_in_concept_pairs(self, path: str, concept_pairs: list[tuple[str, str]]) -> bool:
        """Verify `path` is inside one of the configured DPO concept folders.
        Prevents an arbitrary-filesystem write via /dpo/apply-caption."""
        try:
            target = os.path.realpath(path)
        except (OSError, ValueError):
            return False
        for chosen_path, rejected_path in concept_pairs:
            for base in (chosen_path, rejected_path):
                try:
                    base_real = os.path.realpath(base)
                except (OSError, ValueError):
                    continue
                # Use commonpath so a sibling-prefix match (e.g. /a/foo vs /a/foobar)
                # does not falsely allow access.
                try:
                    if os.path.commonpath([target, base_real]) == base_real:
                        return True
                except ValueError:
                    continue
        return False

    def check_caption_mismatches(self) -> dict:
        from modules.util.dpo_curation_util import find_caption_mismatches

        all_pairs = self._all_concept_pairs()
        if not all_pairs:
            return {"ok": False, "error": "No DPO concept pairs found"}
        mismatches = find_caption_mismatches(all_pairs)
        return {"ok": True, "mismatches": mismatches}

    def correct_all_captions_to_chosen(self) -> dict:
        from modules.util.dpo_curation_util import (
            correct_all_captions_to_chosen,
            find_caption_mismatches,
        )

        all_pairs = self._all_concept_pairs()
        if not all_pairs:
            return {"ok": False, "error": "No DPO concept pairs found"}
        mismatches = find_caption_mismatches(all_pairs)
        corrected = correct_all_captions_to_chosen(mismatches)
        return {"ok": True, "corrected": corrected}

    def repair_rejected(self, quality_floor: float = 0.85) -> dict:
        """Start a re-pair-by-similarity job in the background and return
        immediately. Embedding a whole dataset with DINOv2 takes a while, so the
        client polls :meth:`repair_status` for live progress instead of blocking
        on one long request."""
        from modules.util.dpo_curation_util import repair_rejected_pairs

        all_pairs = self._all_concept_pairs()
        if not all_pairs:
            return {"ok": False, "error": self._no_pairs_reason(self._resolve_concepts())}

        with self._repair_lock:
            if self._repair_state.get("running"):
                return {"ok": False, "error": "A re-pair is already running.", "running": True}
            self._repair_state = {
                "running": True,
                "phase": "starting",
                "images_done": 0,
                "images_total": 0,
                "groups_done": 0,
                "groups_total": 0,
                "pairs_repaired": 0,
                "summary": None,
                "error": None,
            }

        def _progress(p: dict) -> None:
            with self._repair_lock:
                for key in ("phase", "images_done", "images_total", "groups_done", "groups_total", "pairs_repaired"):
                    if key in p:
                        self._repair_state[key] = p[key]

        def _worker() -> None:
            try:
                summary = repair_rejected_pairs(all_pairs, quality_floor=quality_floor, progress=_progress)
                with self._repair_lock:
                    self._repair_state.update(running=False, phase="done", summary=summary)
            except Exception as ex:
                logger.exception("DPO re-pair failed")
                with self._repair_lock:
                    self._repair_state.update(running=False, phase="error", error=str(ex))

        self._repair_thread = threading.Thread(target=_worker, daemon=True)
        self._repair_thread.start()
        return {"ok": True, "started": True}

    def repair_status(self) -> dict:
        with self._repair_lock:
            return {"ok": True, **self._repair_state}

    def apply_caption(self, chosen_image: str, rejected_image: str, caption: str) -> dict:
        from modules.util.dpo_curation_util import apply_caption_to_pair

        all_pairs = self._all_concept_pairs()
        if not all_pairs:
            return {"ok": False, "error": "No DPO concept pairs found"}

        for image_path in (chosen_image, rejected_image):
            if not image_path:
                continue
            if not self._is_path_in_concept_pairs(image_path, all_pairs):
                return {"ok": False, "error": f"Path is outside configured DPO concept folders: {image_path}"}

        try:
            apply_caption_to_pair(chosen_image, rejected_image, caption)
        except OSError as ex:
            return {"ok": False, "error": str(ex)}
        return {"ok": True}

    def bucket_analysis(
        self,
        concept_path: str,
        batch_size: int,
        target_resolutions: list[int],
        quantization: int,
    ) -> dict:
        from modules.util.dpo_bucket_analysis_util import analyze_concept

        try:
            result = analyze_concept(
                concept_path=concept_path,
                batch_size=batch_size,
                target_resolutions=target_resolutions,
                quantization=quantization,
            )
        except (ValueError, FileNotFoundError) as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "result": result}

    # ---- Curation Session ----

    def start_session(
        self,
        source_folder: str,
        output_dir: str,
        pairs_per_group: int = 1,
        mode: str = "selection",
        group_order: str = "most_realistic",
        caption_filter: str = "",
    ) -> dict:
        from modules.util.dpo_curation_util import (
            GROUP_ORDERS,
            DpoScanCache,
            load_manifest,
            parse_caption_keywords,
            prune_orphaned_pairs,
        )

        if mode not in ("selection", "swiss", "triage"):
            return {"ok": False, "error": "mode must be 'selection', 'swiss' or 'triage'"}
        if group_order not in GROUP_ORDERS:
            return {"ok": False, "error": f"group_order must be one of {', '.join(GROUP_ORDERS)}"}

        with self._lock:
            if self._session_active:
                return {"ok": False, "error": "Session already active"}
            self._session_active = True

        self._source_folder = source_folder
        self._output_dir = output_dir
        # Triage drains a whole stack per pass and creates min(good, bad)
        # pairs per group, so pairs_per_group must never pre-skip groups —
        # the sentinel disables the cap in next_group and the scan worker.
        self._pairs_per_group = 10**9 if mode == "triage" else max(1, pairs_per_group)
        self._mode = mode
        self._group_order = group_order
        self._caption_keywords = parse_caption_keywords(caption_filter)
        self._manifest = load_manifest(output_dir)
        pruned = prune_orphaned_pairs(output_dir, self._manifest)
        existing = len(self._manifest.get("pairs", []))

        self._hash_cache = DpoScanCache(os.path.join(output_dir, ".dpo_hash_cache.json"))
        self._used_pixel_hashes = set()
        self._foreign_used_sources = set()

        self._scan_count = 0
        self._scan_total = 0
        self._hash_count = 0
        self._groups_queued = 0
        self._groups_shown = 0
        self._worker_finished = False
        self._current_group = None
        self._remaining_images = []
        self._exported_in_group = []
        self._ready_queue = Queue(maxsize=10)
        self._worker_stop = threading.Event()

        self._worker_thread = threading.Thread(target=self._background_scan, daemon=True)
        self._worker_thread.start()

        return {
            "ok": True,
            "existing_pairs": existing,
            "pruned": pruned,
        }

    def get_session_status(self) -> dict:
        return {
            "active": self._session_active,
            "scan_count": self._scan_count,
            "scan_total": self._scan_total,
            "hash_count": self._hash_count,
            "cache_hits": self._hash_cache.hits if self._hash_cache is not None else 0,
            "groups_queued": self._groups_queued,
            "groups_shown": self._groups_shown,
            "worker_finished": self._worker_finished,
            "has_group": self._current_group is not None,
            "total_pairs": len(self._manifest.get("pairs", [])),
        }

    def next_group(self) -> dict:
        """Advance to the next group. Returns the group data with image paths."""
        from modules.util.dpo_curation_util import is_source_used, manifest_pair_counts, manifest_used_sources

        while True:
            try:
                group = self._ready_queue.get_nowait()
            except Empty:
                if self._worker_finished:
                    return {"done": True, "total_pairs": len(self._manifest.get("pairs", []))}
                return {"waiting": True}

            existing_counts = manifest_pair_counts(self._manifest)
            group_key = (group["prompt"], group["aspectratio"])
            pairs_done = existing_counts.get(group_key, 0)
            is_unconditional = group["prompt"] == "UNCONDITIONAL"
            if not is_unconditional and pairs_done >= self._pairs_per_group:
                continue

            # Re-filter against the manifest at present-time. The scan-time
            # filter is best-effort (the manifest may have grown since), and
            # for groups that are already partway through pairs_per_group we
            # need to drop any sources that were committed in earlier passes.
            # The pixel-hash check additionally drops byte/pixel-identical
            # copies committed under a different path (e.g. a duplicate that
            # lives in another group); these lookups hit the in-memory cache
            # because the scan worker already hashed every queued image.
            used_sources = manifest_used_sources(self._manifest) | self._foreign_used_sources
            available_images = [
                i for i in group["images"] if not is_source_used(used_sources, i) and not self._is_pixel_hash_used(i)
            ]
            if len(available_images) < 2:
                continue

            self._current_group = group
            self._groups_shown += 1
            self._pairs_created_in_group = pairs_done
            self._remaining_images = available_images
            self._exported_in_group = []
            self._selection_phase = "best"
            self._selected_best = None
            self._pending_pair = None

            if self._mode == "swiss":
                self._swiss = SwissTournament(list(self._remaining_images))
                self._ranked_order = None

            return {
                "group": {
                    "prompt": group["prompt"],
                    "aspectratio": group["aspectratio"],
                    "images": available_images,
                    "group_index": self._groups_shown,
                    "total_groups": self._groups_queued,
                    "pairs_done": pairs_done,
                    "pairs_target": self._pairs_per_group,
                    "mode": self._mode,
                },
            }

    def select_image(self, path: str) -> dict:
        """Handle a selection pick: first pick = best, second = worst -> creates pair."""
        from modules.util.dpo_curation_util import export_single_pair

        if not self._current_group:
            return {"ok": False, "error": "No active group"}

        if self._selection_phase == "best":
            self._selected_best = path
            self._selection_phase = "worst"
            remaining = [i for i in self._remaining_images if i != path]
            return {
                "ok": True,
                "phase": "worst",
                "best": path,
                "remaining_images": remaining,
            }
        else:
            chosen = self._selected_best
            rejected = path
            # Defensive: never let the same image land on both sides of a
            # pair. The frontend filters the best out of the worst-pick grid,
            # but a bug there shouldn't be able to corrupt the export.
            if chosen == rejected:
                return {"ok": False, "error": "Chosen and rejected images must differ"}
            group = self._current_group

            # If there'd still be ≥2 images left in the group after this pair,
            # defer the write so the UI can offer Confirm / Pick More / Cancel
            # (matches Ctk). Cancel must be a real undo — so we don't mutate
            # _remaining_images, _pairs_created_in_group, _selection_phase, or
            # _selected_best until confirm_pair commits. On the last-possible
            # pair (can_continue == False) there's no meaningful Cancel, so
            # commit inline — same as Ctk, which also skips the dialog there.
            potential_remaining = [i for i in self._remaining_images if i not in {chosen, rejected}]
            can_continue = len(potential_remaining) >= 2

            if can_continue:
                self._pending_pair = {"chosen": chosen, "rejected": rejected}
                return {
                    "ok": True,
                    "pair_pending": True,
                    "chosen": chosen,
                    "rejected": rejected,
                }

            export_single_pair(
                self._output_dir,
                self._manifest,
                chosen,
                rejected,
                group["prompt"],
                group["aspectratio"],
            )
            self._mark_pair_used(chosen, rejected)
            self._exported_in_group.append((chosen, rejected))
            self._remaining_images = potential_remaining
            self._pairs_created_in_group += 1
            self._selection_phase = "best"
            self._selected_best = None
            return {
                "ok": True,
                "pair_created": True,
                "chosen": chosen,
                "rejected": rejected,
                "continue_group": False,
                "pairs_done": self._pairs_created_in_group,
            }

    def _retire_group_sources(self, rest: list[str]) -> int:
        """Move this group's source images out of the scan tree: each pair
        exported from the group into ``.chosen``/``.rejected``, everything in
        ``rest`` into ``.discard``. Dot-prefixed folders are pruned by the
        scanner, so nothing relocated here can resurface in a later session.
        Returns the number of images discarded. Callers hold ``self._lock``."""
        from modules.util.dpo_curation_util import move_to_dot_bucket

        # export_single_pair already copied the paired sources into the output
        # dir, so relocating the originals now is safe.
        move_to_dot_bucket([chosen for chosen, _ in self._exported_in_group], ".chosen")
        move_to_dot_bucket([rejected for _, rejected in self._exported_in_group], ".rejected")
        return move_to_dot_bucket(rest, ".discard")

    def _release_group(self) -> None:
        """Clear per-group state so fetch_next_group advances. Callers hold
        ``self._lock`` (or own the session, as skip_group does)."""
        self._current_group = None
        self._pending_pair = None
        self._selected_best = None
        self._selection_phase = "best"
        self._exported_in_group = []
        self._swiss = None
        self._ranked_order = None

    def confirm_pair(self, continue_scoring: bool, discard_rest: bool = False) -> dict:
        """Commit the pending pair picked via select_image and decide whether to
        stay in the current group or advance. Mirrors Ctk Yes (keep scoring) /
        No (next group) from the askyesnocancel dialog.

        With ``discard_rest`` the group is also retired from the source tree
        after the write: every source paired in this group moves into
        ``.chosen``/``.rejected`` and every leftover into ``.discard``, so the
        group can't reappear in a later session. Overrides ``continue_scoring``
        — there is nothing left to keep scoring.
        """
        from modules.util.dpo_curation_util import export_single_pair

        with self._lock:
            if not self._pending_pair:
                return {"ok": False, "error": "No pending pair to confirm"}
            if not self._current_group:
                return {"ok": False, "error": "No active group"}

            chosen = self._pending_pair["chosen"]
            rejected = self._pending_pair["rejected"]
            group = self._current_group

            export_single_pair(
                self._output_dir,
                self._manifest,
                chosen,
                rejected,
                group["prompt"],
                group["aspectratio"],
            )
            self._mark_pair_used(chosen, rejected)
            self._exported_in_group.append((chosen, rejected))

            self._remaining_images = [i for i in self._remaining_images if i not in {chosen, rejected}]
            self._pairs_created_in_group += 1
            self._pending_pair = None

            if discard_rest:
                discarded = self._retire_group_sources(list(self._remaining_images))
                pairs_done = self._pairs_created_in_group
                self._remaining_images = []
                self._release_group()
                return {
                    "ok": True,
                    "committed": True,
                    "continue_group": False,
                    "pairs_done": pairs_done,
                    "discarded": discarded,
                }

            is_unconditional = group["prompt"] == "UNCONDITIONAL"
            keep_going = continue_scoring or is_unconditional or self._pairs_created_in_group < self._pairs_per_group
            can_continue = len(self._remaining_images) >= 2

            if keep_going and can_continue:
                self._selection_phase = "best"
                self._selected_best = None
                return {
                    "ok": True,
                    "committed": True,
                    "continue_group": True,
                    "remaining_images": self._remaining_images,
                    "pairs_done": self._pairs_created_in_group,
                }
            return {
                "ok": True,
                "committed": True,
                "continue_group": False,
                "pairs_done": self._pairs_created_in_group,
            }

    def cancel_pending_pair(self) -> dict:
        """Discard the pending pair without writing it. _selection_phase stays
        'worst' and _selected_best is preserved, so the user returns to the
        worst-pick UI with the same best image and can try a different worst —
        same as Ctk's Cancel branch (messagebox.askyesnocancel result is None)."""
        with self._lock:
            if not self._pending_pair:
                return {"ok": False, "error": "No pending pair to cancel"}
            self._pending_pair = None
            remaining_for_worst = [i for i in self._remaining_images if i != self._selected_best]
            return {
                "ok": True,
                "phase": self._selection_phase,
                "best": self._selected_best,
                "remaining_images": remaining_for_worst,
            }

    def skip_group(self) -> dict:
        # Clear per-group selection state so a stale _pending_pair or
        # _selected_best can't leak into the next group if, for any reason,
        # fetch_next_group isn't the immediate next call.
        self._release_group()
        return {"ok": True}

    def triage_align(self, good: list[str], bad: list[str]) -> dict:
        """Chosen-anchored similarity pairing for a triage group: pair each good
        (chosen) image with its most visually-similar bad (rejected) image. Pure
        suggestion — writes nothing; the client applies the result and the user
        can still rearrange before committing. Validates both pools against the
        current group's images so an arbitrary path can't be embedded."""
        from modules.util.dpo_curation_util import align_pool_by_similarity

        with self._lock:
            if not self._current_group:
                return {"ok": False, "error": "No active group"}
            valid = set(self._current_group.get("images", []))

        for path in list(good) + list(bad):
            if path not in valid:
                return {"ok": False, "error": "Image is not part of the current group"}

        # Embedding (a model call) runs outside the lock so the UI stays responsive.
        try:
            result = align_pool_by_similarity(good, bad)
        except RuntimeError as ex:
            return {"ok": False, "error": str(ex)}

        return {
            "ok": True,
            "pairs": [{"chosen": c, "rejected": r} for c, r in result["pairs"]],
            "chosen_pool": result["chosen_pool"],
            "rejected_pool": result["rejected_pool"],
        }

    def _scoring_disabled(self) -> bool:
        """True when the user has turned image scoring off for this sitting.

        Every entry point that could reach the model consults this, so "off"
        means the weights are never loaded — not merely that the scores are
        hidden. That is the whole point: the setting exists for when the GPU is
        busy with something else.
        """
        from modules.util.realism_detect_util import REALISM_DEVICE_OFF

        return self._realism_device == REALISM_DEVICE_OFF

    def realism_scores(self, paths: list[str]) -> dict:
        """P(AI-generated) per path, for sorting a group most-realistic-first.

        Same shape as :meth:`triage_align`: paths are validated against the
        current group under the lock, then the model call runs outside it so the
        UI stays responsive. Reuses the session's scan cache, so a group costs
        the model once and is free on every revisit.

        A missing model is reported rather than raised — the tool stays usable
        without the weights, just without scores.
        """
        from modules.util.realism_detect_util import score_images_realism

        # Not an error: with scoring off the grid simply keeps source order, so
        # report "no scores" rather than a failure the UI would have to explain.
        if self._scoring_disabled():
            return {"ok": True, "scores": {}, "disabled": True}

        with self._lock:
            if not self._current_group:
                return {"ok": False, "error": "No active group"}
            valid = set(self._current_group.get("images", []))
            cache = self._hash_cache
            device = self._realism_device

        for path in paths:
            if path not in valid:
                return {"ok": False, "error": "Image is not part of the current group"}

        try:
            scores = score_images_realism(list(paths), cache=cache, device=device)
        except RuntimeError as ex:
            return {"ok": False, "error": str(ex)}

        return {"ok": True, "scores": scores}

    # ------------------------------------------------------------------ #
    # Realism backfill
    # ------------------------------------------------------------------ #

    #: Images per `score_images_realism` call. This is a correctness bound, not
    #: a tuning knob: that function preprocesses its ENTIRE argument list before
    #: running a batch, holding one ~3 MB tensor per image. Handing it a whole
    #: source tree asks for tens of gigabytes — 11.9k images reached a 39 GB
    #: working set on a 64 GB machine before being killed. Chunking caps peak
    #: memory at roughly `CHUNK x 3 MB`. Scores are identical either way: each
    #: TTA draw is keyed on the file path, so nothing depends on which images
    #: share a call.
    _BACKFILL_CHUNK = 200

    def start_realism_backfill(self) -> dict:
        """Score every image in the session's source tree, in the background.

        Selection sorts a group most-photorealistic-first, so the first view of
        each group otherwise pays for the model. Doing the whole tree once up
        front means every group is a cache hit from then on.

        Idempotent by construction: scores already cached under the current
        model version are served from the cache, so re-running after adding new
        images only pays for the new ones.
        """
        from modules.util.dpo_curation_util import walk_skipping_dotted
        from modules.util.path_util import supported_image_extensions

        if self._scoring_disabled():
            return {"ok": False, "error": "Image scoring is turned off"}

        with self._lock:
            if not self._session_active or not self._source_folder:
                return {"ok": False, "error": "No active session"}
            if self._backfill_thread is not None and self._backfill_thread.is_alive():
                return {"ok": False, "error": "A realism backfill is already running"}
            source = self._source_folder
            cache = self._hash_cache

        supported = supported_image_extensions()
        paths: list[str] = []
        for root, files in walk_skipping_dotted(source):
            paths.extend(
                os.path.join(root, filename)
                for filename in sorted(files)
                if os.path.splitext(filename)[1].lower() in supported
            )

        if not paths:
            return {"ok": False, "error": f"No images found under {source}"}

        self._backfill_stop.clear()
        self._backfill_state = {
            "running": True,
            "done": 0,
            "total": len(paths),
            "scored": 0,
            "cached": 0,
            "error": None,
            "finished": False,
        }
        self._backfill_thread = threading.Thread(target=self._realism_backfill_worker, args=(paths, cache), daemon=True)
        self._backfill_thread.start()
        return {"ok": True, "total": len(paths)}

    def _realism_backfill_worker(self, paths: list[str], cache) -> None:
        from modules.util.realism_detect_util import score_images_realism

        state = self._backfill_state
        try:
            for start in range(0, len(paths), self._BACKFILL_CHUNK):
                # Same chunk boundary the device switch is honoured at, for the
                # same reason: turning scoring off mid-run has to stop the work,
                # not just stop displaying it.
                if self._backfill_stop.is_set() or self._scoring_disabled():
                    break
                chunk = paths[start : start + self._BACKFILL_CHUNK]
                before_hits = cache.hits if cache is not None else 0
                # Read per chunk, not once up front: the session is created per
                # call, so flipping the device mid-run takes effect at the next
                # chunk boundary instead of needing the backfill restarted.
                score_images_realism(chunk, cache=cache, device=self._realism_device)
                if cache is not None:
                    state["cached"] += cache.hits - before_hits
                state["done"] = start + len(chunk)
                state["scored"] = state["done"] - state["cached"]
                # Persist as we go: a tree this size is minutes of work and a
                # late failure should not throw all of it away. `save()` is a
                # no-op when nothing changed.
                if cache is not None:
                    cache.save()
        except RuntimeError as ex:
            # The model is missing — reported, not raised, same as realism_scores.
            state["error"] = str(ex)
        except Exception as ex:  # noqa: BLE001 — a backfill must never kill the session
            state["error"] = f"{type(ex).__name__}: {ex}"
        finally:
            if cache is not None:
                cache.save()
            state["running"] = False
            state["finished"] = True

    def realism_backfill_status(self) -> dict:
        state = dict(self._backfill_state)
        state["ok"] = True
        state["device"] = self._realism_device
        return state

    def realism_device(self) -> dict:
        return {"ok": True, "device": self._realism_device}

    def set_realism_device(self, device: str) -> dict:
        """Pin the realism head to the CPU, turn scoring off, or let it choose.

        Costs nothing to change and needs no teardown: every scoring call builds
        its own ONNX session, so the next one — including the next backfill
        chunk — simply picks different providers.

        Switching to ``off`` also stops a backfill in flight. Leaving one
        running would contradict the setting outright, and the work it has
        already written to the cache is kept either way, so a later run resumes
        rather than starting over.
        """
        from modules.util.realism_detect_util import REALISM_DEVICE_OFF, REALISM_DEVICE_SETTINGS

        if device not in REALISM_DEVICE_SETTINGS:
            return {"ok": False, "error": f"Unknown device {device!r}", "device": self._realism_device}
        self._realism_device = device
        if device == REALISM_DEVICE_OFF:
            self._backfill_stop.set()
        return {"ok": True, "device": device}

    def stop_realism_backfill(self) -> dict:
        """Ask the backfill to stop after the current chunk. Work already
        written to the cache is kept, so a later run resumes from it."""
        self._backfill_stop.set()
        return {"ok": True}

    def commit_triage_pairs(self, pairs: list[tuple[str, str]], discard_rest: bool = False) -> dict:
        """Batch-commit a triage group's pairs, then release the group.

        Triage voting/pairing happens entirely client-side, so this is the
        only write call for the whole group. Every pair is validated against
        _remaining_images (the dedup-filtered whitelist served by next_group)
        before anything is exported — a malformed batch writes nothing. An
        empty list just releases the group, equivalent to skip_group.

        When ``discard_rest`` is set, the whole group is retired from the source
        tree after the pairs are exported: each paired source image is moved into
        a dot-prefixed ``.chosen``/``.rejected`` subfolder and every unpaired
        leftover (unscored, skipped, or surplus) into ``.discard``. Dot-prefixed
        folders are pruned by the scanner, so nothing in the group can reappear
        in a later session.
        """
        from modules.util.dpo_curation_util import export_single_pair

        with self._lock:
            if not self._current_group:
                return {"ok": False, "error": "No active group"}
            group = self._current_group

            valid = set(self._remaining_images)
            seen: set[str] = set()
            for chosen, rejected in pairs:
                if chosen == rejected:
                    return {"ok": False, "error": "Chosen and rejected images must differ"}
                if chosen not in valid or rejected not in valid:
                    return {"ok": False, "error": "Pair references an image not in the current group"}
                if chosen in seen or rejected in seen:
                    return {"ok": False, "error": "An image is used in more than one pair"}
                seen.add(chosen)
                seen.add(rejected)

            for chosen, rejected in pairs:
                export_single_pair(
                    self._output_dir,
                    self._manifest,
                    chosen,
                    rejected,
                    group["prompt"],
                    group["aspectratio"],
                )
                self._mark_pair_used(chosen, rejected)
                self._exported_in_group.append((chosen, rejected))
                self._pairs_created_in_group += 1

            rest = [i for i in self._remaining_images if i not in seen]

            discarded = self._retire_group_sources(rest) if discard_rest else 0

            self._remaining_images = [] if discard_rest else rest
            pairs_done = self._pairs_created_in_group
            # Release the group like skip_group so fetch_next_group advances.
            self._release_group()
            return {"ok": True, "pairs_done": pairs_done, "discarded": discarded}

    def discard_group(self) -> dict:
        """Retire the current group without saving any pair: move every remaining
        source image into a dot-prefixed ``.discard`` subfolder and release the
        group. Unlike :meth:`skip_group`, the images are physically removed from
        the source tree (dot folders are pruned by the scanner), so a discarded
        group never reappears in a later session. Mode-agnostic — the header
        button sits next to Skip Group in every curation mode. Pairs already
        exported from the group (selection/tournament modes commit as they go)
        are not unwritten — their sources move to ``.chosen``/``.rejected``
        rather than ``.discard`` so the whole group leaves the scan tree."""
        with self._lock:
            if not self._current_group:
                return {"ok": False, "error": "No active group"}
            discarded = self._retire_group_sources(list(self._remaining_images))
            self._remaining_images = []
            self._release_group()
            return {"ok": True, "discarded": discarded}

    def finalize_session(self, val_percentage: float = 0.0) -> dict:
        from modules.util.dpo_curation_util import finalize_export

        if not self._output_dir:
            return {"ok": False, "error": "No session active"}
        if self._pending_pair is not None:
            # Refuse to finalize while a pair is waiting on the user —
            # otherwise that selection would be silently dropped.
            return {"ok": False, "error": "Confirm or cancel the pending pair first"}

        train_count, val_count = finalize_export(self._output_dir, self._manifest, val_percentage=val_percentage)
        total = len(self._manifest.get("pairs", []))

        self._session_active = False
        self._worker_stop.set()
        if self._hash_cache is not None:
            self._hash_cache.save()

        return {
            "ok": True,
            "total_pairs": total,
            "train_count": train_count,
            "val_count": val_count,
        }

    def cancel_session(self) -> dict:
        self._worker_stop.set()
        self._session_active = False
        self._current_group = None
        self._pending_pair = None
        self._exported_in_group = []
        self._swiss = None
        self._ranked_order = None
        # Keep the hash work done so far — a restarted session over the same
        # output dir resumes from the cache instead of re-decoding everything.
        if self._hash_cache is not None:
            self._hash_cache.save()
        return {"ok": True}

    def serve_image(self, path: str) -> str | None:
        """Return the absolute path if the file exists and is within the source/output dirs."""
        if not os.path.isfile(path):
            return None
        return os.path.abspath(path)

    # ---- Swiss tournament mode ----

    def _swiss_guard(self) -> dict | None:
        if not self._current_group or self._mode != "swiss" or self._swiss is None:
            return {"ok": False, "error": "No active tournament group"}
        return None

    def _swiss_scores(self) -> dict[str, dict[str, float]]:
        return {
            image: {"score": self._swiss.player(image).score, "elo": round(self._swiss.player(image).elo, 1)}
            for image in self._swiss.standings()
        }

    def swiss_state(self) -> dict:
        """Current match plus round/progress, or finished=True after the last vote."""
        if error := self._swiss_guard():
            return error
        match = self._swiss.next_match()
        return {
            "ok": True,
            "finished": match is None,
            "match": list(match) if match else None,
            "round": self._swiss.current_round,
            "total_rounds": self._swiss.total_rounds,
            "matches_played": self._swiss.matches_played(),
            "matches_total": self._swiss.matches_total(),
            "scores": self._swiss_scores(),
        }

    def swiss_vote(self, a: str, b: str, winner: str) -> dict:
        """Resolve the current match. winner: 'a' | 'b' | 'tie' (relative to a/b as sent)."""
        if error := self._swiss_guard():
            return error
        if winner not in ("a", "b", "tie"):
            return {"ok": False, "error": "winner must be 'a', 'b', or 'tie'"}
        current = self._swiss.next_match()
        if current is None:
            return {"ok": False, "error": "Tournament already finished"}
        if {a, b} != {current[0], current[1]}:
            return {"ok": False, "error": "Vote does not match the current match"}
        # report() expects the exact (a, b) order of next_match; translate the
        # winner if the client sent the pair flipped.
        if (a, b) != current and winner in ("a", "b"):
            winner = "b" if winner == "a" else "a"
        self._swiss.report(current[0], current[1], winner)
        return self.swiss_state()

    def swiss_finish_early(self) -> dict:
        """Jump straight to the ranked review with the votes cast so far."""
        if error := self._swiss_guard():
            return error
        return self.swiss_ranking()

    def swiss_ranking(self) -> dict:
        """Best-to-worst standings for the review step (computed once per group)."""
        if error := self._swiss_guard():
            return error
        if self._ranked_order is None:
            self._ranked_order = self._swiss.standings()
        max_pairs = len(self._ranked_order) // 2
        is_unconditional = self._current_group["prompt"] == "UNCONDITIONAL"
        remaining_target = (
            max_pairs if is_unconditional else max(1, self._pairs_per_group - self._pairs_created_in_group)
        )
        return {
            "ok": True,
            "order": list(self._ranked_order),
            "scores": self._swiss_scores(),
            "max_pairs": max_pairs,
            "default_pairs": min(remaining_target, max_pairs),
        }

    def swiss_reorder(self, order: list[str]) -> dict:
        """Persist a manual re-ranking from the review grid (same image multiset)."""
        if error := self._swiss_guard():
            return error
        if self._ranked_order is None:
            return {"ok": False, "error": "Ranking not computed yet"}
        if sorted(order) != sorted(self._ranked_order):
            return {"ok": False, "error": "Reorder must contain exactly the ranked images"}
        self._ranked_order = list(order)
        return {"ok": True}

    def swiss_export(self, pair_count: int, discard_rest: bool = False) -> dict:
        """Export the outermost pairs of the (possibly reordered) ranking, then
        release the group. pair_count=0 just skips the group.

        With ``discard_rest`` the group is also retired from the source tree:
        exported sources move into ``.chosen``/``.rejected`` and every
        unexported image into ``.discard``, so the mid-ranked leftovers this
        mode never uses can't come back in a later session."""
        from modules.util.dpo_curation_util import export_single_pair

        if error := self._swiss_guard():
            return error
        if self._ranked_order is None:
            return {"ok": False, "error": "Ranking not computed yet"}

        with self._lock:
            group = self._current_group
            pairs = outermost_pairs(self._ranked_order, max(0, int(pair_count)))

            for chosen, rejected in pairs:
                export_single_pair(
                    self._output_dir,
                    self._manifest,
                    chosen,
                    rejected,
                    group["prompt"],
                    group["aspectratio"],
                )
                self._mark_pair_used(chosen, rejected)
                self._exported_in_group.append((chosen, rejected))
                self._pairs_created_in_group += 1

            exported = {img for pair in pairs for img in pair}
            rest = [i for i in self._remaining_images if i not in exported]

            discarded = self._retire_group_sources(rest) if discard_rest else 0

            self._remaining_images = [] if discard_rest else rest
            pairs_done = self._pairs_created_in_group
            # Release the group like skip_group so fetch_next_group advances.
            self._release_group()
            return {"ok": True, "exported": len(pairs), "pairs_done": pairs_done, "discarded": discarded}

    # ---- Background worker ----

    @staticmethod
    def _metadata_worker_count() -> int:
        # Metadata extraction is I/O-bound (file open + small read + parse), so
        # oversubscribing the CPU pays off — but cap it so we don't thrash on
        # HDDs or hit Windows' per-process thread limits.
        return min(16, (os.cpu_count() or 4) * 2)

    @staticmethod
    def _hash_worker_count() -> int:
        # Pixel hashing is CPU-bound (full image decode + BLAKE3); both PIL and
        # blake3 release the GIL, so threads scale near-linearly up to the core
        # count. Cap below it to keep the machine responsive while curating.
        return max(2, min(8, os.cpu_count() or 4))

    def _is_pixel_hash_used(self, path: str) -> bool:
        if self._hash_cache is None or not self._used_pixel_hashes:
            return False
        digest = self._hash_cache.get_pixel_hash(path)
        return digest is not None and digest in self._used_pixel_hashes

    def _mark_pair_used(self, chosen: str, rejected: str) -> None:
        """Record the pixel hashes of a just-committed pair so byte/pixel-identical
        copies elsewhere in the source tree can't be presented again — covers
        duplicates living in other groups (same pixels, different metadata
        prompt) that the path-based filter misses. Cache hits make this free:
        both images were hashed during the scan's dedup pass."""
        if self._hash_cache is None:
            return
        for path in (chosen, rejected):
            digest = self._hash_cache.get_pixel_hash(path)
            if digest is not None:
                self._used_pixel_hashes.add(digest)

    def _exported_image_paths(self) -> list[str]:
        """All image files already exported under the output dir's chosen/ and
        rejected/ trees (including train/val splits after finalize)."""
        from modules.util import path_util

        supported = path_util.supported_image_extensions()
        paths: list[str] = []
        for subdir in ("chosen", "rejected"):
            base = os.path.join(self._output_dir, subdir)
            if not os.path.isdir(base):
                continue
            for root, _dirs, files in os.walk(base):
                paths.extend(
                    os.path.join(root, fname) for fname in files if os.path.splitext(fname)[1].lower() in supported
                )
        return paths

    def _foreign_concept_usage(self) -> tuple[set[str], list[str]]:
        """Source paths and exported image files of every configured DPO
        concept other than this session's output dir. Read errors (a concept
        with a bad pattern, an unreadable manifest) degrade to nothing foreign
        rather than failing the scan."""
        from modules.util.dpo_curation_util import concept_usage_for_exclusion
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        try:
            sources, _fingerprints, paths = concept_usage_for_exclusion(
                dpo_concept_pattern_dirs(self._resolve_concepts()), exclude_dir=self._output_dir
            )
            return sources, paths
        except Exception as ex:
            logger.warning("could not read configured DPO concepts for image exclusion: %s", ex)
            return set(), []

    def _hash_paths(self, pool, paths: list[str]) -> dict[str, str]:
        """Hash ``paths`` through the persistent cache on ``pool``. Returns
        ``{path: hash}`` for files that could be hashed; returns early
        (partial) when the worker is stopped mid-flight."""
        from concurrent.futures import as_completed

        result: dict[str, str] = {}
        futures = {pool.submit(self._hash_cache.get_pixel_hash, p): p for p in paths}
        for future in as_completed(futures):
            if self._worker_stop.is_set():
                for f in futures:
                    f.cancel()
                break
            path = futures[future]
            try:
                digest = future.result()
            except Exception:
                digest = None
            self._hash_count += 1
            if digest is not None:
                result[path] = digest
        return result

    def _background_scan(self) -> None:
        try:
            self._background_scan_inner()
        finally:
            # Persist hash work even on cancel/error so the next session over
            # the same output dir skips straight past everything already done.
            if self._hash_cache is not None:
                self._hash_cache.save()

    def _background_scan_inner(self) -> None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from modules.util import path_util
        from modules.util.dpo_curation_util import (
            group_matches_keywords,
            is_source_used,
            manifest_pair_counts,
            manifest_used_sources,
            normalize_prompt_for_grouping,
            resolve_aspect_ratio,
            walk_skipping_dotted,
        )
        from modules.util.image_metadata_util import extract_metadata

        supported = path_util.supported_image_extensions()

        # Pass 1: enumerate candidate paths up front. os.walk uses scandir under
        # the hood so this is cheap even on large trees, and collecting paths
        # first lets us fan the expensive per-file metadata reads out across a
        # thread pool (mirrors DPOCurationWindow._background_scan_and_dedup).
        # Dot-prefixed subdirectories (.thumbnails, .cache, ...) are pruned.
        candidate_paths: list[str] = []
        for root, files in walk_skipping_dotted(self._source_folder):
            if self._worker_stop.is_set():
                return
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext in supported:
                    candidate_paths.append(os.path.join(root, filename))
        self._scan_total = len(candidate_paths)

        def extract_group_key(path: str) -> tuple[str, str]:
            meta = extract_metadata(path)
            # The aspect component is the trainer bucket label ("7:4", "4:7",
            # ...) derived from actual pixel dimensions, so images that crop
            # to the same AspectBucketing bucket group together even when
            # their exact ratios differ (1344x768 vs 1680x960, metadata
            # "16:9" vs derived). Metadata is only a fallback for undecodable
            # files.
            ar = resolve_aspect_ratio(meta.get("aspectratio", ""), path)
            prompt = normalize_prompt_for_grouping(meta.get("prompt", ""))
            return prompt, ar

        # Pass 2: parallel metadata extraction, served from the persistent
        # cache for unchanged files — extraction reads (and for unmarked files,
        # fully scans) each file, so on a rescan this pass collapses to one
        # os.stat per file. The grouping dict is mutated only on the consumer
        # side of `as_completed`, so no lock is needed.
        groups_dict: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
        if candidate_paths:
            with ThreadPoolExecutor(max_workers=self._metadata_worker_count()) as pool:
                future_to_path = {
                    pool.submit(self._hash_cache.get_group_key, p, extract_group_key): p for p in candidate_paths
                }
                for future in as_completed(future_to_path):
                    if self._worker_stop.is_set():
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
                    try:
                        group_key = future.result()
                    except Exception:
                        group_key = None
                    self._scan_count += 1
                    if group_key is not None:
                        groups_dict[group_key].append(future_to_path[future])

        # as_completed scrambles arrival order; re-sort so group contents (and
        # therefore dedup keep-order and UI display) stay deterministic.
        raw_groups = [
            {"prompt": prompt, "aspectratio": ar, "images": sorted(images)}
            for (prompt, ar), images in groups_dict.items()
            if len(images) >= 2
        ]
        # A caption filter narrows the session to groups whose prompt carries
        # every keyword; it runs before the ordering so the order applies to
        # the filtered list rather than to everything the scan found.
        if self._caption_keywords:
            keywords = self._caption_keywords
            raw_groups = [g for g in raw_groups if group_matches_keywords(g, keywords)]
        # Shuffle first so ties and unscored groups are not presented in
        # filesystem order, then rank: the sort is stable, so the shuffle
        # survives wherever the scores cannot separate two groups (and is the
        # whole order when the session asked for "random").
        random.shuffle(raw_groups)
        raw_groups = self._order_by_realism(raw_groups)

        existing_counts = manifest_pair_counts(self._manifest)
        foreign_sources, foreign_paths = self._foreign_concept_usage()
        self._foreign_used_sources = foreign_sources
        used_sources = manifest_used_sources(self._manifest) | foreign_sources

        with ThreadPoolExecutor(max_workers=self._hash_worker_count()) as hash_pool:
            # Fingerprint everything already exported — here and in every other
            # configured DPO concept — so a byte/pixel-identical copy of a used
            # image can never be presented again, no matter how it was renamed
            # or where it lives in the source tree. The path-based used_sources
            # filter alone misses those.
            exported = self._hash_paths(hash_pool, self._exported_image_paths() + foreign_paths)
            self._used_pixel_hashes.update(exported.values())
            if self._worker_stop.is_set():
                return

            for group in raw_groups:
                if self._worker_stop.is_set():
                    return

                group_key = (group["prompt"], group["aspectratio"])
                is_unconditional = group["prompt"] == "UNCONDITIONAL"
                if not is_unconditional and existing_counts.get(group_key, 0) >= self._pairs_per_group:
                    continue

                # Drop images already committed in any prior pair before dedup so
                # the content-hash pass doesn't waste work on sources we'll discard.
                fresh = [i for i in group["images"] if not is_source_used(used_sources, i)]
                if len(fresh) < 2:
                    continue

                deduped = self._dedup_by_content_hash(fresh, hash_pool)
                if len(deduped) >= 2:
                    group["images"] = deduped
                    self._groups_queued += 1
                    while not self._worker_stop.is_set():
                        try:
                            self._ready_queue.put(group, timeout=0.5)
                            break
                        except Full:
                            continue

        self._worker_finished = True

    def _order_by_realism(self, groups: list[dict]) -> list[dict]:
        """Rank groups by realism in the session's ``group_order`` (most
        photorealistic first by default), from cached scores only.

        Reads the cache and never the model: the scan is CPU work that must not
        take VRAM from the generator, and a score that is not cached is simply
        treated as unknown. Run the backfill and every group is ranked; skip it
        and the order degrades to the shuffle it always was.
        """
        from modules.util.dpo_curation_util import GROUP_ORDER_RANDOM, order_groups_by_realism
        from modules.util.realism_detect_util import realism_model_version

        # Random is the shuffle the caller already applied; nothing to rank.
        if self._group_order == GROUP_ORDER_RANDOM:
            return groups
        # Off means off, cache or no cache. Reading cached scores would cost
        # nothing, but a group order the user cannot see the reason for — no
        # badges, no model — is worse than the shuffle they asked for.
        if self._scoring_disabled():
            return groups

        cache = self._hash_cache
        if cache is None:
            return groups
        version = realism_model_version()
        if version is None:
            return groups
        return order_groups_by_realism(groups, lambda path: cache.peek_realism(path, version), self._group_order)

    def _dedup_by_content_hash(self, images: list[str], hash_pool) -> list[str]:
        """Drop pixel-identical files within a group, keeping the latest
        mtime copy, and drop any image whose pixel content already exists in
        the output dataset (``_used_pixel_hashes``). DPO is *meant* to
        discriminate between near-duplicates from different seeds, so the hash
        covers the decoded pixel buffer rather than the raw file: PNG text
        chunks, EXIF, ICC profiles, and re-encoded container metadata don't
        count as a difference, but a single different pixel does (see
        ``compute_pixel_hash``). Hashes come from the persistent per-output-dir
        cache, so only new/changed files pay the decode cost; files that can't
        be hashed at all (unreadable) are kept rather than silently dropped."""
        hashes = self._hash_paths(hash_pool, images)
        seen: dict[str, int] = {}
        unique: list[str] = []
        for path in images:
            h = hashes.get(path)
            if h is None:
                unique.append(path)
                continue
            if h in self._used_pixel_hashes:
                continue
            if h not in seen:
                seen[h] = len(unique)
                unique.append(path)
            else:
                existing_idx = seen[h]
                try:
                    if os.path.getmtime(path) > os.path.getmtime(unique[existing_idx]):
                        unique[existing_idx] = path
                except OSError:
                    continue
        return unique
