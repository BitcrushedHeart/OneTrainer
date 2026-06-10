import logging
import math
import os
import random
import threading
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from queue import Empty, Full, Queue

from web.backend.services._singleton import SingletonMixin
from web.backend.services.config_service import ConfigService

logger = logging.getLogger(__name__)


class DPOService(SingletonMixin):
    _ELO_K = 32.0
    _ELO_BASE = 1500.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ws_broadcast: Callable[[dict], None] | None = None

        # Curation session state
        self._session_active = False
        self._source_folder: str | None = None
        self._output_dir: str | None = None
        self._manifest: dict = {"pairs": []}
        self._pairs_per_group = 1
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
        self._groups_queued = 0
        self._groups_shown = 0
        self._current_group: dict | None = None
        self._remaining_images: list[str] = []
        self._mode: str = "selection"  # "selection" | "elo"
        self._selection_phase = "best"  # "best" or "worst"
        self._selected_best: str | None = None
        self._pairs_created_in_group = 0
        # Held when the user has picked both best and worst but has not yet
        # confirmed the pair. No file is written and no counters advance
        # until confirm_pair commits or cancel_pending_pair discards it —
        # matches the Ctk yes/no/cancel dialog semantics.
        self._pending_pair: dict | None = None

        # ELO mode state (per-group)
        self._elo_ratings: dict[str, float] = {}
        self._elo_done: int = 0
        self._elo_pair: tuple[str, str] | None = None

    def set_ws_broadcast(self, fn: Callable[[dict], None]) -> None:
        self._ws_broadcast = fn

    def _broadcast(self, message: dict) -> None:
        if self._ws_broadcast is not None:
            with suppress(Exception):
                self._ws_broadcast(message)

    def check_pairs(self) -> dict:
        from modules.util.dpo_curation_util import check_dpo_pairs
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []

        concept_pairs = dpo_concept_pattern_dirs(concepts)
        if not concept_pairs:
            return {"ok": False, "error": "No DPO concepts found. Set the chosen/rejected patterns on a concept."}

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

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []
        exts = supported_image_extensions()

        concept_pairs = dpo_concept_pattern_dirs(concepts)
        if not concept_pairs:
            return {"ok": False, "error": "No DPO concepts found. Set the chosen/rejected patterns on a concept."}

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

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []

        all_pairs = dpo_concept_pattern_dirs(concepts)
        if not all_pairs:
            return {"ok": False, "error": "No DPO concept pairs found"}

        pairs = scan_finalized_pairs(all_pairs)
        return {"ok": True, "pairs": pairs}

    def remove_pair(self, chosen_path: str | None, rejected_path: str | None) -> dict:
        from modules.util.dpo_curation_util import remove_finalized_pair

        remove_finalized_pair(chosen_path, rejected_path)
        return {"ok": True}

    def fix_multiline_captions(self) -> dict:
        from modules.util.dpo_curation_util import fix_multiline_captions
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []

        all_pairs = dpo_concept_pattern_dirs(concepts)
        fixed = fix_multiline_captions(all_pairs)
        return {"ok": True, "fixed": fixed}

    def _all_concept_pairs(self) -> list[tuple[str, str]]:
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []
        return dpo_concept_pattern_dirs(concepts)

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
    ) -> dict:
        from modules.util.dpo_curation_util import DpoScanCache, load_manifest, prune_orphaned_pairs

        if mode not in ("selection", "elo", "triage"):
            return {"ok": False, "error": "mode must be 'selection', 'elo' or 'triage'"}

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
        self._manifest = load_manifest(output_dir)
        pruned = prune_orphaned_pairs(output_dir, self._manifest)
        existing = len(self._manifest.get("pairs", []))

        self._hash_cache = DpoScanCache(os.path.join(output_dir, ".dpo_hash_cache.json"))
        self._used_pixel_hashes = set()

        self._scan_count = 0
        self._scan_total = 0
        self._hash_count = 0
        self._groups_queued = 0
        self._groups_shown = 0
        self._worker_finished = False
        self._current_group = None
        self._remaining_images = []
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
            used_sources = manifest_used_sources(self._manifest)
            available_images = [
                i for i in group["images"] if not is_source_used(used_sources, i) and not self._is_pixel_hash_used(i)
            ]
            if len(available_images) < 2:
                continue

            self._current_group = group
            self._groups_shown += 1
            self._pairs_created_in_group = pairs_done
            self._remaining_images = available_images
            self._selection_phase = "best"
            self._selected_best = None
            self._pending_pair = None

            if self._mode == "elo":
                self._elo_init(self._remaining_images)
                self._elo_next_pair()

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

    def confirm_pair(self, continue_scoring: bool) -> dict:
        """Commit the pending pair picked via select_image and decide whether to
        stay in the current group or advance. Mirrors Ctk Yes (keep scoring) /
        No (next group) from the askyesnocancel dialog."""
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

            self._remaining_images = [i for i in self._remaining_images if i not in {chosen, rejected}]
            self._pairs_created_in_group += 1
            self._pending_pair = None

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
        self._current_group = None
        self._pending_pair = None
        self._selected_best = None
        self._selection_phase = "best"
        return {"ok": True}

    def commit_triage_pairs(self, pairs: list[tuple[str, str]]) -> dict:
        """Batch-commit a triage group's pairs, then release the group.

        Triage voting/pairing happens entirely client-side, so this is the
        only write call for the whole group. Every pair is validated against
        _remaining_images (the dedup-filtered whitelist served by next_group)
        before anything is exported — a malformed batch writes nothing. An
        empty list just releases the group, equivalent to skip_group.
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
                self._pairs_created_in_group += 1

            self._remaining_images = [i for i in self._remaining_images if i not in seen]
            # Release the group like skip_group so fetch_next_group advances.
            self._current_group = None
            self._pending_pair = None
            self._selected_best = None
            self._selection_phase = "best"
            return {"ok": True, "pairs_done": self._pairs_created_in_group}

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

    # ---- ELO mode ----

    def _elo_init(self, images: list[str]) -> None:
        """Initialize ELO ratings for a group of images."""
        self._elo_ratings = dict.fromkeys(images, self._ELO_BASE)
        self._elo_done = 0
        self._elo_pair = None

    def _elo_suggested_count(self, n: int | None = None) -> int:
        """Suggested number of comparisons: max(15, ceil(n * log2(n)))."""
        if n is None:
            n = len(self._elo_ratings)
        return max(15, math.ceil(n * math.log2(max(n, 2))))

    def _elo_next_pair(self) -> tuple[str, str] | None:
        """Pick two consecutive images from the rating-sorted list."""
        if len(self._elo_ratings) < 2:
            self._elo_pair = None
            return None
        sorted_imgs = sorted(self._elo_ratings, key=lambda x: self._elo_ratings[x])
        idx = random.randint(0, len(sorted_imgs) - 2)
        self._elo_pair = (sorted_imgs[idx], sorted_imgs[idx + 1])
        return self._elo_pair

    def _elo_vote(self, a: str, b: str, winner: str) -> None:
        """Apply an ELO update for a single pairwise vote.

        winner: "a" | "b" | "tie"
        """
        if a not in self._elo_ratings or b not in self._elo_ratings:
            raise ValueError("Both images must be initialised in ELO ratings")
        ra = self._elo_ratings[a]
        rb = self._elo_ratings[b]

        if winner == "a":
            sa, sb = 1.0, 0.0
        elif winner == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea
        self._elo_ratings[a] = ra + self._ELO_K * (sa - ea)
        self._elo_ratings[b] = rb + self._ELO_K * (sb - eb)
        self._elo_done += 1

    def _elo_finish_round(self) -> tuple[str, str] | None:
        """Return (best, worst) by final ratings; None if fewer than 2 images."""
        if len(self._elo_ratings) < 2:
            return None
        sorted_imgs = sorted(self._elo_ratings, key=lambda x: self._elo_ratings[x], reverse=True)
        return sorted_imgs[0], sorted_imgs[-1]

    def elo_current_pair(self) -> dict:
        """Return the current ELO pair plus progress, or {finished: True} when done."""
        if not self._current_group or self._mode != "elo":
            return {"ok": False, "error": "No active ELO group"}

        # Lazy init if no pair yet
        if self._elo_pair is None and self._elo_ratings:
            self._elo_next_pair()

        suggested = self._elo_suggested_count()
        if self._elo_pair is None:
            return {
                "ok": True,
                "finished": True,
                "done": self._elo_done,
                "suggested": suggested,
                "ratings": dict(self._elo_ratings),
            }
        a, b = self._elo_pair
        return {
            "ok": True,
            "finished": False,
            "pair": [a, b],
            "ratings": {a: self._elo_ratings[a], b: self._elo_ratings[b]},
            "done": self._elo_done,
            "suggested": suggested,
        }

    def elo_vote(self, a: str, b: str, winner: str) -> dict:
        """Public ELO vote endpoint. Picks the next pair and returns updated state."""
        if not self._current_group or self._mode != "elo":
            return {"ok": False, "error": "No active ELO group"}
        if winner not in ("a", "b", "tie"):
            return {"ok": False, "error": "winner must be 'a', 'b', or 'tie'"}
        try:
            self._elo_vote(a, b, winner)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        self._elo_next_pair()
        return self.elo_current_pair()

    def elo_accept_pair(self, continue_scoring: bool = False) -> dict:
        """Finalize the current ELO round into a chosen/rejected pair on disk."""
        from modules.util.dpo_curation_util import export_single_pair

        if not self._current_group or self._mode != "elo":
            return {"ok": False, "error": "No active ELO group"}
        result = self._elo_finish_round()
        if result is None:
            return {"ok": False, "error": "Need at least 2 images to accept pair"}
        chosen, rejected = result
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

        self._remaining_images = [i for i in self._remaining_images if i not in {chosen, rejected}]
        self._pairs_created_in_group += 1

        is_unconditional = group["prompt"] == "UNCONDITIONAL"
        keep_going = continue_scoring or is_unconditional or self._pairs_created_in_group < self._pairs_per_group
        can_continue = len(self._remaining_images) >= 2

        if keep_going and can_continue:
            self._elo_init(self._remaining_images)
            self._elo_next_pair()
            return {
                "ok": True,
                "pair_created": True,
                "chosen": chosen,
                "rejected": rejected,
                "continue_group": True,
                "pairs_done": self._pairs_created_in_group,
            }
        return {
            "ok": True,
            "pair_created": True,
            "chosen": chosen,
            "rejected": rejected,
            "continue_group": False,
            "pairs_done": self._pairs_created_in_group,
        }

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
        random.shuffle(raw_groups)

        existing_counts = manifest_pair_counts(self._manifest)
        used_sources = manifest_used_sources(self._manifest)

        with ThreadPoolExecutor(max_workers=self._hash_worker_count()) as hash_pool:
            # Fingerprint everything already exported so a byte/pixel-identical
            # copy of a used image can never be presented again, no matter how
            # it was renamed or where it lives in the source tree. The
            # path-based used_sources filter alone misses those.
            exported = self._hash_paths(hash_pool, self._exported_image_paths())
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
