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
        self._groups_queued = 0
        self._groups_shown = 0
        self._current_group: dict | None = None
        self._remaining_images: list[str] = []
        self._mode: str = "selection"  # "selection" | "elo"
        self._selection_phase = "best"  # "best" or "worst"
        self._selected_best: str | None = None
        self._pairs_created_in_group = 0

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
        from modules.util.dpo_curation_util import check_dpo_pairs, dpo_concept_pairs

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []

        try:
            concept_pairs = dpo_concept_pairs(concepts, is_validation=False)
        except RuntimeError:
            try:
                concept_pairs = dpo_concept_pairs(concepts, is_validation=True)
            except RuntimeError as e:
                return {"ok": False, "error": str(e)}

        result = check_dpo_pairs(concept_pairs)
        return {"ok": True, "result": result}

    def remove_strays(self) -> dict:
        import os

        from modules.util.dpo_curation_util import (
            check_dpo_pairs,
            dpo_concept_pairs,
            dpo_pair_key,
            remove_finalized_pair,
        )
        from modules.util.path_util import supported_image_extensions

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []
        exts = supported_image_extensions()

        try:
            concept_pairs = dpo_concept_pairs(concepts, is_validation=False)
        except RuntimeError:
            concept_pairs = dpo_concept_pairs(concepts, is_validation=True)

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
        import contextlib

        from modules.util.dpo_curation_util import dpo_concept_pairs, scan_finalized_pairs

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []

        all_pairs: list[tuple[str, str]] = []
        with contextlib.suppress(RuntimeError):
            all_pairs.extend(dpo_concept_pairs(concepts, is_validation=False))
        with contextlib.suppress(RuntimeError):
            all_pairs.extend(dpo_concept_pairs(concepts, is_validation=True))

        if not all_pairs:
            return {"ok": False, "error": "No DPO concept pairs found"}

        pairs = scan_finalized_pairs(all_pairs)
        return {"ok": True, "pairs": pairs}

    def remove_pair(self, chosen_path: str | None, rejected_path: str | None) -> dict:
        from modules.util.dpo_curation_util import remove_finalized_pair
        remove_finalized_pair(chosen_path, rejected_path)
        return {"ok": True}

    def fix_multiline_captions(self) -> dict:
        import contextlib

        from modules.util.dpo_curation_util import dpo_concept_pairs, fix_multiline_captions

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []

        all_pairs: list[tuple[str, str]] = []
        with contextlib.suppress(RuntimeError):
            all_pairs.extend(dpo_concept_pairs(concepts, is_validation=False))
        with contextlib.suppress(RuntimeError):
            all_pairs.extend(dpo_concept_pairs(concepts, is_validation=True))

        fixed = fix_multiline_captions(all_pairs)
        return {"ok": True, "fixed": fixed}

    # ---- Curation Session ----

    def start_session(
        self,
        source_folder: str,
        output_dir: str,
        pairs_per_group: int = 1,
        mode: str = "selection",
    ) -> dict:
        from modules.util.dpo_curation_util import load_manifest, prune_orphaned_pairs

        if mode not in ("selection", "elo"):
            return {"ok": False, "error": "mode must be 'selection' or 'elo'"}

        with self._lock:
            if self._session_active:
                return {"ok": False, "error": "Session already active"}
            self._session_active = True

        self._source_folder = source_folder
        self._output_dir = output_dir
        self._pairs_per_group = max(1, pairs_per_group)
        self._mode = mode
        self._manifest = load_manifest(output_dir)
        pruned = prune_orphaned_pairs(output_dir, self._manifest)
        existing = len(self._manifest.get("pairs", []))

        self._scan_count = 0
        self._groups_queued = 0
        self._groups_shown = 0
        self._worker_finished = False
        self._current_group = None
        self._remaining_images = []
        self._ready_queue = Queue(maxsize=10)
        self._worker_stop = threading.Event()

        self._worker_thread = threading.Thread(
            target=self._background_scan, daemon=True
        )
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
            "groups_queued": self._groups_queued,
            "groups_shown": self._groups_shown,
            "worker_finished": self._worker_finished,
            "has_group": self._current_group is not None,
            "total_pairs": len(self._manifest.get("pairs", [])),
        }

    def next_group(self) -> dict:
        """Advance to the next group. Returns the group data with image paths."""
        from modules.util.dpo_curation_util import manifest_pair_counts

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

            self._current_group = group
            self._groups_shown += 1
            self._pairs_created_in_group = pairs_done
            self._remaining_images = list(group["images"])
            self._selection_phase = "best"
            self._selected_best = None

            if self._mode == "elo":
                self._elo_init(self._remaining_images)
                self._elo_next_pair()

            return {
                "group": {
                    "prompt": group["prompt"],
                    "aspectratio": group["aspectratio"],
                    "images": group["images"],
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
            group = self._current_group

            export_single_pair(
                self._output_dir, self._manifest,
                chosen, rejected,
                group["prompt"], group["aspectratio"],
            )

            self._remaining_images = [
                i for i in self._remaining_images if i not in {chosen, rejected}
            ]
            self._pairs_created_in_group += 1

            is_unconditional = group["prompt"] == "UNCONDITIONAL"
            keep_going = (
                is_unconditional or self._pairs_created_in_group < self._pairs_per_group
            )
            can_continue = len(self._remaining_images) >= 2

            if keep_going and can_continue:
                self._selection_phase = "best"
                self._selected_best = None
                return {
                    "ok": True,
                    "pair_created": True,
                    "chosen": chosen,
                    "rejected": rejected,
                    "continue_group": True,
                    "remaining_images": self._remaining_images,
                    "pairs_done": self._pairs_created_in_group,
                }
            else:
                return {
                    "ok": True,
                    "pair_created": True,
                    "chosen": chosen,
                    "rejected": rejected,
                    "continue_group": False,
                    "pairs_done": self._pairs_created_in_group,
                }

    def skip_group(self) -> dict:
        self._current_group = None
        return {"ok": True}

    def finalize_session(self, val_percentage: float = 0.0) -> dict:
        from modules.util.dpo_curation_util import finalize_export

        if not self._output_dir:
            return {"ok": False, "error": "No session active"}

        train_count, val_count = finalize_export(
            self._output_dir, self._manifest, val_percentage=val_percentage
        )
        total = len(self._manifest.get("pairs", []))

        self._session_active = False
        self._worker_stop.set()

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
        sorted_imgs = sorted(
            self._elo_ratings, key=lambda x: self._elo_ratings[x]
        )
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
        sorted_imgs = sorted(
            self._elo_ratings, key=lambda x: self._elo_ratings[x], reverse=True
        )
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
            self._output_dir, self._manifest,
            chosen, rejected,
            group["prompt"], group["aspectratio"],
        )

        self._remaining_images = [
            i for i in self._remaining_images if i not in {chosen, rejected}
        ]
        self._pairs_created_in_group += 1

        is_unconditional = group["prompt"] == "UNCONDITIONAL"
        keep_going = (
            continue_scoring
            or is_unconditional
            or self._pairs_created_in_group < self._pairs_per_group
        )
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

    def _background_scan(self) -> None:

        from modules.util import path_util
        from modules.util.dpo_curation_util import manifest_pair_counts
        from modules.util.image_metadata_util import extract_metadata, strip_angle_bracket_segments


        supported = path_util.supported_image_extensions()
        groups_dict: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

        for root, _, files in os.walk(self._source_folder):
            for filename in sorted(files):
                if self._worker_stop.is_set():
                    return
                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported:
                    continue
                path = os.path.join(root, filename)
                meta = extract_metadata(path)
                prompt = meta.get("prompt", "").strip()
                ar = meta.get("aspectratio", "").strip()
                if prompt:
                    prompt = strip_angle_bracket_segments(prompt)
                if not prompt:
                    prompt = "UNCONDITIONAL"
                groups_dict[(prompt, ar)].append(path)
                self._scan_count += 1

        raw_groups = [
            {"prompt": prompt, "aspectratio": ar, "images": images}
            for (prompt, ar), images in groups_dict.items()
            if len(images) >= 2
        ]
        random.shuffle(raw_groups)

        existing_counts = manifest_pair_counts(self._manifest)

        for group in raw_groups:
            if self._worker_stop.is_set():
                return

            group_key = (group["prompt"], group["aspectratio"])
            is_unconditional = group["prompt"] == "UNCONDITIONAL"
            if not is_unconditional and existing_counts.get(group_key, 0) >= self._pairs_per_group:
                continue

            deduped = self._dedup_by_dhash(group["images"])
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

    @staticmethod
    def _dedup_by_dhash(images: list[str]) -> list[str]:
        from PIL import Image

        seen: dict[int, str] = {}
        unique = []
        for path in images:
            try:
                with Image.open(path) as img:
                    img = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                    pixels = list(img.getdata())
                bits = 0
                for row in range(8):
                    for col in range(8):
                        idx = row * 9 + col
                        if pixels[idx] < pixels[idx + 1]:
                            bits |= 1 << (row * 8 + col)
                h = bits
            except Exception:
                unique.append(path)
                continue
            if h not in seen:
                seen[h] = path
                unique.append(path)
        return unique
