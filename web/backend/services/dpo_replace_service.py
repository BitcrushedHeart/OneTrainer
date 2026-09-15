"""Replace Pairs: regenerate the sides of existing DPO pairs with the current
checkpoint and swap the better result in.

Session shape, mirroring the off-policy builder:

1. ``start`` reads every pair of a DPO concept from disk in a background
   scan — the caption the trainer will read, who generated each side, cached
   realism scores — and ranks them by staleness.
2. ``generate`` sends the first N ranked pairs (or an explicit list) to
   SwarmUI with the settings of the user's most recent generation, scores the
   fresh images with the realism head as each job lands, and turns each into
   a proposal: a harder negative, a promotion (fresh image becomes chosen,
   old chosen becomes rejected — only when the chosen side is itself a
   generation), or keep.
3. Each pair becomes a selection group — old chosen, old rejected, and the
   fresh images side by side — and the reviewer picks the best as chosen and
   the second-best as rejected (``next`` / ``apply`` / ``keep`` / ``undo``).
   Realism scores pre-select a suggestion; with auto-apply on, a suggestion
   that leaves the chosen side alone is applied without review. Every pair
   is AI-vs-AI here: any image in the group can become chosen.

Every displaced file is kept under ``<concept>/.replaced`` and the manifest
records the model and adapters behind every side, so the history of a pair
across checkpoints is readable later.
"""

import logging
import os
import threading
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

from modules.util.dpo_replace_util import (
    ACTION_KEEP,
    ORDER_STALE_FIRST,
    ORDERS,
    ROLE_FRESH,
    GroupImage,
    PairState,
    Proposal,
    apply_selection,
    backfill_provenance,
    build_proposal,
    local_path_for_view,
    model_groups,
    provenance_from_params,
    rank_pairs,
    read_pair_state,
    undo_selection,
)
from modules.util.swarm_client import DEFAULT_BASE_URL, SwarmClient, SwarmError
from modules.util.swarm_generate_util import (
    DEFAULT_QUANTIZATION,
    DEFAULT_TARGET_RESOLUTION,
    merge_params,
    plan_generations,
    read_last_generation_params,
    sanitize_source_params,
)
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_CONCURRENCY = 3
_MAX_CONCURRENCY = 8
_MAX_IMAGES_PER_PAIR = 8


class DPOReplacePairsService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: Serialises realism-model calls: each builds its own ONNX session.
        self._score_lock = threading.Lock()
        self._active = False
        self._generation = 0
        self._concept_dir: str | None = None
        self._generated_folder: str | None = None
        self._base_url = DEFAULT_BASE_URL
        self._token = ""
        self._order = ORDER_STALE_FIRST
        self._allow_promote = True
        self._target_resolution = DEFAULT_TARGET_RESOLUTION
        self._quantization = DEFAULT_QUANTIZATION
        self._manifest: dict = {"pairs": []}
        self._cache = None
        self._states: list[PairState] = []
        self._by_id: dict[int, PairState] = {}
        self._phase = "idle"
        self._progress_done = 0
        self._progress_total = 0
        self._scan_finished = False
        self._error: str | None = None
        self._warning: str | None = None
        self._stop = threading.Event()
        self._scan_thread: threading.Thread | None = None
        # --- generation run ---
        self._run_thread: threading.Thread | None = None
        self._running = False
        self._run_phase = "idle"
        self._params: dict = {}
        self._param_source = ""
        self._planned = 0
        self._done = 0
        self._failed = 0
        self._images = 0
        self._run_error: str | None = None
        self._auto_apply = False
        self._images_per_pair = 1
        self._in_flight: set[int] = set()
        # --- review ---
        self._ready: deque[Proposal] = deque()
        self._current: Proposal | None = None
        self._undo_stack: list[dict] = []
        self._applied = 0
        self._auto_applied = 0
        self._kept = 0
        self._promoted = 0
        self._unscored = 0
        self._shown = 0

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    @staticmethod
    def _layout_error(concept_dir: str) -> str | None:
        if not os.path.isdir(concept_dir):
            return f"Concept folder not found: {concept_dir}"
        if not os.path.isfile(os.path.join(concept_dir, "manifest.json")):
            return (
                "No manifest.json in the concept folder. Replace Pairs works on concepts exported by the "
                "DPO curation tool or the off-policy builder."
            )
        for sub in ("chosen", "rejected"):
            if not os.path.isdir(os.path.join(concept_dir, sub)):
                return f"Concept folder has no {sub}/ subfolder"
        return None

    def concepts(self) -> dict:
        """Configured DPO concepts whose folders carry a manifest, as a pick list."""
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
        from web.backend.services.dpo_service import DPOService

        roots: list[str] = []
        try:
            concepts = DPOService.get_instance()._resolve_concepts()
            for chosen_dir, _rejected_dir in dpo_concept_pattern_dirs(concepts):
                root = os.path.dirname(chosen_dir)
                if os.path.isfile(os.path.join(root, "manifest.json")) and root not in roots:
                    roots.append(root)
        except Exception as ex:  # a missing config is not an error here
            logger.warning("could not list configured DPO concepts: %s", ex)
            return {"ok": True, "concepts": [], "warning": str(ex)}
        return {"ok": True, "concepts": roots}

    def start(
        self,
        concept_dir: str,
        generated_folder: str,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        order: str = ORDER_STALE_FIRST,
        allow_promote: bool = True,
        target_resolution: int = DEFAULT_TARGET_RESOLUTION,
        quantization: int = DEFAULT_QUANTIZATION,
    ) -> dict:
        from modules.util.dpo_curation_util import DpoScanCache, load_manifest, prune_orphaned_pairs

        error = self._layout_error(concept_dir)
        if error:
            return {"ok": False, "error": error}
        if not os.path.isdir(generated_folder):
            return {"ok": False, "error": f"Generated folder not found: {generated_folder}"}
        if order not in ORDERS:
            return {"ok": False, "error": f"order must be one of {', '.join(ORDERS)}"}

        with self._lock:
            if self._active:
                return {"ok": False, "error": "Session already active"}
            self._active = True
            self._stop.set()
            self._generation += 1
            generation = self._generation

        try:
            manifest = load_manifest(concept_dir)
            pruned = prune_orphaned_pairs(concept_dir, manifest)
        except (OSError, RuntimeError, ValueError) as ex:
            with self._lock:
                self._active = False
            return {"ok": False, "error": f"Could not open concept: {ex}"}

        with self._lock:
            self._concept_dir = concept_dir
            self._generated_folder = generated_folder
            self._base_url = base_url
            self._token = token
            self._order = order
            self._allow_promote = allow_promote
            self._target_resolution = max(256, int(target_resolution))
            self._quantization = max(8, int(quantization))
            self._manifest = manifest
            # The same cache file the curation tool and the off-policy builder
            # keep in a concept folder, so realism scores are shared.
            self._cache = DpoScanCache(os.path.join(concept_dir, ".dpo_hash_cache.json"))
            self._states = []
            self._by_id = {}
            self._phase = "scanning"
            self._progress_done = 0
            self._progress_total = len(manifest.get("pairs", []))
            self._scan_finished = False
            self._error = None
            self._warning = None
            self._running = False
            self._run_phase = "idle"
            self._params = {}
            self._param_source = ""
            self._planned = self._done = self._failed = self._images = 0
            self._run_error = None
            self._in_flight = set()
            self._ready = deque()
            self._current = None
            self._undo_stack = []
            self._applied = self._auto_applied = self._kept = self._promoted = self._unscored = self._shown = 0
            self._stop = threading.Event()
            self._scan_thread = threading.Thread(target=self._scan, args=(generation, self._stop), daemon=True)
            self._scan_thread.start()
        return {"ok": True, "pairs": len(manifest.get("pairs", [])), "pruned": pruned}

    def cancel(self) -> dict:
        with self._lock:
            self._stop.set()
            self._active = False
            self._running = False
            self._phase = "idle"
            self._run_phase = "idle"
            self._ready.clear()
            self._current = None
            if self._cache is not None:
                self._cache.save()
        return {"ok": True}

    def _live(self, generation: int, stop: threading.Event) -> bool:
        return self._generation == generation and not stop.is_set()

    def status(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "active": self._active,
                "phase": self._phase,
                "progress_done": self._progress_done,
                "progress_total": self._progress_total,
                "scan_finished": self._scan_finished,
                "error": self._error,
                "warning": self._warning,
                "pairs": len(self._states),
                "stale_captions": sum(1 for s in self._states if s.stale_caption),
                "order": self._order,
                "allow_promote": self._allow_promote,
                # generation run
                "running": self._running,
                "run_phase": self._run_phase,
                "params": self._params,
                "param_source": self._param_source,
                "planned": self._planned,
                "done": self._done,
                "failed": self._failed,
                "images": self._images,
                "run_error": self._run_error,
                "auto_apply": self._auto_apply,
                # review
                "ready": len(self._ready),
                "has_proposal": self._current is not None,
                "shown": self._shown,
                "applied": self._applied,
                "auto_applied": self._auto_applied,
                "promoted": self._promoted,
                "kept": self._kept,
                "unscored": self._unscored,
                "can_undo": bool(self._undo_stack),
                "undo_action": self._undo_stack[-1]["record"]["action"] if self._undo_stack else None,
            }

    # ------------------------------------------------------------------ #
    # Scan
    # ------------------------------------------------------------------ #

    def _score_peek(self):
        from modules.util.realism_detect_util import realism_model_version

        try:
            version = realism_model_version()
        except Exception:
            version = None
        cache = self._cache
        if cache is None or version is None:
            return lambda _path: None
        return lambda path: cache.peek_realism(path, version)

    def _scan(self, generation: int, stop: threading.Event) -> None:
        try:
            self._scan_inner(generation, stop)
        except Exception as ex:
            logger.exception("Replace Pairs scan failed")
            if self._generation == generation:
                self._error = str(ex)
                self._phase = "error"
        finally:
            if self._generation == generation:
                self._scan_finished = True
                if self._phase == "scanning":
                    self._phase = "ready"

    def _scan_inner(self, generation: int, stop: threading.Event) -> None:
        from modules.util.dpo_curation_util import save_manifest

        concept_dir = self._concept_dir
        peek = self._score_peek()
        states: list[PairState] = []
        backfilled = 0
        workers = min(16, (os.cpu_count() or 4) * 2)
        entries = list(self._manifest.get("pairs", []))

        def read(entry: dict) -> PairState | None:
            return read_pair_state(concept_dir, entry, score_of=peek)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(read, entry): entry for entry in entries}
            for future in as_completed(futures):
                if not self._live(generation, stop):
                    pool.shutdown(wait=False, cancel_futures=True)
                    return
                entry = futures[future]
                try:
                    state = future.result()
                except Exception as ex:  # one unreadable pair must not end the scan
                    logger.warning("could not read pair %s: %s", entry.get("pair_id"), ex)
                    state = None
                self._progress_done += 1
                if state is None:
                    continue
                if backfill_provenance(entry, state):
                    backfilled += 1
                states.append(state)

        if not self._live(generation, stop):
            return
        ranked = rank_pairs(states, self._order)
        with self._lock:
            self._states = ranked
            self._by_id = {s.pair_id: s for s in ranked}
        if backfilled:
            try:
                save_manifest(concept_dir, self._manifest)
            except OSError as ex:
                self._warning = f"Could not save provenance backfill: {ex}"

    def pairs(self, offset: int = 0, limit: int = 200) -> dict:
        """The ranked pair list (a page of it) plus the per-model summary."""
        if not self._active:
            return {"ok": False, "error": "No active session"}
        with self._lock:
            if not self._scan_finished:
                return {"ok": True, "waiting": True, "pairs": [], "total": 0, "models": []}
            states = self._states
            page = states[max(0, offset) : max(0, offset) + max(1, limit)]
            return {
                "ok": True,
                "pairs": [s.to_dict() for s in page],
                "total": len(states),
                "offset": max(0, offset),
                "models": model_groups(states),
            }

    # ------------------------------------------------------------------ #
    # Generation settings
    # ------------------------------------------------------------------ #

    def params(self) -> dict:
        """The settings the next batch would use: the user's last generation."""
        if not self._generated_folder:
            return {"ok": False, "error": "No active session"}
        source = read_last_generation_params(self._generated_folder)
        params = sanitize_source_params(source)
        return {
            "ok": True,
            "params": params,
            "model": source.get("model", "") if isinstance(source, dict) else "",
            "found": bool(source),
        }

    def _resolve_params(self) -> dict:
        source = read_last_generation_params(self._generated_folder)
        params = merge_params(source, {"model": source.get("model")} if isinstance(source, dict) else {})
        self._param_source = "last generation" if source else "none"
        return params

    # ------------------------------------------------------------------ #
    # Generation run
    # ------------------------------------------------------------------ #

    def generate(
        self,
        count: int = 10,
        pair_ids: list[int] | None = None,
        images_per_pair: int = 2,
        concurrency: int = DEFAULT_CONCURRENCY,
        auto_apply: bool = False,
    ) -> dict:
        if not self._active:
            return {"ok": False, "error": "No active session"}
        with self._lock:
            if not self._scan_finished:
                return {"ok": False, "error": "Still reading the concept — try again in a moment"}
            if self._running:
                return {"ok": False, "error": "A generation run is already in progress"}
            if pair_ids:
                wanted = [self._by_id[i] for i in pair_ids if i in self._by_id]
                if not wanted:
                    return {"ok": False, "error": "None of those pair ids are in this concept"}
            else:
                # Pairs already refreshed this session, or awaiting review,
                # are skipped so repeated batches walk down the ranking.
                skip = set(self._in_flight) | {p.state.pair_id for p in self._ready}
                if self._current is not None:
                    skip.add(self._current.state.pair_id)
                skip |= {r["record"]["pair_id"] for r in self._undo_stack}
                wanted = [s for s in self._states if s.pair_id not in skip][: max(1, int(count))]
            if not wanted:
                return {"ok": False, "error": "Nothing left to generate for in this ranking"}
            params = self._resolve_params()
            if not params.get("model"):
                return {
                    "ok": False,
                    "error": "No model: no previous generation with settings was found in the generated folder. "
                    "Generate once in SwarmUI with the checkpoint and adapters you want, then try again.",
                }
            self._params = params
            self._images_per_pair = max(1, min(int(images_per_pair), _MAX_IMAGES_PER_PAIR))
            self._auto_apply = bool(auto_apply)
            self._running = True
            self._run_phase = "generating"
            self._planned = len(wanted)
            self._done = self._failed = self._images = 0
            self._run_error = None
            self._in_flight |= {s.pair_id for s in wanted}
            generation = self._generation
            stop = self._stop
            options = {
                "concurrency": max(1, min(int(concurrency), _MAX_CONCURRENCY)),
                "params": params,
            }
        self._run_thread = threading.Thread(target=self._run, args=(generation, stop, wanted, options), daemon=True)
        self._run_thread.start()
        return {"ok": True, "planned": len(wanted), "model": params.get("model", ""), "params": params}

    def _run(self, generation: int, stop: threading.Event, states: list[PairState], options: dict) -> None:
        try:
            self._run_inner(generation, stop, states, options)
        except Exception as ex:
            logger.exception("Replace Pairs generation run failed")
            if self._generation == generation:
                self._run_error = str(ex)
                self._run_phase = "error"
        finally:
            if self._generation == generation:
                with self._lock:
                    self._running = False
                    if self._run_phase == "generating":
                        self._run_phase = "done"
                if self._cache is not None:
                    self._cache.save()

    def _run_inner(self, generation: int, stop: threading.Event, states: list[PairState], options: dict) -> None:
        params = options["params"]
        extra = {key: value for key, value in params.items() if key != "model"}
        provenance = provenance_from_params(params)
        entries = [
            {"caption": s.caption, "real_size": s.chosen_size or (0, 0), "real_path": s.chosen_path, "state": s}
            for s in states
        ]
        jobs, _stats = plan_generations(
            entries,
            images_per_caption=self._images_per_pair,
            exclude_tag_style=False,
            min_caption_words=0,
            target_resolution=self._target_resolution,
            quantization=self._quantization,
        )
        by_path = {s.chosen_path: s for s in states}
        planned_paths = {job["real_path"] for job in jobs}
        for state in states:
            if state.chosen_path not in planned_paths:
                # No size (unreadable image) or empty caption: nothing to ask for.
                with self._lock:
                    self._failed += 1
                    self._done += 1
                    self._in_flight.discard(state.pair_id)
        thread_state = threading.local()

        def worker_client() -> SwarmClient:
            existing = getattr(thread_state, "client", None)
            if existing is None:
                existing = SwarmClient(self._base_url, self._token)
                thread_state.client = existing
            return existing

        def run_job(job: dict) -> list[str]:
            return worker_client().generate(
                prompt=job["caption"],
                width=job["width"],
                height=job["height"],
                model=params["model"],
                images=job["images"],
                extra_params=extra,
            )

        failures = 0
        with ThreadPoolExecutor(max_workers=options["concurrency"]) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                if not self._live(generation, stop):
                    pool.shutdown(wait=False, cancel_futures=True)
                    self._run_phase = "cancelled"
                    return
                job = futures[future]
                state = by_path[job["real_path"]]
                try:
                    views = future.result()
                except CancelledError:
                    continue
                except SwarmError as ex:
                    with self._lock:
                        failures += 1
                        self._failed += 1
                        self._done += 1
                        self._in_flight.discard(state.pair_id)
                    if failures >= _MAX_CONSECUTIVE_FAILURES:
                        self._run_error = f"Stopped after {failures} consecutive SwarmUI failures: {ex}"
                        self._run_phase = "error"
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
                    continue
                failures = 0
                paths = [p for p in (local_path_for_view(v, self._generated_folder) for v in views) if p]
                if not paths:
                    with self._lock:
                        self._failed += 1
                        self._done += 1
                        self._in_flight.discard(state.pair_id)
                    self._warning = (
                        "SwarmUI reported images that were not found under the generated folder. Is the "
                        "generated folder SwarmUI's raw output root (…/Output/local/raw)?"
                    )
                    continue
                proposal = self._propose(state, job["caption"], paths, provenance)
                self._settle(proposal)
                with self._lock:
                    self._images += len(paths)
                    self._done += 1

    # ------------------------------------------------------------------ #
    # Scoring and proposals
    # ------------------------------------------------------------------ #

    def _score(self, paths: list[str]) -> dict[str, float]:
        """P(AI) for each path, via the shared cache; ``{}`` when the head is
        unavailable or turned off (reported once as a warning)."""
        from modules.util.realism_detect_util import REALISM_DEVICE_OFF, score_images_realism
        from web.backend.services.dpo_service import DPOService

        try:
            device = DPOService.get_instance().realism_device().get("device", "auto")
        except Exception:
            device = "auto"
        if device == REALISM_DEVICE_OFF:
            self._warning = "Realism scoring is off; groups are unscored — pick by eye."
            return {}
        with self._score_lock:
            try:
                return score_images_realism(paths, cache=self._cache, device=device)
            except RuntimeError as ex:
                self._warning = f"Realism model unavailable ({ex}); groups are unscored — pick by eye."
                return {}

    def _propose(self, state: PairState, caption: str, paths: list[str], provenance) -> Proposal:
        from modules.util.dpo_offpolicy_util import image_size

        scores = self._score(list(paths) + [state.chosen_path, state.rejected_path])
        if state.chosen_path in scores:
            state.chosen_score = scores[state.chosen_path]
        if state.rejected_path in scores:
            state.rejected_score = scores[state.rejected_path]
        fresh = [GroupImage(path=p, role=ROLE_FRESH, size=image_size(p), score=scores.get(p)) for p in paths]
        proposal = build_proposal(state, caption, fresh, provenance, allow_promote=self._allow_promote)
        if not proposal.scored:
            with self._lock:
                self._unscored += 1
        return proposal

    def _settle(self, proposal: Proposal) -> None:
        """Auto-apply a scored suggestion that keeps the chosen side, when
        asked; queue everything else for a person."""
        auto = (
            self._auto_apply
            and proposal.scored
            and not proposal.suggestion_is_keep
            and not proposal.suggestion_moves_chosen
        )
        if auto:
            with self._lock:
                try:
                    record = apply_selection(
                        self._concept_dir,
                        self._manifest,
                        proposal,
                        proposal.suggested_chosen,
                        proposal.suggested_rejected,
                    )
                except (OSError, ValueError) as ex:
                    self._warning = f"Auto-apply failed for pair {proposal.state.pair_id}: {ex}"
                    self._ready.append(proposal)
                    return
                self._undo_stack.append({"record": record, "proposal": proposal, "auto": True})
                self._applied += 1
                self._auto_applied += 1
                self._in_flight.discard(proposal.state.pair_id)
            return
        with self._lock:
            self._ready.append(proposal)

    # ------------------------------------------------------------------ #
    # Review
    # ------------------------------------------------------------------ #

    def _payload(self, proposal: Proposal) -> dict:
        data = proposal.to_dict()
        data["index"] = self._shown
        data["queued"] = len(self._ready)
        return data

    def next(self) -> dict:
        if not self._active:
            return {"ok": False, "error": "No active session"}
        with self._lock:
            if not self._ready:
                if not self._running:
                    self._current = None
                    return {"ok": True, "done": True}
                return {"ok": True, "waiting": True}
            proposal = self._ready.popleft()
            self._current = proposal
            self._shown += 1
            return {"ok": True, "proposal": self._payload(proposal)}

    def current(self) -> dict:
        if not self._active:
            return {"ok": False, "error": "No active session"}
        with self._lock:
            if self._current is None:
                return {"ok": True, "proposal": None}
            return {"ok": True, "proposal": self._payload(self._current)}

    def apply(self, chosen_path: str | None = None, rejected_path: str | None = None) -> dict:
        """Make two images of the group on screen the pair's sides. Without
        arguments the suggestion is applied."""
        with self._lock:
            proposal = self._current
            if proposal is None:
                return {"ok": False, "error": "No proposal on screen"}
            chosen = chosen_path or proposal.suggested_chosen
            rejected = rejected_path or proposal.suggested_rejected
            if chosen == proposal.old_chosen and rejected == proposal.old_rejected:
                return {"ok": False, "error": "That is the pair as it already is — use keep"}
            try:
                record = apply_selection(self._concept_dir, self._manifest, proposal, chosen, rejected)
            except (OSError, ValueError) as ex:
                return {"ok": False, "error": f"Apply failed: {ex}"}
            self._undo_stack.append({"record": record, "proposal": proposal, "auto": False})
            self._applied += 1
            if chosen != proposal.old_chosen:
                self._promoted += 1
            self._in_flight.discard(proposal.state.pair_id)
            self._current = None
            return {"ok": True, "action": record["action"], "pair_id": proposal.state.pair_id}

    def keep(self) -> dict:
        """Leave the pair as it is; the fresh generations stay in the
        generated folder untouched."""
        with self._lock:
            proposal = self._current
            if proposal is None:
                return {"ok": False, "error": "No proposal on screen"}
            self._undo_stack.append(
                {
                    "record": {"action": ACTION_KEEP, "pair_id": proposal.state.pair_id},
                    "proposal": proposal,
                    "auto": False,
                }
            )
            self._kept += 1
            self._in_flight.discard(proposal.state.pair_id)
            self._current = None
            return {"ok": True, "pair_id": proposal.state.pair_id}

    def undo(self) -> dict:
        with self._lock:
            if not self._active:
                return {"ok": False, "error": "No active session"}
            if not self._undo_stack:
                return {"ok": False, "error": "Nothing to undo"}
            item = self._undo_stack.pop()
            record, proposal = item["record"], item["proposal"]
            action = record["action"]
            if action == ACTION_KEEP:
                self._kept = max(0, self._kept - 1)
            else:
                try:
                    undo_selection(self._concept_dir, self._manifest, record)
                except (OSError, KeyError) as ex:
                    self._undo_stack.append(item)
                    return {"ok": False, "error": f"Undo failed: {ex}"}
                self._applied = max(0, self._applied - 1)
                if item.get("auto"):
                    self._auto_applied = max(0, self._auto_applied - 1)
                if record.get("chosen_from") not in (None, proposal.old_chosen):
                    self._promoted = max(0, self._promoted - 1)
            self._in_flight.add(proposal.state.pair_id)
            if self._current is not None:
                self._ready.appendleft(self._current)
            self._current = proposal
            return {"ok": True, "undone": action, "proposal": self._payload(proposal)}
