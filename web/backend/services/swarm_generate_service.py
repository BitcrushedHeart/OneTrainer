"""Generate off-policy negatives directly through SwarmUI.

Picks captions from the real photo set that have no usable generation yet,
asks SwarmUI for images at each photo's exact trainer bucket size, and leaves
the results in SwarmUI's own output folder — which is the folder the
off-policy builder scans, so the new images appear as candidates on its next
run. There is no approval step here on purpose: the builder's own
confirm/skip screen is where a pair is accepted or thrown away.

The scan reuses the off-policy builder's caches (the real-photo index and the
DPO scan cache in the same output folder), so a second run costs almost
nothing.
"""

import logging
import os
import random
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from contextlib import suppress

from modules.util.caption_style import classify_caption_type
from modules.util.swarm_client import DEFAULT_BASE_URL, SwarmClient, SwarmError
from modules.util.swarm_generate_util import (
    DEFAULT_QUANTIZATION,
    DEFAULT_TARGET_RESOLUTION,
    filter_pool_lines,
    merge_params,
    plan_generations,
    read_last_generation_params,
    sanitize_source_params,
)
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

#: Consecutive SwarmUI failures before a run gives up. A single bad prompt
#: should not stop a batch; a server that has gone away should.
_MAX_CONSECUTIVE_FAILURES = 5

#: Requests kept in flight at once. SwarmUI queues what it cannot start
#: immediately, so having the next request already waiting is what removes the
#: idle gap between one batch finishing and the next arriving. More than a
#: handful buys nothing — the GPU is the bottleneck — and costs a longer tail
#: of work that a cancel has to unwind.
DEFAULT_CONCURRENCY = 3
_MAX_CONCURRENCY = 8


def _job_summary(options: dict) -> dict:
    """The few fields that identify a queued run in the UI."""
    return {
        "keywords": options.get("keywords", ""),
        "captions": options.get("captions", 0),
        "images_per_caption": options.get("images_per_caption", 1),
        "dry_run": options.get("dry_run", False),
    }


class SwarmGenerateService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._generation = 0
        self._state: dict = self._idle_state()
        #: Guards the counters and result list, which the generation workers
        #: now update from several threads at once.
        self._results_lock = threading.Lock()
        #: Runs waiting to start. Queuing several (different keywords,
        #: different counts) lets a batch be lined up in one sitting
        #: instead of babysitting one run at a time.
        self._queue: list[dict] = []
        self._rng = random.Random()

    @staticmethod
    def _idle_state() -> dict:
        return {
            "running": False,
            "phase": "idle",
            "planned": 0,
            "done": 0,
            "images": 0,
            "failed": 0,
            "skipped_tag_style": 0,
            "skipped_short": 0,
            "pool": 0,
            "error": None,
            "results": [],
            "dry_run": False,
            "params": {},
            "param_source": "",
            # Which queued run is executing, and what is still waiting.
            "job_index": 0,
            "job_total": 0,
            "current_job": None,
            "queued": [],
        }

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def check(self, base_url: str = DEFAULT_BASE_URL, token: str = "") -> dict:
        """Confirm SwarmUI is reachable and list the models it can load."""
        client = SwarmClient(base_url, token, timeout=20)
        try:
            info = client.ping()
            models = client.list_models()
        except SwarmError as ex:
            return {"ok": False, "error": str(ex)}
        # LoRAs and the loaded checkpoint are conveniences: a server that
        # cannot answer them is still perfectly usable for generation.
        try:
            loras = client.list_models(subtype="LoRA")
        except SwarmError:
            loras = []
        try:
            loaded = client.list_loaded_models()
        except SwarmError:
            loaded = []
        try:
            presets = client.presets()
        except SwarmError:
            presets = []
        return {
            "ok": True,
            "version": info.get("version"),
            "user_id": info.get("user_id"),
            "models": models,
            "loras": loras,
            "loaded": loaded,
            "presets": [preset["title"] for preset in presets],
        }

    # ------------------------------------------------------------------ #
    # Caption pool (shares the off-policy builder's caches)
    # ------------------------------------------------------------------ #

    def _collect_pool(self, real_folder: str, generated_folder: str, output_dir: str, aspect_tolerance: float) -> list:
        from modules.util import path_util
        from modules.util.dpo_curation_util import (
            DpoScanCache,
            load_manifest,
            manifest_used_sources,
            normalize_prompt_for_grouping,
            resolve_aspect_ratio,
            walk_skipping_dotted,
        )
        from modules.util.dpo_offpolicy_util import (
            REAL_INDEX_FILENAME,
            GeneratedImage,
            RealPhotoIndex,
            caption_match_key,
            entry_stamp,
            find_sidecar_entry,
            image_size,
            load_permanent_skips,
            photo_from_entry,
            read_caption_lines,
            sidecar_map,
            suggestion_pool,
            used_caption_keys_from_manifest,
            walk_entries_skipping_dotted,
        )
        from modules.util.image_metadata_util import extract_metadata

        supported = path_util.supported_image_extensions()
        workers = min(16, (os.cpu_count() or 4) * 2)
        index = RealPhotoIndex(os.path.join(output_dir, REAL_INDEX_FILENAME))
        cache = DpoScanCache(os.path.join(output_dir, ".dpo_hash_cache.json"))

        # --- real photos, served from the shared index where possible ------
        self._state["phase"] = "indexing"
        photos = []
        seen: set[str] = set()
        misses: list[tuple] = []
        for _root, files in walk_entries_skipping_dotted(real_folder):
            if self._stop.is_set():
                return []
            sidecars = sidecar_map(files)
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() not in supported:
                    continue
                sidecar = find_sidecar_entry(files, name, sidecars)
                if sidecar is None:
                    continue
                image_stamp, sidecar_stamp = entry_stamp(files[name]), entry_stamp(sidecar)
                if image_stamp is None or sidecar_stamp is None:
                    continue
                path = files[name].path
                key = RealPhotoIndex.key(path)
                seen.add(key)
                cached = index.get(key, image_stamp, sidecar_stamp)
                if cached is not None:
                    photo = photo_from_entry(path, sidecar.path, cached)
                    if photo is not None:
                        photos.append(photo)
                else:
                    misses.append((path, sidecar.path, key, image_stamp, sidecar_stamp))

        def index_photo(item):
            path, caption_path, key, image_stamp, sidecar_stamp = item
            lines = read_caption_lines(caption_path)
            size = image_size(path) if lines else None
            return photo_from_entry(path, caption_path, index.put(key, image_stamp, sidecar_stamp, size, lines))

        if misses:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in as_completed({pool.submit(index_photo, item) for item in misses}):
                    if self._stop.is_set():
                        pool.shutdown(wait=False, cancel_futures=True)
                        index.save(seen)
                        return []
                    with suppress(Exception):
                        photo = future.result()
                        if photo is not None:
                            photos.append(photo)
        index.save(seen)
        photos.sort(key=lambda p: p.path)

        # --- generations already on disk -----------------------------------
        self._state["phase"] = "scanning"

        def group_key(path: str):
            meta = extract_metadata(path)
            return (
                normalize_prompt_for_grouping(meta.get("prompt", "")),
                resolve_aspect_ratio(meta.get("aspectratio", ""), path),
            )

        def index_generated(path: str):
            key = cache.get_group_key(path, group_key)
            size = cache.get_image_size(path, image_size) if key else None
            if key is None or size is None:
                return None
            return GeneratedImage(path, caption_match_key(key[0]), size[0], size[1])

        gen_paths = []
        for root, files in walk_skipping_dotted(generated_folder):
            if self._stop.is_set():
                return []
            gen_paths.extend(
                os.path.join(root, f) for f in sorted(files) if os.path.splitext(f)[1].lower() in supported
            )
        generated = []
        if gen_paths:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in as_completed({pool.submit(index_generated, p) for p in gen_paths}):
                    if self._stop.is_set():
                        pool.shutdown(wait=False, cancel_futures=True)
                        cache.save()
                        return []
                    with suppress(Exception):
                        gen = future.result()
                        if gen is not None and gen.prompt_key:
                            generated.append(gen)
        cache.save()

        # --- what still needs generating -----------------------------------
        self._state["phase"] = "planning"
        manifest = load_manifest(output_dir)
        return suggestion_pool(
            photos,
            generated,
            excluded_caption_keys=used_caption_keys_from_manifest(manifest),
            used_sources=manifest_used_sources(manifest),
            skipped_sources=load_permanent_skips(output_dir),
            aspect_tolerance=aspect_tolerance,
        )

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #

    def start(
        self,
        real_folder: str,
        generated_folder: str,
        output_dir: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        captions: int = 10,
        images_per_caption: int = 1,
        keywords: str = "",
        exclude_tag_style: bool = True,
        min_caption_words: int = 0,
        steps: int | None = None,
        cfg_scale: float | None = None,
        negative_prompt: str = "",
        loras: list[dict] | None = None,
        # Where the generation settings come from: "last" reuses the newest
        # image in the generated folder (what the user is generating with right
        # now), "preset" a saved SwarmUI preset, "manual" only the fields here.
        param_source: str = "last",
        preset_name: str = "",
        concurrency: int = DEFAULT_CONCURRENCY,
        aspect_tolerance: float = 0.15,
        target_resolution: int = DEFAULT_TARGET_RESOLUTION,
        quantization: int = DEFAULT_QUANTIZATION,
        dry_run: bool = False,
    ) -> dict:
        for label, folder in (("Real photo", real_folder), ("Generated", generated_folder)):
            if not os.path.isdir(folder):
                return {"ok": False, "error": f"{label} folder not found: {folder}"}
        if not output_dir:
            return {"ok": False, "error": "Output folder is required"}
        if param_source not in ("last", "preset", "manual"):
            return {"ok": False, "error": "param_source must be 'last', 'preset' or 'manual'"}
        if param_source == "manual" and not dry_run and not model:
            return {"ok": False, "error": "Pick a model"}
        if param_source == "preset" and not preset_name:
            return {"ok": False, "error": "Pick a preset"}
        if captions < 1 or images_per_caption < 1:
            return {"ok": False, "error": "Captions and images per caption must be at least 1"}

        options = {
            "real_folder": real_folder,
            "generated_folder": generated_folder,
            "output_dir": output_dir,
            "model": model,
            "base_url": base_url,
            "token": token,
            "captions": captions,
            "images_per_caption": images_per_caption,
            "keywords": keywords,
            "exclude_tag_style": exclude_tag_style,
            "min_caption_words": min_caption_words,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "negative_prompt": negative_prompt,
            "loras": [entry for entry in (loras or []) if entry.get("name")],
            "param_source": param_source,
            "preset_name": preset_name,
            "concurrency": max(1, min(int(concurrency), _MAX_CONCURRENCY)),
            "aspect_tolerance": aspect_tolerance,
            "target_resolution": target_resolution,
            "quantization": quantization,
            "dry_run": dry_run,
        }

        with self._lock:
            self._queue.append(options)
            position = len(self._queue)
            if self._state["running"]:
                # A run is in flight: this one waits its turn, and the
                # worker picks it up without any idle time in between.
                self._state["job_total"] += 1
                self._state["queued"] = [_job_summary(job) for job in self._queue]
                return {"ok": True, "queued": True, "position": position}
            self._generation += 1
            generation = self._generation
            self._stop = threading.Event()
            self._state = self._idle_state()
            self._state.update(
                {
                    "running": True,
                    "phase": "starting",
                    "dry_run": dry_run,
                    "job_total": len(self._queue),
                    "queued": [_job_summary(job) for job in self._queue],
                }
            )
            stop = self._stop
        self._thread = threading.Thread(target=self._drain, args=(generation, stop), daemon=True)
        self._thread.start()
        return {"ok": True, "queued": False, "position": position}

    def _live(self, generation: int, stop: threading.Event) -> bool:
        return self._generation == generation and not stop.is_set()

    def _drain(self, generation: int, stop: threading.Event) -> None:
        """Run every queued job in turn, newest additions included.

        Counters and results accumulate across the whole queue, so a
        batch of keyword runs reads as one session. A failure in one job
        stops the queue: whatever broke (a dead server, a bad model) will
        almost certainly break the rest too."""
        try:
            while True:
                with self._lock:
                    if not self._queue or self._generation != generation or stop.is_set():
                        break
                    options = self._queue.pop(0)
                    self._state["job_index"] += 1
                    self._state["current_job"] = _job_summary(options)
                    self._state["queued"] = [_job_summary(job) for job in self._queue]
                    self._state["dry_run"] = options["dry_run"]
                self._run_inner(generation, stop, options)
                if self._state["phase"] in ("error", "cancelled"):
                    break
        except Exception as ex:  # surfaced, never swallowed
            logger.exception("SwarmUI generation run failed")
            if self._generation == generation:
                self._state["error"] = str(ex)
                self._state["phase"] = "error"
        finally:
            if self._generation == generation:
                with self._lock:
                    self._queue.clear()
                    self._state["queued"] = []
                    self._state["current_job"] = None
                    self._state["running"] = False
                    if self._state["phase"] not in ("error", "cancelled"):
                        self._state["phase"] = "done"

    def _run_inner(self, generation: int, stop: threading.Event, options: dict) -> None:
        from modules.util.dpo_curation_util import parse_caption_keywords
        from modules.util.dpo_offpolicy_util import filter_pool_by_keywords, suggest_captions

        pool = self._collect_pool(
            options["real_folder"], options["generated_folder"], options["output_dir"], options["aspect_tolerance"]
        )
        if not self._live(generation, stop):
            self._state["phase"] = "cancelled"
            return
        pool = filter_pool_by_keywords(pool, parse_caption_keywords(options["keywords"]))
        # Tag-style and word-minimum filters narrow each photo's eligible
        # *lines* before the draw, so a photo is never judged by whichever of
        # its caption lines the random pick landed on.
        pool, stats = filter_pool_lines(
            pool, exclude_tag_style=options["exclude_tag_style"], min_caption_words=options["min_caption_words"]
        )
        self._state["pool"] = len(pool)

        entries = suggest_captions(pool, self._rng, options["captions"], base_resolution=options["target_resolution"])
        jobs, plan_stats = plan_generations(
            entries,
            images_per_caption=options["images_per_caption"],
            exclude_tag_style=options["exclude_tag_style"],
            min_caption_words=options["min_caption_words"],
            target_resolution=options["target_resolution"],
            quantization=options["quantization"],
        )
        jobs = jobs[: options["captions"]]
        self._state["skipped_tag_style"] = stats["tag_style"] + plan_stats.get("tag_style", 0)
        self._state["skipped_short"] = stats["too_short"] + plan_stats.get("too_short", 0)
        self._state["planned"] = len(jobs)

        if options["dry_run"]:
            if options["param_source"] == "last":
                self._state["params"] = merge_params(
                    read_last_generation_params(options["generated_folder"]),
                    {"model": options["model"]} if options["model"] else {},
                )
                self._state["param_source"] = "last generation"
            self._state["results"] = [
                {
                    "caption": job["caption"],
                    "real_path": job["real_path"],
                    "size": [job["width"], job["height"]],
                    "exact_size": job.get("exact_size", False),
                    "images": job["images"],
                    "caption_type": classify_caption_type(job["caption"]).value,
                    "ok": True,
                    "planned_only": True,
                }
                for job in jobs
            ]
            self._state["phase"] = "done"
            return

        # Only used to look a preset up; generation uses per-thread clients.
        client = SwarmClient(options["base_url"], options["token"])
        try:
            params = self._resolve_params(client, options)
        except SwarmError as ex:
            self._state["error"] = str(ex)
            self._state["phase"] = "error"
            return
        if not params.get("model"):
            self._state["error"] = (
                "No model: the settings source did not name one and none was picked. "
                "Generate once in SwarmUI, choose a preset, or pick a model here."
            )
            self._state["phase"] = "error"
            return
        self._state["params"] = params
        self._state["phase"] = "generating"

        extra = {key: value for key, value in params.items() if key != "model"}
        # One client per worker thread: a client owns a session and an HTTP
        # connection, neither of which wants to be shared mid-request.
        thread_state = threading.local()

        def worker_client() -> SwarmClient:
            existing = getattr(thread_state, "client", None)
            if existing is None:
                existing = SwarmClient(options["base_url"], options["token"])
                thread_state.client = existing
            return existing

        def run_job(job: dict) -> tuple[dict, list[str]]:
            return job, worker_client().generate(
                prompt=job["caption"],
                width=job["width"],
                height=job["height"],
                model=params["model"],
                images=job["images"],
                extra_params=extra,
            )

        failures_since_success = 0
        aborted = False
        # Every job is submitted up front so SwarmUI always has the next
        # request queued behind the one it is running — no idle GPU between
        # batches — while the pool keeps only `concurrency` in flight.
        with ThreadPoolExecutor(max_workers=options["concurrency"]) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                if not self._live(generation, stop):
                    pool.shutdown(wait=False, cancel_futures=True)
                    self._state["phase"] = "cancelled"
                    return
                job = futures[future]
                record = {
                    "caption": job["caption"],
                    "real_path": job["real_path"],
                    "size": [job["width"], job["height"]],
                    "exact_size": job.get("exact_size", False),
                    "images": job["images"],
                    "caption_type": classify_caption_type(job["caption"]).value,
                }
                try:
                    _job, paths = future.result()
                except CancelledError:
                    continue
                except SwarmError as ex:
                    record.update({"ok": False, "error": str(ex)})
                    with self._results_lock:
                        failures_since_success += 1
                        self._state["failed"] += 1
                        self._state["done"] += 1
                        self._state["results"].append(record)
                        give_up = failures_since_success >= _MAX_CONSECUTIVE_FAILURES
                    if give_up:
                        self._state["error"] = (
                            f"Stopped after {failures_since_success} consecutive SwarmUI failures: {ex}"
                        )
                        self._state["phase"] = "error"
                        pool.shutdown(wait=False, cancel_futures=True)
                        aborted = True
                        break
                    continue
                record.update({"ok": True, "paths": paths})
                with self._results_lock:
                    failures_since_success = 0
                    self._state["results"].append(record)
                    self._state["images"] += len(paths)
                    self._state["done"] += 1
        if not aborted:
            self._state["phase"] = "done"

    def _resolve_params(self, client: SwarmClient, options: dict) -> dict:
        """The parameter set every generation in this run uses.

        A source (the user's last generation, or a saved preset) supplies
        everything model-specific — sampler, scheduler, sigma shift, CFG
        variants, LoRAs — and the fields set in this tool are layered on top.
        That way the negatives are produced the way the user actually
        generates, rather than with this tool's idea of defaults."""
        source: dict = {}
        source_label = "manual"
        if options["param_source"] == "last":
            source = read_last_generation_params(options["generated_folder"])
            source_label = "last generation" if source else "manual (no previous generation found)"
        elif options["param_source"] == "preset":
            wanted = options["preset_name"]
            for preset in client.presets():
                if preset["title"] == wanted:
                    source = preset["params"]
                    source_label = f"preset {wanted!r}"
                    break
            else:
                raise SwarmError(f"No SwarmUI preset named {wanted!r}")

        overrides: dict = {}
        if options["model"]:
            overrides["model"] = options["model"]
        if options["steps"] is not None:
            overrides["steps"] = options["steps"]
        if options["cfg_scale"] is not None:
            overrides["cfgscale"] = options["cfg_scale"]
        if options["negative_prompt"]:
            overrides["negativeprompt"] = options["negative_prompt"]
        if options["loras"]:
            overrides["loras"] = [entry["name"] for entry in options["loras"]]
            overrides["loraweights"] = [str(entry.get("weight", 1.0)) for entry in options["loras"]]
        params = merge_params(source, overrides)
        self._state["param_source"] = source_label
        return params

    def preview_params(self, generated_folder: str) -> dict:
        """What "reuse my last generation" would apply, for the UI to show
        before anything is generated."""
        params = sanitize_source_params(read_last_generation_params(generated_folder))
        return {"ok": True, "params": params}

    def status(self) -> dict:
        state = dict(self._state)
        # The result list can grow long; the UI only renders the tail.
        state["results"] = state["results"][-50:]
        return {"ok": True, **state}

    def cancel(self) -> dict:
        """Stop the current run and drop everything still queued."""
        with self._lock:
            self._stop.set()
            dropped = len(self._queue)
            self._queue.clear()
            self._state["queued"] = []
            if self._state["running"]:
                self._state["phase"] = "cancelling"
        return {"ok": True, "dropped": dropped}

    def clear_queue(self) -> dict:
        """Drop the waiting runs but let the current one finish."""
        with self._lock:
            dropped = len(self._queue)
            self._queue.clear()
            self._state["queued"] = []
            self._state["job_total"] = self._state["job_index"]
        return {"ok": True, "dropped": dropped}
