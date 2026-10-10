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
    BLACK_FOLDER,
    DEFAULT_BLACK_MAX_LEVEL,
    DEFAULT_QUANTIZATION,
    DEFAULT_TARGET_RESOLUTION,
    describe_black_image,
    filter_pool_lines,
    find_black_images,
    is_black_image,
    merge_params,
    plan_generations,
    read_last_generation_params,
    sanitize_source_params,
    save_generation_locally,
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

#: Extra attempts a job gets when every image it produced came back black.
#: A fresh seed each time; after these the job is a failure like any other,
#: so a backend that only writes black still trips the consecutive-failure
#: stop instead of looping.
_BLACK_RETRIES = 2


def split_keyword_jobs(entries: list[str] | None) -> list[str]:
    """Bulk keyword entries as one keyword filter per run: trimmed, blanks
    and repeats dropped, and ``a+b`` turned into the all-required ``a, b``."""
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries or []:
        terms = [term.strip() for term in str(entry).replace(",", "+").split("+") if term.strip()]
        keywords = ", ".join(terms)
        if not keywords or keywords.casefold() in seen:
            continue
        seen.add(keywords.casefold())
        out.append(keywords)
    return out


def _job_summary(options: dict) -> dict:
    """The few fields that identify a queued run in the UI."""
    return {
        "mode": options.get("mode", "generate"),
        # Free text from the caller (the character a run is for, say).
        "label": options.get("label", ""),
        "keywords": options.get("keywords", ""),
        "captions": options.get("captions", 0),
        "images_per_caption": options.get("images_per_caption", 1),
        "images": len(options.get("jobs") or []),
        "dry_run": options.get("dry_run", False),
    }


#: Where a run's settings come from. "last" and "folder" both read the newest
#: image's parameter block — from the generated folder, or from any folder the
#: caller names (a reference set generated the way the user likes).
PARAM_SOURCES = ("last", "preset", "manual", "folder")


def strip_source_loras(params: dict) -> dict:
    """``params`` without any adapter keys SwarmUI records (``loras``,
    ``loraweights``, ``lorasectionconfs``, ``loratencweights``, ...), so only
    the LoRAs a caller passes explicitly reach the generation."""
    return {key: value for key, value in (params or {}).items() if not str(key).lower().startswith("lora")}


def _folder_source_label(folder: str, found: bool) -> str:
    name = os.path.basename(os.path.normpath(folder)) or folder
    return f"last generation in {name}" if found else f"manual (no generation found in {name})"


def last_generation_source(folder: str, search_limit: int = 40) -> tuple[str | None, dict]:
    """``(path, params)`` of the newest image under ``folder`` that carries a
    SwarmUI parameter block — the same search as
    :func:`read_last_generation_params`, with the file it settled on, so the
    UI can name it. ``(None, {})`` when there is none."""
    from modules.util.path_util import supported_image_extensions
    from modules.util.swarm_generate_util import _params_from_image

    supported = supported_image_extensions()
    candidates: list[tuple[float, str]] = []
    if os.path.isdir(folder or ""):
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for name in files:
                if os.path.splitext(name)[1].lower() not in supported:
                    continue
                path = os.path.join(root, name)
                with suppress(OSError):
                    candidates.append((os.path.getmtime(path), path))
    candidates.sort(reverse=True)
    for _mtime, path in candidates[:search_limit]:
        params = _params_from_image(path)
        if params:
            return path, params
    return None, {}


_BLACK_OUTPUT = f"SwarmUI wrote a black image (backend failure); moved to {BLACK_FOLDER}"


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
        #: Makes finding black images and retiring them one step, so a manual
        #: requeue and a run's own start-of-run scan cannot both claim the
        #: same file and generate it twice.
        self._black_lock = threading.Lock()

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
            # Black images found in the tree at the start of a run and
            # retired to .black, and in-run black outputs retried.
            "black_retired": 0,
            "black_retried": 0,
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
            yolo = client.list_yolo_models()
        except SwarmError:
            yolo = []
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
            "yolo": yolo,
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
        # now), "preset" a saved SwarmUI preset, "manual" only the fields here,
        # "folder" the newest image in ``param_folder``.
        param_source: str = "last",
        preset_name: str = "",
        param_folder: str = "",
        # Drop the source's own LoRAs so only ``loras`` applies — a reference
        # folder generated with one character's adapter must not carry it
        # into another character's negatives.
        drop_source_loras: bool = False,
        # Free text shown against the queued run (the character it is for).
        label: str = "",
        # When set, SwarmUI is asked not to save (``donotsave``) and every
        # result is written into this folder as a self-describing PNG instead
        # of landing in SwarmUI's output tree. Used for per-character folders.
        save_to: str = "",
        concurrency: int = DEFAULT_CONCURRENCY,
        aspect_tolerance: float = 0.15,
        target_resolution: int = DEFAULT_TARGET_RESOLUTION,
        quantization: int = DEFAULT_QUANTIZATION,
        dry_run: bool = False,
        # Retire and regenerate black images already in the tree before the
        # new captions, and retry an in-run black output with a fresh seed.
        auto_requeue_black: bool = True,
        # Bulk: one run per entry, each with its own ``captions`` budget and
        # the entry as its keyword filter ("a+b" requires both). Replaces
        # ``keywords`` when it holds anything.
        keyword_jobs: list[str] | None = None,
    ) -> dict:
        for what, folder in (("Real photo", real_folder), ("Generated", generated_folder)):
            if not os.path.isdir(folder):
                return {"ok": False, "error": f"{what} folder not found: {folder}"}
        if not output_dir:
            return {"ok": False, "error": "Output folder is required"}
        if param_source not in PARAM_SOURCES:
            return {"ok": False, "error": "param_source must be 'last', 'preset', 'manual' or 'folder'"}
        if param_source == "manual" and not dry_run and not model:
            return {"ok": False, "error": "Pick a model"}
        if param_source == "preset" and not preset_name:
            return {"ok": False, "error": "Pick a preset"}
        if param_source == "folder" and not os.path.isdir(param_folder or ""):
            return {"ok": False, "error": f"Parameter folder not found: {param_folder}"}
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
            "param_folder": param_folder,
            "drop_source_loras": bool(drop_source_loras),
            "label": label or "",
            "save_to": save_to or "",
            "concurrency": max(1, min(int(concurrency), _MAX_CONCURRENCY)),
            "aspect_tolerance": aspect_tolerance,
            "target_resolution": target_resolution,
            "quantization": quantization,
            "dry_run": dry_run,
            "auto_requeue_black": bool(auto_requeue_black),
        }

        bulk = split_keyword_jobs(keyword_jobs)
        if not bulk:
            return self._enqueue(options)
        # Runs drain in order and each reads the pool afresh, so a caption two
        # keywords share is generated once, by the first run that reaches it.
        result = self._enqueue(*[{**options, "keywords": keywords} for keywords in bulk])
        return {**result, "jobs": len(bulk)}

    def _enqueue(self, *jobs: dict) -> dict:
        """Add runs to the queue, starting the worker if none is draining.
        ``position`` is the first new run's place in the queue."""
        options = jobs[0]
        with self._lock:
            position = len(self._queue) + 1
            self._queue.extend(jobs)
            if self._state["running"]:
                # A run is in flight: this one waits its turn, and the
                # worker picks it up without any idle time in between.
                self._state["job_total"] += len(jobs)
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
                    "dry_run": options.get("dry_run", False),
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
                    self._state["dry_run"] = options.get("dry_run", False)
                if options.get("mode") == "requeue":
                    self._requeue_inner(generation, stop, options)
                else:
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
        from modules.util.dpo_curation_util import move_to_dot_bucket, parse_caption_keywords
        from modules.util.dpo_offpolicy_util import caption_match_key, filter_pool_by_keywords, suggest_captions

        # Black images already in the tree (an earlier backend failure) are
        # retired before the pool is read, so their captions count as
        # un-generated again, and regenerated ahead of the new captions. A
        # plan-only run lists them without touching them.
        black_jobs: list[dict] = []
        black_keys: set[str] = set()
        if options.get("auto_requeue_black"):
            self._state["phase"] = "scanning"
            with self._black_lock:
                black, black_jobs, _unresendable = self._plan_black_jobs(options["generated_folder"], options["model"])
                if black and not options["dry_run"]:
                    self._state["black_retired"] += move_to_dot_bucket(black, BLACK_FOLDER)
            black_keys = {caption_match_key(job["caption"]) for job in black_jobs}
            if not self._live(generation, stop):
                self._state["phase"] = "cancelled"
                return

        pool = self._collect_pool(
            options["real_folder"], options["generated_folder"], options["output_dir"], options["aspect_tolerance"]
        )
        if not self._live(generation, stop):
            self._state["phase"] = "cancelled"
            return
        pool = filter_pool_by_keywords(pool, parse_caption_keywords(options["keywords"]))
        if black_keys:
            # A caption freed by retiring its black image is already covered
            # by the requeue; it must not be drawn a second time here.
            pool = [
                (photo, kept)
                for photo, lines in pool
                if (kept := [line for line in lines if caption_match_key(line) not in black_keys])
            ]
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
        jobs = black_jobs + jobs[: options["captions"]]
        self._state["skipped_tag_style"] = stats["tag_style"] + plan_stats.get("tag_style", 0)
        self._state["skipped_short"] = stats["too_short"] + plan_stats.get("too_short", 0)
        self._state["planned"] = len(jobs)

        if options["dry_run"]:
            if options["param_source"] in ("last", "folder"):
                source, source_label = self._image_source_params(options)
                self._state["params"] = merge_params(source, {"model": options["model"]} if options["model"] else {})
                self._state["param_source"] = source_label
            self._state["results"] = [
                {
                    "caption": job["caption"],
                    "real_path": job["real_path"],
                    "requeued_from": job.get("requeued_from"),
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
        save_to = options.get("save_to") or ""
        if save_to:
            extra = {**extra, "donotsave": True}

        def run_job(client: SwarmClient, job: dict) -> list[str]:
            if "_kwargs" in job:
                # A retired black image: its own recorded parameters.
                kwargs = job["_kwargs"]
                if save_to:
                    kwargs = {**kwargs, "extra_params": {**(kwargs.get("extra_params") or {}), "donotsave": True}}
                views = client.generate(**kwargs)
                request = {
                    **(kwargs.get("extra_params") or {}),
                    "prompt": kwargs.get("prompt", ""),
                    "model": kwargs.get("model", ""),
                }
            else:
                views = client.generate(
                    prompt=job["caption"],
                    width=job["width"],
                    height=job["height"],
                    model=params["model"],
                    images=job["images"],
                    extra_params=extra,
                )
                request = {
                    **extra,
                    "prompt": job["caption"],
                    "model": params["model"],
                    "width": job["width"],
                    "height": job["height"],
                }
            if not save_to:
                return views
            return self._store_locally(client, views, save_to, request)

        self._execute(generation, stop, options, jobs, run_job)

    @staticmethod
    def _store_locally(client: SwarmClient, views: list[str], save_to: str, request: dict) -> list[str]:
        """Pull each ``donotsave`` result back (data URI or a download) and write
        it into ``save_to``; returns the local paths in place of SwarmUI views."""
        import base64
        import hashlib
        import time

        request = {key: value for key, value in request.items() if key != "donotsave"}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        digest = hashlib.sha1(str(request.get("prompt", "")).encode("utf-8")).hexdigest()[:8]
        stored = []
        for index, view in enumerate(views, start=1):
            data = base64.b64decode(view.split(",", 1)[1]) if view.startswith("data:") else client.fetch_view(view)
            stored.append(save_generation_locally(data, save_to, f"{stamp}-{digest}-{index}", request))
        return stored

    # ------------------------------------------------------------------ #
    # Black (failed) generations
    # ------------------------------------------------------------------ #

    def scan_black(self, generated_folder: str, max_level: int = DEFAULT_BLACK_MAX_LEVEL) -> dict:
        """Every black image in the generated tree, with what it recorded.

        A generation that failed inside SwarmUI's backend still lands on disk
        as an all-black file, and because it carries the prompt it claims
        that caption as generated: the off-policy builder would never
        suggest it again and could only ever pair the photo with a black
        rejected image. These are what :meth:`requeue_black` retires."""
        if not os.path.isdir(generated_folder):
            return {"ok": False, "error": f"Generated folder not found: {generated_folder}"}
        paths = find_black_images(generated_folder, max_level=max_level)
        images = [describe_black_image(path) for path in paths]
        return {"ok": True, "count": len(images), "images": images}

    def _plan_black_jobs(
        self,
        generated_folder: str,
        model: str = "",
        paths: list[str] | None = None,
        max_level: int = DEFAULT_BLACK_MAX_LEVEL,
    ) -> tuple[list[str], list[dict], list[dict]]:
        """``(black, jobs, unresendable)``: the black images found (or those
        of ``paths`` that really are black and inside the tree), one resend
        job per image that carries a parameter block, and the rest with the
        reason. Nothing is moved here."""
        from modules.util.dpo_regen_util import build_generate_kwargs, resend_params_from_image

        if paths:
            root = os.path.normcase(os.path.abspath(generated_folder)).rstrip(os.sep) + os.sep
            black: list[str] = []
            for path in paths:
                full = os.path.abspath(str(path))
                inside = os.path.normcase(full).startswith(root)
                if inside and os.path.isfile(full) and is_black_image(full, max_level):
                    black.append(full)
        else:
            black = find_black_images(generated_folder, max_level=max_level)

        override = (model or "").strip()
        jobs: list[dict] = []
        unresendable: list[dict] = []
        for path in black:
            params = resend_params_from_image(path)
            params.pop("seed", None)  # a fresh sample, like the run that made it
            prompt = str(params.get("prompt") or "").strip()
            if not params or not prompt:
                unresendable.append({"path": path, "error": "No SwarmUI parameter block in this image"})
                continue
            try:
                kwargs = build_generate_kwargs(params, prompt, {}, override)
            except ValueError as ex:
                unresendable.append({"path": path, "error": str(ex)})
                continue
            jobs.append(
                {
                    "caption": prompt,
                    "real_path": None,
                    "requeued_from": path,
                    "width": kwargs["width"],
                    "height": kwargs["height"],
                    "images": 1,
                    "_kwargs": kwargs,
                }
            )
        return black, jobs, unresendable

    def requeue_black(
        self,
        generated_folder: str,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        model: str = "",
        concurrency: int = DEFAULT_CONCURRENCY,
        paths: list[str] | None = None,
        max_level: int = DEFAULT_BLACK_MAX_LEVEL,
    ) -> dict:
        """Retire the black images and queue a fresh generation for each.

        Every image is moved into ``.black`` beside itself first, whether or
        not it can be resent, so its caption is free again for the builder
        and for an ordinary run even if the resend fails. The resend uses the
        image's own recorded parameter block (prompt, size, sampler, LoRAs,
        model unless overridden) with a fresh seed, the way the original run
        would have made it. ``paths`` limits the requeue to those images;
        each must still be black — a non-black file is never retired.

        An ordinary run does the same on its own when ``auto_requeue_black``
        is on; this is the way to do it without generating anything new."""
        from modules.util.dpo_curation_util import move_to_dot_bucket

        if not os.path.isdir(generated_folder):
            return {"ok": False, "error": f"Generated folder not found: {generated_folder}"}
        with self._black_lock:
            black, jobs, unresendable = self._plan_black_jobs(generated_folder, model, paths, max_level)
            if not black:
                return {"ok": True, "queued": False, "found": 0, "moved": 0, "requeued": 0, "unresendable": []}
            moved = move_to_dot_bucket(black, BLACK_FOLDER)
        if not jobs:
            return {
                "ok": True,
                "queued": False,
                "found": len(black),
                "moved": moved,
                "requeued": 0,
                "unresendable": unresendable,
            }
        options = {
            "mode": "requeue",
            "generated_folder": generated_folder,
            "base_url": base_url,
            "token": token,
            "model": (model or "").strip(),
            "concurrency": max(1, min(int(concurrency), _MAX_CONCURRENCY)),
            "jobs": jobs,
            "dry_run": False,
        }
        result = self._enqueue(options)
        result.update({"found": len(black), "moved": moved, "requeued": len(jobs), "unresendable": unresendable})
        return result

    def _requeue_inner(self, generation: int, stop: threading.Event, options: dict) -> None:
        jobs = options["jobs"]
        self._state["planned"] += len(jobs)
        self._state["param_source"] = "each image's own parameters"
        self._state["phase"] = "generating"

        def run_job(client: SwarmClient, job: dict) -> list[str]:
            return client.generate(**job["_kwargs"])

        self._execute(generation, stop, options, jobs, run_job)

    # ------------------------------------------------------------------ #
    # Shared run loop
    # ------------------------------------------------------------------ #

    def _execute(self, generation: int, stop: threading.Event, options: dict, jobs: list[dict], run_job) -> None:
        """Send every job through a worker pool and record the outcomes.

        ``run_job(client, job)`` returns SwarmUI's ``View/...`` paths. Each
        returned image is checked on disk: a black file is a backend failure
        that SwarmUI reported as success, so it is retired to ``.black`` at
        once. A job whose every image came back black is retried with a fresh
        seed (:data:`_BLACK_RETRIES` times, unless ``auto_requeue_black`` is
        off) and only then counted as a failure — a dead backend then stops
        the run like any other consecutive-failure streak, instead of filling
        the tree with black images that claim their captions."""
        from modules.util.dpo_curation_util import move_to_dot_bucket
        from modules.util.dpo_replace_util import local_path_for_view

        generated_folder = options.get("generated_folder") or ""
        # One client per worker thread: a client owns a session and an HTTP
        # connection, neither of which wants to be shared mid-request.
        thread_state = threading.local()

        def worker_client() -> SwarmClient:
            existing = getattr(thread_state, "client", None)
            if existing is None:
                existing = SwarmClient(options["base_url"], options["token"])
                thread_state.client = existing
            return existing

        def split_black(views: list[str]) -> tuple[list[str], list[str]]:
            good, black = [], []
            for view in views:
                local = local_path_for_view(view, generated_folder) if generated_folder else None
                if local and is_black_image(local):
                    black.append(local)
                else:
                    good.append(view)
            return good, black

        retries = _BLACK_RETRIES if options.get("auto_requeue_black", True) else 0

        def submit(job: dict) -> tuple[dict, list[str], int, int]:
            """``(job, good views, black outputs retired, retries made)``."""
            client = worker_client()
            retired = 0
            for attempt in range(1 + retries):
                good, black = split_black(run_job(client, job))
                if black:
                    move_to_dot_bucket(black, BLACK_FOLDER)
                    retired += len(black)
                if good or not black or not self._live(generation, stop):
                    return job, good, retired, attempt
            return job, [], retired, retries

        failures_since_success = 0
        aborted = False
        # Every job is submitted up front so SwarmUI always has the next
        # request queued behind the one it is running — no idle GPU between
        # batches — while the pool keeps only `concurrency` in flight.
        with ThreadPoolExecutor(max_workers=options["concurrency"]) as pool:
            futures = {pool.submit(submit, job): job for job in jobs}
            for future in as_completed(futures):
                if not self._live(generation, stop):
                    pool.shutdown(wait=False, cancel_futures=True)
                    self._state["phase"] = "cancelled"
                    return
                job = futures[future]
                record = {
                    "caption": job["caption"],
                    "real_path": job.get("real_path"),
                    "requeued_from": job.get("requeued_from"),
                    "size": [job["width"], job["height"]],
                    "exact_size": job.get("exact_size", False),
                    "images": job["images"],
                    "caption_type": classify_caption_type(job["caption"]).value,
                }
                error: str | None = None
                paths: list[str] = []
                try:
                    _job, paths, retired, retried = future.result()
                except CancelledError:
                    continue
                except SwarmError as ex:
                    error = str(ex)
                if error is None:
                    if retired:
                        record["black"] = retired
                        with self._results_lock:
                            self._state["black_retried"] += retried
                    if not paths:
                        error = _BLACK_OUTPUT
                if error is not None:
                    record.update({"ok": False, "error": error})
                    with self._results_lock:
                        failures_since_success += 1
                        self._state["failed"] += 1
                        self._state["done"] += 1
                        self._state["results"].append(record)
                        give_up = failures_since_success >= _MAX_CONSECUTIVE_FAILURES
                    if give_up:
                        self._state["error"] = (
                            f"Stopped after {failures_since_success} consecutive SwarmUI failures: {error}"
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
        if options["param_source"] in ("last", "folder"):
            source, source_label = self._image_source_params(options)
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

    @staticmethod
    def _image_source_params(options: dict) -> tuple[dict, str]:
        """``(source params, label)`` for the sources read from an image's
        parameter block: the generated folder ("last") or ``param_folder``
        ("folder"). Adapter keys are stripped when ``drop_source_loras`` is
        set, so only the run's explicit ``loras`` override applies."""
        if options["param_source"] == "folder":
            folder = options["param_folder"]
            source = read_last_generation_params(folder)
            source_label = _folder_source_label(folder, bool(source))
        else:
            source = read_last_generation_params(options["generated_folder"])
            source_label = "last generation" if source else "manual (no previous generation found)"
        if options.get("drop_source_loras"):
            source = strip_source_loras(source)
        return source, source_label

    def preview_params(self, generated_folder: str) -> dict:
        """What "reuse my last generation" would apply, for the UI to show
        before anything is generated."""
        return self.preview_params_for(generated_folder, False)

    def preview_params_for(self, param_folder: str, drop_source_loras: bool = False) -> dict:
        """The settings the newest image in ``param_folder`` would supply, as
        a run with ``param_source="folder"`` (or "last") would reuse them —
        minus adapters when ``drop_source_loras`` is set — and the name of
        the image they came from (``source_file``, None when there is none)."""
        source_path, source = last_generation_source(param_folder)
        params = sanitize_source_params(source)
        if drop_source_loras:
            params = strip_source_loras(params)
        return {"ok": True, "params": params, "source_file": os.path.basename(source_path) if source_path else None}

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
