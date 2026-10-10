"""Fix & Regenerate: resend images of the current curation group to SwarmUI
with ``<segment:...>`` fixes and add the results to the group.

One run at a time, in the background:

1. ``preview`` reads each image's own SwarmUI parameter block and shows what
   would be sent — the composed prompt, seed, model, size — and which images
   cannot be regenerated (no block, no model).
2. ``start`` sends one request per image with the original seed and every
   recorded parameter, plus the segment tags and the segment-refining
   settings. SwarmUI writes the file into its own output tree; the run finds
   it there (curation source folder first, then the configured generated
   folder) and only downloads it beside the source when neither knows it.
3. Each finished image is handed to ``DPOService.add_images_to_current_group``.
   The group token taken at ``start`` guards against results landing after
   the user moved on: those are kept on disk and reported as dropped.
"""

import logging
import os
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

from modules.util.dpo_regen_util import (
    FixSpec,
    build_generate_kwargs,
    compose_prompt,
    fix_output_path,
    resend_params_from_image,
    sanitize_segment_params,
)
from modules.util.dpo_replace_util import local_path_for_view
from modules.util.swarm_client import DEFAULT_BASE_URL, SwarmClient, SwarmError
from web.backend.services._singleton import SingletonMixin
from web.backend.services.dpo_service import DPOService

logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_CONCURRENCY = 2
_MAX_CONCURRENCY = 4

_NO_PARAMS = "No SwarmUI parameter block in this image"


class DPORegenService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._phase = "idle"
        self._group_index = 0
        self._planned = 0
        self._done = 0
        self._failed = 0
        self._items: dict[str, dict] = {}
        self._added: list[str] = []
        self._dropped: list[str] = []
        self._error: str | None = None
        self._warning: str | None = None

    # ------------------------------------------------------------------ #
    # Preview
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fixes(raw_fixes: list[dict]) -> list[FixSpec]:
        fixes = []
        for raw in raw_fixes or []:
            fix = FixSpec.from_dict(raw if isinstance(raw, dict) else {})
            if fix.target.strip():
                fixes.append(fix)
        return fixes

    def _plan(self, paths: list[str], raw_fixes: list[dict], segment_params: dict, model_override: str) -> dict:
        """Shared by preview and start: the group check plus one item per image."""
        snapshot = DPOService.get_instance().group_snapshot()
        if snapshot is None:
            return {"ok": False, "error": "No active group"}
        if snapshot["mode"] != "selection":
            return {"ok": False, "error": "Fix & Regenerate is only available in selection mode"}
        paths = [str(p) for p in paths if p]
        if not paths:
            return {"ok": False, "error": "No images selected"}
        outside = [p for p in paths if p not in snapshot["images"]]
        if outside:
            return {"ok": False, "error": "Image is not part of the current group"}
        fixes = self._fixes(raw_fixes)
        if not fixes:
            return {"ok": False, "error": "Pick at least one fix"}
        try:
            for fix in fixes:
                compose_prompt("", [fix])
        except ValueError as ex:
            return {"ok": False, "error": str(ex)}
        segment = sanitize_segment_params(segment_params)
        override = (model_override or "").strip()
        sub_prompt_used = any((fix.sub_prompt or "").strip() for fix in fixes)

        items = []
        for path in paths:
            params = resend_params_from_image(path)
            item: dict = {
                "path": path,
                "ok": False,
                "error": None,
                "warnings": [],
                "original_prompt": "",
                "prompt": "",
                "seed": None,
                "model": "",
                "width": None,
                "height": None,
                "steps": params.get("steps"),
                "cfgscale": params.get("cfgscale"),
                "sampler": params.get("sampler"),
                "scheduler": params.get("scheduler"),
                "loras": params.get("loras") or [],
                "negativeprompt": params.get("negativeprompt") or "",
            }
            if not params:
                item["error"] = _NO_PARAMS
                items.append(item)
                continue
            original = str(params.get("prompt") or "")
            item["original_prompt"] = original
            item["prompt"] = compose_prompt(original, fixes)
            item["seed"] = params.get("seed")
            item["width"] = params.get("width")
            item["height"] = params.get("height")
            item["model"] = override or str(params.get("model") or "")
            try:
                kwargs = build_generate_kwargs(params, item["prompt"], segment, override)
            except ValueError as ex:
                item["error"] = str(ex)
                items.append(item)
                continue
            if "seed" not in params:
                item["warnings"].append("No seed recorded — SwarmUI will pick a random one")
            if sub_prompt_used:
                item["warnings"].append("A sub-prompt changes this image's group on a future scan")
            item["ok"] = True
            item["_kwargs"] = kwargs
            items.append(item)
        return {
            "ok": True,
            "group_index": snapshot["group_index"],
            "token": snapshot["token"],
            "source_folder": snapshot["source_folder"],
            "items": items,
            "segment_params": segment,
            "fixes": [fix.__dict__ for fix in fixes],
        }

    def preview(self, paths: list[str], fixes: list[dict], segment_params: dict, model_override: str = "") -> dict:
        plan = self._plan(paths, fixes, segment_params, model_override)
        if not plan.get("ok"):
            return plan
        return {
            "ok": True,
            "group_index": plan["group_index"],
            "items": [{k: v for k, v in item.items() if not k.startswith("_")} for item in plan["items"]],
            "segment_params": plan["segment_params"],
            "fixes": plan["fixes"],
        }

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def start(
        self,
        paths: list[str],
        fixes: list[dict],
        segment_params: dict,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        generated_folder: str = "",
        concurrency: int = DEFAULT_CONCURRENCY,
        model_override: str = "",
    ) -> dict:
        with self._lock:
            if self._running:
                return {"ok": False, "error": "A regeneration is already in progress"}
        plan = self._plan(paths, fixes, segment_params, model_override)
        if not plan.get("ok"):
            return plan
        jobs = [item for item in plan["items"] if item["ok"]]
        if not jobs:
            return {"ok": False, "error": "None of the selected images carry SwarmUI parameters"}
        generated = (generated_folder or "").strip()
        with self._lock:
            if self._running:
                return {"ok": False, "error": "A regeneration is already in progress"}
            self._generation += 1
            generation = self._generation
            self._stop = threading.Event()
            stop = self._stop
            self._running = True
            self._phase = "generating"
            self._group_index = plan["group_index"]
            self._planned = len(plan["items"])
            self._done = self._failed = 0
            self._added = []
            self._dropped = []
            self._error = None
            self._warning = None
            self._items = {}
            for item in plan["items"]:
                if item["ok"]:
                    self._items[item["path"]] = {
                        "status": "queued",
                        "output": None,
                        "error": None,
                        "prompt": item["prompt"],
                    }
                else:
                    self._items[item["path"]] = {
                        "status": "failed",
                        "output": None,
                        "error": item["error"],
                        "prompt": item["prompt"],
                    }
                    self._done += 1
                    self._failed += 1
            options = {
                "token": plan["token"],
                "source_folder": plan["source_folder"] or "",
                "generated_folder": generated if generated and os.path.isdir(generated) else "",
                "base_url": base_url or DEFAULT_BASE_URL,
                "auth": token or "",
                "concurrency": max(1, min(int(concurrency or DEFAULT_CONCURRENCY), _MAX_CONCURRENCY)),
            }
        self._thread = threading.Thread(target=self._run, args=(generation, stop, jobs, options), daemon=True)
        self._thread.start()
        return {"ok": True, "planned": len(jobs), "group_index": plan["group_index"]}

    def cancel(self) -> dict:
        with self._lock:
            self._stop.set()
            if self._running:
                self._phase = "cancelled"
            self._running = False
        return {"ok": True}

    def status(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "running": self._running,
                "phase": self._phase,
                "group_index": self._group_index,
                "planned": self._planned,
                "done": self._done,
                "failed": self._failed,
                "items": {path: dict(item) for path, item in self._items.items()},
                "added": list(self._added),
                "dropped": list(self._dropped),
                "error": self._error,
                "warning": self._warning,
            }

    def _live(self, generation: int, stop: threading.Event) -> bool:
        return self._generation == generation and not stop.is_set()

    def _run(self, generation: int, stop: threading.Event, jobs: list[dict], options: dict) -> None:
        try:
            self._run_inner(generation, stop, jobs, options)
        except Exception as ex:
            logger.exception("Fix & Regenerate run failed")
            if self._generation == generation:
                with self._lock:
                    self._error = str(ex)
                    self._phase = "error"
        finally:
            if self._generation == generation:
                with self._lock:
                    self._running = False
                    if self._phase == "generating":
                        self._phase = "done"

    def _set_item(self, path: str, **fields) -> None:
        with self._lock:
            item = self._items.get(path)
            if item is not None:
                item.update(fields)

    @staticmethod
    def _locate(view: str, source_path: str, options: dict, client: SwarmClient) -> str:
        """The local file behind a ``View/...`` path, downloading as a last resort."""
        folders = [options.get("source_folder") or ""]
        generated = options.get("generated_folder") or ""
        if generated and os.path.normcase(os.path.abspath(generated)) not in {
            os.path.normcase(os.path.abspath(f)) for f in folders if f
        }:
            folders.append(generated)
        for folder in folders:
            if folder and os.path.isdir(folder):
                found = local_path_for_view(view, folder)
                if found:
                    return found
        try:
            data = client.fetch_view(view)
        except SwarmError as ex:
            raise SwarmError(
                f"SwarmUI reported {view} but it was not found under the source or generated folder, "
                f"and downloading it failed: {ex}"
            ) from ex
        target = fix_output_path(source_path)
        with open(target, "wb") as f:
            f.write(data)
        return target

    def _run_inner(self, generation: int, stop: threading.Event, jobs: list[dict], options: dict) -> None:
        thread_state = threading.local()

        def worker_client() -> SwarmClient:
            existing = getattr(thread_state, "client", None)
            if existing is None:
                existing = SwarmClient(options["base_url"], options["auth"])
                thread_state.client = existing
            return existing

        def run_job(job: dict) -> str:
            self._set_item(job["path"], status="running")
            client = worker_client()
            views = client.generate(**job["_kwargs"])
            if not views:
                raise SwarmError("SwarmUI returned no images")
            return self._locate(views[0], job["path"], options, client)

        failures = 0
        dropped = 0
        service = DPOService.get_instance()
        with ThreadPoolExecutor(max_workers=options["concurrency"]) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(futures):
                if not self._live(generation, stop):
                    pool.shutdown(wait=False, cancel_futures=True)
                    return
                job = futures[future]
                path = job["path"]
                try:
                    output = future.result()
                except CancelledError:
                    continue
                except SwarmError as ex:
                    with self._lock:
                        failures += 1
                        self._failed += 1
                        self._done += 1
                        item = self._items.get(path)
                        if item is not None:
                            item.update(status="failed", error=str(ex))
                    if failures >= _MAX_CONSECUTIVE_FAILURES:
                        with self._lock:
                            self._error = f"Stopped after {failures} consecutive SwarmUI failures: {ex}"
                            self._phase = "error"
                        pool.shutdown(wait=False, cancel_futures=True)
                        return
                    continue
                failures = 0
                result = service.add_images_to_current_group([output], options["token"])
                with self._lock:
                    self._done += 1
                    item = self._items.get(path)
                    if not result.get("ok"):
                        dropped += 1
                        self._dropped.append(output)
                        if item is not None:
                            item.update(status="dropped", output=output, error=result.get("error"))
                        self._warning = f"{dropped} image(s) finished after the group changed; they were kept on disk"
                    elif output in result.get("added", []):
                        self._added.append(output)
                        if item is not None:
                            item.update(status="done", output=output)
                    else:
                        reason = result.get("skipped", {}).get(output, "skipped")
                        message = (
                            "Identical to an image already in the dataset"
                            if reason == "identical_to_exported"
                            else "Already in the group"
                            if reason == "already_present"
                            else reason
                        )
                        if item is not None:
                            item.update(status="skipped", output=output, error=message)
