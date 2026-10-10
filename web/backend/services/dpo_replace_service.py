"""Two independent queues: caption-faithful generation and blind human review.

A review group (the pair's old chosen and rejected plus up to
``MAX_IMAGES_PER_PAIR`` fresh images) can yield any number of pairs: the
first replaces the reviewed pair, the rest are added to the dataset. An
image may be chosen in several pairs or rejected in several, never both.
Old images the review drops, and fresh ones SwarmUI saved to its output
tree, are retired in the manifest so the curation tool never offers them
again. Fresh images are requested with ``donotsave`` so new ones never land
in that tree at all.
"""

import base64
import copy
import io
import logging
import os
import random
import shutil
import threading
import time
import uuid
from pathlib import Path

from modules.util import dpo_replace_session as session
from modules.util.dpo_curation_util import (
    RETIRED_KEY,
    compute_pixel_hash,
    find_exported_file,
    load_manifest,
    retire_images,
    save_manifest,
)
from modules.util.dpo_replace_util import local_path_for_view, provenance_from_params
from modules.util.swarm_client import DEFAULT_BASE_URL, SwarmClient, SwarmError
from modules.util.swarm_generate_util import (
    bucket_size_for_photo,
    read_last_generation_params,
    sanitize_source_params,
)
from web.backend.services._dpo_session_lock import dpo_session_admission
from web.backend.services._singleton import SingletonMixin

from PIL import Image

logger = logging.getLogger(__name__)

#: Fresh images generated for each pair under review.
MAX_IMAGES_PER_PAIR = 10


class DPOReplacePairsService(SingletonMixin):
    def __init__(self):
        self._lock = threading.RLock()
        self._active = False
        self._concept_dir = ""
        self._ledger = None
        self._store = None
        self._run_thread = None
        self._running = False
        self._pause = True
        self._close = False
        self._token = ""
        self._error = None

    def concepts(self):
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs
        from web.backend.services.dpo_service import DPOService

        try:
            roots = []
            for chosen, _ in dpo_concept_pattern_dirs(DPOService.get_instance()._resolve_concepts()):
                root = os.path.abspath(os.path.dirname(chosen))
                if (Path(root) / "manifest.json").is_file() and root not in roots:
                    roots.append(root)
            return {"ok": True, "concepts": roots}
        except Exception as ex:
            return {"ok": True, "concepts": [], "warning": str(ex)}

    def _resolution(self, folder):
        from web.backend.services.config_service import ConfigService
        from web.backend.services.dpo_service import DPOService

        config = ConfigService.get_instance().get_config_for_training()
        value = config.resolution
        for concept in DPOService.get_instance()._resolve_concepts():
            if os.path.normcase(os.path.abspath(concept.path)) == os.path.normcase(folder):
                if concept.image.enable_resolution_override:
                    value = concept.image.resolution_override
                break
        return max(int(part.strip()) for part in str(value).split(","))

    def _save(self):
        self._store.save(self._ledger)

    def _new_ledger(self, count, resolution):
        items = []
        for entry in load_manifest(self._concept_dir).get("pairs", []):
            chosen = find_exported_file(str(Path(self._concept_dir) / "chosen"), entry.get("chosen_file", ""))
            try:
                created = session.creation_time(chosen) if chosen else time.time()
            except OSError:
                created = time.time()
            items.append(
                {
                    "id": uuid.uuid4().hex,
                    "pair_id": entry["pair_id"],
                    "created": created,
                    "state": "queued",
                    "candidates": [],
                    "best": None,
                    "error": None,
                }
            )
        items = session.weighted_shuffle(items, random.Random())
        return {
            "version": 1,
            "id": uuid.uuid4().hex,
            "images_per_pair": count,
            "resolution": resolution,
            "order": [item["id"] for item in items],
            "items": {item["id"]: item for item in items},
            "current": None,
            "undo": [],
            "settings": {},
        }

    @dpo_session_admission
    def start(
        self,
        concept_dir,
        generated_folder,
        base_url=DEFAULT_BASE_URL,
        token="",
        images_per_pair=2,
        new_pass=False,
        restore_only=False,
    ):
        with self._lock:
            try:
                if self._active or self._running:
                    return {"ok": False, "error": "A Replace Pairs session is already open"}
                folder = os.path.abspath(concept_dir)
                if not all((Path(folder) / name).exists() for name in ("manifest.json", "chosen", "rejected")):
                    raise ValueError("Select an exported DPO concept with manifest.json, chosen/ and rejected/")
                if not isinstance(images_per_pair, int) or not 1 <= images_per_pair <= MAX_IMAGES_PER_PAIR:
                    raise ValueError(f"Images per pair must be 1 to {MAX_IMAGES_PER_PAIR}")
                store = session.SessionStore(folder)
                ledger = store.load()
                if restore_only and ledger is None:
                    return {"ok": True, "active": False}
                if (
                    new_pass
                    and ledger
                    and any(i["state"] not in ("applied", "skipped") for i in ledger["items"].values())
                ):
                    raise ValueError("Finish the current pass before starting a new one")
                self._concept_dir, self._store = folder, store
                if ledger is None or new_pass:
                    ledger = self._new_ledger(images_per_pair, self._resolution(folder))
                self._ledger = ledger
                self._token = token
                if not restore_only:
                    ledger["settings"].update({"generated_folder": generated_folder, "base_url": base_url})
                for item in ledger["items"].values():
                    if item["state"] == "generating":
                        item["state"] = "queued"
                self._active, self._pause, self._close = True, True, False
                self._error = None
                self._save()
                if not restore_only:
                    result = self.resume(ledger["id"], token)
                    if not result["ok"]:
                        self._error = result["error"]
                return self.status()
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def status(self):
        with self._lock:
            if not self._ledger or not self._active:
                return {"ok": True, "active": False}
            items = list(self._ledger["items"].values())
            counts = {
                state: sum(i["state"] == state for i in items)
                for state in ("queued", "generating", "ready", "applied", "skipped", "failed")
            }
            return {
                "ok": True,
                "active": True,
                "session_id": self._ledger["id"],
                "concept_dir": self._concept_dir,
                "generated_folder": self._ledger["settings"].get("generated_folder", ""),
                "base_url": self._ledger["settings"].get("base_url", DEFAULT_BASE_URL),
                "running": self._running,
                "paused": self._pause,
                "closing": self._close,
                "total": len(items),
                "counts": counts,
                "complete": all(i["state"] in ("applied", "skipped") for i in items),
                "params": self._ledger["settings"].get("params", {}),
                "images_per_pair": self._ledger["images_per_pair"],
                "images": sum(sum(c["role"] == "fresh" for c in i["candidates"]) for i in items),
                "error": self._error,
                "can_undo": bool(self._ledger["undo"]),
                "errors": [
                    {"group_id": i["id"], "pair_id": i["pair_id"], "error": i["error"]}
                    for i in items
                    if i["state"] == "failed" and i.get("error")
                ],
            }

    def _check_session(self, session_id):
        if not self._active or not self._ledger or session_id != self._ledger["id"]:
            raise ValueError("This session is no longer active")

    def resume(self, session_id, token=""):
        with self._lock:
            try:
                self._check_session(session_id)
                if self._running:
                    raise ValueError("Wait for the current image to finish")
                params = sanitize_source_params(
                    read_last_generation_params(self._ledger["settings"]["generated_folder"])
                )
                if not params.get("model"):
                    raise ValueError("No last-generation settings found. Generate an image in SwarmUI first")
                self._ledger["settings"]["params"] = params
                self._token = token
                self._pause, self._close, self._error = False, False, None
                self._save()
                self._ensure_worker()
                return {"ok": True}
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def pause(self, session_id, close=False):
        with self._lock:
            try:
                self._check_session(session_id)
                self._pause = True
                self._close = close
                if close and not self._running:
                    self._active = False
                self._save()
                return {"ok": True}
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def cancel(self):
        # Also used by application/test teardown; never deletes durable queues.
        with self._lock:
            return self.pause(self._ledger["id"], close=True) if self._active else {"ok": True}

    def _source(self, item):
        entry = next((e for e in load_manifest(self._concept_dir)["pairs"] if e["pair_id"] == item["pair_id"]), None)
        if entry is None:
            raise ValueError("Pair is no longer present in the manifest")
        paths = {}
        for side in ("chosen", "rejected"):
            path = find_exported_file(str(Path(self._concept_dir) / side), entry[side + "_file"])
            if not path:
                raise ValueError(f"The pair's {side} image is missing")
            paths[side] = path
        caption_path = str(Path(paths["chosen"]).with_suffix(".txt"))
        caption = session.caption_text(caption_path)
        return entry, paths, caption

    def _reset(self, item):
        item.update({"state": "queued", "candidates": [], "best": None, "error": None})
        item.pop("caption", None)
        if self._ledger["current"] == item["id"]:
            self._ledger["current"] = None
        self._ensure_worker()

    def _ensure_worker(self):
        # Called while holding the service lock. A naturally drained queue
        # remains enabled; caption changes/retries can add work to it again.
        if (
            self._active
            and not self._running
            and not self._pause
            and any(i["state"] == "queued" for i in self._ledger["items"].values())
        ):
            self._running = True
            self._run_thread = threading.Thread(target=self._run, daemon=True)
            self._run_thread.start()

    def _validate(self, item):
        entry, paths, caption = self._source(item)
        if "caption" in item and caption != item["caption"]:
            self._reset(item)
            self._save()
            return False
        if "caption" in item and (
            entry != item["entry"]
            or any(
                paths[side] != item["originals"][side] or session.fingerprint(paths[side]) != item["fingerprints"][side]
                for side in ("chosen", "rejected")
            )
        ):
            raise ValueError("The original pair changed outside this session; retry to reload it")
        return True

    def _stage(self, item, source, role, caption, params=None, data=None):
        candidate_id = uuid.uuid4().hex
        if data is None:
            suffix = Path(source).suffix.lower()
        else:
            # Derive the extension from the downloaded header, not a URL that
            # may contain escaped characters or a cache-busting query string.
            with Image.open(io.BytesIO(data)) as downloaded:
                suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(downloaded.format)
                if suffix is None:
                    suffix = next(
                        (ext for ext, fmt in Image.registered_extensions().items() if fmt == downloaded.format), None
                    )
            if suffix is None:
                raise ValueError("SwarmUI returned an unsupported image format")
        target = self._store.root / "candidates" / self._ledger["id"] / (candidate_id + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if data is None:
            shutil.copy2(source, target)
        else:
            target.write_bytes(data)
        with Image.open(target) as img:
            if role == "fresh" and list(img.size) != item["size"]:
                raise ValueError("SwarmUI returned different dimensions from the requested trainer bucket")
            img.verify()
        session.write_caption(target.with_suffix(".txt"), caption)
        candidate = {
            "id": candidate_id,
            "path": str(target),
            "source": str(source) if data is None else str(target),
            "role": role,
            "params": params or {},
            "generated_at": time.time(),
        }
        item["candidates"].append(candidate)
        return candidate

    def _prepare(self, item):
        entry, paths, caption = self._source(item)
        with Image.open(paths["chosen"]) as img:
            size = bucket_size_for_photo(*img.size, target_resolution=self._ledger["resolution"], quantization=64)
        item.update(
            {
                "entry": copy.deepcopy(entry),
                "originals": paths,
                "caption": caption,
                "size": list(size),
                "fingerprints": {side: session.fingerprint(path) for side, path in paths.items()},
            }
        )
        for side, path in paths.items():
            self._stage(item, path, side, caption)

    def _run(self):
        failures = 0
        try:
            while True:
                with self._lock:
                    if self._pause:
                        break
                    item = next(
                        (
                            self._ledger["items"][key]
                            for key in self._ledger["order"]
                            if self._ledger["items"][key]["state"] == "queued"
                        ),
                        None,
                    )
                    if item is None:
                        break
                    try:
                        if not self._validate(item):
                            continue
                        if "caption" not in item:
                            self._prepare(item)
                        item["state"] = "generating"
                        settings = copy.deepcopy(self._ledger["settings"])
                        params = settings["params"]
                        caption = item["caption"]
                        self._save()
                    except Exception as ex:
                        self._fail(item, ex)
                        continue
                # No lock during the remote request: review and pause remain usable.
                try:
                    client = SwarmClient(settings["base_url"], self._token)
                    views = client.generate(
                        prompt=caption,
                        width=item["size"][0],
                        height=item["size"][1],
                        model=params["model"],
                        images=1,
                        seed=-1,
                        # Kept out of SwarmUI's output tree: the curation tool
                        # scans it, and these images belong to this review only.
                        extra_params={**{k: v for k, v in params.items() if k != "model"}, "donotsave": True},
                    )
                    if len(views) != 1:
                        raise ValueError("SwarmUI did not return exactly one image")
                    path = None
                    data = None
                    if views[0].startswith("data:"):
                        data = base64.b64decode(views[0].split(",", 1)[1])
                    else:
                        path = local_path_for_view(views[0], settings["generated_folder"])
                    if not path and data is None:
                        try:
                            data = client.fetch_view(views[0])
                        except SwarmError as ex:
                            raise SwarmError(
                                "The generated image was not available locally and downloading it from SwarmUI failed: "
                                f"{ex}"
                            ) from ex
                    with self._lock:
                        if not self._validate(item):
                            continue
                        self._stage(item, path or views[0], "fresh", caption, params, data=data)
                        count = sum(c["role"] == "fresh" for c in item["candidates"])
                        if count == self._ledger["images_per_pair"]:
                            random.shuffle(item["candidates"])
                            item["state"] = "ready"
                        else:
                            item["state"] = "queued"
                        failures = 0
                        self._save()
                except Exception as ex:
                    with self._lock:
                        self._fail(item, ex)
                        failures += 1
                        if failures >= 3:
                            self._pause = True
                            self._error = f"Generation paused after three consecutive failures: {ex}"
        except Exception as ex:
            logger.exception("Replace Pairs worker failed")
            with self._lock:
                self._pause, self._error = True, str(ex)
        finally:
            with self._lock:
                self._running = False
                if self._close:
                    self._active = False
                self._ensure_worker()

    def _fail(self, item, ex):
        item["state"], item["error"] = "failed", str(ex)
        if self._ledger["current"] == item["id"]:
            self._ledger["current"] = None
        self._save()

    def retry(self, session_id, group_id):
        with self._lock:
            try:
                self._check_session(session_id)
                item = self._ledger["items"][group_id]
                if item["state"] != "failed":
                    raise ValueError("Only failed pairs can be retried")
                # Keep successful partial outputs after a transient SwarmUI
                # failure. Changed sources or missing files require a rebuild.
                try:
                    reusable = (
                        "caption" in item
                        and self._validate(item)
                        and len(item["candidates"]) >= 2
                        and all(Path(c["path"]).is_file() for c in item["candidates"])
                    )
                except (OSError, ValueError):
                    reusable = False
                if reusable:
                    full = sum(c["role"] == "fresh" for c in item["candidates"]) >= self._ledger["images_per_pair"]
                    item.update({"state": "ready" if full else "queued", "error": None, "best": None})
                    self._ensure_worker()
                else:
                    self._reset(item)
                self._save()
                return {"ok": True}
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def current(self):
        with self._lock:
            if not self._active:
                return {"ok": True, "group": None}
            for key in ([self._ledger["current"]] if self._ledger["current"] else []) + self._ledger["order"]:
                item = self._ledger["items"][key]
                if item["state"] != "ready":
                    continue
                try:
                    if not self._validate(item):
                        continue
                    if any(not Path(c["path"]).is_file() for c in item["candidates"]):
                        raise ValueError("A saved candidate is missing; retry to regenerate this group")
                except Exception as ex:
                    self._fail(item, ex)
                    continue
                if self._ledger["current"] != key:
                    self._ledger["current"] = key
                    self._save()
                return {"ok": True, "group": self._payload(item)}
            return {"ok": True, "group": None}

    def _payload(self, item):
        return {
            "session_id": self._ledger["id"],
            "group_id": item["id"],
            "caption": item["caption"],
            "best_id": item["best"],
            "size": item["size"],
            "images": [
                {
                    "id": c["id"],
                    "url": f"/dpo/replace/image/{self._ledger['id']}/{item['id']}/{c['id']}",
                    "current_rejected": c["role"] == "rejected",
                }
                for c in item["candidates"]
            ],
        }

    def image_path(self, session_id, group_id, candidate_id):
        with self._lock:
            self._check_session(session_id)
            item = self._ledger["items"][group_id]
            return next(c["path"] for c in item["candidates"] if c["id"] == candidate_id)

    def _group(self, session_id, group_id):
        self._check_session(session_id)
        if group_id != self._ledger["current"]:
            raise ValueError("This review group is no longer current")
        item = self._ledger["items"][group_id]
        if item["state"] != "ready":
            raise ValueError("This group is no longer ready")
        if not self._validate(item):
            raise ValueError("Caption changed; this group has returned to Queue 1")
        return item

    def select_best(self, session_id, group_id, best_id=None):
        with self._lock:
            try:
                item = self._group(session_id, group_id)
                if best_id is not None and not any(c["id"] == best_id for c in item["candidates"]):
                    raise ValueError("Image is not in this group")
                item["best"] = best_id
                self._save()
                return {"ok": True, "group": self._payload(item)}
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def apply(self, session_id, group_id, best_id, worst_id, override_reversal=False):
        with self._lock:
            try:
                item = self._group(session_id, group_id)
                candidates = {c["id"]: c for c in item["candidates"]}
                if (
                    best_id != item["best"]
                    or best_id == worst_id
                    or best_id not in candidates
                    or worst_id not in candidates
                ):
                    raise ValueError("Select two different images from the current group, BEST first")
                return self._apply_checked(item, [(candidates[best_id], candidates[worst_id])], override_reversal)
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def apply_pairs(self, session_id, group_id, pairs, override_reversal=False):
        """Save any number of pairs from the current group. ``pairs`` is a
        list of ``(chosen_id, rejected_id)``; the first replaces the reviewed
        pair and the rest are added. An image may be chosen in several pairs
        or rejected in several, but never both."""
        with self._lock:
            try:
                item = self._group(session_id, group_id)
                candidates = {c["id"]: c for c in item["candidates"]}
                pairs = [(str(chosen), str(rejected)) for chosen, rejected in pairs or []]
                if not pairs:
                    raise ValueError("Select at least one pair")
                if any(c not in candidates or r not in candidates for c, r in pairs):
                    raise ValueError("Image is not in this group")
                if any(c == r for c, r in pairs):
                    raise ValueError("A pair needs two different images")
                if len(set(pairs)) != len(pairs):
                    raise ValueError("The same pair was selected twice")
                both = {c for c, _r in pairs} & {r for _c, r in pairs}
                if both:
                    raise ValueError(
                        "An image cannot be chosen in one pair and rejected in another; "
                        "the dataset would then teach both directions"
                    )
                return self._apply_checked(item, [(candidates[c], candidates[r]) for c, r in pairs], override_reversal)
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def _apply_checked(self, item, pairs, override_reversal):
        # The old pair turned around is legitimate but easy to click by accident.
        reversed_pair = any(c["role"] == "rejected" and r["role"] == "chosen" for c, r in pairs)
        if reversed_pair and not override_reversal:
            return {"ok": False, "requires_reversal_confirmation": True}
        self._commit_pairs(item, pairs)
        return {"ok": True, "pairs": len(pairs)}

    def _side_fields(self, entry, side, candidate, item, target):
        origin = candidate["role"]
        entry[side + "_file"] = target.name
        entry[side + "_sha256"] = session.fingerprint(target)
        entry[side + "_size"] = target.stat().st_size
        entry[side + "_source"] = (
            candidate.get("source", candidate["path"]) if origin == "fresh" else item["entry"].get(origin + "_source")
        )
        entry[side + "_provenance"] = (
            provenance_from_params(candidate["params"], candidate.get("generated_at")).to_dict()
            if origin == "fresh"
            else item["entry"].get(origin + "_provenance")
        )

    def _in_swarm_tree(self, candidate):
        """A fresh image SwarmUI saved to its output tree (before generation
        asked it not to): the curation tool would find it there."""
        source = Path(candidate.get("source") or candidate["path"])
        try:
            source.relative_to(self._store.root)
            return False
        except ValueError:
            return source != Path(candidate["path"])

    def _retirees(self, item, used_ids):
        """What this group puts out of reach of every DPO tool: the old
        pair's images the review dropped, and fresh images that sit in
        SwarmUI's output tree but went unused."""
        records = []
        for candidate in item["candidates"]:
            if candidate["id"] in used_ids:
                continue
            role = candidate["role"]
            if role == "fresh" and not self._in_swarm_tree(candidate):
                continue
            path = candidate["path"]
            records.append(
                {
                    "source": item["entry"].get(role + "_source") if role != "fresh" else candidate.get("source"),
                    "sha256": session.fingerprint(path),
                    "size": os.path.getsize(path) if os.path.isfile(path) else None,
                    "pixel_hash": compute_pixel_hash(path) if os.path.isfile(path) else None,
                    "reason": "replace_pairs_dropped" if role != "fresh" else "replace_pairs_unused",
                    "pair_id": item["pair_id"],
                }
            )
        return records

    def _commit_pairs(self, item, pairs):
        manifest = load_manifest(self._concept_dir)
        entry = next(e for e in manifest["pairs"] if e["pair_id"] == item["pair_id"])
        retired_before = copy.deepcopy(manifest.get(RETIRED_KEY)) if RETIRED_KEY in manifest else None
        (chosen, rejected), extra = pairs[0], pairs[1:]
        targets = {}
        for side, candidate in (("chosen", chosen), ("rejected", rejected)):
            old = Path(item["originals"][side])
            targets[side] = str(old.with_suffix(Path(candidate["path"]).suffix))
            if targets[side] != str(old) and Path(targets[side]).exists():
                raise ValueError("Replacement filename already exists; no files were changed")
        # Added pairs get the next free ids and the export's own file names.
        next_id = max((e["pair_id"] for e in manifest["pairs"]), default=-1) + 1
        added_targets = []
        for offset, (added_chosen, added_rejected) in enumerate(extra):
            pair_id = next_id + offset
            name = f"pair_{pair_id:04d}"
            files = {
                side: str(Path(self._concept_dir) / side / (name + Path(candidate["path"]).suffix.lower()))
                for side, candidate in (("chosen", added_chosen), ("rejected", added_rejected))
            }
            if any(Path(p).exists() or Path(p).with_suffix(".txt").exists() for p in files.values()):
                raise ValueError(f"Files for new pair {name} already exist; no files were changed")
            added_targets.append((pair_id, files))
        paths = list(item["originals"].values()) + list(targets.values())
        paths += [p for _pair_id, files in added_targets for p in files.values()]
        paths = list(dict.fromkeys(paths + [str(Path(p).with_suffix(".txt")) for p in paths]))
        retirees = self._retirees(item, {c["id"] for pair in pairs for c in pair})
        journal = self._store.begin(paths)
        try:
            for side, candidate in (("chosen", chosen), ("rejected", rejected)):
                target = Path(targets[side])
                shutil.copy2(candidate["path"], target)
                session.write_caption(target.with_suffix(".txt"), item["caption"])
                old = Path(item["originals"][side])
                if old != target:
                    old.unlink()
                self._side_fields(entry, side, candidate, item, target)
            entry["prompt"] = item["caption"]
            entry.setdefault("history", []).append(
                {
                    "at": time.time(),
                    "action": "blind_replacement",
                    "caption": item["caption"],
                    "best_id": chosen["id"],
                    "worst_id": rejected["id"],
                    **({"added_pairs": [pair_id for pair_id, _files in added_targets]} if added_targets else {}),
                }
            )
            added = []
            for (pair_id, files), (added_chosen, added_rejected) in zip(added_targets, extra, strict=True):
                new = {
                    key: copy.deepcopy(value)
                    for key, value in item["entry"].items()
                    if not key.startswith(("chosen_", "rejected_")) and key not in ("pair_id", "history")
                }
                new["pair_id"] = pair_id
                for side, candidate in (("chosen", added_chosen), ("rejected", added_rejected)):
                    target = Path(files[side])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate["path"], target)
                    self._side_fields(new, side, candidate, item, target)
                # As the export writes a pair: the trainer reads the caption from the chosen side.
                session.write_caption(Path(files["chosen"]).with_suffix(".txt"), item["caption"])
                new["prompt"] = item["caption"]
                new["history"] = [
                    {
                        "at": time.time(),
                        "action": "blind_replacement_added",
                        "from_pair": item["pair_id"],
                        "caption": item["caption"],
                        "best_id": added_chosen["id"],
                        "worst_id": added_rejected["id"],
                    }
                ]
                manifest["pairs"].append(new)
                added.append(new)
            retire_images(manifest, retirees)
            save_manifest(self._concept_dir, manifest)
            backups = {p: b for p, b in journal["backups"].items() if p in paths}
            self._ledger["undo"].append(
                {
                    "group_id": item["id"],
                    "action": "apply",
                    "backups": backups,
                    "entry": item["entry"],
                    "after_entry": copy.deepcopy(entry),
                    "added": copy.deepcopy(added),
                    "retired_changed": True,
                    "retired_before": retired_before,
                    "after": {p: session.fingerprint(p) for p in paths},
                }
            )
            item["state"] = "applied"
            self._ledger["current"] = None
            self._save()
            self._store.commit(journal)
        except Exception:
            self._store.recover()
            self._ledger = self._store.load()
            raise

    def _retire_skipped(self, item):
        """A skipped group leaves the pair alone; fresh images SwarmUI saved
        to its output tree are still retired. Returns the undo fields."""
        records = self._retirees(item, {c["id"] for c in item["candidates"] if c["role"] != "fresh"})
        if not records:
            return {}
        manifest = load_manifest(self._concept_dir)
        before = copy.deepcopy(manifest.get(RETIRED_KEY)) if RETIRED_KEY in manifest else None
        if not retire_images(manifest, records):
            return {}
        save_manifest(self._concept_dir, manifest)
        return {"retired_changed": True, "retired_before": before}

    def _restore_retired(self, manifest, record):
        if not record.get("retired_changed"):
            return
        if record.get("retired_before") is None:
            manifest.pop(RETIRED_KEY, None)
        else:
            manifest[RETIRED_KEY] = record["retired_before"]

    def skip(self, session_id, group_id):
        with self._lock:
            try:
                self._check_session(session_id)
                item = self._ledger["items"][group_id]
                if item["state"] != "failed":
                    item = self._group(session_id, group_id)
                retired = self._retire_skipped(item) if item.get("candidates") and "caption" in item else {}
                self._ledger["undo"].append({"group_id": group_id, "action": "skip", "state": item["state"], **retired})
                item["state"] = "skipped"
                if self._ledger["current"] == group_id:
                    self._ledger["current"] = None
                self._save()
                return {"ok": True}
            except Exception as ex:
                return {"ok": False, "error": str(ex)}

    def undo(self, session_id):
        with self._lock:
            try:
                self._check_session(session_id)
                if not self._ledger["undo"]:
                    raise ValueError("Nothing to undo")
                record = self._ledger["undo"][-1]
                item = self._ledger["items"][record["group_id"]]
                paths = list(record.get("backups", {}))
                manifest = load_manifest(self._concept_dir)
                added_ids = {e["pair_id"] for e in record.get("added", [])}
                if record["action"] == "apply":
                    entry = next(e for e in manifest["pairs"] if e["pair_id"] == item["pair_id"])
                    current_added = {e["pair_id"]: e for e in manifest["pairs"] if e["pair_id"] in added_ids}
                    if (
                        entry != record["after_entry"]
                        or any(current_added.get(e["pair_id"]) != e for e in record.get("added", []))
                        or any(session.fingerprint(p) != h for p, h in record["after"].items())
                    ):
                        raise ValueError("The saved pair changed outside this session; Undo cannot overwrite it")
                journal = self._store.begin(paths)
                try:
                    if record["action"] == "apply":
                        session.restore_files(record["backups"])
                        entry.clear()
                        entry.update(record["entry"])
                        manifest["pairs"] = [e for e in manifest["pairs"] if e["pair_id"] not in added_ids]
                    if record.get("retired_changed"):
                        self._restore_retired(manifest, record)
                    if record["action"] == "apply" or record.get("retired_changed"):
                        save_manifest(self._concept_dir, manifest)
                    item["state"] = record.get("state", "ready")
                    item["best"] = None
                    self._ledger["current"] = item["id"] if item["state"] == "ready" else None
                    self._ledger["undo"].pop()
                    self._save()
                    self._store.commit(journal)
                except Exception:
                    self._store.recover()
                    self._ledger = self._store.load()
                    raise
                return {"ok": True}
            except Exception as ex:
                return {"ok": False, "error": str(ex)}
