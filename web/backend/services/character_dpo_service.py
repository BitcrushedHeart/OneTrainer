"""Character DPO: per-character dataset overview, pair import from the
shared DPO concepts, finalize (split + masks + concept file) and queue entry
building. The pure pieces live in ``modules.util.dpo_character_util``.
"""

import logging
import os
import shutil
import threading
import time
from contextlib import suppress

from modules.util import dpo_character_util as cu
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.dpo_curation_util import compute_file_sha256, export_single_pair, load_manifest, save_manifest
from modules.util.path_util import supported_image_extensions
from web.backend.paths import PROJECT_ROOT
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

#: Where finished character LoRAs are looked for when a request names no
#: folders: the trainer's own output folder, then the folder they are moved
#: to once done. ``;``-separated; every route's ``models_dir`` takes the same
#: form and :func:`cu.resolve_lora` searches the folders in order.
DEFAULT_MODELS_DIR = cu.MODELS_DIR_SEPARATOR.join(
    (os.path.join(PROJECT_ROOT, "models"), "E:/AI/Data/Models/Lora/SoRealPeople!")
)
DEFAULT_MASK_MODEL = "Rembg-Human"
DEFAULT_MASK_OPTIONS = {"threshold": 0.3, "smooth": 5, "expand": 10, "alpha": 1.0}
#: Images SwarmUI generates per caption when the one-click flow is not told otherwise.
DEFAULT_IMAGES_PER_CAPTION = 2
#: SwarmUI rescans its model folders on a timer; a LoRA copied in a moment
#: ago may not be listed on the first ask. One more try after this pause.
SWARM_LIST_RETRY_DELAY = 2.0


def _count_captions(folder: str) -> int:
    """``.txt`` sidecars under ``folder`` (dot-folders skipped)."""
    if not os.path.isdir(folder):
        return 0
    count = 0
    for _root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        count += sum(1 for f in files if f.lower().endswith(".txt"))
    return count


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_within(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([_norm(path), _norm(folder)]) == _norm(folder)
    except ValueError:
        return False


def _count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    supported = supported_image_extensions()
    count = 0
    for _root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        count += sum(1 for f in files if os.path.splitext(f)[1].lower() in supported and not f.endswith(cu.MASK_SUFFIX))
    return count


def _read_text(path: str | None) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with suppress(OSError, UnicodeDecodeError), open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
    return ""


def _provenance_dict(entry: dict, side: str, path: str) -> dict | None:
    stored = entry.get(f"{side}_provenance")
    if isinstance(stored, dict):
        return stored
    if os.path.splitext(path)[1].lower() != ".png":
        return None
    try:
        from modules.util.dpo_replace_util import provenance_of_file

        prov = provenance_of_file(path)
    except Exception:
        return None
    if prov.model or prov.loras:
        return prov.to_dict()
    return None


class CharacterDPOService(SingletonMixin):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: Finalize jobs by character key (normalized root + name).
        self._jobs: dict[str, dict] = {}
        #: Concept dirs seen by the last import scan, for the image endpoint.
        self._scanned_concept_dirs: set[str] = set()
        #: Character roots seen by any request, for the image endpoint.
        self._roots: set[str] = set()
        #: The one-click "prepare" batch (finalize for several characters).
        self._prepare: dict = self._idle_prepare()

    @staticmethod
    def _idle_prepare() -> dict:
        return {
            "running": False,
            "current": None,
            "done": 0,
            "total": 0,
            "stage": "idle",
            "message": "",
            "error": None,
            "results": {},
            # The per-character finalize job in flight (not reported as such).
            "job": None,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _models_dir(models_dir: str | None) -> str:
        return models_dir or DEFAULT_MODELS_DIR

    @staticmethod
    def _job_key(root: str, name: str) -> str:
        return _norm(os.path.join(root, name))

    def _remember_root(self, root: str) -> None:
        if root:
            with self._lock:
                self._roots.add(_norm(root))

    @staticmethod
    def _queue_entries() -> list:
        try:
            from web.backend.services.queue_service import QueueService

            service = QueueService.get_instance()
            manager = getattr(service, "manager", None) or getattr(service, "_manager", None)
            return list(getattr(manager, "entries", None) or [])
        except Exception as ex:  # queue unavailable (tests, broken queue.json)
            logger.debug("queue entries unavailable: %s", ex)
            return []

    @staticmethod
    def _global_concepts() -> list[dict]:
        """Concept dicts from the active config (its list, else its file)."""
        try:
            from web.backend.services.config_service import ConfigService

            config = ConfigService.get_instance().config
        except Exception as ex:
            logger.debug("config unavailable: %s", ex)
            return []
        concepts = getattr(config, "concepts", None)
        if isinstance(concepts, list) and concepts:
            return [concept.to_dict() if hasattr(concept, "to_dict") else dict(concept) for concept in concepts]
        file_name = getattr(config, "concept_file_name", "") or ""
        path = file_name if os.path.isabs(file_name) else os.path.join(PROJECT_ROOT, file_name)
        if file_name and os.path.isfile(path):
            with suppress(OSError, ValueError):
                return cu.load_concept_dicts(path)
        return []

    @staticmethod
    def _global_workspace_dir() -> str:
        try:
            from web.backend.services.config_service import ConfigService

            return ConfigService.get_instance().config.workspace_dir or "workspace/run"
        except Exception:
            return "workspace/run"

    @staticmethod
    def _swarm_lora_names(swarm_url: str | None, swarm_token: str | None) -> list[str]:
        if not swarm_url:
            return []
        try:
            from modules.util.swarm_client import SwarmClient

            return SwarmClient(swarm_url, swarm_token or None, timeout=10).list_models(subtype="LoRA")
        except Exception as ex:
            logger.debug("SwarmUI LoRA listing failed: %s", ex)
            return []

    @staticmethod
    def _swarm_name_for(stem: str, swarm_loras: list[str]) -> str | None:
        wanted = stem.lower()
        for entry in swarm_loras:
            base = os.path.splitext(os.path.basename(str(entry).replace("\\", "/")))[0]
            if base.lower() == wanted:
                return entry
        return None

    # ------------------------------------------------------------------ #
    # List
    # ------------------------------------------------------------------ #

    def list_characters(
        self,
        root: str,
        models_dir: str | None = None,
        swarm_url: str | None = None,
        swarm_token: str | None = None,
    ) -> dict:
        models_dir = self._models_dir(models_dir)
        self._remember_root(root)
        characters = []
        if not root or not os.path.isdir(root):
            return {"root": root, "models_dir": models_dir, "characters": characters}
        entries = self._queue_entries()
        swarm_loras = self._swarm_lora_names(swarm_url, swarm_token)
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if name.startswith(".") or not os.path.isdir(os.path.join(folder, "train")):
                continue
            characters.append(self._character_summary(root, name, models_dir, entries, swarm_loras))
        return {"root": root, "models_dir": models_dir, "characters": characters}

    def _character_summary(self, root: str, name: str, models_dir: str, entries: list, swarm_loras: list[str]) -> dict:
        folder = cu.character_dir(root, name)
        lora_path, kind = cu.resolve_lora(models_dir, name)
        summary = {
            "name": name,
            "path": folder,
            "train_count": _count_images(os.path.join(folder, "train")),
            "val_count": _count_images(os.path.join(folder, "val")),
            "generated_count": _count_images(os.path.join(folder, "dpo", "generated")),
            "lora": {
                "path": lora_path,
                # Which of the ``models_dir`` folders it was found in.
                "dir": os.path.dirname(lora_path) if lora_path else None,
                "kind": kind,
                "swarm_name": self._swarm_name_for(cu.lora_stem(lora_path), swarm_loras) if lora_path else None,
            },
            "queue_entry": None,
            "finalized": False,
            "concepts_file": None,
        }
        for pair_kind in cu.PAIR_KINDS:
            pair_folder = cu.pair_dir(root, name, pair_kind)
            counts = {"train": 0, "val": 0, "unsplit": 0, "missing_masks": 0}
            if os.path.isfile(os.path.join(pair_folder, "manifest.json")):
                try:
                    counts = cu.pair_folder_summary(pair_folder, load_manifest(pair_folder))
                except Exception as ex:
                    logger.warning("could not read %s: %s", pair_folder, ex)
            summary[pair_kind] = counts
        entry = cu.find_sft_queue_entry(entries, name)
        if entry is not None:
            summary["queue_entry"] = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
        concepts_file = os.path.join(cu.dpo_dir(root, name), cu.CONCEPTS_FILENAME)
        if os.path.isfile(concepts_file):
            summary["concepts_file"] = concepts_file
            summary["finalized"] = True
        return summary

    # ------------------------------------------------------------------ #
    # LoRA publish
    # ------------------------------------------------------------------ #

    def publish_lora(
        self, name: str, root: str, swarm_lora_dir: str, models_dir: str | None = None, force: bool = False
    ) -> dict:
        models_dir = self._models_dir(models_dir)
        lora_path, _kind = cu.resolve_lora(models_dir, name)
        if not lora_path:
            return {"ok": False, "path": None, "error": f"No LoRA for {name} in {models_dir}"}
        if not swarm_lora_dir or not os.path.isdir(swarm_lora_dir):
            return {"ok": False, "path": None, "error": f"SwarmUI LoRA folder not found: {swarm_lora_dir}"}
        # Already where SwarmUI scans (possibly in a subfolder it lists with
        # a prefix): nothing to copy.
        if _is_within(lora_path, swarm_lora_dir):
            return {"ok": True, "path": os.path.abspath(lora_path), "unchanged": True}
        dest = os.path.join(swarm_lora_dir, cu.lora_stem(lora_path) + ".safetensors")
        if os.path.isfile(dest):
            if compute_file_sha256(dest) == compute_file_sha256(lora_path):
                return {"ok": True, "path": dest, "unchanged": True}
            if not force:
                return {
                    "ok": False,
                    "path": dest,
                    "error": f"{os.path.basename(dest)} already exists in the SwarmUI LoRA folder with different contents",
                }
        try:
            shutil.copy2(lora_path, dest)
        except OSError as ex:
            return {"ok": False, "path": dest, "error": f"Copy failed: {ex}"}
        return {"ok": True, "path": dest}

    # ------------------------------------------------------------------ #
    # One-click negatives: publish each LoRA, queue a SwarmUI run per character
    # ------------------------------------------------------------------ #

    @staticmethod
    def train_captions(name: str, root: str) -> dict:
        """How many caption sidecars the character's train set holds — the
        number of captions a generate run with ``captions=0`` covers."""
        return {"count": _count_captions(os.path.join(cu.character_dir(root, name), "train"))}

    @staticmethod
    def _find_swarm_lora(stem: str, swarm_url: str, swarm_token: str | None) -> tuple[str | None, str | None]:
        """``(SwarmUI model name, error)`` for the published LoRA. SwarmUI
        may not have noticed the new file yet, so the listing is asked twice."""
        from modules.util.swarm_client import SwarmClient, SwarmError

        error = None
        for attempt in range(2):
            if attempt:
                time.sleep(SWARM_LIST_RETRY_DELAY)
            try:
                listed = SwarmClient(swarm_url, swarm_token or None, timeout=20).list_models(subtype="LoRA")
            except SwarmError as ex:
                error = f"SwarmUI LoRA listing failed: {ex}"
                continue
            found = CharacterDPOService._swarm_name_for(stem, listed)
            if found is not None:
                return found, None
            error = f"{stem} is not in SwarmUI's LoRA list after publishing; refresh its model list and try again"
        return None, error

    def generate_negatives(
        self,
        names: list[str],
        root: str,
        swarm_url: str,
        swarm_lora_dir: str,
        param_folder: str,
        models_dir: str | None = None,
        swarm_token: str = "",
        images_per_caption: int = DEFAULT_IMAGES_PER_CAPTION,
        captions: int = 0,
        concurrency: int | None = None,
    ) -> dict:
        """Queue one SwarmUI run per character, generating negatives for its
        train captions with its own LoRA and the settings of the newest image
        in ``param_folder`` (minus that image's adapters). ``captions`` 0 means
        every caption in the train set. One character's failure is reported in
        its job and the rest still go ahead; progress is the shared
        ``/swarm/status``."""
        from modules.util.swarm_generate_util import read_last_generation_params
        from web.backend.services.swarm_generate_service import DEFAULT_CONCURRENCY, SwarmGenerateService

        self._remember_root(root)
        models_dir = self._models_dir(models_dir)
        if not os.path.isdir(param_folder or ""):
            return {"ok": False, "error": f"Parameter folder not found: {param_folder}", "jobs": []}
        if not read_last_generation_params(param_folder):
            return {"ok": False, "error": f"No SwarmUI generation with a parameter block in {param_folder}", "jobs": []}
        jobs = []
        for name in names or []:
            job = {"name": name, "ok": False}
            jobs.append(job)
            folder = cu.character_dir(root, name)
            train = os.path.join(folder, "train")
            lora_path, _kind = cu.resolve_lora(models_dir, name)
            if not lora_path:
                job["error"] = f"No LoRA for {name} in {models_dir}"
                continue
            # A changed file under the same stem is this character's newer
            # LoRA: the published copy is what the negatives must use.
            published = self.publish_lora(name, root, swarm_lora_dir, models_dir=models_dir, force=True)
            if not published["ok"]:
                job["error"] = published["error"]
                continue
            swarm_lora, error = self._find_swarm_lora(cu.lora_stem(lora_path), swarm_url, swarm_token)
            if swarm_lora is None:
                job["error"] = error
                continue
            job["swarm_lora"] = swarm_lora
            wanted = int(captions) if captions else _count_captions(train)
            if wanted < 1:
                job["error"] = f"No captions in {train}"
                continue
            generated = os.path.join(folder, "dpo", "generated")
            try:
                os.makedirs(generated, exist_ok=True)
            except OSError as ex:
                job["error"] = f"Could not create {generated}: {ex}"
                continue
            started = SwarmGenerateService.get_instance().start(
                real_folder=train,
                generated_folder=generated,
                output_dir=cu.pair_dir(root, name, "offpolicy"),
                model="",
                base_url=swarm_url,
                token=swarm_token or "",
                captions=wanted,
                images_per_caption=max(1, int(images_per_caption)),
                loras=[{"name": swarm_lora, "weight": 1.0}],
                param_source="folder",
                param_folder=param_folder,
                drop_source_loras=True,
                label=name,
                save_to=generated,
                concurrency=int(concurrency) if concurrency else DEFAULT_CONCURRENCY,
            )
            if not started.get("ok"):
                job["error"] = started.get("error") or "SwarmUI run could not be queued"
                continue
            job["ok"] = True
            job["position"] = started.get("position")
        return {"ok": all(job["ok"] for job in jobs), "jobs": jobs}

    # ------------------------------------------------------------------ #
    # Import
    # ------------------------------------------------------------------ #

    @staticmethod
    def _concept_dirs_from_files(concept_files: list[str]) -> list[tuple[str, str]]:
        """``(concept_dir, concept_name)`` per enabled DPO pattern concept."""
        from modules.util.dpo_pattern_util import dpo_concept_pattern_dirs

        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for file in concept_files or []:
            try:
                dicts = cu.load_concept_dicts(file)
            except (OSError, ValueError) as ex:
                logger.warning("could not read concept file %s: %s", file, ex)
                continue
            for data in dicts:
                try:
                    concept = ConceptConfig.default_values().from_dict(data)
                    pairs = dpo_concept_pattern_dirs([concept])
                except Exception as ex:
                    logger.warning("skipping concept %r: %s", data.get("name"), ex)
                    continue
                for chosen_dir, _rejected_dir in pairs:
                    concept_dir = None
                    for candidate in (
                        concept.path,
                        os.path.dirname(chosen_dir),
                        os.path.dirname(os.path.dirname(chosen_dir)),
                    ):
                        if candidate and os.path.isfile(os.path.join(candidate, "manifest.json")):
                            concept_dir = os.path.abspath(candidate)
                            break
                    if concept_dir is None or _norm(concept_dir) in seen:
                        continue
                    seen.add(_norm(concept_dir))
                    out.append((concept_dir, str(concept.name or os.path.basename(concept_dir))))
        return out

    def _imported_hashes(self, root: str, name: str) -> set[str]:
        hashes: set[str] = set()
        for kind in cu.PAIR_KINDS:
            folder = cu.pair_dir(root, name, kind)
            if os.path.isfile(os.path.join(folder, "manifest.json")):
                with suppress(Exception):
                    for entry in load_manifest(folder).get("pairs", []):
                        if entry.get("chosen_sha256"):
                            hashes.add(entry["chosen_sha256"])
        for record in cu.load_character_state(root, name).get("imported", []):
            if isinstance(record, dict) and record.get("sha256"):
                hashes.add(record["sha256"])
        return hashes

    @staticmethod
    def _candidate(
        name: str, stem: set[str], concept_dir: str, concept_name: str, entry: dict, imported: set
    ) -> dict | None:
        from modules.util.dpo_replace_util import file_is_generation

        chosen, rejected = cu.locate_pair_files(concept_dir, entry)
        if not chosen or not rejected:
            return None
        caption = _read_text(os.path.splitext(chosen)[0] + ".txt")
        prompt = str(entry.get("prompt") or "")
        by_name = cu.name_matches(name, prompt, caption)
        by_lora = cu.lora_matches(
            stem, _provenance_dict(entry, "rejected", rejected), _provenance_dict(entry, "chosen", chosen)
        )
        match = cu.match_kind(by_name, by_lora)
        if match is None:
            return None
        kind = "aivai" if file_is_generation(chosen) else "offpolicy"
        return {
            "key": f"{concept_dir}|{entry.get('pair_id')}",
            "concept_dir": concept_dir,
            "concept_name": concept_name,
            "pair_id": entry.get("pair_id"),
            "kind": kind,
            "match": match,
            "prompt": prompt,
            "caption": caption,
            "chosen_path": chosen,
            "rejected_path": rejected,
            "already_imported": bool(entry.get("chosen_sha256")) and entry["chosen_sha256"] in imported,
        }

    def scan_imports(self, name: str, root: str, concept_files: list[str], models_dir: str | None = None) -> dict:
        self._remember_root(root)
        models_dir = self._models_dir(models_dir)
        lora_path, _kind = cu.resolve_lora(models_dir, name)
        stem = cu.lora_match_stems(name, lora_path)
        concept_dirs = self._concept_dirs_from_files(concept_files)
        with self._lock:
            self._scanned_concept_dirs = {_norm(d) for d, _n in concept_dirs}
        imported = self._imported_hashes(root, name)
        own = {_norm(cu.pair_dir(root, name, kind)) for kind in cu.PAIR_KINDS}
        candidates = []
        for concept_dir, concept_name in concept_dirs:
            if _norm(concept_dir) in own:
                continue
            try:
                manifest = load_manifest(concept_dir)
            except Exception as ex:
                logger.warning("could not read manifest in %s: %s", concept_dir, ex)
                continue
            for entry in manifest.get("pairs", []):
                if not isinstance(entry, dict):
                    continue
                with suppress(Exception):
                    candidate = self._candidate(name, stem, concept_dir, concept_name, entry, imported)
                    if candidate is not None:
                        candidates.append(candidate)
        return {"candidates": candidates}

    def apply_imports(self, name: str, root: str, picks: list[dict]) -> dict:
        from modules.util.dpo_replace_util import file_is_generation

        self._remember_root(root)
        imported = skipped = 0
        errors: list[str] = []
        state = cu.load_character_state(root, name)
        known = self._imported_hashes(root, name)
        manifests: dict[str, dict] = {}
        source_cache: dict[str, dict] = {}
        for pick in picks or []:
            concept_dir = str(pick.get("concept_dir") or "")
            pair_id = pick.get("pair_id")
            if not concept_dir or not os.path.isfile(os.path.join(concept_dir, "manifest.json")):
                errors.append(f"{concept_dir}: no manifest")
                continue
            try:
                if _norm(concept_dir) not in source_cache:
                    source_cache[_norm(concept_dir)] = load_manifest(concept_dir)
                source = source_cache[_norm(concept_dir)]
            except Exception as ex:
                errors.append(f"{concept_dir}: {ex}")
                continue
            entry = next((e for e in source.get("pairs", []) if e.get("pair_id") == pair_id), None)
            if entry is None:
                errors.append(f"{concept_dir}: pair {pair_id} not found")
                continue
            chosen, rejected = cu.locate_pair_files(concept_dir, entry)
            if not chosen or not rejected:
                errors.append(f"{concept_dir}: pair {pair_id} files missing")
                continue
            digest = entry.get("chosen_sha256") or compute_file_sha256(chosen)
            if digest and digest in known:
                skipped += 1
                continue
            kind = "aivai" if file_is_generation(chosen) else "offpolicy"
            target = cu.pair_dir(root, name, kind)
            try:
                os.makedirs(target, exist_ok=True)
                if target not in manifests:
                    manifests[target] = load_manifest(target)
                manifest = manifests[target]
                caption = _read_text(os.path.splitext(chosen)[0] + ".txt") or str(entry.get("prompt") or "")
                export_single_pair(target, manifest, chosen, rejected, caption, str(entry.get("aspectratio") or ""))
                new_entry = manifest["pairs"][-1]
                for key in ("chosen_provenance", "rejected_provenance"):
                    if isinstance(entry.get(key), dict):
                        new_entry[key] = dict(entry[key])
                # The copy came from the shared concept's export, but the
                # photo (and its mask) live where that concept found them.
                for key in ("chosen_source", "rejected_source"):
                    if entry.get(key):
                        new_entry[key] = entry[key]
                new_entry["source_concept"] = concept_dir
                new_entry["source_pair_id"] = pair_id
                save_manifest(target, manifest)
            except Exception as ex:
                errors.append(f"{concept_dir}: pair {pair_id}: {ex}")
                continue
            if digest:
                known.add(digest)
            state["imported"].append({"concept_dir": concept_dir, "pair_id": pair_id, "sha256": digest})
            imported += 1
        with suppress(OSError):
            cu.save_character_state(root, name, state)
        return {"ok": not errors, "imported": imported, "skipped": skipped, "errors": errors}

    # ------------------------------------------------------------------ #
    # Finalize
    # ------------------------------------------------------------------ #

    def finalize(
        self,
        name: str,
        root: str,
        val_pct: float = 10.0,
        mask_model: str = DEFAULT_MASK_MODEL,
        mask_options: dict | None = None,
        dpo_resolution: str = cu.DEFAULT_DPO_RESOLUTION,
    ) -> dict:
        self._remember_root(root)
        if not os.path.isdir(cu.character_dir(root, name)):
            return {"ok": False, "error": f"Character folder not found: {cu.character_dir(root, name)}"}
        with self._lock:
            if self._prepare["running"]:
                return {"ok": False, "error": "A prepare batch is running", "conflict": True}
            job = self._register_finalize_job(root, name)
            if job is None:
                return {"ok": False, "error": "Finalize already running for this character", "conflict": True}
        thread = threading.Thread(
            target=self._finalize_job,
            args=(job, name, root, float(val_pct), mask_model, dict(mask_options or {}), dpo_resolution),
            daemon=True,
            name="OneTrainerWeb-character-dpo-finalize",
        )
        thread.start()
        return {"ok": True, "job_id": job["job_id"]}

    def _register_finalize_job(self, root: str, name: str) -> dict | None:
        """Caller holds ``_lock``. The per-character job record that
        ``finalize_status`` reports and that blocks a second finalize of the
        same character; None when one is already running."""
        key = self._job_key(root, name)
        job = self._jobs.get(key)
        if job and job.get("running"):
            return None
        job = {
            "running": True,
            "stage": "starting",
            "done": 0,
            "total": 0,
            "message": "",
            "error": None,
            "result": None,
            "job_id": f"{key}:{threading.get_ident()}:{len(self._jobs)}",
        }
        self._jobs[key] = job
        return job

    def finalize_status(self, name: str, root: str) -> dict:
        with self._lock:
            job = self._jobs.get(self._job_key(root, name))
        if job is None:
            return {
                "running": False,
                "stage": "idle",
                "done": 0,
                "total": 0,
                "message": "",
                "error": None,
                "result": None,
            }
        return {k: job[k] for k in ("running", "stage", "done", "total", "message", "error", "result")}

    def _finalize_job(
        self,
        job: dict,
        name: str,
        root: str,
        val_pct: float,
        mask_model: str,
        mask_options: dict,
        dpo_resolution: str = cu.DEFAULT_DPO_RESOLUTION,
    ):
        try:
            result = self._finalize_inner(job, name, root, val_pct, mask_model, mask_options, dpo_resolution)
            job["result"] = result
            job["stage"] = "done"
            job["message"] = "Finalized"
        except Exception as ex:
            logger.exception("character DPO finalize failed")
            job["error"] = str(ex)
            job["stage"] = "error"
        finally:
            job["running"] = False

    # ------------------------------------------------------------------ #
    # One-click prepare: finalize several characters in one background job
    # ------------------------------------------------------------------ #

    @staticmethod
    def needs_finalize(root: str, name: str) -> bool:
        """False when the character already has its concept file and every
        pair in both folders is split and masked — finalizing it again would
        only rewrite the same concepts."""
        if not os.path.isfile(os.path.join(cu.dpo_dir(root, name), cu.CONCEPTS_FILENAME)):
            return True
        for kind in cu.PAIR_KINDS:
            folder = cu.pair_dir(root, name, kind)
            if not os.path.isfile(os.path.join(folder, "manifest.json")):
                continue
            try:
                counts = cu.pair_folder_summary(folder, load_manifest(folder))
            except Exception as ex:
                logger.warning("could not read %s: %s", folder, ex)
                return True
            if counts["unsplit"] or counts["missing_masks"]:
                return True
        return False

    def prepare(
        self,
        names: list[str],
        root: str,
        models_dir: str | None = None,
        val_pct: float = 10.0,
        mask_model: str = DEFAULT_MASK_MODEL,
        mask_options: dict | None = None,
        dpo_resolution: str = cu.DEFAULT_DPO_RESOLUTION,
    ) -> dict:
        """Finalize every named character in turn (split, masks, concepts) in
        one background job. A character that is already fully finalized is
        skipped. Refused while a prepare batch or any single finalize runs."""
        self._remember_root(root)
        names = [name for name in (names or []) if name]
        if not names:
            return {"ok": False, "error": "No characters selected"}
        with self._lock:
            if self._prepare["running"]:
                return {"ok": False, "error": "A prepare batch is already running", "conflict": True}
            if any(job.get("running") for job in self._jobs.values()):
                return {"ok": False, "error": "A finalize is running for a character", "conflict": True}
            self._prepare = self._idle_prepare() | {"running": True, "total": len(names), "stage": "starting"}
            state = self._prepare
        thread = threading.Thread(
            target=self._prepare_job,
            args=(state, names, root, float(val_pct), mask_model, dict(mask_options or {}), dpo_resolution),
            daemon=True,
            name="OneTrainerWeb-character-dpo-prepare",
        )
        thread.start()
        return {"ok": True, "total": len(names)}

    def prepare_status(self) -> dict:
        with self._lock:
            state = self._prepare
            out = {k: state[k] for k in ("running", "current", "done", "total", "stage", "message", "error")}
            out["results"] = dict(state["results"])
            # While a character is being finalized its own job carries the
            # live stage and message; one poll here is enough for the UI.
            job = state.get("job")
            if state["running"] and job is not None and job.get("running"):
                out["stage"], out["message"] = job["stage"], job["message"]
        return out

    def _prepare_job(
        self,
        state: dict,
        names: list[str],
        root: str,
        val_pct: float,
        mask_model: str,
        mask_options: dict,
        dpo_resolution: str = cu.DEFAULT_DPO_RESOLUTION,
    ):
        try:
            for name in names:
                state["current"] = name
                state["message"] = ""
                if not os.path.isdir(cu.character_dir(root, name)):
                    state["results"][name] = {"error": f"Character folder not found: {cu.character_dir(root, name)}"}
                    state["done"] += 1
                    continue
                if not self.needs_finalize(root, name):
                    state["results"][name] = {"skipped": True}
                    state["stage"] = "skipped"
                    state["done"] += 1
                    continue
                with self._lock:
                    job = self._register_finalize_job(root, name)
                    state["job"] = job
                if job is None:  # cannot happen: prepare refuses to start beside a finalize
                    state["results"][name] = {"error": "Finalize already running for this character"}
                    state["done"] += 1
                    continue
                # The very same code path as a single finalize, so the result
                # shape and the per-character finalize status are identical.
                self._finalize_job(job, name, root, val_pct, mask_model, mask_options, dpo_resolution)
                state["stage"], state["message"] = job["stage"], job["message"]
                state["results"][name] = {"error": job["error"]} if job["error"] else dict(job["result"] or {})
                state["done"] += 1
            state["stage"] = "done"
        except Exception as ex:
            logger.exception("character DPO prepare batch failed")
            state["error"] = str(ex)
            state["stage"] = "error"
        finally:
            state["current"] = None
            state["job"] = None
            state["running"] = False

    def _finalize_inner(
        self, job, name, root, val_pct, mask_model, mask_options, dpo_resolution=cu.DEFAULT_DPO_RESOLUTION
    ) -> dict:
        from modules.util.dpo_replace_util import file_is_generation

        counts: dict[str, dict[str, int]] = {}
        to_generate: list[str] = []
        masks_copied = 0
        for kind in cu.PAIR_KINDS:
            folder = cu.pair_dir(root, name, kind)
            if not os.path.isfile(os.path.join(folder, "manifest.json")):
                counts[kind] = {"train": 0, "val": 0}
                continue
            job["stage"] = f"split {kind}"
            manifest = load_manifest(folder)
            split = cu.split_new_pairs(folder, manifest, val_pct)
            counts[kind] = {"train": split["train"], "val": split["val"]}
            job["stage"] = f"masks {kind}"
            for entry, side, path in cu.pairs_missing_masks(folder, manifest):
                source_mask = self._existing_mask_for(entry, side, file_is_generation)
                if source_mask is not None:
                    try:
                        shutil.copy2(source_mask, cu.mask_path_for(path))
                        masks_copied += 1
                        continue
                    except OSError as ex:
                        logger.warning("could not copy mask %s: %s", source_mask, ex)
                to_generate.append(path)
        masks_generated = 0
        if to_generate:
            job["stage"] = "generating masks"
            job["total"] = len(to_generate)
            job["done"] = 0
            masks_generated = self._generate_masks(to_generate, mask_model, mask_options, job)
        job["stage"] = "writing concepts"
        concepts_file = cu.write_character_concepts(root, name, counts, dpo_resolution)
        state = cu.load_character_state(root, name)
        state["last_finalize"] = {
            "offpolicy": counts["offpolicy"],
            "aivai": counts["aivai"],
            "masks_generated": masks_generated,
            "masks_copied": masks_copied,
            "val_pct": val_pct,
            "mask_model": mask_model,
            "dpo_resolution": str(dpo_resolution),
        }
        cu.save_character_state(root, name, state)
        return {
            "offpolicy": counts["offpolicy"],
            "aivai": counts["aivai"],
            "masks_generated": masks_generated,
            "masks_copied": masks_copied,
            "concepts_file": concepts_file,
        }

    @staticmethod
    def _existing_mask_for(entry: dict, side: str, file_is_generation) -> str | None:
        """A mask already drawn for this side's image: next to the real photo
        it was exported from, or next to the identical copy in the concept it
        was imported from."""
        candidates = []
        source = entry.get(f"{side}_source")
        if source and os.path.isfile(source) and not file_is_generation(source):
            candidates.append(source)
        source_concept = entry.get("source_concept")
        if source_concept and entry.get("source_pair_id") is not None:
            with suppress(Exception):
                origin = next(
                    (
                        e
                        for e in load_manifest(source_concept).get("pairs", [])
                        if e.get("pair_id") == entry["source_pair_id"]
                    ),
                    None,
                )
                if origin is not None:
                    chosen, rejected = cu.locate_pair_files(source_concept, origin)
                    exported = chosen if side == "chosen" else rejected
                    if exported:
                        candidates.append(exported)
        for candidate in candidates:
            mask = cu.mask_path_for(candidate)
            if os.path.isfile(mask):
                return mask
        return None

    def _generate_masks(self, paths: list[str], mask_model: str, mask_options: dict, job: dict) -> int:
        """Run one mask model over ``paths`` (``<stem>-masklabel.png`` next to
        each) and release it. Tests monkeypatch this."""
        import importlib

        from web.backend.services.tool_service import MASK_MODEL_MAP

        import torch

        class_name = MASK_MODEL_MAP.get(mask_model)
        if class_name is None or class_name == "YOLOMaskAdapter":
            raise ValueError(f"Unsupported mask model: {mask_model}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        module = importlib.import_module(f"modules.module.{class_name}")
        model = getattr(module, class_name)(device, torch.float32)
        options = dict(DEFAULT_MASK_OPTIONS) | {k: v for k, v in (mask_options or {}).items() if v is not None}
        prompts = [options["prompt"]] if options.get("prompt") else []
        failed: list[str] = []

        def progress(done: int, total: int) -> None:
            job["done"], job["total"] = done, total

        try:
            model.mask_images(
                paths,
                prompts,
                mode="fill",
                alpha=float(options["alpha"]),
                threshold=float(options["threshold"]),
                smooth_pixels=int(options["smooth"]),
                expand_pixels=int(options["expand"]),
                progress_callback=progress,
                error_callback=failed.append,
            )
        finally:
            del model
            with suppress(Exception):
                from modules.util.torch_util import torch_gc

                torch_gc()
        if failed:
            job["message"] = f"{len(failed)} mask(s) failed: {', '.join(os.path.basename(p) for p in failed[:5])}"
        return sum(1 for p in paths if os.path.isfile(cu.mask_path_for(p)))

    # ------------------------------------------------------------------ #
    # Queue
    # ------------------------------------------------------------------ #

    def build_queue(
        self,
        names: list[str],
        root: str,
        models_dir: str | None = None,
        add_on_weight: float = 0.7,
        mode: str = "preview",
        dpo_beta: float = cu.DEFAULT_DPO_BETA,
    ) -> dict:
        self._remember_root(root)
        models_dir = self._models_dir(models_dir)
        queue_entries = self._queue_entries()
        global_concepts = None
        workspace = self._global_workspace_dir()
        entries, errors, added = [], [], 0
        for name in names or []:
            sft_entry = cu.find_sft_queue_entry(queue_entries, name)
            if sft_entry is None and global_concepts is None:
                global_concepts = self._global_concepts()
            try:
                built = cu.build_dpo_entry(
                    name,
                    root,
                    models_dir,
                    workspace,
                    sft_queue_entry=sft_entry,
                    global_concepts=global_concepts or [],
                    add_on_weight=add_on_weight,
                    dpo_beta=dpo_beta,
                )
            except ValueError as ex:
                errors.append({"name": name, "error": str(ex)})
                continue
            except Exception as ex:
                logger.exception("queue entry build failed for %s", name)
                errors.append({"name": name, "error": str(ex)})
                continue
            entries.append(built)
            if mode == "add":
                try:
                    from web.backend.services.queue_service import QueueService

                    QueueService.get_instance().add_entry(built["name"], built["overrides"])
                    added += 1
                except Exception as ex:
                    errors.append({"name": name, "error": f"Could not add to the queue: {ex}"})
        return {"entries": entries, "errors": errors, "added": added}

    # ------------------------------------------------------------------ #
    # Images
    # ------------------------------------------------------------------ #

    def serve_image(self, path: str) -> str | None:
        if not path or not os.path.isfile(path):
            return None
        if os.path.splitext(path)[1].lower() not in supported_image_extensions():
            return None
        with self._lock:
            folders = set(self._roots) | set(self._scanned_concept_dirs)
        for folder in folders:
            if _is_within(path, folder):
                return os.path.abspath(path)
        return None
