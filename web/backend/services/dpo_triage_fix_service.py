"""Triage & Fix: tag the chosen images of an exported DPO dataset with the
fixes they need, repair them through SwarmUI in the background, review the
candidates and commit each decision.

One session at a time, on one dataset. The ledger in the dataset folder is
the only persistent state: every tag, candidate and decision is written
through as it happens, so closing the tool (or the app) and reopening it on
the same folder resumes where it stopped. Images that were confirmed,
skipped or decided never come back.

A single drain thread turns confirmed items into candidate jobs on a small
pool as they arrive, so the user keeps triaging while earlier images are
being repaired; results appear on the compare screen in the order they
finished. While sending is held (SwarmUI busy with other work) confirms
still queue, but nothing new goes out until it is released.

Several sessions can run side by side (one per fixer model, say anatomy
on one and Klein for everything else), each with its own ledger, settings
and hold. Sessions on the same dataset share its manifest and take turns on
an image both have queued: the one that confirmed it first repairs it, and
the other holds its copy back until that is decided, then repairs the image
as it stands. Fixes that would repaint the same area are refused.
"""

import contextlib
import logging
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from modules.util import dpo_triage_fix_util as tf
from modules.util.dpo_curation_util import compute_file_sha256, load_manifest, prune_orphaned_pairs
from modules.util.dpo_regen_util import parse_segment_string, sanitize_segment_params
from modules.util.swarm_client import DEFAULT_BASE_URL, SwarmClient, SwarmError
from web.backend.services._dpo_session_lock import dpo_session_admission
from web.backend.services._singleton import SingletonMixin

logger = logging.getLogger(__name__)

DEFAULT_OPTIONS_PER_IMAGE = 2
MAX_OPTIONS_PER_IMAGE = 4
DEFAULT_CONCURRENCY = 2
MAX_CONCURRENCY = 4
#: Candidate failures in a row before the run stops itself. Queued items stay
#: queued; a fresh start on the same folder picks them up.
MAX_CONSECUTIVE_FAILURES = 5

#: One lock for every session: sessions on the same dataset share its
#: manifest, and each reads the others' ledgers to take turns on an image.
_SHARED_LOCK = threading.RLock()


class DPOTriageFixService(SingletonMixin):
    #: Every session made in this process, by id.
    _registry: dict[str, "DPOTriageFixService"] = {}

    @classmethod
    def get_instance(cls) -> "DPOTriageFixService":
        """The main session."""
        return cls.session(tf.MAIN_SESSION)

    @classmethod
    def session(cls, session_id: str | None) -> "DPOTriageFixService":
        """The session with this id, made on first use. Raises ``ValueError``
        on an id that cannot be a folder name."""
        session_id = (session_id or tf.MAIN_SESSION).strip().lower()
        if not tf.valid_session_id(session_id):
            raise ValueError(f"Invalid session id: {session_id!r}")
        with _SHARED_LOCK:
            instance = cls._registry.get(session_id)
            return instance if instance is not None else cls(session_id)

    @classmethod
    def all_sessions(cls) -> list["DPOTriageFixService"]:
        with _SHARED_LOCK:
            return list(cls._registry.values())

    @classmethod
    def overview(cls) -> dict:
        """Every session this process knows, for the session tabs."""
        with _SHARED_LOCK:
            out = []
            for session in cls._registry.values():
                ledger = session._ledger
                out.append(
                    {
                        "id": session._session,
                        "name": session._name,
                        "active": session._active,
                        "held": session._held,
                        "concept_dir": session._concept_dir,
                        "model": session._settings.get("model", ""),
                        "counts": tf.ledger_counts(ledger, session._stems) if ledger else None,
                    }
                )
        return {"ok": True, "sessions": out}

    def __init__(self, session_id: str = tf.MAIN_SESSION) -> None:
        self._session = session_id
        self._name = session_id
        self._lock = _SHARED_LOCK
        self._wake = threading.Condition(self._lock)
        self._generation = 0
        self._stop = threading.Event()
        self._active = False
        # Sending held: confirmed images wait in the queue instead of going to SwarmUI.
        self._held = False
        self._phase = "idle"
        self._concept_dir: str | None = None
        self._manifest: dict = {"pairs": []}
        self._entries: dict[str, dict] = {}
        self._manifest_entry: dict[str, dict] = {}
        self._stems: list[str] = []
        self._ledger: dict | None = None
        self._settings: dict = {}
        self._fix_dir = ""
        self._fix_manifest: dict | None = None
        # Other fix-pair folders an earlier round wrote to, by normcased path.
        self._other_fix_manifests: dict[str, dict] = {}
        self._client_opts: dict = {}
        self._detector: tf.FixDetector | None = None
        self._worker: threading.Thread | None = None
        self._scanner: threading.Thread | None = None
        self._in_flight: dict[tuple[str, int], Future] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._contexts: dict[str, dict] = {}
        self._undo_stack: list[dict] = []
        self._error: str | None = None
        self._warning: str | None = None
        self._consecutive_failures = 0
        with _SHARED_LOCK:
            type(self)._registry[session_id] = self

    def _save_ledger(self) -> None:
        tf.save_ledger(self._concept_dir, self._ledger, self._session)

    # ------------------------------------------------------------------ #
    # Other sessions
    # ------------------------------------------------------------------ #

    def _running_on(self, folder: str | None) -> list["DPOTriageFixService"]:
        """Other active sessions on ``folder``. Caller holds the lock."""
        if not folder:
            return []
        key = os.path.normcase(os.path.abspath(folder))
        return [
            session
            for session in type(self)._registry.values()
            if session is not self
            and session._active
            and session._ledger is not None
            and session._concept_dir
            and os.path.normcase(session._concept_dir) == key
        ]

    def _peers(self) -> list["DPOTriageFixService"]:
        """Other active sessions on this session's dataset. Caller holds the lock."""
        return self._running_on(self._concept_dir)

    @staticmethod
    def _claim_key(record: dict, session: "DPOTriageFixService") -> tuple:
        """Which of two sessions queuing the same image goes first: the one
        that confirmed it first. A re-roll keeps its place."""
        at = record.get("queued_at") or record.get("tagged_at") or 0.0
        return (float(at), session._session)

    def _blocked_by_locked(self, stem: str, record: dict) -> "DPOTriageFixService | None":
        """The session this image has to wait for: one that is repairing it
        (generating, awaiting a decision or failed), or queued it first."""
        mine = self._claim_key(record, self)
        for peer in self._peers():
            other = peer._ledger["items"].get(stem)
            status = (other or {}).get("status")
            if status in (tf.STATUS_GENERATING, tf.STATUS_READY, tf.STATUS_FAILED):
                return peer
            if status == tf.STATUS_QUEUED and self._claim_key(other, peer) < mine:
                return peer
        return None

    def _pending_peer_locked(self, stem: str) -> "DPOTriageFixService | None":
        for peer in self._peers():
            if (peer._ledger["items"].get(stem) or {}).get("status") in tf.PENDING_STATUSES:
                return peer
        return None

    def _overlap_error_locked(self, stem: str, fixes: list[dict], mask_file: str | None) -> str | None:
        """Refuse fixes that would repaint an area another session has queued
        on the same image. Caller holds the lock."""
        for peer in self._peers():
            other = peer._ledger["items"].get(stem) or {}
            if other.get("status") not in tf.PENDING_STATUSES:
                continue
            other_mask = tf.absolute(peer._concept_dir, other.get("mask")) if other.get("mask") else None
            reason = tf.fixes_overlap(fixes, mask_file, other.get("fixes") or [], other_mask)
            if reason:
                return (
                    f'Cannot proceed: session "{peer._name}" already has {stem} queued with {reason}. '
                    "Decide it there first, or pick fixes for a different area."
                )
        return None

    def _dispatch_problem_locked(self, stem: str, record: dict) -> str | None:
        """Why a queued image cannot go out after all: its pair is gone, or
        its painted mask lands on what another session repaired while it
        waited. Caller holds the lock."""
        entry = self._manifest_entry.get(stem)
        path = self._entries[stem]["chosen_path"]
        if entry is None or not os.path.isfile(path):
            return "The pair is gone from the dataset"
        mask_file = tf.absolute(self._concept_dir, record.get("mask")) if record.get("mask") else None
        if not mask_file or not any(f.get("id") == tf.CATEGORY_MASK for f in record.get("fixes") or []):
            return None
        waited_from = float(record.get("queued_at") or record.get("tagged_at") or 0.0)
        for peer in self._peers():
            other = peer._ledger["items"].get(stem) or {}
            decision = other.get("decision") or {}
            if other.get("status") != tf.STATUS_FIXED or float(other.get("decided_at") or 0.0) < waited_from:
                continue
            before = tf.absolute(peer._concept_dir, decision.get("backup"))
            if before and tf.mask_overlaps_change(mask_file, before, path):
                return (
                    f'Cannot proceed: the painted mask overlaps what session "{peer._name}" repaired on this image. '
                    "Paint it again outside that area, or skip."
                )
        return None

    def _loaded_fix_manifest(self, fix_dir: str) -> dict | None:
        """A fix-pair manifest another active session already holds, so two
        sessions writing pairs to one folder never overwrite each other."""
        key = os.path.normcase(os.path.abspath(fix_dir))
        for session in type(self)._registry.values():
            if session is self or not session._active:
                continue
            if session._fix_dir and os.path.normcase(session._fix_dir) == key and session._fix_manifest is not None:
                return session._fix_manifest
            if key in session._other_fix_manifests:
                return session._other_fix_manifests[key]
        return None

    def _folder_clash(self, concept_dir: str, fix_dir: str) -> str | None:
        """A session's dataset cannot be another's fix-pair folder."""
        concept_key = os.path.normcase(concept_dir)
        fix_key = os.path.normcase(fix_dir) if fix_dir else ""
        for session in type(self)._registry.values():
            if session is self or not session._active:
                continue
            if fix_key and session._concept_dir and os.path.normcase(session._concept_dir) == fix_key:
                return f'The fix pair folder is the dataset of session "{session._name}"'
            if session._fix_dir and os.path.normcase(session._fix_dir) == concept_key:
                return f'This dataset is the fix pair folder of session "{session._name}"'
        return None

    # ------------------------------------------------------------------ #
    # Session
    # ------------------------------------------------------------------ #

    def concepts(self) -> dict:
        from web.backend.services.dpo_replace_service import DPOReplacePairsService

        return DPOReplacePairsService.get_instance().concepts()

    @staticmethod
    def _layout_error(concept_dir: str) -> str | None:
        if not concept_dir or not os.path.isdir(concept_dir):
            return "Dataset folder not found"
        if not os.path.isfile(os.path.join(concept_dir, "manifest.json")):
            return "Dataset folder has no manifest.json (only exported DPO datasets can be triaged)"
        for sub in ("chosen", "rejected"):
            if not os.path.isdir(os.path.join(concept_dir, sub)):
                return f"Dataset folder has no {sub}/ subfolder"
        return None

    def inspect(self, concept_dir: str) -> dict:
        """What a start on this folder would find: pair count and, when a
        ledger exists, how far the previous session got."""
        error = self._layout_error(concept_dir)
        if error:
            return {"ok": False, "error": error}
        try:
            manifest = load_manifest(concept_dir)
            ledger = tf.load_ledger(concept_dir, self._session)
        except RuntimeError as ex:
            return {"ok": False, "error": str(ex)}
        items = tf.enumerate_items(concept_dir, manifest)
        stems = [item["stem"] for item in items]
        counts = tf.ledger_counts(ledger, stems) if ledger else tf.ledger_counts(tf.new_ledger(0), stems)
        return {
            "ok": True,
            "total": len(stems),
            "has_ledger": ledger is not None,
            "counts": counts,
            "shuffle_seed": ledger["shuffle_seed"] if ledger else None,
            "settings": ledger.get("settings", {}) if ledger else {},
        }

    @staticmethod
    def _normalize_settings(raw: dict | None) -> dict:
        raw = raw or {}
        klein = tf.klein_settings(raw)
        try:
            options = int(raw.get("options_per_image") or DEFAULT_OPTIONS_PER_IMAGE)
        except (TypeError, ValueError):
            options = DEFAULT_OPTIONS_PER_IMAGE
        try:
            concurrency = int(raw.get("concurrency") or DEFAULT_CONCURRENCY)
        except (TypeError, ValueError):
            concurrency = DEFAULT_CONCURRENCY
        template = str(raw.get("prompt_template") or "").strip() or tf.DEFAULT_PROMPT_TEMPLATE
        return {
            **klein,
            "options_per_image": max(1, min(MAX_OPTIONS_PER_IMAGE, options)),
            "concurrency": max(1, min(MAX_CONCURRENCY, concurrency)),
            "segment_params": {
                **tf.DEFAULT_SEGMENT_PARAMS,
                **sanitize_segment_params(raw.get("segment_params") or {}),
            },
            "mask_params": tf.sanitize_mask_params(raw.get("mask_params") or {}),
            "categories": tf.normalize_categories(raw.get("categories")),
            "prompt_template": template,
        }

    @dpo_session_admission
    def start(
        self,
        concept_dir: str,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        settings: dict | None = None,
        fix_pair_dir: str = "",
        shuffle_seed: int | None = None,
        detector: str = "",
        hold: bool = False,
        name: str = "",
    ) -> dict:
        with self._lock:
            if self._active:
                return {"ok": False, "error": "This Triage & Fix session is already running; pause it first"}
        concept_dir = os.path.abspath((concept_dir or "").strip()) if concept_dir else ""
        error = self._layout_error(concept_dir)
        if error:
            return {"ok": False, "error": error}
        settings = self._normalize_settings(settings)
        if not settings["model"]:
            return {"ok": False, "error": "Pick the fixer model (Flux.2 Klein) first"}
        fix_dir = os.path.abspath(fix_pair_dir.strip()) if (fix_pair_dir or "").strip() else ""
        if fix_dir:
            error = tf.fix_pair_folder_error(fix_dir, concept_dir)
            if error:
                return {"ok": False, "error": error}
        with self._lock:
            error = self._folder_clash(concept_dir, fix_dir)
            # A session already on this dataset holds its manifest: share it,
            # so a pick in one is never undone by a save in the other.
            joined = next(iter(self._running_on(concept_dir)), None)
        if error:
            return {"ok": False, "error": error}
        try:
            detector_obj = tf.load_detector(detector)
        except NotImplementedError as ex:
            return {"ok": False, "error": str(ex)}
        try:
            if joined is None:
                manifest = load_manifest(concept_dir)
                pruned = prune_orphaned_pairs(concept_dir, manifest)
            else:
                manifest, pruned = joined._manifest, 0
            ledger = tf.load_ledger(concept_dir, self._session)
        except RuntimeError as ex:
            return {"ok": False, "error": str(ex)}
        items = tf.enumerate_items(concept_dir, manifest) if joined is None else None
        if joined is None and not items:
            return {"ok": False, "error": "The dataset has no pairs with a chosen image on disk"}
        resumed = ledger is not None
        if ledger is None:
            seed = int(shuffle_seed) if shuffle_seed is not None else random.randrange(1, 2**31 - 1)
            ledger = tf.new_ledger(seed, settings)
        # Queued fixes written under an older default wording follow the defaults.
        tf.migrate_ledger_instructions(ledger)
        ledger["settings"] = settings
        # A run interrupted mid-generation restarts those items from the queue.
        for record in ledger["items"].values():
            if record.get("status") == tf.STATUS_GENERATING:
                tf.delete_candidate_files(concept_dir, record)
                record["candidates"] = []
                record["status"] = tf.STATUS_QUEUED
        fix_manifest = None
        if fix_dir:
            with self._lock:
                fix_manifest = self._loaded_fix_manifest(fix_dir)
            if fix_manifest is None:
                try:
                    fix_manifest = tf.load_fix_manifest(fix_dir)
                except RuntimeError as ex:
                    return {"ok": False, "error": str(ex)}

        with self._lock:
            if self._active:
                return {"ok": False, "error": "A Triage & Fix session is already running; pause it first"}
            self._generation += 1
            generation = self._generation
            self._stop = threading.Event()
            stop = self._stop
            self._active = True
            self._held = bool(hold)
            self._phase = "running"
            self._name = (name or "").strip() or self._session
            self._concept_dir = concept_dir
            self._manifest = manifest
            if joined is not None and joined._active and joined._manifest is manifest:
                # Picks in either session move the chosen file; both see it.
                self._entries = joined._entries
                self._stems = joined._stems
                self._manifest_entry = joined._manifest_entry
            else:
                if items is None:
                    self._active = False
                    self._phase = "idle"
                    return {"ok": False, "error": "The other session on this dataset stopped; start again"}
                self._entries = {item["stem"]: item for item in items}
                self._stems = [item["stem"] for item in items]
                self._manifest_entry = {}
                for entry in manifest.get("pairs", []):
                    stem = os.path.splitext(str(entry.get("chosen_file") or ""))[0]
                    if stem in self._entries:
                        self._manifest_entry[stem] = entry
            self._ledger = ledger
            self._settings = settings
            self._fix_dir = fix_dir
            self._fix_manifest = fix_manifest
            self._other_fix_manifests = {}
            self._client_opts = {"base_url": base_url or DEFAULT_BASE_URL, "auth": token or ""}
            self._detector = detector_obj
            self._in_flight = {}
            self._undo_stack = []
            self._error = None
            self._warning = f"{pruned} manifest pair(s) without files were pruned" if pruned else None
            self._consecutive_failures = 0
            self._save_ledger()
            counts = tf.ledger_counts(ledger, self._stems)
        self._worker = threading.Thread(target=self._drain, args=(generation, stop), daemon=True)
        self._worker.start()
        if detector_obj is not None:
            self._scanner = threading.Thread(target=self._scan, args=(generation, stop), daemon=True)
            self._scanner.start()
        return {
            "ok": True,
            "total": counts["total"],
            "undecided": counts["undecided"],
            "ready": counts["ready"],
            "resumed": resumed,
            "shuffle_seed": ledger["shuffle_seed"],
            "held": bool(hold),
        }

    def set_hold(self, held: bool) -> dict:
        """Hold or release sending. Holding stops new images going to
        SwarmUI; one still waiting for a pool worker goes back to the queue,
        one SwarmUI already has finishes all its options. Releasing sends the
        queue in the order it was confirmed."""
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            self._held = bool(held)
            withdrawn = 0
            if self._held:
                for (stem, k), future in list(self._in_flight.items()):
                    # Extra options follow a first option that already ran; let them finish.
                    if k != 0 or not future.cancel():
                        continue
                    self._in_flight.pop((stem, k), None)
                    self._contexts.pop(stem, None)
                    record = self._ledger["items"].get(stem)
                    if record is not None and record.get("status") == tf.STATUS_GENERATING:
                        record["candidates"] = []
                        record["status"] = tf.STATUS_QUEUED
                        withdrawn += 1
                if withdrawn:
                    self._save_ledger()
            self._wake.notify_all()
            return {"ok": True, "held": self._held, "withdrawn": withdrawn}

    def cancel(self) -> dict:
        """Pause the session: stop generating, put in-flight items back in
        the queue, keep the ledger. Start again on the folder to resume."""
        with self._lock:
            self._stop.set()
            self._generation += 1
            was_active = self._active
            self._active = False
            self._phase = "idle"
            for stem, _k in list(self._in_flight):
                record = self._ledger["items"].get(stem) if self._ledger else None
                if record is not None and record.get("status") == tf.STATUS_GENERATING:
                    tf.delete_candidate_files(self._concept_dir, record)
                    record["candidates"] = []
                    record["status"] = tf.STATUS_QUEUED
            self._in_flight = {}
            self._contexts = {}
            if was_active and self._ledger is not None and self._concept_dir:
                self._save_ledger()
            self._wake.notify_all()
        return {"ok": True}

    def status(self) -> dict:
        with self._lock:
            ledger = self._ledger
            counts = tf.ledger_counts(ledger, self._stems) if ledger else tf.ledger_counts(tf.new_ledger(0), [])
            undo = self._undo_stack[-1] if self._undo_stack else None
            waiting = 0
            if ledger is not None and self._active:
                waiting = sum(
                    1
                    for stem, record in ledger["items"].items()
                    if record.get("status") == tf.STATUS_QUEUED
                    and stem in self._entries
                    and self._blocked_by_locked(stem, record) is not None
                )
            return {
                "ok": True,
                "session": self._session,
                "name": self._name,
                "peers": [peer._name for peer in self._peers()] if self._active else [],
                # Queued images held back while another session repairs them.
                "waiting": waiting,
                "active": self._active,
                "held": self._held,
                "phase": self._phase,
                "concept_dir": self._concept_dir,
                "fix_pair_dir": self._fix_dir,
                "shuffle_seed": ledger["shuffle_seed"] if ledger else None,
                "options_per_image": self._settings.get("options_per_image", DEFAULT_OPTIONS_PER_IMAGE),
                "has_detector": self._detector is not None,
                "counts": counts,
                "in_flight": len(self._in_flight),
                "can_undo": undo is not None,
                "undo_label": (f"{undo['kind']} {undo['stem']}" if undo else None),
                "error": self._error,
                "warning": self._warning,
            }

    def items(self) -> dict:
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            out = []
            peers = self._peers()
            for stem in self._stems:
                item = self._entries[stem]
                record = self._ledger["items"].get(stem) or {}
                waiting_on = None
                if peers and record.get("status") == tf.STATUS_QUEUED:
                    blocker = self._blocked_by_locked(stem, record)
                    waiting_on = blocker._name if blocker is not None else None
                out.append(
                    {
                        "stem": stem,
                        "pair_id": item["pair_id"],
                        "chosen_path": item["chosen_path"],
                        "caption": item["caption"],
                        "caption_source": item["caption_source"],
                        "width": item["width"],
                        "height": item["height"],
                        "status": record.get("status"),
                        "detected": record.get("detected"),
                        "fixes": record.get("fixes") or [],
                        "error": record.get("error"),
                        "attempt": record.get("attempt", 0),
                        "round": len(record.get("rounds") or []) + 1,
                        "decided_at": record.get("decided_at"),
                        "original_path": tf.absolute(self._concept_dir, tf.first_original(record)),
                        "has_fix_pair": bool(
                            (record.get("decision") or {}).get("fix_pair") or tf.prior_fix_pair(record)
                        ),
                        "second_pass": tf.second_pass_state(record),
                        # What an image taken up by the second round was before.
                        "second_from": ((record.get("second_pass") or {}).get("prev") or {}).get("status")
                        or (tf.STATUS_FIXED if record.get("rounds") else None),
                        # The other session this queued image waits for.
                        "waiting_on": waiting_on,
                        # Other sessions that have this image queued or in progress.
                        "claimed_by": [
                            peer._name
                            for peer in peers
                            if (peer._ledger["items"].get(stem) or {}).get("status") in tf.PENDING_STATUSES
                        ],
                    }
                )
            return {"ok": True, "items": out, "categories": self._settings.get("categories", [])}

    # ------------------------------------------------------------------ #
    # Triage
    # ------------------------------------------------------------------ #

    def _require(self, stem: str) -> tuple[dict | None, str | None]:
        if not self._active or self._ledger is None:
            return None, "No active session"
        if stem not in self._entries:
            return None, "Unknown image"
        return self._ledger["items"].setdefault(stem, {}), None

    def _queue(self, stem: str, record: dict, fixes: list[tf.TriageFix], sha: str | None) -> None:
        record.pop("overrides", None)
        record.update(
            {
                "status": tf.STATUS_QUEUED,
                "pair_id": self._entries[stem]["pair_id"],
                "chosen_file": self._entries[stem]["chosen_file"],
                "chosen_sha256": sha,
                "fixes": [fix.to_dict() for fix in fixes],
                "seed": tf.item_seed(self._ledger["shuffle_seed"], stem),
                "attempt": 1,
                "tagged_at": time.time(),
                # Where the image stands against another session's copy; a re-roll keeps it.
                "queued_at": time.time(),
                "ready_at": None,
                "decided_at": None,
                "candidates": [],
                "decision": None,
                "error": None,
            }
        )

    def confirm(self, stem: str, fixes: list[dict], second_round: bool = False) -> dict:
        """Queue an image with its fixes. From the second round a fixed,
        skipped or kept image is taken up again on the way."""
        allowed = tf.TRIAGE_STATUSES | (tf.SECOND_ROUND_STATUSES if second_round else frozenset())
        try:
            normalized = tf.normalize_fixes(fixes, self._settings.get("categories"))
        except ValueError as ex:
            return {"ok": False, "error": str(ex)}
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            if record.get("status") not in allowed:
                return {"ok": False, "error": f"Image is already {record.get('status')}"}
        sha = compute_file_sha256(self._entries[stem]["chosen_path"])
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            if record.get("status") not in allowed:
                return {"ok": False, "error": f"Image is already {record.get('status')}"}
            overlap = self._overlap_error_locked(stem, [fix.to_dict() for fix in normalized], None)
            if overlap:
                return {"ok": False, "error": overlap, "overlap": True}
            if not self._take_up_locked(stem, record, second_round):
                return {"ok": False, "error": f"Image is already {record.get('status')}"}
            self._queue(stem, record, normalized, sha)
            self._save_ledger()
            self._wake.notify_all()
        return {"ok": True, "status": tf.STATUS_QUEUED}

    def confirm_mask(
        self,
        stem: str,
        mask: str,
        instruction: str = "",
        creativity: float | None = None,
        second_round: bool = False,
        color_correction: str | None = None,
        crop_reference: float | None = None,
    ) -> dict:
        """Quick Mask: a painted region plus an instruction, sent straight
        off. From triage it queues the image; from the compare screen (ready,
        failed or kept) it re-queues it with the new mask; from the second
        round it takes a fixed, skipped or kept image up again.
        ``color_correction`` and ``crop_reference`` override the session's
        for this image (kept for its re-rolls)."""
        if not (mask or "").strip():
            return {"ok": False, "error": "Paint the area to repair first"}
        raw_fix: dict = {"id": tf.CATEGORY_MASK, "instruction": instruction or ""}
        if creativity is not None:
            raw_fix["creativity"] = creativity
        try:
            fixes = tf.normalize_fixes([raw_fix], self._settings.get("categories"))
        except ValueError as ex:
            return {"ok": False, "error": str(ex)}
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            status = record.get("status")
            allowed = tf.TRIAGE_STATUSES | {tf.STATUS_READY, tf.STATUS_FAILED, tf.STATUS_KEPT}
            if second_round:
                allowed |= tf.SECOND_ROUND_STATUSES
            if status not in allowed:
                return {"ok": False, "error": f"Image is already {status}"}
            item = self._entries[stem]
            size = (int(item["width"] or 0), int(item["height"] or 0))
            target = tf.mask_path(self._concept_dir, stem, self._session)
        if not size[0] or not size[1]:
            return {"ok": False, "error": "Cannot read the image size"}
        # Painted beside the current mask until it is known not to clash with
        # another session's; a refused mask leaves the queued one as it was.
        pending_mask = f"{os.path.splitext(target)[0]}.pending.png"
        try:
            tf.save_mask(mask, pending_mask, size)
        except (ValueError, OSError) as ex:
            return {"ok": False, "error": str(ex)}
        sha = compute_file_sha256(item["chosen_path"])
        with self._lock:
            record, error = self._require(stem)
            if error:
                with contextlib.suppress(OSError):
                    os.remove(pending_mask)
                return {"ok": False, "error": error}
            overlap = self._overlap_error_locked(stem, [fix.to_dict() for fix in fixes], pending_mask)
            if overlap:
                with contextlib.suppress(OSError):
                    os.remove(pending_mask)
                return {"ok": False, "error": overlap, "overlap": True}
            status = record.get("status")
            if status not in tf.TRIAGE_STATUSES | {tf.STATUS_READY, tf.STATUS_FAILED, tf.STATUS_KEPT} and not (
                second_round and status in tf.SECOND_ROUND_STATUSES
            ):
                with contextlib.suppress(OSError):
                    os.remove(pending_mask)
                return {"ok": False, "error": f"Image is already {status}"}
            os.replace(pending_mask, target)
            if status in tf.TRIAGE_STATUSES or (second_round and status in tf.SECOND_ROUND_STATUSES):
                self._take_up_locked(stem, record, second_round)
                self._queue(stem, record, fixes, sha)
            elif status in (tf.STATUS_READY, tf.STATUS_FAILED, tf.STATUS_KEPT):
                tf.delete_candidate_files(self._concept_dir, record)
                self._undo_stack = [u for u in self._undo_stack if u["stem"] != stem]
                record.update(
                    {
                        "status": tf.STATUS_QUEUED,
                        "chosen_sha256": sha,
                        "fixes": [fix.to_dict() for fix in fixes],
                        "seed": random.randrange(0, 2**31 - 1),
                        "attempt": int(record.get("attempt", 1)) + 1,
                        "candidates": [],
                        "ready_at": None,
                        "decided_at": None,
                        "decision": None,
                        "error": None,
                        "tagged_at": time.time(),
                    }
                )
            else:
                return {"ok": False, "error": f"Image is already {status}"}
            record["mask"] = tf.rel(self._concept_dir, target)
            overrides = tf.sanitize_overrides({"color_correction": color_correction, "crop_reference": crop_reference})
            if overrides:
                record["overrides"] = overrides
            self._save_ledger()
            self._wake.notify_all()
        return {"ok": True, "status": tf.STATUS_QUEUED}

    def confirm_all(self, mode: str, fixes: list[dict] | None = None) -> dict:
        """Queue every undecided image at once. ``default`` gives them all the
        same fixes; ``detected`` uses each image's detector tags and leaves
        images the detector found clean in triage."""
        categories = self._settings.get("categories")
        default_fixes: list[tf.TriageFix] = []
        if mode == "default":
            try:
                default_fixes = tf.normalize_fixes(fixes or [], categories)
            except ValueError as ex:
                return {"ok": False, "error": str(ex)}
        elif mode != "detected":
            return {"ok": False, "error": f"Unknown confirm-all mode: {mode}"}
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            plan: list[tuple[str, list[tf.TriageFix]]] = []
            for stem in self._stems:
                record = self._ledger["items"].get(stem) or {}
                if record.get("status") not in tf.TRIAGE_STATUSES:
                    continue
                if mode == "default":
                    plan.append((stem, default_fixes))
                else:
                    detected = tf.categorize(parse_segment_string(record.get("detected") or ""), categories or [])
                    if detected:
                        plan.append((stem, detected))
        hashes = {stem: compute_file_sha256(self._entries[stem]["chosen_path"]) for stem, _fixes in plan}
        queued = 0
        conflicts: dict[str, str] = {}
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            for stem, item_fixes in plan:
                record = self._ledger["items"].setdefault(stem, {})
                if record.get("status") not in tf.TRIAGE_STATUSES:
                    continue
                overlap = self._overlap_error_locked(stem, [fix.to_dict() for fix in item_fixes], None)
                if overlap:
                    conflicts[stem] = overlap
                    continue
                self._queue(stem, record, item_fixes, hashes[stem])
                queued += 1
            self._save_ledger()
            self._wake.notify_all()
        return {"ok": True, "queued": queued, "conflicts": conflicts}

    def skip(self, stem: str) -> dict:
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            if record.get("status") not in tf.TRIAGE_STATUSES | {tf.STATUS_FAILED}:
                return {"ok": False, "error": f"Image is already {record.get('status')}"}
            if record.get("status") == tf.STATUS_FAILED:
                tf.delete_candidate_files(self._concept_dir, record)
                record["candidates"] = []
            record.update({"status": tf.STATUS_SKIPPED, "decided_at": time.time()})
            self._save_ledger()
        return {"ok": True}

    def skip_many(self, stems: list[str]) -> dict:
        """Triage grid view: skip several images for good. Only ones still
        in triage are skipped; the rest are refused."""
        skipped: list[str] = []
        refused: list[str] = []
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            now = time.time()
            for stem in dict.fromkeys(stems or []):
                if (
                    stem not in self._entries
                    or self._ledger["items"].get(stem, {}).get("status") not in tf.TRIAGE_STATUSES
                ):
                    refused.append(stem)
                    continue
                record = self._ledger["items"].setdefault(stem, {})
                record.update({"status": tf.STATUS_SKIPPED, "decided_at": now})
                skipped.append(stem)
            if skipped:
                self._save_ledger()
        return {"ok": True, "skipped": skipped, "refused": refused}

    def undo_triage(self, stem: str) -> dict:
        """Take a confirm or skip back, as long as nothing has started."""
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            status = record.get("status")
            if status not in (tf.STATUS_QUEUED, tf.STATUS_SKIPPED):
                return {"ok": False, "error": "Already generating; decide it on the compare screen"}
            if record.get("detected"):
                record.update({"status": tf.STATUS_TAGGED, "fixes": [], "decided_at": None})
            else:
                self._ledger["items"].pop(stem, None)
            self._save_ledger()
        return {"ok": True, "status": tf.status_of(self._ledger, stem)}

    # ------------------------------------------------------------------ #
    # Compare
    # ------------------------------------------------------------------ #

    def _ready_item(self, stem: str) -> dict:
        record = self._ledger["items"][stem]
        item = self._entries[stem]
        candidates = [
            {**candidate, "path": tf.absolute(self._concept_dir, candidate.get("path"))}
            for candidate in record.get("candidates") or []
        ]
        candidates.sort(key=lambda c: int(c.get("k", 0)))
        return {
            "stem": stem,
            "pair_id": item["pair_id"],
            "caption": item["caption"],
            "original_path": item["chosen_path"],
            "width": item["width"],
            "height": item["height"],
            "fixes": record.get("fixes") or [],
            "attempt": record.get("attempt", 1),
            "seed": record.get("seed"),
            "candidates": candidates,
            "all_no_change": tf.all_no_change(record),
            "ready_at": record.get("ready_at"),
            "mask": tf.absolute(self._concept_dir, record.get("mask")) if record.get("mask") else None,
        }

    def next_ready(self) -> dict:
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            ready = tf.ready_stems(self._ledger)
            if ready:
                return {"ok": True, "item": self._ready_item(ready[0])}
            pending = any(
                r.get("status") in (tf.STATUS_QUEUED, tf.STATUS_GENERATING) for r in self._ledger["items"].values()
            )
            if not pending:
                return {"ok": True, "done": True}
            return {"ok": True, "waiting": True, "held": self._held}

    def pick(self, stem: str, k: int) -> dict:
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            if record.get("status") != tf.STATUS_READY:
                return {"ok": False, "error": "Image is not awaiting a decision"}
            candidate = next((c for c in record.get("candidates") or [] if int(c.get("k", -1)) == int(k)), None)
            if candidate is None or candidate.get("error") or not candidate.get("path"):
                return {"ok": False, "error": "That candidate is not available"}
            candidate_abs = tf.absolute(self._concept_dir, candidate["path"])
            if not candidate_abs or not os.path.isfile(candidate_abs):
                return {"ok": False, "error": "Candidate file is missing; re-queue the image"}
            item = self._entries[stem]
            entry = self._manifest_entry.get(stem)
            if entry is None:
                return {"ok": False, "error": "Manifest entry is missing"}
            if record.get("chosen_sha256") and compute_file_sha256(item["chosen_path"]) != record["chosen_sha256"]:
                return {"ok": False, "error": "The chosen file changed since it was tagged; re-queue the image"}
            note = {"fixes": record.get("fixes"), "mode": candidate.get("mode"), "seed": candidate.get("seed")}
            try:
                replace = tf.replace_chosen(self._concept_dir, self._manifest, entry, candidate_abs, note)
            except OSError as ex:
                return {"ok": False, "error": f"Could not replace the chosen image: {ex}"}
            placed = tf.absolute(self._concept_dir, replace["placed"])
            try:
                fix_pair = self._update_fix_pair(record, entry, item, placed, replace["backup"])
            except (OSError, RuntimeError) as ex:
                tf.undo_replace_chosen(self._concept_dir, self._manifest, entry, replace)
                return {"ok": False, "error": f"Could not write the fix pair: {ex}"}
            # The chosen file on disk is now the candidate; keep that in step.
            item["chosen_path"] = tf.absolute(self._concept_dir, replace["placed"])
            item["chosen_file"] = os.path.basename(item["chosen_path"])
            record.update(
                {
                    "status": tf.STATUS_FIXED,
                    "decided_at": time.time(),
                    "decision": {
                        "kind": "fixed",
                        "k": int(k),
                        **replace,
                        "fix_pair": fix_pair,
                        # Undo only while no other session has built on it.
                        "placed_sha256": entry.get("chosen_sha256"),
                    },
                }
            )
            self._undo_stack.append({"kind": "fixed", "stem": stem})
            self._save_ledger()
        return {"ok": True, "backup": replace["backup"], "fix_pair_id": fix_pair["pair_id"] if fix_pair else None}

    def _fix_manifest_for(self, fix_dir: str) -> dict:
        """The session's fix-pair manifest, or another folder's that an earlier
        round wrote to (loaded once). Caller holds the lock."""
        if self._fix_dir and os.path.normcase(os.path.abspath(fix_dir)) == os.path.normcase(self._fix_dir):
            return self._fix_manifest
        key = os.path.normcase(os.path.abspath(fix_dir))
        if key not in self._other_fix_manifests:
            self._other_fix_manifests[key] = self._loaded_fix_manifest(fix_dir) or tf.load_fix_manifest(fix_dir)
        return self._other_fix_manifests[key]

    def _peer_fixed_locked(self, stem: str) -> list[dict]:
        """Other sessions' records that repaired ``stem``, earliest first."""
        records = []
        for peer in self._peers():
            other = peer._ledger["items"].get(stem) or {}
            if tf.first_original(other):
                records.append(other)
        records.sort(key=lambda r: float(r.get("decided_at") or 0.0))
        return records

    def _update_fix_pair(self, record: dict, entry: dict, item: dict, placed: str, backup: str) -> dict | None:
        """Mirror an accepted repair into the fix-pair dataset. A later round
        replaces the chosen side of the pair an earlier round wrote, so the
        pair stays (latest repair, untouched original). Without one, a new
        pair against the original is written when the session has a fix-pair
        folder. Caller holds the lock."""
        prior = tf.prior_fix_pair(record)
        peer_fixed = self._peer_fixed_locked(item["stem"])
        if not prior:
            # Another session already paired this image: its pair takes the
            # combined repair, so the pair stays (latest, untouched original).
            for other in reversed(peer_fixed):
                prior = (other.get("decision") or {}).get("fix_pair") or tf.prior_fix_pair(other)
                if prior:
                    break
        if prior:
            manifest = self._fix_manifest_for(prior["dir"])
            if tf.find_fix_pair(manifest, int(prior["pair_id"])) is not None:
                return tf.replace_fix_pair_chosen(
                    prior["dir"], manifest, int(prior["pair_id"]), placed, tf.fix_pair_backup_dir(self._concept_dir)
                )
        if not self._fix_dir or self._fix_manifest is None:
            return None
        # Rejected is the image before any round, in any session, touched it.
        earliest = tf.first_original(peer_fixed[0]) if peer_fixed else None
        original = tf.absolute(self._concept_dir, tf.first_original(record) or earliest or backup)
        # The pair's .txt may have been corrected since the session
        # started; the transferred pair carries the caption as it is now.
        caption = tf.current_caption(self._concept_dir, entry, item["caption"])
        pair_id = tf.write_fix_pair(
            self._fix_dir,
            self._fix_manifest,
            placed,
            original,
            caption,
            item["aspectratio"],
            unconditional=item["unconditional"],
        )
        return {"dir": self._fix_dir, "pair_id": pair_id}

    def keep(self, stem: str) -> dict:
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            if record.get("status") != tf.STATUS_READY:
                return {"ok": False, "error": "Image is not awaiting a decision"}
            record.update({"status": tf.STATUS_KEPT, "decided_at": time.time(), "decision": {"kind": "kept"}})
            self._undo_stack.append({"kind": "kept", "stem": stem})
            self._save_ledger()
        return {"ok": True}

    def requeue(
        self, stem: str, fixes: list[dict] | None = None, lower: bool = False, lower_by: float | None = None
    ) -> dict:
        """Fresh candidates with a new seed and nothing else changed. With
        ``lower`` the re-roll also turns every fix's creativity down (floored
        at 0.1). The default notch is one sampling step, ``1 / steps``:
        SwarmUI starts a segment at ``round(steps * (1 - creativity))``, so at
        Klein's 4 steps anything smaller rounds back to the same request.
        ``lower_by`` overrides the notch. Re-queuing with other fixes takes
        the category values instead."""
        normalized: list[tf.TriageFix] | None = None
        if fixes:
            try:
                normalized = tf.normalize_fixes(fixes, self._settings.get("categories"))
            except ValueError as ex:
                return {"ok": False, "error": str(ex)}
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            if record.get("status") not in (tf.STATUS_READY, tf.STATUS_FAILED, tf.STATUS_KEPT):
                return {"ok": False, "error": f"Cannot re-queue an image that is {record.get('status')}"}
            if normalized is not None:
                overlap = self._overlap_error_locked(stem, [fix.to_dict() for fix in normalized], None)
                if overlap:
                    return {"ok": False, "error": overlap, "overlap": True}
            tf.delete_candidate_files(self._concept_dir, record)
            self._undo_stack = [u for u in self._undo_stack if u["stem"] != stem]
            record.update(
                {
                    "status": tf.STATUS_QUEUED,
                    "seed": random.randrange(0, 2**31 - 1),
                    "attempt": int(record.get("attempt", 1)) + 1,
                    "candidates": [],
                    "ready_at": None,
                    "decided_at": None,
                    "decision": None,
                    "error": None,
                    "tagged_at": time.time(),
                }
            )
            if normalized is not None:
                record["fixes"] = [fix.to_dict() for fix in normalized]
            elif lower:
                # A "Segment steps" override is what the segment pass really runs.
                segment_steps = (self._settings.get("segment_params") or {}).get("segmentsteps")
                steps = max(1, int(segment_steps or self._settings.get("steps") or tf.DEFAULT_KLEIN["steps"]))
                notch = float(lower_by) if lower_by else 1.0 / steps
                for fix in record.get("fixes") or []:
                    current = float(fix.get("creativity", tf.FIX_CREATIVITY))
                    fix["creativity"] = round(max(0.1, current - notch), 3)
            self._save_ledger()
            self._wake.notify_all()
        return {"ok": True}

    def retriage(self, stem: str) -> dict:
        """Send an image from the compare screen back to triage. A ready,
        failed or kept image loses its candidates and keeps its tags. A
        fixed image is taken up by the second round instead: a new round on
        top of the accepted repair, tags empty, and skipping it there (or
        undoing) restores the accepted fix."""
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            error = self._retriage_locked(stem, record)
            if error:
                return {"ok": False, "error": error}
            self._save_ledger()
            rounds = len(record.get("rounds") or [])
        return {"ok": True, "status": tf.STATUS_TAGGED, "round": rounds + 1}

    # ------------------------------------------------------------------ #
    # Second round
    # ------------------------------------------------------------------ #

    def _open_second_locked(self, stem: str, record: dict) -> None:
        """Take a fixed, skipped or kept image up again. What it was is kept
        in ``second_pass.prev`` so skipping or undoing puts it back; a fixed
        image's decision moves to ``rounds`` so the next pick is a later round.
        Caller holds the lock and sets the new status."""
        status = record.get("status")
        prev = {
            "status": status,
            "decision": record.get("decision"),
            "decided_at": record.get("decided_at"),
            "fixes": record.get("fixes") or [],
            "mask": record.get("mask"),
            "chosen_sha256": record.get("chosen_sha256"),
            "rounds": len(record.get("rounds") or []),
        }
        tf.delete_candidate_files(self._concept_dir, record)
        self._undo_stack = [u for u in self._undo_stack if u["stem"] != stem]
        if status == tf.STATUS_FIXED:
            record.setdefault("rounds", []).append(record.get("decision"))
        record.pop("mask", None)
        record.update(
            {
                "fixes": [],
                "candidates": [],
                "ready_at": None,
                "decided_at": None,
                "decision": None,
                "error": None,
                "second_pass": {"state": tf.SECOND_OPEN, "at": time.time(), "prev": prev},
            }
        )

    def _close_second_locked(self, record: dict, state: str | None) -> None:
        """Put an image taken up by the second round back as it was, then mark
        it ``state`` (``None`` = not visited). Caller holds the lock."""
        prev = (record.get("second_pass") or {}).get("prev")
        rounds = record.get("rounds") or []
        if prev is None:
            # Sent back before the second round kept a snapshot: the last
            # round is the accepted fix.
            prev = {
                "status": tf.STATUS_FIXED,
                "decision": rounds[-1],
                "decided_at": time.time(),
                "rounds": len(rounds) - 1,
            }
        del rounds[int(prev.get("rounds") or 0) :]
        if rounds:
            record["rounds"] = rounds
        else:
            record.pop("rounds", None)
        record.pop("mask", None)
        if prev.get("mask"):
            record["mask"] = prev["mask"]
        record.update(
            {
                "status": prev.get("status"),
                "decision": prev.get("decision"),
                "decided_at": prev.get("decided_at"),
                "fixes": prev.get("fixes") or [],
                "chosen_sha256": prev.get("chosen_sha256"),
                "candidates": [],
                "ready_at": None,
                "error": None,
            }
        )
        if state is None:
            record.pop("second_pass", None)
        else:
            record["second_pass"] = {"state": state, "at": time.time()}

    def skip_second(self, stems: list[str]) -> dict:
        """Pass images over in the second round. One not yet taken up is only
        marked; one taken up (tagged or queued) returns to what it was."""
        done: list[str] = []
        refused: list[str] = []
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            for stem in dict.fromkeys(stems or []):
                record = self._ledger["items"].get(stem) if stem in self._entries else None
                if record is None:
                    refused.append(stem)
                    continue
                status = record.get("status")
                state = tf.second_pass_state(record)
                if status in tf.SECOND_ROUND_STATUSES:
                    record["second_pass"] = {"state": tf.SECOND_SKIPPED, "at": time.time()}
                elif state == tf.SECOND_OPEN and status in tf.TRIAGE_STATUSES | {tf.STATUS_QUEUED}:
                    self._close_second_locked(record, tf.SECOND_SKIPPED)
                else:
                    refused.append(stem)
                    continue
                done.append(stem)
            if done:
                self._save_ledger()
        return {"ok": True, "skipped": done, "refused": refused}

    def clear_second(self) -> dict:
        """Start the second round over: every decided image it has handled
        (skipped there, or confirmed there and decided since) waits in it
        again. Images still in progress are left alone; repairs, rounds and
        fix pairs are untouched."""
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            cleared = 0
            for stem, record in self._ledger["items"].items():
                if stem in self._entries and record.get("status") in tf.SECOND_ROUND_STATUSES:
                    if record.pop("second_pass", None) is not None:
                        cleared += 1
            if cleared:
                self._save_ledger()
        return {"ok": True, "cleared": cleared}

    def undo_second(self, stem: str) -> dict:
        """Take a second-round confirm or skip back, as long as nothing has
        started: the image returns to the second round untouched."""
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            status = record.get("status")
            state = tf.second_pass_state(record)
            if state == tf.SECOND_SKIPPED and status in tf.SECOND_ROUND_STATUSES:
                record.pop("second_pass", None)
            elif state == tf.SECOND_OPEN and status in tf.TRIAGE_STATUSES | {tf.STATUS_QUEUED}:
                self._close_second_locked(record, None)
            else:
                return {"ok": False, "error": "Already generating; decide it on the compare screen"}
            self._save_ledger()
        return {"ok": True, "status": record.get("status")}

    def confirm_many(self, stems: list[str], fixes: list[dict], second_round: bool = True) -> dict:
        """Grid view: queue several images with the same fixes. Outside the
        second round a decided image is refused rather than taken up."""
        try:
            normalized = tf.normalize_fixes(fixes, self._settings.get("categories"))
        except ValueError as ex:
            return {"ok": False, "error": str(ex)}
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            stems = [s for s in dict.fromkeys(stems or []) if s in self._entries]
        hashes = {stem: compute_file_sha256(self._entries[stem]["chosen_path"]) for stem in stems}
        queued: list[str] = []
        refused: list[str] = []
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            fix_dicts = [fix.to_dict() for fix in normalized]
            conflicts: dict[str, str] = {}
            for stem in stems:
                record = self._ledger["items"].setdefault(stem, {})
                overlap = self._overlap_error_locked(stem, fix_dicts, None)
                if overlap:
                    conflicts[stem] = overlap
                    refused.append(stem)
                    continue
                if not self._take_up_locked(stem, record, second_round):
                    refused.append(stem)
                    continue
                self._queue(stem, record, normalized, hashes[stem])
                queued.append(stem)
            if queued:
                self._save_ledger()
                self._wake.notify_all()
        return {"ok": True, "queued": queued, "refused": refused, "conflicts": conflicts}

    def _take_up_locked(self, stem: str, record: dict, second_round: bool) -> bool:
        """Whether ``record`` may be queued from (second-round) triage; a
        decided image is taken up by the second round on the way."""
        status = record.get("status")
        if status in tf.TRIAGE_STATUSES:
            return True
        if second_round and status in tf.SECOND_ROUND_STATUSES:
            self._open_second_locked(stem, record)
            return True
        return False

    def add_fix_pairs(self, stems: list[str]) -> dict:
        """Write the fix pairs accepted repairs would have produced had the
        session had a fix-pair folder then: for a run started without one.
        Pairs are written in decision order, exactly as ``pick`` would have."""
        added: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            if not self._fix_dir or self._fix_manifest is None:
                return {"ok": False, "error": "Start the session with a fix pair folder first"}
            todo = []
            for stem in dict.fromkeys(stems or []):
                record = self._ledger["items"].get(stem) if stem in self._entries else None
                decision = (record or {}).get("decision") or {}
                if record is None or record.get("status") != tf.STATUS_FIXED or decision.get("fix_pair"):
                    skipped.append(stem)
                    continue
                todo.append((float(record.get("decided_at") or 0.0), stem, record, decision))
            for _at, stem, record, decision in sorted(todo, key=lambda row: (row[0], row[1])):
                item = self._entries[stem]
                entry = self._manifest_entry.get(stem)
                original = tf.absolute(self._concept_dir, tf.first_original(record))
                if entry is None or not original or not os.path.isfile(original):
                    errors.append(f"{stem}: the original image is gone")
                    continue
                if not os.path.isfile(item["chosen_path"]):
                    errors.append(f"{stem}: the fixed image is missing")
                    continue
                try:
                    decision["fix_pair"] = self._update_fix_pair(
                        record, entry, item, item["chosen_path"], decision.get("backup")
                    )
                except (OSError, RuntimeError, KeyError) as ex:
                    errors.append(f"{stem}: {ex}")
                    continue
                added.append(stem)
            if added:
                self._save_ledger()
        return {"ok": True, "added": added, "skipped": skipped, "errors": errors}

    def _retriage_locked(self, stem: str, record: dict) -> str | None:
        status = record.get("status")
        if status == tf.STATUS_FIXED:
            self._open_second_locked(stem, record)
            record.update({"status": tf.STATUS_TAGGED, "chosen_sha256": None})
            return None
        if status not in (tf.STATUS_READY, tf.STATUS_FAILED, tf.STATUS_KEPT):
            return f"Cannot send back an image that is {status}"
        tf.delete_candidate_files(self._concept_dir, record)
        self._undo_stack = [u for u in self._undo_stack if u["stem"] != stem]
        record.update(
            {
                "status": tf.STATUS_TAGGED,
                "candidates": [],
                "ready_at": None,
                "decided_at": None,
                "decision": None,
                "error": None,
            }
        )
        return None

    def delete_pair(self, stem: str) -> dict:
        """The pair is unfixable: take it out of the dataset. Reversible
        through undo while the session lasts; the files sit in
        ``.triage_fix/deleted/`` afterwards."""
        with self._lock:
            record, error = self._require(stem)
            if error:
                return {"ok": False, "error": error}
            status = record.get("status")
            if status == tf.STATUS_GENERATING or any(s == stem for s, _k in self._in_flight):
                return {"ok": False, "error": "Still generating; wait for it or pause first"}
            if status == tf.STATUS_DELETED:
                return {"ok": False, "error": "Already deleted"}
            peer = self._pending_peer_locked(stem)
            if peer is not None:
                return {"ok": False, "error": f'Session "{peer._name}" has this image queued; decide it there first'}
            entry = self._manifest_entry.get(stem)
            if entry is None:
                return {"ok": False, "error": "Manifest entry is missing"}
            tf.delete_candidate_files(self._concept_dir, record)
            try:
                removed = tf.delete_pair(self._concept_dir, self._manifest, entry)
            except (OSError, KeyError) as ex:
                return {"ok": False, "error": f"Could not delete the pair: {ex}"}
            self._undo_stack = [u for u in self._undo_stack if u["stem"] != stem]
            record.update(
                {
                    "status": tf.STATUS_DELETED,
                    "candidates": [],
                    "ready_at": None,
                    "decided_at": time.time(),
                    "decision": {"kind": "deleted", "before": status, **removed},
                }
            )
            self._undo_stack.append({"kind": "deleted", "stem": stem})
            self._save_ledger()
        return {"ok": True, "folder": removed["folder"]}

    def undo_decision(self) -> dict:
        with self._lock:
            if not self._active or self._ledger is None:
                return {"ok": False, "error": "No active session"}
            if not self._undo_stack:
                return {"ok": False, "error": "Nothing to undo"}
            last = self._undo_stack[-1]
            stem = last["stem"]
            record = self._ledger["items"].get(stem) or {}
            if record.get("status") != last["kind"]:
                self._undo_stack.pop()
                return {"ok": False, "error": "That decision was already superseded"}
            peer = self._pending_peer_locked(stem)
            if peer is not None:
                return {
                    "ok": False,
                    "error": f'Session "{peer._name}" has taken {stem} up since; decide it there first',
                }
            placed_sha = (record.get("decision") or {}).get("placed_sha256")
            entry_now = self._manifest_entry.get(stem) or {}
            if last["kind"] == "fixed" and placed_sha and entry_now.get("chosen_sha256") != placed_sha:
                return {
                    "ok": False,
                    "error": f"Another session has repaired {stem} on top of this; undo it there first",
                }
            if last["kind"] == "deleted":
                decision = record.get("decision") or {}
                try:
                    entry = tf.undo_delete_pair(self._concept_dir, self._manifest, decision)
                except OSError as ex:
                    return {"ok": False, "error": f"Could not undo: {ex}"}
                self._manifest_entry[stem] = entry
                self._undo_stack.pop()
                # Its candidates are gone, so it comes back to triage with its tags.
                if record.get("fixes"):
                    record.update({"status": tf.STATUS_TAGGED, "decided_at": None, "decision": None})
                else:
                    record.pop("status", None)
                    record.update({"decided_at": None, "decision": None})
                self._save_ledger()
                return {"ok": True, "undone": "deleted", "stem": stem}
            if last["kind"] == "fixed":
                decision = record.get("decision") or {}
                entry = self._manifest_entry.get(stem)
                fix_pair = decision.get("fix_pair")
                try:
                    if fix_pair and fix_pair.get("replaced"):
                        tf.undo_replace_fix_pair_chosen(self._fix_manifest_for(fix_pair["dir"]), fix_pair)
                    elif fix_pair and self._fix_manifest is not None:
                        tf.undo_fix_pair(fix_pair["dir"], self._fix_manifest, int(fix_pair["pair_id"]))
                    tf.undo_replace_chosen(self._concept_dir, self._manifest, entry, decision)
                except OSError as ex:
                    return {"ok": False, "error": f"Could not undo: {ex}"}
                item = self._entries[stem]
                item["chosen_path"] = tf.absolute(self._concept_dir, decision["old_path"])
                item["chosen_file"] = os.path.basename(item["chosen_path"])
            self._undo_stack.pop()
            record.update({"status": tf.STATUS_READY, "decided_at": None, "decision": None})
            self._save_ledger()
        return {"ok": True, "undone": last["kind"], "stem": stem}

    # ------------------------------------------------------------------ #
    # Background work
    # ------------------------------------------------------------------ #

    def _live(self, generation: int, stop: threading.Event) -> bool:
        return self._generation == generation and not stop.is_set()

    def _next_queued(self) -> str | None:
        """The oldest queued image no other session is holding."""
        best: tuple[float, str] | None = None
        peers = self._peers()
        for stem, record in self._ledger["items"].items():
            if record.get("status") != tf.STATUS_QUEUED or stem not in self._entries:
                continue
            if peers and self._blocked_by_locked(stem, record) is not None:
                continue
            key = (float(record.get("tagged_at") or 0.0), stem)
            if best is None or key < best:
                best = key
        return best[1] if best else None

    def _drain(self, generation: int, stop: threading.Event) -> None:
        concurrency = int(self._settings.get("concurrency", DEFAULT_CONCURRENCY))
        options = int(self._settings.get("options_per_image", DEFAULT_OPTIONS_PER_IMAGE))
        pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="triage-fix")
        with self._lock:
            self._pool = pool
            self._contexts = {}
        try:
            while True:
                with self._wake:
                    stem = None
                    while self._live(generation, stop):
                        if not self._held and len(self._in_flight) < concurrency * 2:
                            stem = self._next_queued()
                            if stem is not None:
                                break
                        self._wake.wait(1.0)
                    if not self._live(generation, stop):
                        return
                    record = self._ledger["items"][stem]
                    problem = self._dispatch_problem_locked(stem, record)
                    if problem:
                        record.update({"status": tf.STATUS_FAILED, "candidates": [], "error": problem})
                        self._save_ledger()
                        continue
                    record["status"] = tf.STATUS_GENERATING
                    record["candidates"] = []
                    # The file this run repairs: another session may have
                    # replaced it since the confirm, and a pick checks it.
                    record["chosen_sha256"] = compute_file_sha256(self._entries[stem]["chosen_path"])
                    self._save_ledger()
                    context = {
                        "stem": stem,
                        "item": dict(self._entries[stem]),
                        "fixes": [tf.TriageFix.from_dict(f) for f in record.get("fixes") or []],
                        "seed": int(record.get("seed") or 0),
                        "attempt": int(record.get("attempt", 1)),
                        "settings": self._settings,
                        "concept_dir": self._concept_dir,
                        "session": self._session,
                        "client": dict(self._client_opts),
                        "mask": tf.absolute(self._concept_dir, record.get("mask")) if record.get("mask") else None,
                        "overrides": tf.sanitize_overrides(record.get("overrides")),
                    }
                    # Only the first option goes out now. If segmentation finds
                    # nothing it comes back unchanged and the other options
                    # would too, so they are submitted only after a real change.
                    self._contexts[stem] = context
                    self._submit_locked(generation, stem, 0, options)
        except Exception as ex:  # noqa: BLE001
            logger.exception("Triage & Fix worker failed")
            with self._lock:
                if self._generation == generation:
                    self._error = str(ex)
                    self._phase = "error"
                    self._active = False
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _fetch(client: SwarmClient, views: list[str]) -> bytes:
        if not views:
            raise SwarmError("SwarmUI returned no images")
        view = views[0]
        if view.startswith("data:"):
            import base64

            return base64.b64decode(view.split(",", 1)[1])
        return client.fetch_view(view)

    def _candidate_job(self, context: dict, k: int) -> dict:
        """One candidate: Mode A, then Mode B when an "Other" fix found
        nothing to segment. Runs on the pool; touches no service state."""
        item = context["item"]
        settings = context["settings"]
        fixes: list[tf.TriageFix] = context["fixes"]
        seed = context["seed"] + k
        size = (int(item["width"] or 0), int(item["height"] or 0))
        if not size[0] or not size[1]:
            raise SwarmError("Cannot read the image size")
        client = SwarmClient(context["client"]["base_url"], context["client"]["auth"])
        target = tf.candidate_path(context["concept_dir"], context["stem"], k, context.get("session", tf.MAIN_SESSION))
        provenance = {
            "source": tf.rel(context["concept_dir"], item["chosen_path"]),
            "fixes": [fix.to_dict() for fix in fixes],
            "mode": tf.MODE_SEGMENT,
            "klein": {
                key: settings.get(key)
                for key in (
                    "model",
                    "steps",
                    "cfgscale",
                    "sampler",
                    "scheduler",
                    "sigmashift",
                    "loras",
                    "color_correction",
                    "crop_reference",
                )
            },
            "segment_params": settings.get("segment_params") or {},
            "seed": seed,
            "attempt": context["attempt"],
        }
        uri = tf.image_data_uri(item["chosen_path"])
        mask_fix = next((fix for fix in fixes if fix.id == tf.CATEGORY_MASK), None)
        if mask_fix is not None:
            return self._mask_candidate(context, k, mask_fix, item, settings, size, seed, uri, target, provenance)
        template = settings.get("prompt_template") or tf.DEFAULT_PROMPT_TEMPLATE
        photo = next((fix for fix in fixes if fix.id == tf.CATEGORY_PHOTO), None)
        if photo is None:
            return self._segment_candidate(context, k, fixes, client, size, seed, uri, target, provenance, template)
        # Whole-image Klein edit first; any segment fixes then repair the
        # edited frame, so their no-change ladder compares against it.
        instruction = tf.segment_sub_prompt(photo, item["caption"], template)
        kwargs = tf.build_edit_kwargs(instruction, size, settings, seed, uri, photo.creativity)
        data = self._fetch(client, client.generate(**kwargs))
        provenance = {**provenance, "mode": tf.MODE_EDIT, "instruction": instruction, "creativity": photo.creativity}
        rest = [fix for fix in fixes if fix.id != tf.CATEGORY_PHOTO]
        if not rest:
            tf.write_candidate(data, item["chosen_path"], target, provenance)
            diff, changed = tf.candidate_change(item["chosen_path"], target)
            return {
                "k": k,
                "path": tf.rel(context["concept_dir"], target),
                "seed": seed,
                "mode": tf.MODE_EDIT,
                "rung": tf.RUNG_TAGGED,
                "fallback": "",
                "no_change": tf.is_no_change(changed),
                "diff": round(diff, 3),
                "changed": round(changed, 5),
                "error": None,
            }
        edited = f"{os.path.splitext(target)[0]}.photo-base.png"
        tf.write_candidate(data, item["chosen_path"], edited, provenance)
        try:
            result = self._segment_candidate(
                context, k, rest, client, size, seed, tf.image_data_uri(edited), target, provenance, template, edited
            )
        finally:
            with contextlib.suppress(OSError):
                os.remove(edited)
        # Against the original the edit always counts as a change.
        diff, changed = tf.candidate_change(item["chosen_path"], target)
        return {
            **result,
            "mode": tf.MODE_EDIT,
            "no_change": tf.is_no_change(changed),
            "diff": round(diff, 3),
            "changed": round(changed, 5),
        }

    def _segment_candidate(
        self, context, k, fixes, client, size, seed, uri, target, provenance, template, compare_path=None
    ) -> dict:
        """Mode A on ``uri``, then Mode B when an "Other" fix found nothing.
        ``compare_path`` is the image the no-change check measures against
        (the original unless a whole-image edit ran first)."""
        item = context["item"]
        settings = context["settings"]
        compare_path = compare_path or item["chosen_path"]
        # Every Other selection; together they are the fallback edit's prompt.
        others = [fix for fix in fixes if fix.id == tf.CATEGORY_OTHER]
        mode = tf.MODE_SEGMENT
        segment_error: SwarmError | None = None
        # The first option climbs the detection ladder while nothing changes;
        # the extra seeds reuse the rung it settled on.
        rungs = tf.ladder_rungs(fixes) if k == 0 else [int(context.get("rung", tf.RUNG_TAGGED))]
        rung = rungs[0]
        diff, changed = 0.0, 0.0
        try:
            for rung in rungs:
                rung_fixes = [fix.at_rung(rung) for fix in fixes]
                kwargs = tf.build_segment_kwargs(
                    item["caption"], rung_fixes, size, settings, settings.get("segment_params"), seed, uri, template
                )
                data = self._fetch(client, client.generate(**kwargs))
                tf.write_candidate(data, item["chosen_path"], target, {**provenance, "rung": rung})
                diff, changed = tf.candidate_change(compare_path, target)
                if not tf.is_no_change(changed):
                    break
        except SwarmError as ex:
            if not others:
                raise
            segment_error = ex
            diff, changed = 0.0, 0.0
        no_change = tf.is_no_change(changed)
        if others and (no_change or segment_error is not None):
            kwargs = tf.build_edit_kwargs(tf.edit_instructions(others), size, settings, seed, uri)
            data = self._fetch(client, client.generate(**kwargs))
            provenance = {**provenance, "mode": tf.MODE_EDIT, "instruction": kwargs["prompt"]}
            tf.write_candidate(data, item["chosen_path"], target, provenance)
            diff, changed = tf.candidate_change(compare_path, target)
            no_change = tf.is_no_change(changed)
            mode = tf.MODE_EDIT
        return {
            "k": k,
            "path": tf.rel(context["concept_dir"], target),
            "seed": seed,
            "mode": mode,
            "rung": rung,
            "fallback": tf.RUNG_LABELS.get(rung, ""),
            "no_change": no_change,
            "diff": round(diff, 3),
            "changed": round(changed, 5),
            "error": None,
        }

    def _mask_candidate(self, context, k, fix, item, settings, size, seed, uri, target, provenance) -> dict:
        """Mode C: the painted mask decides the region; nothing to fall back to."""
        mask_file = context.get("mask")
        if not mask_file or not os.path.isfile(mask_file):
            raise SwarmError("The quick mask file is missing; paint it again")
        client = SwarmClient(context["client"]["base_url"], context["client"]["auth"])
        prompt = tf.segment_sub_prompt(
            fix, item["caption"], settings.get("prompt_template") or tf.DEFAULT_PROMPT_TEMPLATE
        )
        overrides = context.get("overrides") or {}
        kwargs = tf.build_mask_kwargs(
            prompt,
            size,
            {**settings, **overrides},
            settings.get("mask_params"),
            seed,
            uri,
            tf.image_data_uri(mask_file),
            fix.creativity,
        )
        data = self._fetch(client, client.generate(**kwargs))
        provenance = {
            **provenance,
            "mode": tf.MODE_MASK,
            "instruction": prompt,
            "mask": tf.rel(context["concept_dir"], mask_file),
            "mask_params": settings.get("mask_params") or {},
            "color_correction": overrides.get("color_correction", settings.get("color_correction")),
            "crop_reference": overrides.get("crop_reference", settings.get("crop_reference")),
        }
        tf.write_candidate(data, item["chosen_path"], target, provenance)
        diff, changed = tf.candidate_change(item["chosen_path"], target)
        return {
            "k": k,
            "path": tf.rel(context["concept_dir"], target),
            "seed": seed,
            "mode": tf.MODE_MASK,
            "no_change": tf.is_no_change(changed),
            "diff": round(diff, 3),
            "changed": round(changed, 5),
            "error": None,
        }

    def _submit_locked(self, generation: int, stem: str, k: int, options: int) -> None:
        """Queue candidate ``k`` of ``stem`` on the pool. Caller holds the lock."""
        context = self._contexts.get(stem)
        if self._pool is None or context is None:
            return
        future = self._pool.submit(self._candidate_job, context, k)
        self._in_flight[(stem, k)] = future
        future.add_done_callback(lambda fut: self._on_candidate(generation, stem, k, options, fut))

    def _on_candidate(self, generation: int, stem: str, k: int, options: int, future: Future) -> None:
        with self._lock:
            if self._generation != generation:
                return
            self._in_flight.pop((stem, k), None)
            # Withdrawn by a hold before it started; set_hold re-queued the item.
            if future.cancelled():
                return
            record = self._ledger["items"].get(stem) if self._ledger else None
            if record is None or record.get("status") != tf.STATUS_GENERATING:
                return
            try:
                result = future.result()
                self._consecutive_failures = 0
            except Exception as ex:  # noqa: BLE001 — any failure is this candidate's
                if not isinstance(ex, SwarmError):
                    logger.exception("candidate %s/%s failed", stem, k)
                result = {"k": k, "path": None, "seed": None, "mode": None, "no_change": False, "error": str(ex)}
                self._consecutive_failures += 1
            record.setdefault("candidates", []).append(result)
            # After the first option: more seeds only when it changed something.
            # A failure or an untouched image would only repeat itself.
            if k == 0 and options > 1 and not result.get("error") and not result.get("no_change"):
                context = self._contexts.get(stem)
                if context is not None:
                    context["rung"] = int(result.get("rung", tf.RUNG_TAGGED))
                for extra in range(1, options):
                    self._submit_locked(generation, stem, extra, options)
            still_running = any(s == stem for s, _k in self._in_flight)
            if not still_running:
                self._contexts.pop(stem, None)
                good = [c for c in record["candidates"] if not c.get("error")]
                if good:
                    record["status"] = tf.STATUS_READY
                    record["ready_at"] = time.time()
                    record["error"] = None
                else:
                    record["status"] = tf.STATUS_FAILED
                    record["error"] = record["candidates"][-1].get("error")
            self._save_ledger()
            if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._error = f"Stopped after {self._consecutive_failures} SwarmUI failures in a row: {result['error']}"
                self._phase = "error"
                self._halt_locked()
            self._wake.notify_all()

    def _halt_locked(self) -> None:
        """Stop generating but keep the session's ledger; queued items wait
        for the next start. Caller holds the lock."""
        self._stop.set()
        self._generation += 1
        self._active = False
        for stem, _k in list(self._in_flight):
            record = self._ledger["items"].get(stem)
            if record is not None and record.get("status") == tf.STATUS_GENERATING:
                tf.delete_candidate_files(self._concept_dir, record)
                record["candidates"] = []
                record["status"] = tf.STATUS_QUEUED
        self._in_flight = {}
        self._contexts = {}
        self._save_ledger()

    def _scan(self, generation: int, stop: threading.Event) -> None:
        """Pre-tag undecided images with the detector's segment string."""
        detector = self._detector
        if detector is None:
            return
        for stem in list(self._stems):
            if not self._live(generation, stop):
                return
            with self._lock:
                record = self._ledger["items"].get(stem) or {}
                if record.get("status") is not None or "detected" in record:
                    continue
                path = self._entries[stem]["chosen_path"]
            try:
                text = detector.detect(path) or ""
            except Exception as ex:  # noqa: BLE001
                logger.warning("detector failed on %s: %s", path, ex)
                continue
            with self._lock:
                if not self._live(generation, stop):
                    return
                record = self._ledger["items"].setdefault(stem, {})
                if record.get("status") is not None:
                    continue
                record["detected"] = text
                if text.strip():
                    record["status"] = tf.STATUS_TAGGED
                self._save_ledger()
