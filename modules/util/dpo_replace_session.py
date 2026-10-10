"""Durable state and recoverable file transactions for blind pair replacement.

The older dpo_replace_util module is also used by other tools; its public
helpers remain intact. This module deliberately has no scoring dependency.
"""

import hashlib
import json
import math
import os
import random
import shutil
import uuid
from contextlib import suppress
from pathlib import Path


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def fingerprint(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def caption_text(path: str | Path) -> str:
    # newline="" is essential: a caption is data, including its line endings.
    with Path(path).open(encoding="utf-8", newline="") as stream:
        caption = stream.read()
    if not caption.strip():
        raise ValueError("The chosen image's .txt caption is empty")
    return caption


def write_caption(path: str | Path, caption: str) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        stream.write(caption)


def creation_time(path: str) -> float:
    stat = os.stat(path)
    return getattr(stat, "st_birthtime", stat.st_ctime if os.name == "nt" else stat.st_mtime)


def weighted_shuffle(items: list[dict], rng: random.Random) -> list[dict]:
    """Exponential race: weighted sampling without replacement, oldest 1.5x."""
    ranked = sorted(items, key=lambda item: item["created"])
    count = len(ranked)
    weighted = []
    for index, item in enumerate(ranked):
        weight = 1.5 - 0.5 * index / max(1, count - 1)
        weighted.append((-math.log(max(rng.random(), 1e-15)) / weight, item))
    return [item for _, item in sorted(weighted, key=lambda value: value[0])]


def snapshot_files(paths: list[str], directory: Path) -> dict[str, str | None]:
    directory.mkdir(parents=True, exist_ok=True)
    backups = {}
    for index, name in enumerate(dict.fromkeys(paths)):
        source = Path(name)
        target = directory / f"{index}{source.suffix}"
        if source.is_file():
            shutil.copy2(source, target)
            backups[str(source)] = str(target)
        else:
            backups[str(source)] = None
    return backups


def restore_files(backups: dict[str, str | None]) -> None:
    for name, backup in backups.items():
        target = Path(name)
        if backup is None:
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".replace-tmp")
            shutil.copy2(backup, tmp)
            os.replace(tmp, target)


class SessionStore:
    def __init__(self, concept: str):
        self.root = Path(concept) / ".replace_pairs"
        self.ledger_path = self.root / "session.json"
        self.journal_path = self.root / "transaction.json"
        self.concept = Path(concept)

    def load(self) -> dict | None:
        self.recover()
        if not self.ledger_path.exists():
            return None
        ledger = read_json(self.ledger_path)
        if ledger.get("version") != 1 or not isinstance(ledger.get("items"), dict):
            raise ValueError("Unsupported Replace Pairs session; existing files were left intact")
        return ledger

    def save(self, ledger: dict) -> None:
        atomic_json(self.ledger_path, ledger)

    def begin(self, paths: list[str]) -> dict:
        self.recover()
        directory = self.concept / ".replaced" / ("replace-" + uuid.uuid4().hex)
        backups = snapshot_files(paths, directory)
        # Manifest/ledger copies are only for crash recovery. Retaining a full
        # dataset ledger per decision would consume quadratic disk space.
        transient = snapshot_files([str(self.concept / "manifest.json"), str(self.ledger_path)], self.root / "rollback")
        journal = {"backups": {**backups, **transient}, "transient": list(transient.values()), "committed": False}
        atomic_json(self.journal_path, journal)
        return journal

    def commit(self, journal: dict) -> None:
        # Commit marker is the transaction boundary. Recovery rolls back any
        # interruption before it, including a crash after saving the ledger.
        atomic_json(self.journal_path, {**journal, "committed": True})
        # A cleanup failure cannot turn a committed decision into a failure.
        with suppress(OSError):
            self._cleanup(journal)

    def _cleanup(self, journal: dict) -> None:
        for name in journal.get("transient", []):
            if name:
                Path(name).unlink(missing_ok=True)
        self.journal_path.unlink(missing_ok=True)

    def recover(self) -> None:
        if self.journal_path.exists():
            journal = read_json(self.journal_path)
            if not journal["committed"]:
                restore_files(journal["backups"])
                # Recovery itself may be interrupted during cleanup. Mark the
                # rollback complete before deleting its temporary snapshots.
                atomic_json(self.journal_path, {**journal, "committed": True})
            self._cleanup(journal)
