"""On-disk cache for the offline text reward's per-image source work.

With ``TextRewardSource.OFFLINE_IMAGES`` every reward sample starts from a
pre-generated image. Its VAE latent and the frozen detector's word boxes (with
their no-grad readings) depend only on the image and a handful of settings,
never on the policy, so they are computed once and kept under
``<cache_dir>/text_reward/`` as one ``.pt`` file per (image, settings) pair.

The entry is plain data — tensors, lists, strings, numbers — so it loads with
``torch.load(..., weights_only=True)`` and no custom class is unpickled. The
full key is hashed into the filename *and* stored in the file; a stored key
that differs from the requested one (or a schema bump, or an unreadable file)
is a miss and the entry is recomputed and overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

#: Bump when the entry layout, the encode preprocessing or the detection
#: procedure (e.g. its multi-scale set) changes, so old entries become misses.
TEXT_REWARD_CACHE_SCHEMA = 1
TEXT_REWARD_CACHE_SUBDIR = "text_reward"


def text_reward_image_identity(image_path: str, sha256: str | None = None) -> str:
    """Stable identity of an offline source image.

    With the dataset tool's content ``sha256`` the identity survives moves and
    copies; the file size is folded in as a cheap guard against the PNG being
    edited in place without its JSON being rewritten. Without a hash it falls
    back to absolute path + size + ``mtime_ns``, so touching or replacing the
    file invalidates its entry."""
    stat = os.stat(image_path)
    if sha256:
        return f"sha256:{sha256.strip().lower()}:{stat.st_size}"
    return f"file:{os.path.normcase(os.path.abspath(image_path))}:{stat.st_size}:{stat.st_mtime_ns}"


def text_reward_cache_key(
    *,
    image_identity: str,
    resolution: int,
    vae_identity: str,
    detector_arch: str,
    recognizer_arch: str,
    min_box_confidence: float,
    min_box_height: float,
) -> dict:
    """Everything the cached latent and detection depend on. ``resolution`` is
    the effective encode target (after the //16 rounding)."""
    return {
        "schema": TEXT_REWARD_CACHE_SCHEMA,
        "image": str(image_identity),
        "resolution": int(resolution),
        "vae": str(vae_identity),
        "detector": str(detector_arch),
        "recognizer": str(recognizer_arch),
        "min_box_confidence": float(min_box_confidence),
        "min_box_height": float(min_box_height),
    }


@dataclass
class CachedTextRewardSource:
    latent: Tensor  #: scaled source latent ``(1, C, h, w)``, float32, CPU
    boxes: list[tuple[float, float, float, float]]  #: relative, reading order
    readings: list[str]  #: detector-time no-grad reading of each box


class TextRewardSourceCache:
    def __init__(self, folder: str):
        self.folder = folder
        self._warned: set[str] = set()

    def _warn_once(self, kind: str, message: str) -> None:
        if kind not in self._warned:
            self._warned.add(kind)
            print(message)

    def path_for(self, key: dict) -> str:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:40]
        return os.path.join(self.folder, digest + ".pt")

    def load(self, key: dict) -> CachedTextRewardSource | None:
        """The entry for ``key``, or ``None`` on a miss, a key/schema
        mismatch, or an unreadable file (reported once per cache)."""
        path = self.path_for(key)
        if not os.path.isfile(path):
            return None
        try:
            entry = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(entry, dict) or entry.get("key") != key:
                return None
            latent, boxes, readings = entry["latent"], entry["boxes"], entry["readings"]
            if not isinstance(latent, Tensor) or latent.dim() != 4:
                raise ValueError("latent is not a 4-d tensor")
            if not isinstance(boxes, Tensor) or boxes.dim() != 2 or boxes.shape[1] != 4:
                raise ValueError("boxes are not an (N, 4) tensor")
            if not isinstance(readings, list) or len(readings) != boxes.shape[0]:
                raise ValueError("one reading per box expected")
            return CachedTextRewardSource(
                latent=latent.float(),
                boxes=[tuple(float(v) for v in row) for row in boxes.tolist()],
                readings=[str(r) for r in readings],
            )
        except Exception as exception:  # noqa: BLE001 - a bad cache entry must never abort a run
            self._warn_once(
                "corrupt",
                f"text reward cache: unreadable entry {path} ({exception}); recomputing "
                "(further unreadable entries are recomputed silently)",
            )
            return None

    def save(
        self,
        key: dict,
        latent: Tensor,
        boxes: Sequence[Sequence[float]],
        readings: Sequence[str],
    ) -> bool:
        """Atomically write (tmp file + ``os.replace``). Returns ``False``
        instead of raising when the write fails."""
        path = self.path_for(key)
        tmp = f"{path}.{os.getpid()}.tmp"
        entry = {
            "key": key,
            "latent": latent.detach().to(device="cpu", dtype=torch.float32).contiguous().clone(),
            "boxes": torch.tensor([list(map(float, b)) for b in boxes], dtype=torch.float64).reshape(-1, 4),
            "readings": [str(r) for r in readings],
        }
        try:
            os.makedirs(self.folder, exist_ok=True)
            torch.save(entry, tmp)
            os.replace(tmp, path)
            return True
        except Exception as exception:  # noqa: BLE001 - caching is an optimisation only
            self._warn_once("write", f"text reward cache: could not write {path} ({exception}); continuing uncached")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return False
