import logging
import threading
import urllib.parse
import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from web.backend.services._singleton import SingletonMixin
from web.backend.services.config_service import ConfigService

logger = logging.getLogger(__name__)


class ValidationService(SingletonMixin):

    def __init__(self) -> None:
        self._scan_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status: str = "idle"  # idle, scanning, done
        self._results: dict = {}
        self._progress: dict = {"label": "", "current": 0, "total": 0}
        self._ws_broadcast: Callable[[dict], None] | None = None

    def set_ws_broadcast(self, fn: Callable[[dict], None]) -> None:
        self._ws_broadcast = fn

    def _broadcast(self, message: dict) -> None:
        if self._ws_broadcast is not None:
            with suppress(Exception):
                self._ws_broadcast(message)

    def get_status(self) -> dict:
        return {
            "status": self._status,
            "progress": self._progress,
        }

    def get_results(self) -> dict:
        return self._results

    def scan_basic(self) -> dict:
        with self._lock:
            if self._status == "scanning":
                return {"ok": False, "error": "Scan already in progress"}
            self._status = "scanning"
            self._stop_event.clear()

        self._scan_thread = threading.Thread(target=self._run_basic_scan, daemon=True)
        self._scan_thread.start()
        return {"ok": True}

    def scan_deep(self, threshold: float = 0.93) -> dict:
        with self._lock:
            if self._status == "scanning":
                return {"ok": False, "error": "Scan already in progress"}
            self._status = "scanning"
            self._stop_event.clear()

        self._scan_thread = threading.Thread(
            target=self._run_deep_scan, args=(threshold,), daemon=True
        )
        self._scan_thread.start()
        return {"ok": True}

    def cancel(self) -> dict:
        self._stop_event.set()
        return {"ok": True}

    def remove_match(self, val_image_path: str) -> dict:
        from modules.util.validation_checker_util import (
            ValidationCheckerImage,
            remove_validation_image,
        )

        # Find the image in cached results
        image = self._find_val_image(val_image_path)
        if image is None:
            return {"ok": False, "error": "Image not found in results"}

        remove_validation_image(image)
        self._remove_matches_for(val_image_path)
        return {"ok": True}

    def move_match(self, val_image_path: str) -> dict:
        from modules.util.validation_checker_util import (
            collect_validation_checker_concepts,
            move_validation_image_to_train,
        )

        image = self._find_val_image(val_image_path)
        if image is None:
            return {"ok": False, "error": "Image not found in results"}

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []
        train_concepts, _ = collect_validation_checker_concepts(concepts)
        if not train_concepts:
            return {"ok": False, "error": "No training concepts found"}

        new_path = move_validation_image_to_train(image, train_concepts[0])
        self._remove_matches_for(val_image_path)
        return {"ok": True, "new_path": new_path}

    # --- Internal ---

    def _progress_callback(self, label: str, current: int, total: int) -> None:
        self._progress = {"label": label, "current": current, "total": total}
        self._broadcast({
            "type": "validation:progress",
            "label": label,
            "current": current,
            "total": total,
        })

    def _run_basic_scan(self) -> None:
        try:
            from modules.util.validation_checker_util import (
                CaptionMatch,
                ValidationCheckerMatch,
                scan_basic_validation_matches,
            )

            config_service = ConfigService.get_instance()
            concepts = config_service.get_config_for_training().concepts or []

            matches: list[dict] = []
            captions: list[dict] = []

            def on_match(match: ValidationCheckerMatch) -> None:
                matches.append(self._serialize_match(match))

            def on_caption(caption_match: CaptionMatch) -> None:
                captions.append({
                    "caption": caption_match.caption,
                    "train_images": [img.display_name for img in caption_match.train_images],
                    "val_images": [img.display_name for img in caption_match.val_images],
                })

            result = scan_basic_validation_matches(
                concepts,
                progress_callback=self._progress_callback,
                match_callback=on_match,
                caption_callback=on_caption,
                stop_event=self._stop_event,
            )

            self._results = {
                "matches": matches,
                "captions": captions,
                "summary": {
                    "hash_matches": result.hash_match_count,
                    "perceptual_matches": result.perceptual_match_count,
                    "filename_matches": result.filename_match_count,
                    "caption_groups": len(captions),
                    "train_images": result.train_image_count,
                    "val_images": result.val_image_count,
                },
                "train_images": [
                    {"path": img.path, "display_name": img.display_name}
                    for img in (result.train_images or [])
                ],
                "val_images": [
                    {"path": img.path, "display_name": img.display_name}
                    for img in (result.val_images or [])
                ],
            }

            self._broadcast({"type": "validation:scan_complete", "scan_type": "basic"})
        except Exception as e:
            logger.exception("Basic scan failed")
            self._results = {"error": str(e)}
        finally:
            with self._lock:
                self._status = "done"

    def _run_deep_scan(self, threshold: float) -> None:
        try:
            from modules.util.validation_checker_util import scan_clip_similarity_matches

            cached = self._results
            train_images_raw = cached.get("train_images", [])
            val_images_raw = cached.get("val_images", [])

            if not train_images_raw or not val_images_raw:
                return

            # We need the actual image objects — re-collect from concepts
            from modules.util.validation_checker_util import (
                collect_validation_checker_concepts,
                collect_validation_checker_images,
            )

            config_service = ConfigService.get_instance()
            concepts = config_service.get_config_for_training().concepts or []
            train_concepts, val_concepts = collect_validation_checker_concepts(concepts)
            train_images = collect_validation_checker_images(train_concepts)
            val_images = collect_validation_checker_images(val_concepts)

            deep_matches = scan_clip_similarity_matches(
                train_images,
                val_images,
                threshold=threshold,
                progress_callback=self._progress_callback,
                stop_event=self._stop_event,
            )

            existing_matches = self._results.get("matches", [])
            for m in deep_matches:
                existing_matches.append(self._serialize_match(m))
            self._results["matches"] = existing_matches
            summary = self._results.get("summary", {})
            summary["clip_matches"] = len(deep_matches)
            self._results["summary"] = summary

            self._broadcast({"type": "validation:scan_complete", "scan_type": "deep"})
        except Exception as e:
            logger.exception("Deep scan failed")
        finally:
            with self._lock:
                self._status = "done"

    @staticmethod
    def _image_url(path: str) -> str:
        return f"/validation/image?path={urllib.parse.quote(path)}"

    @classmethod
    def _serialize_match(cls, match: Any) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "train_image": match.train_image.display_name,
            "train_path": match.train_image.path,
            "train_image_url": cls._image_url(match.train_image.path),
            "val_image": match.val_image.display_name,
            "val_path": match.val_image.path,
            "val_image_url": cls._image_url(match.val_image.path),
            "kind": match.kind,
            "detail": match.detail,
            "score": match.score,
        }

    def remove_matches_batch(self, val_image_paths: list[str]) -> dict:
        """Remove multiple validation matches in one call. Returns count removed."""
        removed = 0
        errors: list[str] = []
        for p in val_image_paths:
            res = self.remove_match(p)
            if res.get("ok"):
                removed += 1
            else:
                errors.append(f"{p}: {res.get('error', 'unknown error')}")
        return {"ok": True, "removed": removed, "errors": errors}

    def _find_val_image(self, path: str) -> Any:
        """Rebuild a ValidationCheckerImage for the given path from current concepts."""
        from modules.util.validation_checker_util import (
            collect_validation_checker_concepts,
            collect_validation_checker_images,
        )

        config_service = ConfigService.get_instance()
        concepts = config_service.get_config_for_training().concepts or []
        _, val_concepts = collect_validation_checker_concepts(concepts)
        val_images = collect_validation_checker_images(val_concepts)
        for img in val_images:
            if img.path == path:
                return img
        return None

    def _remove_matches_for(self, val_path: str) -> None:
        matches = self._results.get("matches", [])
        self._results["matches"] = [m for m in matches if m.get("val_path") != val_path]
