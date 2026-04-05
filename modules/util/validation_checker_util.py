import hashlib
import importlib
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import huggingface_hub

from PIL import Image, ImageOps

from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.path_util import canonical_join, is_supported_image_extension
from modules.util.torch_util import default_device, torch_gc


TRAIN_CONCEPT_TYPES = {
    ConceptType.STANDARD.value,
    ConceptType.PRIOR_PREDICTION.value,
}
VALIDATION_CONCEPT_TYPES = {
    ConceptType.VALIDATION.value,
}


@dataclass(slots=True)
class ValidationCheckerConcept:
    index: int
    name: str
    configured_path: str
    resolved_path: str
    concept_type: str
    include_subdirectories: bool
    is_local: bool


@dataclass(slots=True)
class ValidationCheckerImage:
    path: str
    relative_path: str
    caption_path: str | None
    concept: ValidationCheckerConcept

    @property
    def display_name(self) -> str:
        concept_name = self.concept.name or os.path.basename(self.concept.configured_path) or f"concept {self.concept.index + 1}"
        return f"{concept_name}/{self.relative_path}".replace("\\", "/")


@dataclass(slots=True)
class ValidationCheckerMatch:
    train_image: ValidationCheckerImage
    val_image: ValidationCheckerImage
    kind: str
    detail: str
    score: float | None = None
    train_variant: str | None = None
    val_variant: str | None = None


@dataclass(slots=True)
class CaptionMatch:
    caption: str
    train_images: list[ValidationCheckerImage]
    val_images: list[ValidationCheckerImage]


@dataclass(slots=True)
class BasicScanResult:
    train_images: list[ValidationCheckerImage]
    val_images: list[ValidationCheckerImage]
    exact_matches: list[ValidationCheckerMatch]
    perceptual_matches: list[ValidationCheckerMatch]
    caption_matches: list[CaptionMatch]


ProgressCallback = Callable[[str, int, int], None]
MatchCallback = Callable[[ValidationCheckerMatch], None]
CaptionCallback = Callable[[CaptionMatch], None]


def concept_type_name(concept_type: ConceptType | str) -> str:
    return concept_type.value if hasattr(concept_type, "value") else str(concept_type)


def is_train_concept_type(concept_type: ConceptType | str) -> bool:
    return concept_type_name(concept_type) in TRAIN_CONCEPT_TYPES


def is_validation_concept_type(concept_type: ConceptType | str) -> bool:
    return concept_type_name(concept_type) in VALIDATION_CONCEPT_TYPES


def resolve_concept_path(path: str) -> str | None:
    if os.path.isdir(path):
        return path
    try:
        return huggingface_hub.snapshot_download(repo_id=path, repo_type="dataset", local_files_only=True)
    except Exception:
        return None


def collect_validation_checker_concepts(concepts: list[ConceptConfig]) -> tuple[list[ValidationCheckerConcept], list[ValidationCheckerConcept]]:
    train_concepts = []
    val_concepts = []

    for index, concept in enumerate(concepts):
        if not concept.enabled:
            continue

        concept_type = concept_type_name(concept.type)
        if not (is_train_concept_type(concept_type) or is_validation_concept_type(concept_type)):
            continue

        resolved_path = resolve_concept_path(concept.path)
        if not resolved_path:
            continue

        checker_concept = ValidationCheckerConcept(
            index=index,
            name=concept.name,
            configured_path=concept.path,
            resolved_path=resolved_path,
            concept_type=concept_type,
            include_subdirectories=concept.include_subdirectories,
            is_local=os.path.isdir(concept.path),
        )

        if is_train_concept_type(concept_type):
            train_concepts.append(checker_concept)
        else:
            val_concepts.append(checker_concept)

    return train_concepts, val_concepts


def collect_validation_checker_images(concepts: list[ValidationCheckerConcept]) -> list[ValidationCheckerImage]:
    images = []

    for concept in concepts:
        root_path = Path(concept.resolved_path)
        if concept.include_subdirectories:
            iterator = os.walk(root_path)
            for current_root, dir_names, file_names in iterator:
                dir_names[:] = [dir_name for dir_name in dir_names if not dir_name.startswith(".")]
                current_root_path = Path(current_root)
                for file_name in sorted(file_names):
                    image = _build_validation_image(current_root_path / file_name, root_path, concept)
                    if image is not None:
                        images.append(image)
        else:
            for child in sorted(root_path.iterdir()):
                image = _build_validation_image(child, root_path, concept)
                if image is not None:
                    images.append(image)

    return images


def _build_validation_image(path: Path, root_path: Path, concept: ValidationCheckerConcept) -> ValidationCheckerImage | None:
    if not path.is_file():
        return None
    if path.name.startswith("."):
        return None
    if any(part.startswith(".") for part in path.relative_to(root_path).parent.parts):
        return None
    if path.name.endswith("-masklabel.png") or path.name.endswith("-condlabel.png"):
        return None
    if not is_supported_image_extension(path.suffix):
        return None

    caption_path = path.with_suffix(".txt")
    return ValidationCheckerImage(
        path=canonical_join(str(path)),
        relative_path=str(path.relative_to(root_path)).replace("\\", "/"),
        caption_path=canonical_join(str(caption_path)) if caption_path.is_file() else None,
        concept=concept,
    )


def scan_basic_validation_matches(
    concepts: list[ConceptConfig],
    progress_callback: ProgressCallback | None = None,
    match_callback: MatchCallback | None = None,
    caption_callback: CaptionCallback | None = None,
    stop_event: threading.Event | None = None,
) -> BasicScanResult:
    stop_event = stop_event or threading.Event()
    train_concepts, val_concepts = collect_validation_checker_concepts(concepts)
    train_images = collect_validation_checker_images(train_concepts)
    val_images = collect_validation_checker_images(val_concepts)

    total_images = len(train_images) + len(val_images)
    processed = 0

    train_sha: dict[str, list[ValidationCheckerImage]] = {}
    train_variants: list[tuple[ValidationCheckerImage, str, int]] = []
    train_captions: dict[str, list[ValidationCheckerImage]] = {}

    for image in train_images:
        _raise_if_cancelled(stop_event)
        sha = file_hash(image.path)
        train_sha.setdefault(sha, []).append(image)
        for variant_name, variant_hash in dhash_variants(image.path):
            train_variants.append((image, variant_name, variant_hash))
        caption = read_caption(image.caption_path)
        if caption:
            train_captions.setdefault(caption, []).append(image)
        processed += 1
        _report_progress(progress_callback, "Scanning training images", processed, total_images)

    exact_matches = []
    perceptual_matches = []
    exact_keys = set()
    perceptual_keys = set()
    caption_matches_by_caption: dict[str, CaptionMatch] = {}

    for image in val_images:
        _raise_if_cancelled(stop_event)

        sha = file_hash(image.path)
        for train_image in train_sha.get(sha, []):
            key = (train_image.path, image.path)
            if key in exact_keys:
                continue
            exact_keys.add(key)
            match = ValidationCheckerMatch(
                train_image=train_image,
                val_image=image,
                kind="exact",
                detail="exact file hash match",
            )
            exact_matches.append(match)
            if match_callback:
                match_callback(match)

        val_variants = dhash_variants(image.path)
        best_matches = {}
        for val_variant_name, val_variant_hash in val_variants:
            for train_image, train_variant_name, train_variant_hash in train_variants:
                if train_image.path == image.path:
                    continue
                distance = hamming_distance(train_variant_hash, val_variant_hash)
                if distance > 5:
                    continue
                key = (train_image.path, image.path)
                current = best_matches.get(key)
                if current is None or distance < current[0]:
                    best_matches[key] = (distance, train_variant_name, val_variant_name)

        for key, (distance, train_variant_name, val_variant_name) in sorted(best_matches.items(), key=lambda item: item[1][0]):
            if key in exact_keys or key in perceptual_keys:
                continue
            perceptual_keys.add(key)
            match = ValidationCheckerMatch(
                train_image=_find_image_by_path(train_images, key[0]),
                val_image=image,
                kind="perceptual",
                detail=describe_perceptual_match(train_variant_name, val_variant_name, distance),
                score=float(distance),
                train_variant=train_variant_name,
                val_variant=val_variant_name,
            )
            perceptual_matches.append(match)
            if match_callback:
                match_callback(match)

        caption = read_caption(image.caption_path)
        if caption and caption in train_captions:
            warning = caption_matches_by_caption.get(caption)
            if warning is None:
                warning = CaptionMatch(caption=caption, train_images=list(train_captions[caption]), val_images=[])
                caption_matches_by_caption[caption] = warning
                if caption_callback:
                    caption_callback(warning)
            warning.val_images.append(image)

        processed += 1
        _report_progress(progress_callback, "Scanning validation images", processed, total_images)

    caption_matches = sorted(
        [warning for warning in caption_matches_by_caption.values() if warning.val_images],
        key=lambda warning: len(warning.train_images) + len(warning.val_images),
        reverse=True,
    )

    return BasicScanResult(
        train_images=train_images,
        val_images=val_images,
        exact_matches=exact_matches,
        perceptual_matches=perceptual_matches,
        caption_matches=caption_matches,
    )


def scan_clip_similarity_matches(
    train_images: list[ValidationCheckerImage],
    val_images: list[ValidationCheckerImage],
    threshold: float = 0.93,
    progress_callback: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
) -> list[ValidationCheckerMatch]:
    stop_event = stop_event or threading.Event()
    _report_progress(progress_callback, "Loading CLIP model", 0, max(len(train_images) + len(val_images), 1))

    model_name, clip_model, clip_processor, torch_module = load_clip_components()
    if model_name is None or clip_model is None or clip_processor is None or torch_module is None:
        raise RuntimeError("Deep Scan requires transformers, torch, and a CLIP model.")

    device = default_device

    try:
        model = clip_model.from_pretrained(model_name).to(device)
        processor = clip_processor.from_pretrained(model_name)
        model.eval()

        with torch_module.no_grad():
            train_embeddings = _encode_clip_embeddings(
                model,
                processor,
                torch_module,
                train_images,
                device,
                "Encoding training images",
                progress_callback,
                0,
                len(train_images) + len(val_images),
                stop_event,
            )
            val_embeddings = _encode_clip_embeddings(
                model,
                processor,
                torch_module,
                val_images,
                device,
                "Encoding validation images",
                progress_callback,
                len(train_images),
                len(train_images) + len(val_images),
                stop_event,
            )

        _raise_if_cancelled(stop_event)
        _report_progress(progress_callback, "Computing similarity scores", len(train_images) + len(val_images), len(train_images) + len(val_images))

        similarities = train_embeddings @ val_embeddings.T
        matches = []
        for train_index, train_image in enumerate(train_images):
            for val_index, val_image in enumerate(val_images):
                similarity = float(similarities[train_index, val_index].item())
                if similarity >= threshold:
                    matches.append(
                        ValidationCheckerMatch(
                            train_image=train_image,
                            val_image=val_image,
                            kind="clip",
                            detail=f"cosine similarity {similarity:.3f}",
                            score=similarity,
                        )
                    )

        matches.sort(key=lambda match: match.score if match.score is not None else 0.0, reverse=True)
        return matches
    finally:
        try:
            del model
        except UnboundLocalError:
            pass
        try:
            del processor
        except UnboundLocalError:
            pass
        torch_gc()


def load_clip_components() -> tuple[object | None, object | None, object | None, object | None]:
    try:
        transformers = importlib.import_module("transformers")
        torch_module = importlib.import_module("torch")
    except ImportError:
        return None, None, None, None

    model_name = None
    for candidate in ("openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32"):
        try:
            transformers.CLIPModel.from_pretrained(candidate, local_files_only=True)
            model_name = candidate
            break
        except Exception:
            continue

    if model_name is None:
        model_name = "openai/clip-vit-base-patch32"

    return model_name, transformers.CLIPModel, transformers.CLIPProcessor, torch_module


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_variants(path: str, hash_size: int = 16) -> list[tuple[str, int]]:
    try:
        imagehash = importlib.import_module("imagehash")
    except ImportError as error:
        raise RuntimeError("Validation Checker requires the 'imagehash' package to be installed.") from error
    with Image.open(path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB")
        width, height = image.size
        crop_margin_w = int(width * 0.1)
        crop_margin_h = int(height * 0.1)
        variants = {
            "original": image,
            "horizontal_flip": image.transpose(Image.FLIP_LEFT_RIGHT),
            "center_crop": image.crop((crop_margin_w, crop_margin_h, width - crop_margin_w, height - crop_margin_h)),
        }

        hashes = []
        for variant_name, variant in variants.items():
            hashes.append((variant_name, int(str(imagehash.dhash(variant, hash_size=hash_size)), 16)))
            if variant is not image:
                variant.close()
        image.close()
        return hashes


def hamming_distance(hash_a: int, hash_b: int) -> int:
    return (hash_a ^ hash_b).bit_count()


def describe_perceptual_match(train_variant: str, val_variant: str, distance: int) -> str:
    variant_names = {train_variant, val_variant}
    if "horizontal_flip" in variant_names and "center_crop" in variant_names:
        base = "flipped and center-cropped variant"
    elif "horizontal_flip" in variant_names:
        base = "horizontal flip"
    elif "center_crop" in variant_names:
        base = "center crop"
    else:
        base = "minor edit or recompression"
    return f"{base} ({distance} bit difference)"


def read_caption(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            caption = file.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            caption = file.read().strip()
    return caption or None


def build_explorer_command(path: str) -> list[str]:
    if os.name != "nt":
        raise RuntimeError("Open in Explorer is only supported on Windows.")
    return ["explorer", "/select,", os.path.normpath(path)]


def open_in_explorer(path: str):
    subprocess.Popen(build_explorer_command(path))


def remove_validation_image(image: ValidationCheckerImage):
    _ensure_local_validation_image(image)
    _remove_path_if_exists(image.path)
    if image.caption_path:
        _remove_path_if_exists(image.caption_path)


def move_validation_image_to_train(image: ValidationCheckerImage, target_concept: ValidationCheckerConcept) -> str:
    _ensure_local_validation_image(image)
    if not target_concept.is_local:
        raise RuntimeError("Move to Train is only available for local training concepts.")
    if not os.path.isdir(target_concept.configured_path):
        raise RuntimeError("Target training concept path is not available locally.")

    destination_path = _next_available_destination(os.path.join(target_concept.configured_path, os.path.basename(image.path)))
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(image.path, destination_path)

    if image.caption_path and os.path.isfile(image.caption_path):
        caption_destination = os.path.splitext(destination_path)[0] + ".txt"
        shutil.move(image.caption_path, caption_destination)

    image.path = canonical_join(destination_path)
    image.caption_path = canonical_join(os.path.splitext(destination_path)[0] + ".txt") if os.path.isfile(os.path.splitext(destination_path)[0] + ".txt") else None
    return image.path


def summarize_findings(
    exact_matches: list[ValidationCheckerMatch],
    perceptual_matches: list[ValidationCheckerMatch],
    clip_matches: list[ValidationCheckerMatch],
    caption_matches: list[CaptionMatch],
) -> str:
    leak_count = len(exact_matches) + len(perceptual_matches) + len(clip_matches)
    caption_warning_count = len(caption_matches)
    return f"{leak_count} potential leaks found. {caption_warning_count} caption-match warnings."


def _find_image_by_path(images: list[ValidationCheckerImage], path: str) -> ValidationCheckerImage:
    for image in images:
        if image.path == path:
            return image
    raise RuntimeError(f"Image not found: {path}")


def _encode_clip_embeddings(
    model,
    processor,
    torch_module,
    images: list[ValidationCheckerImage],
    device,
    progress_label: str,
    progress_callback: ProgressCallback | None,
    progress_offset: int,
    progress_total: int,
    stop_event: threading.Event,
):
    embeddings = []
    batch_size = 8

    for start_index in range(0, len(images), batch_size):
        _raise_if_cancelled(stop_event)
        batch = images[start_index:start_index + batch_size]
        pil_images = []
        try:
            for image in batch:
                with Image.open(image.path) as raw_image:
                    pil_images.append(ImageOps.exif_transpose(raw_image).convert("RGB"))
            inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
            batch_embeddings = model.get_image_features(**inputs)
            batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            embeddings.append(batch_embeddings.cpu())
        finally:
            for pil_image in pil_images:
                pil_image.close()

        completed = progress_offset + min(start_index + len(batch), len(images))
        _report_progress(progress_callback, progress_label, completed, progress_total)

    return torch_module.cat(embeddings, dim=0) if embeddings else torch_module.empty((0, 512))


def _ensure_local_validation_image(image: ValidationCheckerImage):
    if not image.concept.is_local:
        raise RuntimeError("This validation concept is not a local directory, so OneTrainer will not modify it.")


def _next_available_destination(path: str) -> str:
    candidate = path
    stem = os.path.splitext(path)[0]
    suffix = os.path.splitext(path)[1]
    index = 1
    while os.path.exists(candidate):
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _remove_path_if_exists(path: str):
    if os.path.isfile(path):
        os.remove(path)


def _raise_if_cancelled(stop_event: threading.Event):
    if stop_event.is_set():
        raise RuntimeError("Validation checker scan cancelled.")


def _report_progress(progress_callback: ProgressCallback | None, label: str, current: int, total: int):
    if progress_callback:
        progress_callback(label, current, max(total, 1))
