import json
import os
import re
import shutil
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from modules.util import path_util


DISTILL_MANIFEST_NAME = "distillation_manifest.json"


@dataclass
class DistillMetadata:
    prompt: str
    negative_prompt: str = ""
    cfg_scale: float = 1.0
    steps: int = 0
    scheduler: str = ""
    seed: int | None = None
    model_id: str = ""
    width: int | None = None
    height: int | None = None
    timestep_schedule: list[float] | None = None
    source_path: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["timestep_schedule"] is None:
            data.pop("timestep_schedule")
        return data


@dataclass
class DistillBuildFilters:
    model_contains: str = ""
    steps: int | None = None
    cfg_scale: float | None = None
    scheduler: str = ""
    width: int | None = None
    height: int | None = None


@dataclass
class DistillBuildRequest:
    source_folder: str
    output_folder: str
    include_subdirectories: bool = True
    val_percentage: float = 0.0
    filters: DistillBuildFilters | None = None


def _read_png_text_chunks(path: str) -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return info
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                length = int.from_bytes(header[:4], "big")
                chunk_type = header[4:8]
                if chunk_type == b"IEND":
                    break
                data = f.read(length)
                f.seek(4, 1)
                try:
                    sep = data.index(b"\x00")
                    if chunk_type == b"tEXt":
                        info[data[:sep].decode("latin-1")] = data[sep + 1 :].decode("latin-1")
                    elif chunk_type == b"zTXt":
                        info[data[:sep].decode("latin-1")] = zlib.decompress(data[sep + 2 :]).decode("latin-1")
                    elif chunk_type == b"iTXt":
                        key = data[:sep].decode("utf-8")
                        compression_flag = data[sep + 1]
                        rest = data[sep + 3 :]
                        sep2 = rest.index(b"\x00")
                        rest = rest[sep2 + 1 :]
                        sep3 = rest.index(b"\x00")
                        text_data = rest[sep3 + 1 :]
                        info[key] = (
                            zlib.decompress(text_data).decode("utf-8")
                            if compression_flag
                            else text_data.decode("utf-8")
                        )
                except Exception:
                    continue
    except Exception:
        return {}
    return info


def _iter_json_objects(text: str):
    depth = 0
    start = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]
                start = None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalise_metadata(raw: dict[str, Any], image_path: str) -> DistillMetadata | None:
    params = raw.get("sui_image_params") if isinstance(raw.get("sui_image_params"), dict) else raw
    if not isinstance(params, dict):
        return None

    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        return None

    model_id = str(params.get("model_id") or params.get("model") or params.get("model_name") or "")
    if not model_id and isinstance(raw.get("sui_models"), list) and raw["sui_models"]:
        first = raw["sui_models"][0]
        if isinstance(first, dict):
            model_id = str(first.get("name") or "")

    seed = _coerce_int(params.get("seed"))
    width = _coerce_int(params.get("width"))
    height = _coerce_int(params.get("height"))
    schedule = params.get("timestep_schedule")
    if not isinstance(schedule, list) or not all(isinstance(v, int | float) for v in schedule):
        schedule = None

    return DistillMetadata(
        prompt=prompt,
        negative_prompt=str(params.get("negative_prompt") or params.get("negativeprompt") or ""),
        cfg_scale=_coerce_float(params.get("cfg_scale", params.get("cfgscale", 1.0)), 1.0),
        steps=_coerce_int(params.get("steps")) or 0,
        scheduler=str(params.get("scheduler") or params.get("sampler") or ""),
        seed=seed,
        model_id=model_id,
        width=width,
        height=height,
        timestep_schedule=[float(v) for v in schedule] if schedule is not None else None,
        source_path=image_path,
    )


def _load_sidecar(image_path: str) -> DistillMetadata | None:
    sidecar = Path(image_path).with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return _normalise_metadata(raw, image_path)


def _load_embedded_json(image_path: str) -> DistillMetadata | None:
    ext = os.path.splitext(image_path)[1].lower()
    blocks: list[str] = []

    if ext == ".png":
        for value in _read_png_text_chunks(image_path).values():
            blocks.extend(_iter_json_objects(value))
            if value.strip().startswith("{"):
                blocks.append(value)

    try:
        raw = Path(image_path).read_bytes()
    except OSError:
        return None

    for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
        text = raw.decode(encoding, errors="surrogateescape" if encoding == "utf-8" else "ignore")
        if '"prompt"' not in text and "sui_image_params" not in text:
            continue
        blocks.extend(_iter_json_objects(text))

    for block in blocks:
        try:
            parsed = json.loads(block)
        except (TypeError, json.JSONDecodeError):
            continue
        metadata = _normalise_metadata(parsed, image_path)
        if metadata is not None:
            return metadata

    # A1111/Forge plain parameters fallback.
    text = raw.decode("utf-8", errors="ignore")
    if "\nSteps:" in text:
        prompt = text.split("\nSteps:", 1)[0].split("\nNegative prompt:", 1)[0].strip()
        if prompt:
            return DistillMetadata(
                prompt=prompt,
                negative_prompt=_extract_a1111_negative(text),
                cfg_scale=_extract_number(text, "CFG scale", 1.0),
                steps=int(_extract_number(text, "Steps", 0)),
                scheduler=_extract_word(text, "Sampler"),
                model_id=_extract_word(text, "Model"),
                source_path=image_path,
            )
    return None


def _extract_a1111_negative(text: str) -> str:
    match = re.search(r"\nNegative prompt:\s*(.*?)(?:\nSteps:|$)", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_number(text: str, key: str, default: float) -> float:
    match = re.search(rf"{re.escape(key)}:\s*([-+]?\d+(?:\.\d+)?)", text)
    return _coerce_float(match.group(1), default) if match else default


def _extract_word(text: str, key: str) -> str:
    match = re.search(rf"{re.escape(key)}:\s*([^,\n]+)", text)
    return match.group(1).strip() if match else ""


def extract_distill_metadata(image_path: str) -> DistillMetadata | None:
    return _load_sidecar(image_path) or _load_embedded_json(image_path)


def _passes_filters(metadata: DistillMetadata, filters: DistillBuildFilters | None) -> tuple[bool, str]:
    if filters is None:
        return True, ""
    if filters.model_contains and filters.model_contains.lower() not in metadata.model_id.lower():
        return False, "model_mismatch"
    if filters.steps is not None and metadata.steps != filters.steps:
        return False, "steps_mismatch"
    if filters.cfg_scale is not None and abs(metadata.cfg_scale - filters.cfg_scale) > 1e-6:
        return False, "cfg_mismatch"
    if filters.scheduler and metadata.scheduler.lower() != filters.scheduler.lower():
        return False, "scheduler_mismatch"
    if filters.width is not None and metadata.width != filters.width:
        return False, "width_mismatch"
    if filters.height is not None and metadata.height != filters.height:
        return False, "height_mismatch"
    return True, ""


def _iter_image_paths(source: Path, include_subdirectories: bool):
    pattern = "**/*" if include_subdirectories else "*"
    exts = path_util.supported_image_extensions()
    for path in sorted(source.glob(pattern)):
        if path.is_file() and path.suffix.lower() in exts and "-masklabel" not in path.stem:
            yield path


def build_distillation_dataset(request: DistillBuildRequest) -> dict[str, Any]:
    source = Path(request.source_folder)
    output = Path(request.output_folder)
    if not source.is_dir():
        return {"ok": False, "error": f"Source folder does not exist: {source}"}
    if source.resolve() == output.resolve():
        return {"ok": False, "error": "Output folder must differ from source folder."}

    filters = request.filters
    accepted: list[tuple[Path, DistillMetadata]] = []
    rejected: dict[str, int] = {}
    scanned = 0

    for image_path in _iter_image_paths(source, request.include_subdirectories):
        scanned += 1
        metadata = extract_distill_metadata(str(image_path))
        if metadata is None:
            rejected["missing_metadata"] = rejected.get("missing_metadata", 0) + 1
            continue
        ok, reason = _passes_filters(metadata, filters)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        accepted.append((image_path, metadata))

    val_count = int(round(len(accepted) * max(min(request.val_percentage, 100.0), 0.0) / 100.0))
    train_count = len(accepted) - val_count
    manifest_items = []

    for index, (image_path, metadata) in enumerate(accepted):
        split = "train" if index < train_count else "val"
        rel = image_path.relative_to(source)
        target = output / split / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target)

        metadata.source_path = str(image_path)
        sidecar_path = target.with_suffix(".json")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_json_dict(), f, indent=2, ensure_ascii=False)
        with open(target.with_suffix(".txt"), "w", encoding="utf-8") as f:
            f.write(metadata.prompt)

        manifest_items.append(
            {
                "split": split,
                "source": str(image_path),
                "image": str(target),
                "sidecar": str(sidecar_path),
                "metadata": metadata.to_json_dict(),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ok": True,
        "source_folder": str(source),
        "output_folder": str(output),
        "scanned": scanned,
        "accepted": len(accepted),
        "train": train_count,
        "val": val_count,
        "rejected": rejected,
        "items": manifest_items,
    }
    with open(output / DISTILL_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
