from web.backend.services.text_alignment_service import (
    DEFAULT_WILDCARD_NAME,
    DEFAULT_WILDCARDS_DIR,
    TextAlignmentDatasetService,
)

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(tags=["text-alignment"])


class ScanRequest(BaseModel):
    # SwarmUI's raw output root (or any folder of its generations).
    source_folder: str
    # The dataset folder the trainer's offline text reward reads.
    output_folder: str
    recursive: bool = True
    # Admit images whose text is missing entirely (no gradient by default).
    keep_absent: bool = False
    # Admit images whose text is already verbatim (nothing to learn by default).
    keep_verbatim: bool = False
    # Target words the training-time detector must box for a keep.
    min_boxed_words: int = 1
    # Minimum agreement between the curation OCR and the training-time reader
    # (0 = off). Below it the image is skipped as reader_disagree.
    min_agreement: float = 0.0
    # 0 = every image; otherwise only the first N (for a quick preview).
    limit: int = 0
    ocr_device: str = "auto"
    detector_device: str = "auto"


class WildcardRequest(BaseModel):
    count: int = 1000
    seed: int = 0
    name: str = DEFAULT_WILDCARD_NAME
    directory: str = DEFAULT_WILDCARDS_DIR
    min_words: int = 1
    max_words: int = 8


@router.post("/text-alignment/scan")
def scan(req: ScanRequest):
    """Dry run: classify every image, nothing written."""
    return TextAlignmentDatasetService.get_instance().scan(
        req.source_folder,
        req.output_folder,
        recursive=req.recursive,
        keep_absent=req.keep_absent,
        keep_verbatim=req.keep_verbatim,
        min_boxed_words=req.min_boxed_words,
        min_agreement=req.min_agreement,
        limit=req.limit,
        ocr_device=req.ocr_device,
        detector_device=req.detector_device,
    )


@router.post("/text-alignment/build")
def build():
    """Copy the kept candidates of the last scan into the dataset folder."""
    return TextAlignmentDatasetService.get_instance().build()


@router.post("/text-alignment/cancel")
def cancel():
    return TextAlignmentDatasetService.get_instance().cancel()


@router.get("/text-alignment/status")
def status():
    return TextAlignmentDatasetService.get_instance().status()


@router.get("/text-alignment/results")
def results(offset: int = 0, limit: int = 500, only_kept: bool = True):
    return TextAlignmentDatasetService.get_instance().results(offset=offset, limit=limit, only_kept=only_kept)


@router.get("/text-alignment/image")
def image(path: str):
    abs_path = TextAlignmentDatasetService.get_instance().serve_image(path)
    if abs_path is None:
        return {"ok": False, "error": "Image not found"}
    return FileResponse(abs_path)


@router.post("/text-alignment/wildcard")
def wildcard(req: WildcardRequest):
    """Write N reward prompts, one per line, as a SwarmUI wildcard file."""
    return TextAlignmentDatasetService.wildcard(
        count=req.count,
        seed=req.seed,
        name=req.name,
        directory=req.directory,
        min_words=req.min_words,
        max_words=req.max_words,
    )
