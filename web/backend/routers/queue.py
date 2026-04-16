import json
from typing import Any

from web.backend.services.queue_service import QueueService

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

router = APIRouter(tags=["queue"])


class QueueActionResponse(BaseModel):
    ok: bool
    error: str | None = None


class CreateEntryRequest(BaseModel):
    name: str = ""
    overrides: dict = {}


class UpdateEntryRequest(BaseModel):
    name: str | None = None
    overrides: dict | None = None


class ReorderRequest(BaseModel):
    entry_id: str
    direction: str  # "up" | "down"


class ImportQueueRequest(BaseModel):
    data: dict


@router.get("/queue")
def get_queue():
    service = QueueService.get_instance()
    return service.get_state()


@router.post("/queue/entry")
def create_entry(req: CreateEntryRequest):
    service = QueueService.get_instance()
    return service.add_entry(name=req.name, overrides=req.overrides)


@router.patch("/queue/entry/{entry_id}")
def update_entry(entry_id: str, req: UpdateEntryRequest):
    service = QueueService.get_instance()
    kwargs: dict[str, Any] = {}
    if req.name is not None:
        kwargs["name"] = req.name
    if req.overrides is not None:
        kwargs["overrides"] = req.overrides
    return service.update_entry(entry_id, **kwargs)


@router.delete("/queue/entry/{entry_id}")
def delete_entry(entry_id: str):
    service = QueueService.get_instance()
    return service.remove_entry(entry_id)


@router.post("/queue/entry/{entry_id}/duplicate")
def duplicate_entry(entry_id: str):
    service = QueueService.get_instance()
    return service.duplicate_entry(entry_id)


@router.post("/queue/reorder")
def reorder(req: ReorderRequest):
    service = QueueService.get_instance()
    return service.reorder(req.entry_id, req.direction)


@router.post("/queue/validate")
def validate_queue():
    service = QueueService.get_instance()
    return service.validate()


@router.post("/queue/execute", response_model=QueueActionResponse)
def execute_queue():
    service = QueueService.get_instance()
    result = service.execute()
    return QueueActionResponse(**result)


@router.post("/queue/stop", response_model=QueueActionResponse)
def stop_current():
    service = QueueService.get_instance()
    result = service.stop_current()
    return QueueActionResponse(**result)


@router.post("/queue/stop-all", response_model=QueueActionResponse)
def stop_all():
    service = QueueService.get_instance()
    result = service.stop_all()
    return QueueActionResponse(**result)


@router.patch("/queue/settings")
def update_settings(settings: dict):
    service = QueueService.get_instance()
    return service.update_settings(settings)


@router.get("/queue/export")
def export_queue():
    service = QueueService.get_instance()
    return service.export_queue()


@router.post("/queue/import")
def import_queue(req: ImportQueueRequest):
    service = QueueService.get_instance()
    return service.import_queue(req.data)


@router.get("/queue/entry/{entry_id}/diff")
def entry_diff(entry_id: str):
    service = QueueService.get_instance()
    return service.entry_diff(entry_id)


@router.post("/queue/entry/from-file")
async def entry_from_file(file: UploadFile = File(...), name: str = ""):
    """Upload a training config JSON; diff against defaults; add as queue entry."""
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Config file must be a JSON object")

    entry_name = name or file.filename or "Imported"
    service = QueueService.get_instance()
    return service.add_entry_from_full_config(payload, name=entry_name)
