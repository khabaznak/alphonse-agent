from datetime import datetime
import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from core.repositories.family_events import (
    create_family_event,
    delete_family_event,
    get_family_event,
    list_family_events,
    update_family_event,
)


router = APIRouter(prefix="/api", tags=["family-events"])


class FamilyEventCreate(BaseModel):
    owner_id: str
    title: str
    event_datetime: datetime
    description: str | None = None
    recurrence: str | None = None
    target_group: str | None = "all"
    push_payload: dict | None = None
    sent_at: datetime | None = None
    error_msg: str | None = None


class FamilyEventUpdate(BaseModel):
    title: str | None = None
    event_datetime: datetime | None = None
    description: str | None = None
    recurrence: str | None = None
    target_group: str | None = None
    push_payload: dict | None = None
    sent_at: datetime | None = None
    error_msg: str | None = None


@router.get("/family-events")
def list_events(limit: int = 50):
    return {"events": list_family_events(limit=limit)}


@router.get("/family-events/{event_id}")
def fetch_event(event_id: str):
    event = get_family_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.post("/family-events", status_code=201)
def create_event(payload: FamilyEventCreate):
    event = create_family_event(payload.model_dump())
    return event


@router.patch("/family-events/{event_id}")
def update_event(event_id: str, payload: FamilyEventUpdate):
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    event = update_family_event(event_id, updates)
    return event


@router.delete("/family-events/{event_id}")
def remove_event(event_id: str):
    event = delete_family_event(event_id)
    return event


@router.post("/webhooks/family-events", status_code=202)
def ingest_family_event(
    payload: FamilyEventCreate,
    x_atrium_webhook_secret: str | None = Header(default=None),
):
    _assert_webhook_secret(x_atrium_webhook_secret)
    event = create_family_event(payload.model_dump())
    return {"status": "accepted", "event": event}


def _assert_webhook_secret(received: str | None) -> None:
    expected = os.getenv("ATRIUM_WEBHOOK_SECRET")
    if expected and received != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
