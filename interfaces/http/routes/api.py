import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from core.repositories.push_devices import deactivate_push_device, upsert_push_device


router = APIRouter(prefix="/api", tags=["push"])
trigger_router = APIRouter(prefix="/api/worker", tags=["worker"])


class PushDeviceUpsert(BaseModel):
    owner_id: str
    token: dict | str
    platform: str = "android"
    active: bool = True

@router.post("/push-devices", status_code=201)
def register_push_device(payload: PushDeviceUpsert):
    return upsert_push_device(payload.model_dump())


@router.delete("/push-devices/{device_id}")
def remove_push_device(device_id: str):
    return deactivate_push_device(device_id)


@trigger_router.post("/notifications/refresh", status_code=202)
def trigger_notifications_refresh(
    x_alphonse_webhook_secret: str | None = Header(default=None),
):
    _assert_webhook_secret(x_alphonse_webhook_secret)
    return {"status": "accepted"}


def _assert_webhook_secret(received: str | None) -> None:
    expected = os.getenv("ALPHONSE_WEBHOOK_SECRET")
    if expected and received != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
