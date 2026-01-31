from __future__ import annotations

from fastapi import APIRouter

from alphonse.agent.runtime import get_runtime

router = APIRouter()


@router.get("/agent/status")
def agent_status() -> dict[str, object]:
    return get_runtime().snapshot()
