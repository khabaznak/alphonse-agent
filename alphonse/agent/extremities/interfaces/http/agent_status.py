from __future__ import annotations

from fastapi import APIRouter

from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_queue_metrics
from alphonse.agent.runtime import get_runtime

router = APIRouter()


@router.get("/agent/status")
def agent_status() -> dict[str, object]:
    snapshot = get_runtime().snapshot()
    snapshot["pdca_metrics"] = get_pdca_queue_metrics()
    return snapshot
