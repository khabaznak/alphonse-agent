from __future__ import annotations

from alphonse.agent.actions.models import ActionResult
from alphonse.agent.extremities.base import Extremity
from alphonse.infrastructure.api_gateway import gateway


class ApiExtremity(Extremity):
    def can_handle(self, result: ActionResult) -> bool:
        return result.intention_key == "NOTIFY_API"

    def execute(self, result: ActionResult, narration: str | None = None) -> None:
        payload = result.payload
        correlation_id = payload.get("correlation_id")
        if not correlation_id:
            return
        response = {
            "message": narration or payload.get("message"),
            "data": payload.get("data"),
        }
        if gateway.exchange:
            gateway.exchange.publish(str(correlation_id), response)
