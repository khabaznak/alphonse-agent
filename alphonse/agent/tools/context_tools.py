from __future__ import annotations

from typing import Any

from alphonse.agent import identity
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.config import settings


class GetMySettingsTool:
    canonical_name: str = "get_my_settings"
    capability: str = "context"

    def execute(self, *, state: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        payload = dict(state or {})
        return {
            "output": {
                "locale": str(payload.get("locale") or "").strip() or None,
                "tone": str(payload.get("tone") or "").strip() or None,
                "address_style": str(payload.get("address_style") or "").strip() or None,
                "timezone": str(payload.get("timezone") or settings.get_timezone() or "").strip() or None,
                "channel_type": str(payload.get("channel_type") or "").strip() or None,
                "execution_mode": str(settings.get_execution_mode() or "").strip() or None,
            },
            "exception": None,
            "metadata": {"tool": "get_my_settings"},
        }


class GetUserDetailsTool:
    canonical_name: str = "get_user_details"
    capability: str = "context"

    def execute(self, *, task_record: TaskRecord) -> dict[str, Any]:        
        user_id = task_record.user_id        
        return {
            "output": {
                
                "user_id": user_id                
            },
            "exception": None,
            "metadata": {"tool": "get_user_details"},
        }
