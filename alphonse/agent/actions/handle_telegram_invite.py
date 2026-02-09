from __future__ import annotations

import logging

from alphonse.agent.actions.base import Action
from alphonse.agent.actions.models import ActionResult
from alphonse.agent.cognition.preferences.store import (
    get_preference,
    get_or_create_scope_principal,
)
from alphonse.agent.nervous_system.telegram_invites import get_invite

logger = logging.getLogger(__name__)


class HandleTelegramInviteAction(Action):
    key = "handle_telegram_invite"

    def execute(self, context: dict) -> ActionResult:
        signal = context.get("signal")
        payload = getattr(signal, "payload", {}) if signal else {}
        chat_id = str(payload.get("chat_id") or "").strip()
        if not chat_id:
            return ActionResult(
                intention_key="NOOP",
                payload={},
                urgency=None,
            )
        invite = get_invite(chat_id)
        system_principal_id = get_or_create_scope_principal("system", "default")
        admin_id = get_preference(system_principal_id, "onboarding.primary.admin_principal_id") if system_principal_id else None
        if not admin_id:
            return ActionResult(
                intention_key="NOOP",
                payload={},
                urgency=None,
            )
        message = "A new Telegram chat requested access. Approve it?"
        if invite and invite.get("from_user_name"):
            message = f"Telegram invite from {invite.get('from_user_name')}. Approve it?"
        return ActionResult(
            intention_key="MESSAGE_READY",
            payload={
                "message": message,
                "channel_hint": "webui",
                "target": "webui",
                "audience": {"kind": "person", "id": str(admin_id)},
                "data": {
                    "invite": invite,
                    "chat_id": chat_id,
                },
            },
            urgency="normal",
            requires_narration=False,
        )
