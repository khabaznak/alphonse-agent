"""Text-only Telegram integration runtime for Alphonse v2."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib import parse, request

from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.io import V2IdentityResolver
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.integrations.presence import PresenceCapabilities
from alphonse.agent_v2.integrations.presence import PresencePhase
from alphonse.agent_v2.integrations.presence import PresenceState
from alphonse.agent_v2.integrations.presence import PresenceProjector
from alphonse.agent_v2.integrations.store import IntegrationConfigRecord


@dataclass(frozen=True)
class TelegramRuntimeStats:
    updates_seen: int = 0
    messages_queued: int = 0
    unknown_users: int = 0
    outbox_delivered: int = 0
    outbox_failed: int = 0


class TelegramHttpClient:
    """Small Telegram Bot API client."""

    def __init__(self, *, bot_token: str, urlopen: Callable[..., Any] | None = None) -> None:
        token = str(bot_token or "").strip()
        if not token:
            raise ValueError("telegram_bot_token_required")
        self.bot_token = token
        self._urlopen = urlopen or request.urlopen

    def get_updates(self, *, offset: int | None = None, timeout: int = 0) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": max(0, int(timeout)),
            "allowed_updates": json.dumps(["message", "edited_message"]),
        }
        if offset is not None:
            payload["offset"] = int(offset)
        result = self._post("getUpdates", payload)
        updates = result.get("result") if isinstance(result, dict) else []
        return [dict(item) for item in updates if isinstance(item, dict)] if isinstance(updates, list) else []

    def send_message(self, *, chat_id: str, text: str) -> str:
        result = self._post("sendMessage", {"chat_id": str(chat_id), "text": str(text)})
        body = result.get("result") if isinstance(result, dict) else {}
        message_id = body.get("message_id") if isinstance(body, dict) else ""
        return str(message_id or "").strip()

    def send_chat_action(self, *, chat_id: str, action: str = "typing") -> None:
        self._post("sendChatAction", {"chat_id": str(chat_id), "action": str(action or "typing")})

    def set_message_reaction(self, *, chat_id: str, message_id: str, emoji: str) -> None:
        reaction = [{"type": "emoji", "emoji": str(emoji)}] if str(emoji or "").strip() else []
        self._post(
            "setMessageReaction",
            {
                "chat_id": str(chat_id),
                "message_id": str(message_id),
                "reaction": json.dumps(reaction, ensure_ascii=False),
            },
        )

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"https://api.telegram.org/bot{self.bot_token}/{endpoint}"
        data = parse.urlencode(payload).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        with self._urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8", errors="ignore")
        parsed = json.loads(body)
        if not isinstance(parsed, dict) or not parsed.get("ok"):
            description = parsed.get("description") if isinstance(parsed, dict) else "invalid response"
            raise RuntimeError(f"telegram_{endpoint}_failed: {description}")
        return parsed


class TelegramIntegrationRuntime:
    """Optional bridge between Telegram and v2 queue/outbox."""

    def __init__(
        self,
        *,
        record: IntegrationConfigRecord,
        channel: CommunicationChannel,
        outbox: SQLiteOutboundStore,
        identity_resolver: V2IdentityResolver,
        owner_user_id: str = "local",
        http_client: TelegramHttpClient | None = None,
        on_message_queued: Callable[[], None] | None = None,
        presence_projector: PresenceProjector | None = None,
    ) -> None:
        self.record = record
        self.channel = channel
        self.outbox = outbox
        self.identity_resolver = identity_resolver
        self.owner_user_id = str(owner_user_id or "local").strip() or "local"
        config = normalize_telegram_config(record)
        self.poll_interval_sec = float(config["poll_interval_sec"])
        self.allowed_chat_ids = set(config["allowed_chat_ids"])
        self._last_update_id: int | None = None
        self._seen_update_ids: set[int] = set()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stats = TelegramRuntimeStats()
        self.http_client = http_client or TelegramHttpClient(bot_token=str(record.secrets.get("bot_token") or ""))
        self._on_message_queued = on_message_queued
        self.presence_projector = presence_projector
        self.presence_adapter = TelegramPresenceAdapter(
            http_client=self.http_client,
            enabled=bool(config["presence_enabled"]),
            reactions_enabled=bool(config["presence_reactions_enabled"]),
            typing_enabled=bool(config["presence_typing_enabled"]),
        )

    @property
    def integration_id(self) -> str:
        return self.record.integration_id

    @property
    def stats(self) -> TelegramRuntimeStats:
        return self._stats

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def poll_once(self) -> TelegramRuntimeStats:
        offset = (self._last_update_id + 1) if self._last_update_id is not None else None
        updates = self.http_client.get_updates(offset=offset, timeout=0)
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._last_update_id = update_id if self._last_update_id is None else max(self._last_update_id, update_id)
            self.handle_update(update)
        self.drain_outbox_once()
        if self.presence_projector is not None:
            self.presence_projector.heartbeat()
        return self._stats

    def handle_update(self, update: dict[str, Any]) -> bool:
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            if update_id in self._seen_update_ids:
                return False
            self._seen_update_ids.add(update_id)
        message = _telegram_message(update)
        if message is None:
            return False
        text = str(message.get("text") or "").strip()
        if not text:
            return False
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        chat_id = str(chat.get("id") or message.get("chat_id") or "").strip()
        if not chat_id:
            return False
        if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
            return False
        from_user = message.get("from") if isinstance(message.get("from"), dict) else {}
        provider_user_id = str(from_user.get("id") or "").strip()
        provider_message_id = str(message.get("message_id") or update.get("update_id") or "").strip()
        reply_to = message.get("reply_to_message") if isinstance(message.get("reply_to_message"), dict) else {}
        reply_to_message_id = str(reply_to.get("message_id") or "").strip()
        self._stats = _replace_stats(self._stats, updates_seen=self._stats.updates_seen + 1)
        resolved = self.identity_resolver.resolve_inbound_user(
            integration_id=self.integration_id,
            provider_key="telegram",
            provider_user_id=provider_user_id,
        )
        if not resolved.resolved:
            self._notify_owner_unresolved_user(
                provider_user_id=provider_user_id,
                chat_id=chat_id,
                reason=resolved.reason,
                update=update,
            )
            self._stats = _replace_stats(self._stats, unknown_users=self._stats.unknown_users + 1)
            return False
        self.channel.queue_message(
            prompt=text,
            user=resolved.alphonse_user_id,
            integration_id=self.integration_id,
            provider_key="telegram",
            provider_user_id=provider_user_id,
            channel_target=chat_id,
            provider_message_id=provider_message_id,
            reply_to_provider_message_id=reply_to_message_id,
            metadata={"provider_raw_message": dict(update)},
            correlation_id=f"{self.integration_id}:{provider_message_id}" if provider_message_id else "",
        )
        self._stats = _replace_stats(self._stats, messages_queued=self._stats.messages_queued + 1)
        self._notify_message_queued()
        return True

    def drain_outbox_once(self, *, limit: int = 20) -> TelegramRuntimeStats:
        selector = OutboundSelector(integration_id=self.integration_id, status="pending")
        for _ in range(max(1, int(limit))):
            outbound = self.outbox.claim_next(selector)
            if outbound is None:
                break
            try:
                provider_message_id = self.http_client.send_message(
                    chat_id=outbound.channel_target,
                    text=outbound.message,
                )
            except Exception as exc:
                self.outbox.mark_failed(outbound.outbox_message_id, error=f"{type(exc).__name__}: {exc}")
                self._stats = _replace_stats(self._stats, outbox_failed=self._stats.outbox_failed + 1)
                continue
            self.outbox.mark_delivered(outbound.outbox_message_id, provider_message_id=provider_message_id)
            self._stats = _replace_stats(self._stats, outbox_delivered=self._stats.outbox_delivered + 1)
        return self._stats

    def _run_loop(self) -> None:
        while self._running:
            try:
                self.poll_once()
            except Exception:
                pass
            time.sleep(max(0.1, self.poll_interval_sec))

    def _notify_owner_unresolved_user(
        self,
        *,
        provider_user_id: str,
        chat_id: str,
        reason: str,
        update: dict[str, Any],
    ) -> None:
        self.outbox.enqueue(
            address=ChannelAddress(
                integration_id="tui",
                provider_key="tui",
                channel_target=self.owner_user_id,
                alphonse_user_id=self.owner_user_id,
                provider_user_id=self.owner_user_id,
            ),
            message=(
                "Telegram message ignored because the sender is not mapped to an Alphonse user: "
                f"provider_user_id={provider_user_id or 'unknown'}, chat_id={chat_id}, reason={reason or 'unknown'}."
            ),
            kind="identity_resolution",
            audience_user_id=self.owner_user_id,
            metadata={
                "provider_key": "telegram",
                "integration_id": self.integration_id,
                "provider_user_id": provider_user_id,
                "channel_target": chat_id,
                "reason": reason,
                "provider_raw_message": dict(update),
            },
        )

    def _notify_message_queued(self) -> None:
        if self._on_message_queued is None:
            return
        try:
            self._on_message_queued()
        except Exception:
            pass


def build_telegram_runtime(
    *,
    record: IntegrationConfigRecord,
    channel: CommunicationChannel,
    outbox: SQLiteOutboundStore,
    identity_resolver: V2IdentityResolver,
    owner_user_id: str = "local",
    on_message_queued: Callable[[], None] | None = None,
    presence_projector: PresenceProjector | None = None,
) -> TelegramIntegrationRuntime:
    return TelegramIntegrationRuntime(
        record=record,
        channel=channel,
        outbox=outbox,
        identity_resolver=identity_resolver,
        owner_user_id=owner_user_id,
        on_message_queued=on_message_queued,
        presence_projector=presence_projector,
    )


def normalize_telegram_config(record: IntegrationConfigRecord) -> dict[str, Any]:
    config = dict(record.config or {})
    allowed = _parse_allowed_chat_ids(config.get("allowed_chat_ids"))
    return {
        "poll_interval_sec": _parse_float(config.get("poll_interval_sec"), default=1.0),
        "allowed_chat_ids": sorted(allowed),
        "owner_user_id": str(config.get("owner_user_id") or "local").strip() or "local",
        "presence_enabled": _parse_bool(config.get("presence_enabled"), default=True),
        "presence_reactions_enabled": _parse_bool(config.get("presence_reactions_enabled"), default=True),
        "presence_typing_enabled": _parse_bool(config.get("presence_typing_enabled"), default=True),
    }


def _telegram_message(update: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("message", "edited_message"):
        value = update.get(key)
        if isinstance(value, dict):
            return value
    return None


def _parse_allowed_chat_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {entry.strip() for entry in str(value or "").split(",") if entry.strip()}


def _parse_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


class TelegramPresenceAdapter:
    """Maps generic presence phases to Telegram typing and reactions."""

    capabilities = PresenceCapabilities(transient_activity=True, reactions=True)
    _REACTIONS = {
        PresencePhase.ACKNOWLEDGED: "👀",
        PresencePhase.THINKING: "🤔",
        PresencePhase.EXECUTING: "⚡",
        PresencePhase.WAITING_USER: "❓",
        PresencePhase.DONE: "👍",
        PresencePhase.FAILED: "👎",
    }

    def __init__(
        self,
        *,
        http_client: TelegramHttpClient,
        enabled: bool = True,
        reactions_enabled: bool = True,
        typing_enabled: bool = True,
    ) -> None:
        self.http_client = http_client
        self.enabled = enabled
        self.reactions_enabled = reactions_enabled
        self.typing_enabled = typing_enabled

    def start(self, presence: PresenceState) -> None:
        self._project(presence)

    def update(self, presence: PresenceState) -> None:
        self._project(presence)

    def heartbeat(self, presence: PresenceState) -> None:
        if self.enabled and self.typing_enabled and presence.phase in _ACTIVE_PRESENCE_PHASES:
            self.http_client.send_chat_action(chat_id=presence.address.channel_target, action="typing")

    def stop(self, presence: PresenceState) -> None:
        return

    def _project(self, presence: PresenceState) -> None:
        if not self.enabled:
            return
        if self.typing_enabled and presence.phase in _ACTIVE_PRESENCE_PHASES:
            self.http_client.send_chat_action(chat_id=presence.address.channel_target, action="typing")
        if self.reactions_enabled and presence.provider_message_id:
            emoji = self._REACTIONS.get(presence.phase)
            if emoji:
                self.http_client.set_message_reaction(
                    chat_id=presence.address.channel_target,
                    message_id=presence.provider_message_id,
                    emoji=emoji,
                )


_ACTIVE_PRESENCE_PHASES = {
    PresencePhase.ACKNOWLEDGED,
    PresencePhase.THINKING,
    PresencePhase.EXECUTING,
}


def _replace_stats(stats: TelegramRuntimeStats, **updates: int) -> TelegramRuntimeStats:
    values = stats.__dict__.copy()
    values.update(updates)
    return TelegramRuntimeStats(**values)
