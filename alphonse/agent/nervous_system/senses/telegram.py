from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphonse.agent import identity
from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.extremities.interfaces.integrations.telegram.telegram_adapter import TelegramAdapter
from alphonse.agent.io import TelegramSenseAdapter
from alphonse.agent.nervous_system.assets import register_uploaded_asset
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.senses.base import Sense, SignalSpec
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.tools.stt_transcribe import SttTranscribeTool

logger = get_component_logger("senses.telegram")
_PROVIDER_KEY = "telegram"


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: str
    allowed_chat_ids: set[int] | None
    poll_interval_sec: float


class TelegramSense(Sense):
    key = "telegram"
    name = "Telegram Sense"
    description = "Receives Telegram messages and emits sense.telegram.message.user.received"
    source_type = "service"
    signals = [
        SignalSpec(key="sense.telegram.message.user.received", name="Telegram User Message Received"),
    ]

    def __init__(self) -> None:
        self._adapter: TelegramAdapter | None = None
        self._running = False
        self._bus: Bus | None = None
        self._sense_adapter = TelegramSenseAdapter()

    def start(self, bus: Bus) -> None:
        if self._running:
            return
        settings = _load_settings()
        if settings is None:
            logger.warning("TelegramSense disabled: missing TELEGRAM_BOT_TOKEN")
            return
        self._bus = bus
        self._adapter = TelegramAdapter(
            {
                "bot_token": settings.bot_token,
                "poll_interval_sec": settings.poll_interval_sec,
                "allowed_chat_ids": list(settings.allowed_chat_ids or [])
                if settings.allowed_chat_ids is not None
                else None,
            }
        )
        self._adapter.on_signal(self._on_signal)
        self._adapter.start()
        self._running = True
        logger.info("TelegramSense started")

    def stop(self) -> None:
        if not self._running:
            return
        if self._adapter:
            self._adapter.stop()
        self._running = False
        logger.info("TelegramSense stopped")

    def _on_signal(self, signal: Signal) -> None:
        if not self._bus:
            return
        if signal.type == "external.telegram.invite_request":
            self._on_invite_request(signal)
            return
        if signal.type != "external.telegram.message":
            return
        self._on_message(signal)

    def _on_invite_request(self, signal: Signal) -> None:
        if not self._bus:
            return
        payload = signal.payload or {}
        chat_id = payload.get("chat_id")
        if chat_id is None:
            return
        from_user = payload.get("from_user")
        from_user_name = payload.get("from_user_name")
        text = payload.get("text") or ""
        from alphonse.agent.nervous_system.telegram_invites import upsert_invite

        upsert_invite(
            {
                "chat_id": str(chat_id),
                "chat_type": payload.get("chat_type"),
                "from_user_id": str(from_user) if from_user is not None else None,
                "from_user_username": payload.get("from_user_username"),
                "from_user_name": from_user_name,
                "last_message": text,
                "status": "pending",
            }
        )
        self._bus.emit(
            Signal(
                type="sense.telegram.message.user.received",
                payload=build_incoming_message_envelope(
                    message_id=str(payload.get("message_id") or payload.get("update_id") or chat_id),
                    channel_type="telegram",
                    channel_target=str(chat_id),
                    provider="telegram",
                    text=text or "Telegram invite request",
                    occurred_at=datetime.now(timezone.utc).isoformat(),
                    correlation_id=str(chat_id),
                    actor_external_user_id=str(from_user) if from_user is not None else None,
                    actor_display_name=str(from_user_name or "").strip() or None,
                    metadata={
                        "message_kind": "invite_request",
                        "invite": {
                            "chat_id": str(chat_id),
                            "chat_type": payload.get("chat_type"),
                            "from_user": from_user,
                            "from_user_name": from_user_name,
                        },
                    },
                ),
                source="telegram",
                correlation_id=str(chat_id),
            )
        )
        logger.info(
            "TelegramSense emitted sense.telegram.message.user.received (invite) chat_id=%s",
            chat_id,
        )

    # Handles all messages received from Telegram which are not invite requests.
    def _on_message(self, signal: Signal) -> None:
        if not self._bus:
            return
        payload = signal.payload or {}
        logger.info(
            "TelegramSense received update chat_id=%s from=%s text=%s",
            payload.get("chat_id"),
            payload.get("from_user"),
            _snippet(str(payload.get("text") or "")),
        )
        normalized = self._sense_adapter.normalize(payload)
        normalized_meta = normalized.metadata if isinstance(normalized.metadata, dict) else {}
        normalized_attachments = [dict(item) for item in normalized.attachments if isinstance(item, dict)]
        ingestion = _ingest_telegram_audio_attachments(
            adapter=self._adapter,
            attachments=normalized_attachments,
            channel_target=normalized.channel_target,
            actor_external_user_id=normalized.user_id,
            message_id=str(payload.get("message_id") or "").strip() or None,
            update_id=str(payload.get("update_id") or "").strip() or None,
        )
        normalized_attachments = ingestion["attachments"]
        transcript_text = str(ingestion.get("prompt_text") or "").strip()
        envelope_text = normalized.text or transcript_text
        normalized_meta = {
            **normalized_meta,
            "attachments": normalized_attachments,
            "telegram_attachment_ingestion": ingestion["metadata"],
        }
        self._bus.emit(
            Signal(
                type="sense.telegram.message.user.received",
                payload=build_incoming_message_envelope(
                    message_id=str(payload.get("message_id") or payload.get("update_id") or normalized.correlation_id or normalized.timestamp),
                    channel_type=normalized.channel_type,
                    channel_target=str(normalized.channel_target).strip(),
                    provider=_PROVIDER_KEY,
                    text=envelope_text,
                    occurred_at=datetime.fromtimestamp(float(normalized.timestamp), tz=timezone.utc).isoformat(),
                    correlation_id=normalized.correlation_id,
                    actor_external_user_id=normalized.user_id,
                    actor_display_name=normalized.user_name,
                    attachments=normalized_attachments,
                    metadata={
                        "normalized_metadata": normalized_meta,
                        "provider_event": payload.get("provider_event")
                        if isinstance(payload.get("provider_event"), dict)
                        else None,
                    },
                    reply_to_message_id=str(payload.get("message_id") or "").strip() or None,
                ),
                source=_PROVIDER_KEY,
                correlation_id=normalized.correlation_id,
            )
        )
        logger.info(
            "TelegramSense emitted sense.telegram.message.user.received chat_id=%s message_id=%s",
            payload.get("chat_id"),
            payload.get("message_id"),
        )


def _load_settings() -> TelegramSettings | None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return None
    allowed = _parse_allowed_chat_ids(
        os.getenv("TELEGRAM_ALLOWED_CHAT_IDS"),
        os.getenv("TELEGRAM_ALLOWED_CHAT_ID"),
    )
    from alphonse.agent.nervous_system.tool_configs import get_active_tool_config

    config = get_active_tool_config("telegram")
    poll_interval = _parse_float(os.getenv("TELEGRAM_POLL_INTERVAL_SEC"), default=1.0)
    if config and isinstance(config.get("config"), dict):
        raw = config.get("config") or {}
        allowed_cfg = raw.get("allowed_chat_ids")
        if isinstance(allowed_cfg, list):
            allowed = {
                int(x)
                for x in allowed_cfg
                if str(x).strip().lstrip("-").isdigit()
            } or None
        poll_value = raw.get("poll_interval_sec")
        poll_interval = _parse_float(str(poll_value), default=poll_interval)
    return TelegramSettings(
        bot_token=bot_token,
        allowed_chat_ids=allowed,
        poll_interval_sec=poll_interval,
    )


def _parse_allowed_chat_ids(primary: str | None, fallback: str | None) -> set[int] | None:
    raw = primary or fallback
    if not raw:
        return None
    ids: set[int] = set()
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            ids.add(int(entry))
        except ValueError:
            continue
    return ids or None


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _ingest_telegram_audio_attachments(
    *,
    adapter: TelegramAdapter | None,
    attachments: list[dict[str, Any]],
    channel_target: str,
    actor_external_user_id: str | None,
    message_id: str | None,
    update_id: str | None,
) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    owner_user_id = _resolve_internal_telegram_user_id(actor_external_user_id)
    for attachment in attachments:
        item = dict(attachment)
        if not _is_telegram_audio_attachment(item):
            enriched.append(item)
            continue
        if not str(item.get("asset_id") or "").strip():
            _register_telegram_audio_attachment(
                adapter=adapter,
                attachment=item,
                channel_target=channel_target,
                owner_user_id=owner_user_id,
                actor_external_user_id=actor_external_user_id,
                message_id=message_id,
                update_id=update_id,
            )
        asset_id = str(item.get("asset_id") or "").strip()
        if asset_id:
            transcript = _transcribe_registered_audio_asset(asset_id)
            item["transcription_status"] = str(transcript.get("status") or "failed")
            if transcript.get("text"):
                item["transcript"] = str(transcript.get("text") or "")
                item["transcription"] = transcript
                transcripts.append(
                    {
                        "asset_id": asset_id,
                        "file_id": str(item.get("file_id") or "").strip() or None,
                        "text": str(transcript.get("text") or ""),
                    }
                )
            elif transcript.get("error"):
                item["transcription_error"] = transcript.get("error")
        enriched.append(item)
    prompt_text = str((transcripts[0] if transcripts else {}).get("text") or "").strip()
    return {
        "attachments": enriched,
        "prompt_text": prompt_text,
        "metadata": {
            "audio_attachment_count": sum(1 for item in enriched if _is_telegram_audio_attachment(item)),
            "transcripts": transcripts,
        },
    }


def _register_telegram_audio_attachment(
    *,
    adapter: TelegramAdapter | None,
    attachment: dict[str, Any],
    channel_target: str,
    owner_user_id: str | None,
    actor_external_user_id: str | None,
    message_id: str | None,
    update_id: str | None,
) -> None:
    if adapter is None:
        attachment["asset_registration_status"] = "failed"
        attachment["asset_registration_error"] = "telegram_adapter_unavailable"
        return
    if not owner_user_id:
        attachment["asset_registration_status"] = "failed"
        attachment["asset_registration_error"] = "owner_user_id_unresolved"
        return
    file_id = str(attachment.get("file_id") or "").strip()
    if not file_id:
        attachment["asset_registration_status"] = "failed"
        attachment["asset_registration_error"] = "telegram_file_id_missing"
        return
    try:
        file_meta = adapter.get_file(file_id=file_id)
        file_path = str(file_meta.get("file_path") or "").strip()
        if not file_path:
            raise RuntimeError("telegram_file_path_missing")
        content = adapter.download_file(file_path=file_path)
        asset = register_uploaded_asset(
            content=content,
            kind="audio",
            mime_type=str(attachment.get("mime_type") or "").strip() or None,
            owner_user_id=owner_user_id,
            provider="telegram",
            channel_type="telegram",
            channel_target=channel_target,
            original_filename=Path(file_path).name or f"{file_id}.bin",
            metadata={
                "telegram_file_id": file_id,
                "telegram_file_path": file_path,
                "telegram_file_unique_id": file_meta.get("file_unique_id"),
                "telegram_file_size": file_meta.get("file_size"),
                "telegram_attachment": dict(attachment),
                "telegram_message_id": message_id,
                "telegram_update_id": update_id,
                "actor_external_user_id": actor_external_user_id,
            },
        )
    except Exception as exc:
        logger.warning(
            "TelegramSense failed to register audio attachment file_id=%s error=%s",
            file_id,
            exc,
        )
        attachment["asset_registration_status"] = "failed"
        attachment["asset_registration_error"] = str(exc) or "asset_registration_failed"
        return
    attachment["asset_registration_status"] = "registered"
    attachment["asset_id"] = str(asset.get("asset_id") or "")
    attachment["asset_kind"] = str(asset.get("kind") or "audio")
    attachment["asset_mime"] = str(asset.get("mime") or "")
    attachment["asset_bytes"] = int(asset.get("bytes") or 0)
    attachment["asset_sha256"] = str(asset.get("sha256") or "")
    attachment["telegram_file_path"] = str(file_meta.get("file_path") or "")
    attachment["telegram_file_unique_id"] = file_meta.get("file_unique_id")
    attachment["telegram_file_size"] = file_meta.get("file_size")


def _transcribe_registered_audio_asset(asset_id: str) -> dict[str, Any]:
    try:
        result = SttTranscribeTool().execute(asset_id=asset_id)
    except Exception as exc:
        logger.warning(
            "TelegramSense failed to transcribe audio asset_id=%s error=%s",
            asset_id,
            exc,
        )
        return {"status": "failed", "asset_id": asset_id, "error": str(exc) or "transcription_failed"}
    if isinstance(result, dict) and result.get("exception") is None:
        output = result.get("output") if isinstance(result.get("output"), dict) else {}
        text = str(output.get("text") or "").strip()
        if text:
            return {
                "status": "ok",
                "asset_id": asset_id,
                "text": text,
                "segments": output.get("segments") if isinstance(output.get("segments"), list) else [],
            }
    error = result.get("exception") if isinstance(result, dict) and isinstance(result.get("exception"), dict) else {}
    code = str(error.get("code") or "transcription_failed").strip()
    return {"status": "failed", "asset_id": asset_id, "error": code}


def _is_telegram_audio_attachment(attachment: dict[str, Any]) -> bool:
    provider = str(attachment.get("provider") or "").strip().lower()
    kind = str(attachment.get("kind") or "").strip().lower()
    return provider == "telegram" and kind in {"voice", "audio"}


def _resolve_internal_telegram_user_id(actor_external_user_id: str | None) -> str | None:
    try:
        return identity.resolve_user_id(
            service_id=TELEGRAM_SERVICE_ID,
            service_user_id=str(actor_external_user_id or "").strip() or None,
        )
    except Exception:
        return None


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."
