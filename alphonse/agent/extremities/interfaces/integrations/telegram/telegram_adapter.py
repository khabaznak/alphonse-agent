"""Telegram integration adapter (polling)."""

from __future__ import annotations

import json
import logging
import threading
import time
from urllib import parse, request
from typing import Any

from alphonse.agent.extremities.interfaces.integrations._contracts import IntegrationAdapter
from alphonse.agent.nervous_system.senses.bus import Signal as BusSignal
from alphonse.agent.nervous_system.telegram_updates_store import mark_update_processed

logger = logging.getLogger(__name__)
_SNIPPET_LIMIT = 80


class TelegramAdapter(IntegrationAdapter):
    """Minimal Telegram adapter using polling."""

    id = "telegram"
    io_type = "io"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._running = False
        self._thread: threading.Thread | None = None
        self._application = None
        self._last_update_id: int | None = None
        self._seen_update_ids: set[int] = set()

        self._bot_token = str(config.get("bot_token") or "").strip()
        if not self._bot_token:
            raise ValueError("TelegramAdapter requires bot_token in config")

        allowed = config.get("allowed_chat_ids")
        if allowed is None:
            self._allowed_chat_ids: set[int] | None = None
        else:
            self._allowed_chat_ids = {int(chat_id) for chat_id in allowed}

        self._poll_interval_sec = float(config.get("poll_interval_sec", 1.0))

    @property
    def id(self) -> str:  # type: ignore[override]
        return "telegram"

    @property
    def io_type(self) -> str:  # type: ignore[override]
        return "io"

    def start(self) -> None:
        logger.info("TelegramAdapter start()")
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_polling, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logger.info("TelegramAdapter stop()")
        self._running = False
        if self._application is not None:
            try:
                self._application.stop_running()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def handle_action(self, action: dict[str, Any]) -> None:
        action_type = action.get("type")
        if action_type == "set_message_reaction":
            payload = action.get("payload") or {}
            chat_id = payload.get("chat_id")
            message_id = payload.get("message_id")
            emoji = str(payload.get("emoji") or "").strip()
            if chat_id is None or message_id is None or not emoji:
                logger.warning(
                    "TelegramAdapter missing chat_id/message_id/emoji in set_message_reaction payload"
                )
                return
            try:
                chat_id_int = int(chat_id)
                message_id_int = int(message_id)
            except (TypeError, ValueError):
                logger.warning(
                    "TelegramAdapter invalid chat_id/message_id in set_message_reaction payload chat_id=%s message_id=%s",
                    chat_id,
                    message_id,
                )
                return
            logger.info(
                "TelegramAdapter setting reaction chat_id=%s message_id=%s emoji=%s",
                chat_id,
                message_id,
                emoji,
            )
            self._set_message_reaction_http(
                chat_id=chat_id_int,
                message_id=message_id_int,
                emoji=emoji,
            )
            return
        if action_type == "send_chat_action":
            payload = action.get("payload") or {}
            chat_id = payload.get("chat_id")
            chat_action = payload.get("action") or "typing"
            if chat_id is None:
                logger.warning("TelegramAdapter missing chat_id in chat_action payload")
                return
            logger.info(
                "TelegramAdapter sending chat_action=%s to %s",
                chat_action,
                chat_id,
            )
            self._send_chat_action_http(chat_id, str(chat_action))
            return
        if action_type != "send_message":
            logger.warning("TelegramAdapter ignoring action: %s", action_type)
            return

        payload = action.get("payload") or {}
        chat_id = payload.get("chat_id")
        text = payload.get("text")
        if chat_id is None or text is None:
            logger.warning("TelegramAdapter missing chat_id/text in action payload")
            return

        logger.info("TelegramAdapter sending message to %s", chat_id)

        self._send_message_http(chat_id, text)

    def _run_polling(self) -> None:
        while self._running:
            offset = (self._last_update_id or 0) + 1
            updates = self._fetch_updates(offset)
            if updates is None:
                time.sleep(self._poll_interval_sec)
                continue
            max_update_id = None
            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    max_update_id = update_id if max_update_id is None else max(max_update_id, update_id)
                self._handle_update(update)
            if max_update_id is not None:
                self._last_update_id = max_update_id
            logger.info(
                "TelegramAdapter getUpdates offset=%s returned=%s max_update_id=%s",
                offset,
                len(updates),
                max_update_id,
            )
            time.sleep(self._poll_interval_sec)

    def _fetch_updates(self, offset: int) -> list[dict[str, Any]] | None:
        endpoint = "getUpdates"
        url = f"https://api.telegram.org/bot{self._bot_token}/{endpoint}"
        data = parse.urlencode({"timeout": 0, "offset": offset}).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        try:
            with request.urlopen(req, timeout=10) as response:
                body = response.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            logger.error("TelegramAdapter %s failed: %s", endpoint, exc)
            return None
        parsed = _parse_json(body)
        if not isinstance(parsed, dict):
            logger.error("TelegramAdapter %s invalid JSON", endpoint)
            return None
        if not parsed.get("ok"):
            logger.error(
                "TelegramAdapter %s ok=false error_code=%s description=%s",
                endpoint,
                parsed.get("error_code"),
                parsed.get("description"),
            )
            return None
        result = parsed.get("result")
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        return []

    def _handle_update(self, update: dict[str, Any]) -> None:
        update_id = update.get("update_id")
        if not isinstance(update_id, int):
            return
        if update_id in self._seen_update_ids:
            logger.info("TelegramAdapter duplicate update_id=%s skipped", update_id)
            return
        self._seen_update_ids.add(update_id)

        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = message.get("text") or ""
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        chat_id = message.get("chat_id") or chat.get("id")
        if chat_id is None:
            return
        if not mark_update_processed(update_id, str(chat_id)):
            logger.info("TelegramAdapter duplicate update_id=%s skipped (db)", update_id)
            return
        if self._allowed_chat_ids is not None and int(chat_id) not in self._allowed_chat_ids:
            logger.info(
                "TelegramAdapter rejected message update_id=%s chat_id=%s reason=not_allowed",
                update_id,
                chat_id,
            )
            if self._should_emit_invite(message):
                self._emit_invite_signal(update, message)
            return

        from_user_payload = message.get("from") if isinstance(message.get("from"), dict) else {}
        from_user = from_user_payload.get("username") or from_user_payload.get("id")
        from_user_name = from_user_payload.get("first_name") or from_user_payload.get("username")
        reply_to = message.get("reply_to_message") if isinstance(message.get("reply_to_message"), dict) else {}
        reply_from = reply_to.get("from") if isinstance(reply_to.get("from"), dict) else {}
        reply_to_user = reply_from.get("id")
        reply_to_user_name = reply_from.get("first_name") or reply_from.get("username")

        logger.info(
            "TelegramAdapter accepted message update_id=%s chat_id=%s from=%s text=%s",
            update_id,
            chat_id,
            from_user,
            _snippet(str(text)),
        )

        signal = BusSignal(
            type="external.telegram.message",
            payload={
                "text": text,
                "chat_id": chat_id,
                "from_user": from_user,
                "from_user_name": from_user_name,
                "reply_to_user": reply_to_user,
                "reply_to_user_name": reply_to_user_name,
                "reply_to_message_id": reply_to.get("message_id") if isinstance(reply_to, dict) else None,
                "message_id": message.get("message_id"),
                "update_id": update_id,
                "provider_event": update,
            },
            source="telegram",
        )
        logger.info(
            "TelegramAdapter emitting external signal update_id=%s chat_id=%s",
            update_id,
            chat_id,
        )
        self.emit_signal(signal)  # type: ignore[arg-type]

    def get_file(self, file_id: str) -> dict[str, Any]:
        endpoint = "getFile"
        url = f"https://api.telegram.org/bot{self._bot_token}/{endpoint}"
        data = parse.urlencode({"file_id": str(file_id)}).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        with request.urlopen(req, timeout=10) as response:
            body = response.read().decode("utf-8", errors="ignore")
        parsed = _parse_json(body)
        if not isinstance(parsed, dict) or not parsed.get("ok") or not isinstance(parsed.get("result"), dict):
            raise RuntimeError(f"TelegramAdapter {endpoint} failed")
        return dict(parsed["result"])

    def download_file(self, file_path: str) -> bytes:
        endpoint = "downloadFile"
        url = f"https://api.telegram.org/file/bot{self._bot_token}/{file_path}"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=20) as response:
            body = response.read()
            logger.info(
                "TelegramAdapter %s status=%s file_path=%s bytes=%s",
                endpoint,
                getattr(response, "status", "unknown"),
                file_path,
                len(body),
            )
            return body

    def _should_emit_invite(self, message: dict[str, Any]) -> bool:
        text = str(message.get("text") or "").strip()
        return bool(text)

    def _emit_invite_signal(self, update: dict[str, Any], message: dict[str, Any]) -> None:
        text = message.get("text") or ""
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        chat_id = message.get("chat_id") or chat.get("id")
        if chat_id is None:
            return
        from_user_payload = message.get("from") if isinstance(message.get("from"), dict) else {}
        from_user = from_user_payload.get("username") or from_user_payload.get("id")
        from_user_name = from_user_payload.get("first_name") or from_user_payload.get("username")
        update_id = update.get("update_id")
        self.emit_signal(
            BusSignal(
                type="external.telegram.invite_request",
                payload={
                    "chat_id": chat_id,
                    "from_user": from_user,
                    "from_user_name": from_user_name,
                    "text": text,
                    "update_id": update_id,
                },
                source="telegram",
            )
        )

    def _send_message_http(self, chat_id: int, text: str) -> None:
        endpoint = "sendMessage"
        url = f"https://api.telegram.org/bot{self._bot_token}/{endpoint}"
        data = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        try:
            with request.urlopen(req, timeout=10) as response:
                body = response.read()
                body_text = body.decode("utf-8", errors="ignore")
                logger.info(
                    "TelegramAdapter %s status=%s chat_id=%s body=%s",
                    endpoint,
                    response.status,
                    chat_id,
                    _snippet(body_text),
                )
                parsed = _parse_json(body_text)
                if not isinstance(parsed, dict):
                    raise RuntimeError("TelegramAdapter invalid JSON response")
                ok = bool(parsed.get("ok"))
                if ok:
                    message_id = None
                    if isinstance(parsed.get("result"), dict):
                        message_id = parsed["result"].get("message_id")
                    logger.info(
                        "TelegramAdapter %s ok=true message_id=%s",
                        endpoint,
                        message_id,
                    )
                    return
                error_code = parsed.get("error_code")
                description = parsed.get("description")
                logger.error(
                    "TelegramAdapter %s ok=false error_code=%s description=%s",
                    endpoint,
                    error_code,
                    description,
                )
                raise RuntimeError(f"TelegramAdapter {endpoint} failed: {error_code} {description}")
        except Exception as exc:
            logger.error("TelegramAdapter %s failed: %s", endpoint, exc)
            raise

    def _send_chat_action_http(self, chat_id: int, action: str) -> None:
        endpoint = "sendChatAction"
        url = f"https://api.telegram.org/bot{self._bot_token}/{endpoint}"
        data = parse.urlencode({"chat_id": chat_id, "action": action}).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        try:
            with request.urlopen(req, timeout=10) as response:
                body = response.read().decode("utf-8", errors="ignore")
                logger.info(
                    "TelegramAdapter %s status=%s chat_id=%s body=%s",
                    endpoint,
                    response.status,
                    chat_id,
                    _snippet(body),
                )
        except Exception as exc:
            logger.error("TelegramAdapter %s failed: %s", endpoint, exc)

    def _set_message_reaction_http(self, chat_id: int, message_id: int, emoji: str) -> None:
        endpoint = "setMessageReaction"
        url = f"https://api.telegram.org/bot{self._bot_token}/{endpoint}"
        reaction_payload = json.dumps(
            [{"type": "emoji", "emoji": emoji}],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        data = parse.urlencode(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reaction": reaction_payload,
                "is_big": "false",
            }
        ).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        try:
            with request.urlopen(req, timeout=10) as response:
                body = response.read().decode("utf-8", errors="ignore")
                logger.info(
                    "TelegramAdapter %s status=%s chat_id=%s message_id=%s body=%s",
                    endpoint,
                    response.status,
                    chat_id,
                    message_id,
                    _snippet(body),
                )
        except Exception as exc:
            logger.error("TelegramAdapter %s failed: %s", endpoint, exc)


def _snippet(text: str) -> str:
    return text if len(text) <= _SNIPPET_LIMIT else f"{text[:_SNIPPET_LIMIT]}..."


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None
