from __future__ import annotations

from types import SimpleNamespace

from alphonse.agent.extremities.interfaces.integrations.telegram import telegram_adapter as telegram_module


def test_registered_private_chat_bypasses_static_allowed_chat_ids(monkeypatch) -> None:
    adapter = telegram_module.TelegramAdapter(
        {
            "bot_token": "fake-token",
            "poll_interval_sec": 0.0,
            "allowed_chat_ids": [8553589429],
        }
    )
    emitted: list[object] = []
    adapter.on_signal(lambda signal: emitted.append(signal))

    monkeypatch.setattr(telegram_module, "mark_update_processed", lambda _update_id, _chat_id: True)
    monkeypatch.setattr(
        telegram_module,
        "evaluate_inbound_access",
        lambda **_: SimpleNamespace(
            allowed=True,
            reason="registered_private",
            emit_invite=False,
            leave_chat=False,
            access=None,
        ),
    )

    update = {
        "update_id": 31223563,
        "message": {
            "message_id": 1,
            "text": "hola",
            "chat": {"id": 8593816828, "type": "private"},
            "from": {"id": 8593816828, "first_name": "Gabriela", "username": "gaby"},
        },
    }

    adapter._handle_update(update)

    assert len(emitted) == 1
    signal = emitted[0]
    assert getattr(signal, "type", "") == "external.telegram.message"
    payload = getattr(signal, "payload", {}) or {}
    assert payload.get("contract_type") == "canonical_inbound_event"
    assert str(payload.get("channel_target")) == "8593816828"
    assert str(payload.get("service_key")) == "telegram"
    assert str(payload.get("provider_user_id_from")) == "8593816828"
    assert str(payload.get("provider_message_id")) == "1"
    assert str(payload.get("event_kind")) == "message"
    assert str(payload.get("dedupe_key")) == "31223563"
    assert payload.get("service_id") is None
    assert payload.get("resolved_user_id") is None
    raw = payload.get("provider_raw_message")
    assert isinstance(raw, dict)
    assert int(raw.get("update_id") or 0) == 31223563


def test_telegram_adapter_maps_voice_payload_into_attachments(monkeypatch) -> None:
    adapter = telegram_module.TelegramAdapter(
        {
            "bot_token": "fake-token",
            "poll_interval_sec": 0.0,
            "allowed_chat_ids": [8553589429],
        }
    )
    emitted: list[object] = []
    adapter.on_signal(lambda signal: emitted.append(signal))

    monkeypatch.setattr(telegram_module, "mark_update_processed", lambda _update_id, _chat_id: True)
    monkeypatch.setattr(
        telegram_module,
        "evaluate_inbound_access",
        lambda **_: SimpleNamespace(
            allowed=True,
            reason="registered_private",
            emit_invite=False,
            leave_chat=False,
            access=None,
        ),
    )

    update = {
        "update_id": 31223564,
        "message": {
            "message_id": 2,
            "chat": {"id": 8593816828, "type": "private"},
            "from": {"id": 8593816828, "first_name": "Gabriela", "username": "gaby"},
            "voice": {"file_id": "voice-123", "duration": 4, "mime_type": "audio/ogg", "file_size": 1200},
        },
    }

    adapter._handle_update(update)

    assert len(emitted) == 1
    signal = emitted[0]
    payload = getattr(signal, "payload", {}) or {}
    raw = payload.get("provider_raw_message")
    assert isinstance(raw, dict)
    message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
    assert isinstance(message.get("voice"), dict)
    attachments = payload.get("attachments")
    assert isinstance(attachments, list)
    assert attachments
    first = attachments[0] if isinstance(attachments[0], dict) else {}
    assert str(first.get("kind") or "") == "voice"
    assert str(first.get("file_id") or "") == "voice-123"


def test_telegram_adapter_maps_contact_payload_into_attachments(monkeypatch) -> None:
    adapter = telegram_module.TelegramAdapter(
        {
            "bot_token": "fake-token",
            "poll_interval_sec": 0.0,
            "allowed_chat_ids": [8553589429],
        }
    )
    emitted: list[object] = []
    adapter.on_signal(lambda signal: emitted.append(signal))

    monkeypatch.setattr(telegram_module, "mark_update_processed", lambda _update_id, _chat_id: True)
    monkeypatch.setattr(
        telegram_module,
        "evaluate_inbound_access",
        lambda **_: SimpleNamespace(
            allowed=True,
            reason="registered_private",
            emit_invite=False,
            leave_chat=False,
            access=None,
        ),
    )

    update = {
        "update_id": 31223565,
        "message": {
            "message_id": 3,
            "chat": {"id": 8593816828, "type": "private"},
            "from": {"id": 8593816828, "first_name": "Gabriela", "username": "gaby"},
            "contact": {"user_id": 222, "first_name": "Maria", "phone_number": "+521555000"},
        },
    }

    adapter._handle_update(update)

    assert len(emitted) == 1
    signal = emitted[0]
    payload = getattr(signal, "payload", {}) or {}
    raw = payload.get("provider_raw_message")
    assert isinstance(raw, dict)
    message = raw.get("message") if isinstance(raw.get("message"), dict) else {}
    assert isinstance(message.get("contact"), dict)
    attachments = payload.get("attachments")
    assert isinstance(attachments, list)
    assert attachments
    first = attachments[0] if isinstance(attachments[0], dict) else {}
    assert str(first.get("kind") or "") == "contact"
    assert str(first.get("contact_user_id") or "") == "222"


def test_telegram_adapter_marks_edited_message_as_edit(monkeypatch) -> None:
    adapter = telegram_module.TelegramAdapter(
        {
            "bot_token": "fake-token",
            "poll_interval_sec": 0.0,
            "allowed_chat_ids": [8553589429],
        }
    )
    emitted: list[object] = []
    adapter.on_signal(lambda signal: emitted.append(signal))

    monkeypatch.setattr(telegram_module, "mark_update_processed", lambda _update_id, _chat_id: True)
    monkeypatch.setattr(
        telegram_module,
        "evaluate_inbound_access",
        lambda **_: SimpleNamespace(
            allowed=True,
            reason="registered_private",
            emit_invite=False,
            leave_chat=False,
            access=None,
        ),
    )

    update = {
        "update_id": 31223566,
        "edited_message": {
            "message_id": 4,
            "text": "edited hola",
            "chat": {"id": 8593816828, "type": "private"},
            "from": {"id": 8593816828, "first_name": "Gabriela", "username": "gaby"},
        },
    }

    adapter._handle_update(update)

    assert len(emitted) == 1
    signal = emitted[0]
    payload = getattr(signal, "payload", {}) or {}
    assert payload.get("contract_type") == "canonical_inbound_event"
    assert str(payload.get("provider_message_id")) == "4"
    assert str(payload.get("event_kind")) == "edit"
