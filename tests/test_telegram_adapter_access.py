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
    assert str(payload.get("chat_id")) == "8593816828"
