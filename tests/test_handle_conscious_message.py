from __future__ import annotations

from alphonse.agent.actions.conscious_message_handler import build_incoming_message_envelope
from alphonse.agent.actions.handle_conscious_message import HandleConsciousMessageAction
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import apply_seed
from alphonse.agent.nervous_system.senses.bus import Bus, Signal
from alphonse.agent.nervous_system.senses.cli import build_cli_user_message_signal


def test_handle_conscious_message_enqueues_pdca_slice(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    called: dict[str, object] = {}
    presence: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "s-1"},
    )
    def _enqueue(**kwargs):
        called.update(kwargs)
        return "task-123"

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        _enqueue,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **kwargs: presence.update(kwargs),
    )

    envelope = build_incoming_message_envelope(
        message_id="m-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-1",
        actor_external_user_id="u-ext",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.intention_key == "NOOP"
    assert result.payload.get("task_id") == "task-123"
    assert called.get("session_user_id") == "u-1"
    assert presence.get("phase") == "acknowledged"
    seeded_state = called.get("state")
    assert isinstance(seeded_state, dict)
    assert seeded_state.get("message_id") == "m-1"


def test_handle_conscious_message_does_not_parse_control_intent(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "s-1"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-cancel-test",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Please cancel task",
        correlation_id="c-1",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.payload.get("task_id") == "task-cancel-test"
    assert captured.get("payload", {}).get("text") == "Please cancel task"


def test_handle_conscious_message_fail_fast_when_slicing_disabled(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: called.update(kwargs) or "unexpected-task",
    )

    envelope = build_incoming_message_envelope(
        message_id="m-2",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-2",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.intention_key == "MESSAGE_READY"
    assert "temporarily unable to process messages" in str(result.payload.get("message") or "")
    assert called == {}


def test_handle_conscious_message_logs_missing_actor_context(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    emitted: list[dict[str, object]] = []

    class _FakeLog:
        def emit(self, **kwargs: object) -> None:
            emitted.append(dict(kwargs))

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "s-1"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **_: "task-ctx-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message._LOG",
        _FakeLog(),
    )

    envelope = build_incoming_message_envelope(
        message_id="m-ctx-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-ctx-1",
    )
    _ = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert any(str(item.get("event") or "") == "incoming_message.context_missing_fields" for item in emitted)


def test_handle_conscious_message_backfills_person_id_from_external_user_id(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "s-1"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-backfill-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.identity.resolve_service_id",
        lambda service_key: 1 if str(service_key or "").strip() == "telegram" else None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.identity.resolve_user_id",
        lambda **kwargs: "owner-1" if kwargs == {"service_id": 1, "service_user_id": "u-ext"} else None,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-backfill-1",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="Hello",
        correlation_id="c-backfill-1",
        actor_external_user_id="u-ext",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.payload.get("task_id") == "task-backfill-1"
    seeded_state = captured.get("state")
    assert isinstance(seeded_state, dict)
    assert seeded_state.get("actor_person_id") == "owner-1"
    assert seeded_state.get("incoming_user_id") == "u-ext"


def test_handle_conscious_message_cli_seeded_identity_has_no_missing_actor_context(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    apply_seed(db_path)

    action = HandleConsciousMessageAction()
    emitted: list[dict[str, object]] = []

    class _FakeLog:
        def emit(self, **kwargs: object) -> None:
            emitted.append(dict(kwargs))

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "owner-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "owner-1|2026-03-12", "user_id": "owner-1", "date": "2026-03-12"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **_: "task-cli-ctx-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message._LOG",
        _FakeLog(),
    )

    signal = build_cli_user_message_signal(
        text="Hi Alphonse!",
        correlation_id="c-cli-1",
        metadata={"source": "test.cli"},
    )
    _ = action.execute({"signal": signal, "ctx": Bus()})
    assert not any(str(item.get("event") or "") == "incoming_message.context_missing_fields" for item in emitted)


def test_handle_conscious_message_writes_through_user_message_to_day_session(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.build_next_session_state",
        lambda **kwargs: captured.update({"build": kwargs}) or {"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.commit_session_state",
        lambda state: captured.update({"committed": state}),
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **_: "task-write-through",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-write-through",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="This should be in history immediately",
        correlation_id="cid-write-through",
    )
    _ = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    build_kwargs = captured.get("build")
    assert isinstance(build_kwargs, dict)
    assert build_kwargs.get("user_message") == "This should be in history immediately"
    assert build_kwargs.get("assistant_message") == ""
    assert "committed" in captured


def test_handle_conscious_message_attachment_only_writes_history_summary(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.build_next_session_state",
        lambda **kwargs: captured.update({"build": kwargs}) or {"session_id": "u-1|2026-03-12", "user_id": "u-1", "date": "2026-03-12"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.commit_session_state",
        lambda state: captured.update({"committed": state}),
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **_: "task-attachment-only",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-attach-only",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="",
        attachments=[{"kind": "photo", "provider": "telegram", "file_id": "photo-1"}],
        correlation_id="cid-attach-only",
    )
    _ = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    build_kwargs = captured.get("build")
    assert isinstance(build_kwargs, dict)
    assert str(build_kwargs.get("user_message") or "").startswith("[attachments:")


def test_handle_conscious_message_transcribes_telegram_voice_when_text_empty(monkeypatch) -> None:
    action = HandleConsciousMessageAction()
    captured: dict[str, object] = {}

    class _FakeTranscribeTool:
        def execute(self, *, file_id: str, language: str | None = None, sandbox_alias: str = "telegram") -> dict[str, object]:
            _ = (language, sandbox_alias)
            return {
                "output": {"text": f"transcript for {file_id}"},
                "exception": None,
                "metadata": {"tool": "transcribe_telegram_audio"},
            }

    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.TranscribeTelegramAudioTool",
        _FakeTranscribeTool,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_user_id",
        lambda **_: "u-1",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_session_timezone",
        lambda *_: "UTC",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.resolve_day_session",
        lambda **_: {"session_id": "s-1"},
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.enqueue_pdca_slice",
        lambda **kwargs: captured.update(kwargs) or "task-transcribed",
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.is_pdca_slicing_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "alphonse.agent.actions.handle_conscious_message.emit_presence_phase_changed",
        lambda **_: None,
    )

    envelope = build_incoming_message_envelope(
        message_id="m-voice-only",
        channel_type="telegram",
        channel_target="123",
        provider="telegram",
        text="",
        attachments=[{"kind": "voice", "provider": "telegram", "file_id": "voice-1"}],
        correlation_id="cid-voice-only",
    )
    result = action.execute({"signal": Signal(type="sense.telegram.message.user.received", payload=envelope), "ctx": Bus()})
    assert result.payload.get("task_id") == "task-transcribed"
    enqueued_payload = captured.get("payload")
    assert isinstance(enqueued_payload, dict)
    assert str(enqueued_payload.get("text") or "") == "transcript for voice-1"
