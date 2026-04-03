from __future__ import annotations

from pathlib import Path

from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system import users as users_store
from alphonse.agent.nervous_system.senses.base import SignalSpec
from alphonse.agent.nervous_system.senses.registry import all_senses
from alphonse.agent.nervous_system.senses.registry import all_signal_specs
from alphonse.agent.nervous_system.senses.user_communication import UserCommunicationSense
from alphonse.agent.nervous_system.senses.user_communication import build_canonical_user_message
from alphonse.agent.nervous_system.migrate import apply_schema


def test_user_communication_sense_is_discovered() -> None:
    sense_keys = {sense_cls.key for sense_cls in all_senses()}
    assert "user_communication" in sense_keys
    assert any(sense_cls is UserCommunicationSense for sense_cls in all_senses())


def test_user_communication_sense_declares_canonical_signal() -> None:
    specs = [
        spec for spec in all_signal_specs()
        if isinstance(spec, SignalSpec) and spec.key == "sense.user_communication.message.user.received"
    ]
    assert specs
    assert any((spec.source or "") == "user_communication" for spec in specs)


def test_canonical_user_message_payload_resolved_case() -> None:
    payload = build_canonical_user_message(
        message_id="m-1",
        correlation_id="c-1",
        occurred_at="2026-04-03T12:00:00+00:00",
        service_key="telegram",
        channel_type="telegram",
        channel_target="123",
        external_user_id="555",
        display_name="Alex",
        text="hello",
        attachments=[{"kind": "audio", "file_id": "f-1"}],
        metadata={"provider_event_id": "evt-1"},
        resolved_user_id="u-1",
    )
    assert payload["service_id"] == 2
    assert payload["service_key"] == "telegram"
    assert payload["identity_resolved"] is True
    assert payload["alphonse_user_id"] == "u-1"
    assert payload["identity"]["resolved"] is True
    assert payload["identity"]["alphonse_user_id"] == "u-1"
    assert payload["identity"]["user_id"] == "u-1"
    assert payload["identity"]["external_user_id"] == "555"
    assert payload["transport"]["service_id"] == 2
    assert payload["transport"]["service_key"] == "telegram"
    assert payload["transport"]["channel_type"] == "telegram"
    assert payload["transport"]["channel_target"] == "123"
    assert payload["content"]["text"] == "hello"
    assert payload["content"]["attachments"][0]["file_id"] == "f-1"


def test_canonical_user_message_payload_unresolved_case() -> None:
    payload = build_canonical_user_message(
        message_id="m-2",
        occurred_at="2026-04-03T12:05:00+00:00",
        service_key="discord",
        channel_type="discord",
        channel_target="chan-1",
        external_user_id="disc-user-9",
        display_name="Unknown User",
        text="ping",
        metadata={"raw_message_id": "provider-22"},
    )
    assert payload["service_id"] is None
    assert payload["service_key"] == "discord"
    assert payload["identity_resolved"] is False
    assert payload["identity"]["resolved"] is False
    assert payload["identity"]["user_id"] is None
    assert payload["identity"]["external_user_id"] == "disc-user-9"
    assert payload["transport"]["service_key"] == "discord"
    assert payload["metadata"]["raw_message_id"] == "provider-22"


def test_user_communication_sense_canonicalizes_inbound_identity(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    users_store.upsert_user(
        {
            "user_id": "u-1",
            "principal_id": "p-1",
            "display_name": "Alex",
            "is_active": True,
        }
    )
    resolvers.upsert_service_resolver(
        user_id="u-1",
        service_id=2,
        service_user_id="8553589429",
        is_active=True,
    )

    sense = UserCommunicationSense()
    payload = sense.canonicalize_message(
        message_id="m-3",
        correlation_id="c-3",
        occurred_at="2026-04-03T12:10:00+00:00",
        service_key="telegram",
        service_user_id="8553589429",
        channel_target="8553589429",
        display_name="Alex",
        text="hello there",
        metadata={"provider_event_id": "evt-3"},
    )

    assert payload["service_id"] == 2
    assert payload["service_key"] == "telegram"
    assert payload["identity_resolved"] is True
    assert payload["alphonse_user_id"] == "u-1"


def test_user_communication_sense_preserves_unresolved_identity() -> None:
    sense = UserCommunicationSense()
    payload = sense.canonicalize_message(
        message_id="m-4",
        occurred_at="2026-04-03T12:11:00+00:00",
        service_key="discord",
        service_user_id="disc-user-9",
        channel_target="chan-1",
        display_name="Unknown User",
        text="ping",
    )

    assert payload["service_key"] == "discord"
    assert payload["service_id"] is None
    assert payload["identity_resolved"] is False
    assert payload["alphonse_user_id"] is None


def test_existing_provider_signals_remain_available() -> None:
    signal_keys = {spec.key for spec in all_signal_specs()}
    assert "sense.telegram.message.user.received" in signal_keys
    assert "sense.api.message.user.received" in signal_keys
