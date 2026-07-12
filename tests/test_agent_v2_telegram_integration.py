from __future__ import annotations

from pathlib import Path

from alphonse.agent import identity as users_store
from alphonse.agent.nervous_system import user_service_resolvers as resolvers
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.io import IntegrationIdentity
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.io import V2IdentityResolver
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.integrations.telegram.runtime import TelegramIntegrationRuntime
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter
from alphonse.agent_v2.services.project_sessions import ProjectSessionKey
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore


class FakeTelegramClient:
    def __init__(self, updates: list[dict] | None = None) -> None:
        self.updates = updates or []
        self.sent: list[dict[str, str]] = []

    def get_updates(self, *, offset: int | None = None, timeout: int = 0) -> list[dict]:
        _ = offset
        _ = timeout
        return list(self.updates)

    def send_message(self, *, chat_id: str, text: str) -> str:
        self.sent.append({"chat_id": chat_id, "text": text})
        return "777"


def test_telegram_inbound_text_queues_canonical_core_message(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "u-alex", "display_name": "Alex", "is_active": True})
    resolvers.upsert_service_resolver(user_id="u-alex", service_id=2, service_user_id="123")
    queue = InMemoryMessageQueue()
    runtime = _runtime(
        queue=queue,
        http_client=FakeTelegramClient(
            [
                {
                    "update_id": 10,
                    "message": {
                        "message_id": 5,
                        "text": "hello",
                        "chat": {"id": "999"},
                        "from": {"id": "123", "first_name": "Alex"},
                    },
                }
            ]
        ),
    )

    runtime.poll_once()

    queued = queue.dequeue()
    assert queued is not None
    assert queued.message.user == "u-alex"
    assert queued.message.prompt == "hello"
    assert queued.message.metadata["channel"]["integration_id"] == "telegram-home"
    assert queued.message.metadata["channel"]["provider_key"] == "telegram"
    assert queued.message.metadata["channel"]["provider_user_id"] == "123"
    assert queued.message.metadata["channel"]["channel_target"] == "999"
    assert queued.message.metadata["channel"]["provider_message_id"] == "5"


def test_telegram_inbound_text_wakes_processor_after_queueing(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "u-alex", "display_name": "Alex", "is_active": True})
    resolvers.upsert_service_resolver(user_id="u-alex", service_id=2, service_user_id="123")
    wakes: list[str] = []
    runtime = _runtime(
        http_client=FakeTelegramClient(
            [
                {
                    "update_id": 10,
                    "message": {
                        "message_id": 5,
                        "text": "hello",
                        "chat": {"id": "999"},
                        "from": {"id": "123", "first_name": "Alex"},
                    },
                }
            ]
        ),
        on_message_queued=lambda: wakes.append("wake"),
    )

    runtime.poll_once()

    assert wakes == ["wake"]
    assert runtime.stats.messages_queued == 1


def test_telegram_unknown_user_notifies_owner_without_queueing_capd_task(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    queue = InMemoryMessageQueue()
    outbox = SQLiteOutboundStore()
    wakes: list[str] = []
    runtime = _runtime(
        queue=queue,
        outbox=outbox,
        http_client=FakeTelegramClient(
            [
                {
                    "update_id": 10,
                    "message": {
                        "message_id": 5,
                        "text": "hello",
                        "chat": {"id": "999"},
                        "from": {"id": "missing"},
                    },
                }
            ]
        ),
        on_message_queued=lambda: wakes.append("wake"),
    )

    runtime.poll_once()

    assert queue.size() == 0
    assert wakes == []
    outbound = outbox.claim_next(OutboundSelector(integration_id="tui", channel_target="local"))
    assert outbound is not None
    assert outbound.kind == "identity_resolution"
    assert "provider_user_id=missing" in outbound.message


def test_telegram_outbound_drains_matching_outbox_message() -> None:
    outbox = SQLiteOutboundStore()
    outbox.enqueue(
        address=ChannelAddress(
            integration_id="telegram-home",
            provider_key="telegram",
            channel_target="999",
            alphonse_user_id="u-alex",
            provider_user_id="123",
        ),
        message="reply",
    )
    client = FakeTelegramClient()
    runtime = _runtime(outbox=outbox, http_client=client)

    runtime.drain_outbox_once()

    assert client.sent == [{"chat_id": "999", "text": "reply"}]
    delivered = outbox.list(OutboundSelector(integration_id="telegram-home", status="delivered"))
    assert len(delivered) == 1
    assert delivered[0].provider_message_id == "777"


def test_telegram_project_command_is_handled_without_queueing_capd_work(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    users_store.upsert_user({"user_id": "u-alex", "display_name": "Alex", "is_active": True})
    resolvers.upsert_service_resolver(user_id="u-alex", service_id=2, service_user_id="123")
    queue = InMemoryMessageQueue()
    outbox = SQLiteOutboundStore()
    projects = ProjectStore(":memory:")
    project = projects.create_project(name="Exercise", root_path=str(tmp_path / "exercise"), owner_user_id="u-alex")
    router = ProjectInboundRouter(
        channel=CommunicationChannel(queue), outbox=outbox, projects=projects, sessions=SQLiteProjectSessionStore(":memory:")
    )
    client = FakeTelegramClient(
        [{"update_id": 10, "message": {"message_id": 5, "text": "/project Exercise", "chat": {"id": "999"}, "from": {"id": "123"}}}]
    )
    runtime = _runtime(queue=queue, outbox=outbox, http_client=client, inbound_router=router)

    runtime.poll_once()

    assert queue.size() == 0
    assert client.sent == [{"chat_id": "999", "text": "Active project: Exercise."}]
    assert router.active_project(ProjectSessionKey("u-alex", "telegram-home", "999")).project_id == project.project_id


def _runtime(
    *,
    queue: InMemoryMessageQueue | None = None,
    outbox: SQLiteOutboundStore | None = None,
    http_client: FakeTelegramClient | None = None,
    on_message_queued=None,
    inbound_router=None,
) -> TelegramIntegrationRuntime:
    store = SQLiteIntegrationStore()
    record = store.upsert(
        integration_id="telegram-home",
        provider_key="telegram",
        display_name="Telegram Home",
        enabled=True,
        config={"poll_interval_sec": 1.0, "owner_user_id": "local"},
        secrets={"bot_token": "token"},
    )
    return TelegramIntegrationRuntime(
        record=record,
        channel=CommunicationChannel(queue or InMemoryMessageQueue()),
        outbox=outbox or SQLiteOutboundStore(),
        identity_resolver=V2IdentityResolver((IntegrationIdentity("telegram-home", "telegram"),)),
        owner_user_id="local",
        http_client=http_client or FakeTelegramClient(),
        on_message_queued=on_message_queued,
        inbound_router=inbound_router,
    )
