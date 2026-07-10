from __future__ import annotations

from dataclasses import dataclass, field

from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.integrations.presence import PresenceCapabilities
from alphonse.agent_v2.integrations.presence import PresencePhase
from alphonse.agent_v2.integrations.presence import PresenceProjector
from alphonse.agent_v2.integrations.presence import PresenceState
from alphonse.agent_v2.integrations.telegram.runtime import TelegramPresenceAdapter


@dataclass
class _RecordingAdapter:
    capabilities: PresenceCapabilities = field(
        default_factory=lambda: PresenceCapabilities(transient_activity=True, reactions=True)
    )
    events: list[tuple[str, PresencePhase, str]] = field(default_factory=list)

    def start(self, presence: PresenceState) -> None:
        self.events.append(("start", presence.phase, presence.address.integration_id))

    def update(self, presence: PresenceState) -> None:
        self.events.append(("update", presence.phase, presence.address.integration_id))

    def heartbeat(self, presence: PresenceState) -> None:
        self.events.append(("heartbeat", presence.phase, presence.address.integration_id))

    def stop(self, presence: PresenceState) -> None:
        self.events.append(("stop", presence.phase, presence.address.integration_id))


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.actions: list[dict[str, str]] = []
        self.reactions: list[dict[str, str]] = []

    def send_chat_action(self, *, chat_id: str, action: str) -> None:
        self.actions.append({"chat_id": chat_id, "action": action})

    def set_message_reaction(self, *, chat_id: str, message_id: str, emoji: str) -> None:
        self.reactions.append({"chat_id": chat_id, "message_id": message_id, "emoji": emoji})


def _queued_message(*, integration_id: str = "telegram-home"):
    queue = InMemoryMessageQueue()
    return CommunicationChannel(queue).queue_message(
        prompt="hello",
        user="u-alex",
        integration_id=integration_id,
        provider_key="telegram" if integration_id != "tui" else "tui",
        provider_user_id="123",
        channel_target="999",
        provider_message_id="55",
        correlation_id="corr-1",
    )


def test_projector_routes_lifecycle_and_activity_to_matching_adapter() -> None:
    projector = PresenceProjector(heartbeat_interval_sec=3)
    telegram = _RecordingAdapter()
    tui = _RecordingAdapter()
    projector.register("telegram-home", telegram)
    projector.register("tui", tui)

    queued = _queued_message()
    with projector.processing(queued):
        projector.on_activity(
            CoreActivityEvent(
                phase=ImprovementPhase.DO,
                label="working",
                message="Running a tool.",
            )
        )
        projector.finish()

    assert telegram.events == [
        ("start", PresencePhase.ACKNOWLEDGED, "telegram-home"),
        ("update", PresencePhase.EXECUTING, "telegram-home"),
        ("update", PresencePhase.DONE, "telegram-home"),
        ("stop", PresencePhase.DONE, "telegram-home"),
    ]
    assert tui.events == []


def test_projector_heartbeats_are_interval_limited() -> None:
    projector = PresenceProjector(heartbeat_interval_sec=3)
    adapter = _RecordingAdapter()
    projector.register("telegram-home", adapter)

    with projector.processing(_queued_message()):
        started_at = next(iter(projector._active.values())).started_at
        projector.heartbeat(now=started_at + 1)
        projector.heartbeat(now=started_at + 2)
        projector.heartbeat(now=started_at + 3)
        projector.heartbeat(now=started_at + 4)

    assert [event[0] for event in adapter.events].count("heartbeat") == 1


def test_projector_isolates_adapter_failures() -> None:
    class _BrokenAdapter(_RecordingAdapter):
        def update(self, presence: PresenceState) -> None:
            raise RuntimeError("provider down")

    projector = PresenceProjector()
    broken = _BrokenAdapter()
    healthy = _RecordingAdapter()
    projector.register("telegram-home", broken)
    projector.register("tui", healthy)

    with projector.processing(_queued_message()):
        projector.on_activity(
            CoreActivityEvent(phase=ImprovementPhase.DO, label="working", message="")
        )
        projector.finish()

    assert healthy.events == []
    assert broken.events[-1][0] == "stop"


def test_telegram_presence_maps_phases_to_typing_and_reactions() -> None:
    client = _FakeTelegramClient()
    adapter = TelegramPresenceAdapter(http_client=client)
    address = ChannelAddress(
        integration_id="telegram-home",
        provider_key="telegram",
        channel_target="999",
        provider_message_id="55",
    )

    for phase in PresencePhase:
        adapter.update(PresenceState(phase=phase, address=address, provider_message_id="55"))

    assert len(client.actions) == 3
    assert [item["emoji"] for item in client.reactions] == ["👀", "🤔", "⚡", "❓", "👍", "👎"]
    assert all(item["chat_id"] == "999" and item["message_id"] == "55" for item in client.reactions)


def test_telegram_presence_skips_reaction_without_provider_message_id() -> None:
    client = _FakeTelegramClient()
    adapter = TelegramPresenceAdapter(http_client=client)
    adapter.update(
        PresenceState(
            phase=PresencePhase.THINKING,
            address=ChannelAddress(
                integration_id="telegram-home",
                provider_key="telegram",
                channel_target="999",
            ),
        )
    )

    assert client.actions == [{"chat_id": "999", "action": "typing"}]
    assert client.reactions == []
