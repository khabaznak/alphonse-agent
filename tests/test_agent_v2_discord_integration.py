from __future__ import annotations

from pathlib import Path

from alphonse.agent_v2.assets import AttachmentDescriptor, SQLiteAssetStore
from alphonse.agent_v2.core.io import ChannelAddress, IntegrationIdentity, OutboundSelector, SQLiteOutboundStore, V2IdentityResolver
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.integrations.discord.runtime import DiscordIntegrationRuntime
from alphonse.agent_v2.integrations.registry import build_default_integration_registry
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.users import V2UserStore


class FakeDiscordClient:
    def __init__(self) -> None:
        self.callback = None
        self.sent: list[dict[str, str]] = []
        self.reactions: list[dict[str, str]] = []
        self.typing: list[str] = []

    def start(self, callback) -> None:
        self.callback = callback

    def stop(self) -> None:
        return

    def send_message(self, *, channel_id: str, text: str) -> str:
        self.sent.append({"channel_id": channel_id, "text": text})
        return "sent-1"

    def send_asset(self, *, channel_id: str, asset, caption: str = "") -> str:
        self.sent.append({"channel_id": channel_id, "text": caption, "asset": asset.asset_id})
        return "sent-asset"

    def send_typing(self, *, channel_id: str) -> None:
        self.typing.append(channel_id)

    def set_message_reaction(self, *, channel_id: str, message_id: str, emoji: str) -> None:
        self.reactions.append({"channel_id": channel_id, "message_id": message_id, "emoji": emoji})

    def describe_inbound(self, raw: dict) -> list[AttachmentDescriptor]:
        return [AttachmentDescriptor("ticket.pdf", "application/pdf", 3, "attachment-1", "", "attachment")] if raw.get("attachments") else []

    def download(self, descriptor: AttachmentDescriptor) -> bytes:
        assert descriptor.provider_file_id == "attachment-1"
        return b"pdf"


def test_default_registry_includes_discord() -> None:
    assert build_default_integration_registry().get("discord") is not None


def test_discord_guild_message_requires_mention_then_continues_thread(tmp_path: Path) -> None:
    users = _users(tmp_path)
    users.bind_address(user_id="u-alex", integration_id="discord-home", provider_key="discord", provider_user_id="123")
    queue = InMemoryMessageQueue()
    runtime = _runtime(queue=queue, users=users)

    assert runtime.handle_message(_message(mentions_bot=False, thread_id="thread-1")) is False
    assert runtime.handle_message(_message(mentions_bot=True, thread_id="thread-1")) is True
    assert runtime.handle_message(_message(message_id="3", mentions_bot=False, thread_id="thread-1")) is True

    first = queue.dequeue()
    second = queue.dequeue()
    assert first is not None and second is not None
    assert first.message.metadata["channel"]["provider_key"] == "discord"
    assert first.message.metadata["guild_id"] == "guild-1"
    assert second.message.metadata["channel"]["thread_id"] == "thread-1"


def test_discord_private_unknown_user_requests_access(tmp_path: Path) -> None:
    users = _users(tmp_path)
    client = FakeDiscordClient()
    runtime = _runtime(users=users, client=client)

    assert runtime.handle_message(_message(guild_id="", is_private=True, author_id="unknown")) is False

    requests = users.list_access_requests()
    assert len(requests) == 1
    assert requests[0].provider_key == "discord"
    assert client.sent == [{"channel_id": "channel-1", "text": "Your request to talk to Alphonse has been sent to the administrator. Please wait for approval."}]


def test_discord_respects_guild_channel_allow_lists(tmp_path: Path) -> None:
    users = _users(tmp_path)
    users.bind_address(user_id="u-alex", integration_id="discord-home", provider_key="discord", provider_user_id="123")
    runtime = _runtime(users=users, config={"allowed_guild_ids": ["guild-allowed"], "allowed_channel_ids": ["channel-allowed"]})

    assert runtime.handle_message(_message(guild_id="guild-other", channel_id="channel-allowed", mentions_bot=True)) is False
    assert runtime.handle_message(_message(guild_id="guild-allowed", channel_id="channel-other", mentions_bot=True)) is False
    assert runtime.handle_message(_message(guild_id="guild-allowed", channel_id="channel-allowed", mentions_bot=True)) is True


def test_discord_registers_attachment_and_delivers_outbox(tmp_path: Path) -> None:
    users = _users(tmp_path)
    users.bind_address(user_id="u-alex", integration_id="discord-home", provider_key="discord", provider_user_id="123")
    assets = SQLiteAssetStore(tmp_path / "assets.sqlite3", tmp_path / "assets")
    queue = InMemoryMessageQueue()
    client = FakeDiscordClient()
    outbox = SQLiteOutboundStore()
    runtime = _runtime(queue=queue, users=users, client=client, outbox=outbox, asset_store=assets)

    assert runtime.handle_message(_message(mentions_bot=True, attachments=[{"id": "x"}])) is True
    queued = queue.dequeue()
    assert queued is not None
    attachment = queued.message.metadata["attachments"][0]
    assert attachment["ingestion_status"] == "registered"

    outbox.enqueue(address=ChannelAddress(integration_id="discord-home", provider_key="discord", channel_target="channel-1", alphonse_user_id="u-alex", provider_user_id="123"), message="reply")
    runtime.drain_outbox_once()
    delivered = outbox.list(OutboundSelector(integration_id="discord-home", status="delivered"))
    assert len(delivered) == 1
    assert delivered[0].provider_message_id == "sent-1"


def _users(tmp_path: Path) -> V2UserStore:
    store = V2UserStore(tmp_path / "users.sqlite3")
    store.set_users_root(tmp_path / "profiles")
    store.create_user(user_id="u-alex", display_name="Alex")
    return store


def _runtime(*, queue: InMemoryMessageQueue | None = None, users: V2UserStore, client: FakeDiscordClient | None = None, outbox: SQLiteOutboundStore | None = None, asset_store=None, config: dict | None = None) -> DiscordIntegrationRuntime:
    record = SQLiteIntegrationStore().upsert(integration_id="discord-home", provider_key="discord", display_name="Discord", enabled=True, config=config or {"owner_user_id": "local"}, secrets={"bot_token": "token"})
    return DiscordIntegrationRuntime(record=record, channel=CommunicationChannel(queue or InMemoryMessageQueue()), outbox=outbox or SQLiteOutboundStore(), identity_resolver=V2IdentityResolver((IntegrationIdentity("discord-home", "discord"),), user_store=users), owner_user_id="local", discord_client=client or FakeDiscordClient(), access_request_store=users, asset_store=asset_store)


def _message(*, message_id: str = "1", author_id: str = "123", channel_id: str = "channel-1", guild_id: str = "guild-1", thread_id: str = "", is_private: bool = False, mentions_bot: bool = False, attachments=None) -> dict:
    return {"id": message_id, "content": "hello", "author_id": author_id, "author_name": "Alex", "channel_id": channel_id, "guild_id": guild_id, "thread_id": thread_id, "is_private": is_private, "mentions_bot": mentions_bot, "attachments": attachments or []}
