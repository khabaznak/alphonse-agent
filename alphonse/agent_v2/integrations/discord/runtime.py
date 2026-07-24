"""Discord Gateway integration runtime for Alphonse v2."""

from __future__ import annotations

import asyncio
import mimetypes
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import request

from alphonse.agent_v2.assets import AttachmentDescriptor
from alphonse.agent_v2.core.io import ChannelAddress, OutboundSelector, SQLiteOutboundStore, V2IdentityResolver
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.integrations.presence import PresenceCapabilities, PresencePhase, PresenceProjector, PresenceState
from alphonse.agent_v2.integrations.store import IntegrationConfigRecord
from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter


@dataclass(frozen=True)
class DiscordRuntimeStats:
    messages_seen: int = 0
    messages_queued: int = 0
    unknown_users: int = 0
    outbox_delivered: int = 0
    outbox_failed: int = 0


class DiscordGatewayClient:
    """Small synchronous facade around ``discord.py``'s Gateway client.

    The facade keeps provider-specific asyncio concerns outside the v2 runtime.
    It is also deliberately duck-typed so integration tests can supply a fake.
    """

    def __init__(self, *, bot_token: str) -> None:
        self.bot_token = str(bot_token or "").strip()
        if not self.bot_token:
            raise ValueError("discord_bot_token_required")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopping = False

    def start(self, on_message: Callable[[dict[str, Any]], None]) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(target=self._run, args=(on_message,), name="alphonse-discord", daemon=True)
        self._thread.start()

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def stop(self) -> None:
        self._stopping = True
        loop, client = self._loop, self._client
        if loop is not None and client is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(client.close(), loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def send_message(self, *, channel_id: str, text: str) -> str:
        return str(self._call(self._send_message(channel_id=channel_id, text=text)) or "")

    def send_asset(self, *, channel_id: str, asset: Any, caption: str = "") -> str:
        return str(self._call(self._send_asset(channel_id=channel_id, asset=asset, caption=caption)) or "")

    def send_typing(self, *, channel_id: str) -> None:
        self._call(self._send_typing(channel_id=channel_id))

    def set_message_reaction(self, *, channel_id: str, message_id: str, emoji: str) -> None:
        self._call(self._set_message_reaction(channel_id=channel_id, message_id=message_id, emoji=emoji))

    def describe_inbound(self, raw: dict[str, Any]) -> list[AttachmentDescriptor]:
        result: list[AttachmentDescriptor] = []
        for item in raw.get("attachments", []) if isinstance(raw.get("attachments"), list) else []:
            if not isinstance(item, dict):
                continue
            result.append(
                AttachmentDescriptor(
                    str(item.get("filename") or "discord-attachment"),
                    str(item.get("content_type") or mimetypes.guess_type(str(item.get("filename") or ""))[0] or "application/octet-stream"),
                    int(item.get("size") or 0),
                    str(item.get("url") or item.get("id") or ""),
                    str(raw.get("content") or ""),
                    "attachment",
                )
            )
        return result

    def download(self, descriptor: AttachmentDescriptor) -> bytes:
        with request.urlopen(descriptor.provider_file_id, timeout=30) as response:
            return bytes(response.read())

    def _run(self, on_message: Callable[[dict[str, Any]], None]) -> None:
        try:
            import discord
        except ImportError:
            return

        gateway = self

        class Client(discord.Client):
            async def on_ready(self) -> None:
                gateway._ready.set()

            async def on_message(self, message: Any) -> None:
                if getattr(message.author, "bot", False):
                    return
                raw = {
                    "id": str(message.id),
                    "content": str(message.content or ""),
                    "channel_id": str(message.channel.id),
                    "guild_id": str(message.guild.id) if message.guild is not None else "",
                    "thread_id": str(message.channel.id) if isinstance(message.channel, discord.Thread) else "",
                    "author_id": str(message.author.id),
                    "author_name": str(getattr(message.author, "display_name", "") or message.author),
                    "is_private": message.guild is None,
                    "mentions_bot": self.user in message.mentions if self.user is not None else False,
                    "reply_to_message_id": str(getattr(getattr(message, "reference", None), "message_id", "") or ""),
                    "attachments": [
                        {"id": str(item.id), "filename": item.filename, "content_type": item.content_type or "", "size": item.size, "url": item.url}
                        for item in message.attachments
                    ],
                }
                on_message(raw)

        async def runner() -> None:
            intents = discord.Intents.none()
            intents.guilds = True
            intents.guild_messages = True
            intents.dm_messages = True
            intents.message_content = True
            self._client = Client(intents=intents)
            await self._client.start(self.bot_token)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(runner())
        except Exception:
            pass
        finally:
            self._ready.clear()
            self._client = None
            self._loop.close()
            self._loop = None

    def _call(self, coroutine: Any) -> Any:
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError("discord_gateway_not_ready")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=30)

    async def _channel(self, channel_id: str) -> Any:
        if self._client is None:
            raise RuntimeError("discord_gateway_not_ready")
        channel = self._client.get_channel(int(channel_id))
        return channel if channel is not None else await self._client.fetch_channel(int(channel_id))

    async def _send_message(self, *, channel_id: str, text: str) -> str:
        message = await (await self._channel(channel_id)).send(str(text))
        return str(message.id)

    async def _send_asset(self, *, channel_id: str, asset: Any, caption: str) -> str:
        import discord
        message = await (await self._channel(channel_id)).send(content=str(caption or "") or None, file=discord.File(str(getattr(asset, "path", "")), filename=str(getattr(asset, "filename", "attachment"))))
        return str(message.id)

    async def _send_typing(self, *, channel_id: str) -> None:
        await (await self._channel(channel_id)).trigger_typing()

    async def _set_message_reaction(self, *, channel_id: str, message_id: str, emoji: str) -> None:
        if emoji:
            await (await self._channel(channel_id)).get_partial_message(int(message_id)).add_reaction(emoji)


class DiscordIntegrationRuntime:
    def __init__(self, *, record: IntegrationConfigRecord, channel: CommunicationChannel, outbox: SQLiteOutboundStore, identity_resolver: V2IdentityResolver, owner_user_id: str = "local", discord_client: Any | None = None, on_message_queued: Callable[[], None] | None = None, on_outbox_delivered: Callable[[Any], None] | None = None, on_outbox_failed: Callable[[Any, str], None] | None = None, presence_projector: PresenceProjector | None = None, inbound_router: ProjectInboundRouter | None = None, access_request_store: Any | None = None, asset_store: Any | None = None) -> None:
        self.record, self.channel, self.outbox, self.identity_resolver = record, channel, outbox, identity_resolver
        self.owner_user_id = str(owner_user_id or "local").strip() or "local"
        self.config = normalize_discord_config(record)
        self.discord_client = discord_client or DiscordGatewayClient(bot_token=str(record.secrets.get("bot_token") or ""))
        self._on_message_queued, self._on_outbox_delivered, self._on_outbox_failed = on_message_queued, on_outbox_delivered, on_outbox_failed
        self.presence_projector, self.inbound_router, self.access_request_store, self.asset_store = presence_projector, inbound_router, access_request_store, asset_store
        self._running = False
        self._drain_thread: threading.Thread | None = None
        self._active_thread_ids: set[str] = set()
        self._stats = DiscordRuntimeStats()
        self.presence_adapter = DiscordPresenceAdapter(discord_client=self.discord_client, enabled=bool(self.config["presence_enabled"]))

    @property
    def integration_id(self) -> str:
        return self.record.integration_id

    @property
    def stats(self) -> DiscordRuntimeStats:
        return self._stats

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.discord_client.start(self.handle_message)
        self._drain_thread = threading.Thread(target=self._drain_loop, name="alphonse-discord-outbox", daemon=True)
        self._drain_thread.start()

    def stop(self) -> None:
        self._running = False
        self.discord_client.stop()
        if self._drain_thread is not None and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=5)

    def handle_message(self, raw: dict[str, Any]) -> bool:
        if not isinstance(raw, dict) or bool(raw.get("author_bot")):
            return False
        self._stats = _stats(self._stats, messages_seen=self._stats.messages_seen + 1)
        content = str(raw.get("content") or "").strip()
        descriptors = self.discord_client.describe_inbound(raw)
        if not content and not descriptors:
            return False
        channel_id, guild_id, thread_id = str(raw.get("channel_id") or "").strip(), str(raw.get("guild_id") or "").strip(), str(raw.get("thread_id") or "").strip()
        provider_user_id = str(raw.get("author_id") or "").strip()
        if not channel_id or not provider_user_id:
            return False
        if guild_id and self.config["allowed_guild_ids"] and guild_id not in self.config["allowed_guild_ids"]:
            return False
        if self.config["allowed_channel_ids"] and channel_id not in self.config["allowed_channel_ids"]:
            return False
        resolved = self.identity_resolver.resolve_inbound_user(integration_id=self.integration_id, provider_key="discord", provider_user_id=provider_user_id)
        if not resolved.resolved:
            self._handle_unknown_user(raw=raw, channel_id=channel_id, guild_id=guild_id, provider_user_id=provider_user_id)
            return False
        is_private = bool(raw.get("is_private")) or not guild_id
        mentioned = bool(raw.get("mentions_bot"))
        if not is_private and not mentioned and thread_id not in self._active_thread_ids:
            return False
        if mentioned and thread_id:
            self._active_thread_ids.add(thread_id)
        asset_ids, attachments = self._ingest_attachments(descriptors, resolved.alphonse_user_id)
        prompt = content or "Please analyze the attached file."
        values = {"prompt": prompt, "user": resolved.alphonse_user_id, "integration_id": self.integration_id, "provider_key": "discord", "provider_user_id": provider_user_id, "channel_target": channel_id, "provider_message_id": str(raw.get("id") or ""), "reply_to_provider_message_id": str(raw.get("reply_to_message_id") or ""), "thread_id": thread_id, "metadata": {"provider_raw_message": dict(raw), "asset_ids": asset_ids, "attachments": attachments, "guild_id": guild_id}, "correlation_id": f"{self.integration_id}:{str(raw.get('id') or '')}"}
        routed = self.inbound_router.ingest(**values) if self.inbound_router is not None else self.channel.queue_message(**values)
        queued = getattr(routed, "queued", routed)
        if queued is not None:
            self._stats = _stats(self._stats, messages_queued=self._stats.messages_queued + 1)
            self._safe(self._on_message_queued)
        return True

    def drain_outbox_once(self, *, limit: int = 20) -> DiscordRuntimeStats:
        if hasattr(self.discord_client, "is_ready") and not bool(self.discord_client.is_ready):
            return self._stats
        for _ in range(max(1, int(limit))):
            outbound = self.outbox.claim_next(OutboundSelector(integration_id=self.integration_id, status="pending"))
            if outbound is None:
                break
            try:
                provider_message_id = self.discord_client.send_message(channel_id=outbound.channel_target, text=outbound.message)
                asset_ids = outbound.metadata.get("asset_ids") if isinstance(outbound.metadata, dict) else []
                if asset_ids and self.asset_store is not None:
                    for asset_id in asset_ids if isinstance(asset_ids, list) else []:
                        asset = self.asset_store.system_get(str(asset_id))
                        if asset is None:
                            raise RuntimeError("attachment_not_found")
                        provider_message_id = self.discord_client.send_asset(channel_id=outbound.channel_target, asset=asset) or provider_message_id
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.outbox.mark_failed(outbound.outbox_message_id, error=error)
                self._stats = _stats(self._stats, outbox_failed=self._stats.outbox_failed + 1)
                self._safe(self._on_outbox_failed, outbound, error)
                continue
            self.outbox.mark_delivered(outbound.outbox_message_id, provider_message_id=str(provider_message_id or ""))
            self._stats = _stats(self._stats, outbox_delivered=self._stats.outbox_delivered + 1)
            self._safe(self._on_outbox_delivered, outbound)
        return self._stats

    def _drain_loop(self) -> None:
        while self._running:
            try:
                self.drain_outbox_once()
                if self.presence_projector is not None:
                    self.presence_projector.heartbeat()
            except Exception:
                pass
            time.sleep(float(self.config["outbox_poll_interval_sec"]))

    def _handle_unknown_user(self, *, raw: dict[str, Any], channel_id: str, guild_id: str, provider_user_id: str) -> None:
        self._stats = _stats(self._stats, unknown_users=self._stats.unknown_users + 1)
        if not guild_id and self.access_request_store is not None:
            try:
                self.access_request_store.record_access_request(integration_id=self.integration_id, provider_key="discord", provider_user_id=provider_user_id, channel_target=channel_id, display_name=str(raw.get("author_name") or ""))
                self.discord_client.send_message(channel_id=channel_id, text="Your request to talk to Alphonse has been sent to the administrator. Please wait for approval.")
            except Exception:
                pass
        self.outbox.enqueue(address=ChannelAddress(integration_id="tui", provider_key="tui", channel_target=self.owner_user_id, alphonse_user_id=self.owner_user_id, provider_user_id=self.owner_user_id), message=f"Discord message ignored because the sender is not mapped to an Alphonse user: provider_user_id={provider_user_id}, channel_id={channel_id}.", kind="identity_resolution", audience_user_id=self.owner_user_id, metadata={"provider_key": "discord", "integration_id": self.integration_id, "provider_user_id": provider_user_id, "channel_target": channel_id, "provider_raw_message": dict(raw)})

    def _ingest_attachments(self, descriptors: list[AttachmentDescriptor], user_id: str) -> tuple[list[str], list[dict[str, Any]]]:
        asset_ids: list[str] = []
        metadata: list[dict[str, Any]] = []
        for descriptor in descriptors:
            item = dict(descriptor.__dict__)
            item["asset_id"], item["ingestion_status"] = "", "not_registered"
            if self.asset_store is not None:
                try:
                    asset = self.asset_store.register_bytes(owner_user_id=user_id, descriptor=descriptor, content=self.discord_client.download(descriptor), source="discord")
                    asset_ids.append(asset.asset_id)
                    item["asset_id"], item["ingestion_status"] = asset.asset_id, "registered"
                except Exception:
                    item["ingestion_status"] = "failed"
            metadata.append(item)
        return asset_ids, metadata

    @staticmethod
    def _safe(callback: Callable[..., None] | None, *args: Any) -> None:
        if callback is not None:
            try:
                callback(*args)
            except Exception:
                pass


class DiscordPresenceAdapter:
    capabilities = PresenceCapabilities(transient_activity=True, reactions=True)
    _REACTIONS = {PresencePhase.ACKNOWLEDGED: "👀", PresencePhase.THINKING: "🤔", PresencePhase.EXECUTING: "⚡", PresencePhase.WAITING_USER: "❓", PresencePhase.DONE: "👍", PresencePhase.FAILED: "👎"}

    def __init__(self, *, discord_client: Any, enabled: bool = True) -> None:
        self.discord_client, self.enabled = discord_client, enabled

    def start(self, presence: PresenceState) -> None: self._project(presence)
    def update(self, presence: PresenceState) -> None: self._project(presence)
    def stop(self, presence: PresenceState) -> None: return
    def heartbeat(self, presence: PresenceState) -> None:
        if self.enabled and presence.phase in {PresencePhase.ACKNOWLEDGED, PresencePhase.THINKING, PresencePhase.EXECUTING}:
            self.discord_client.send_typing(channel_id=presence.address.channel_target)
    def _project(self, presence: PresenceState) -> None:
        self.heartbeat(presence)
        emoji = self._REACTIONS.get(presence.phase)
        if self.enabled and emoji and presence.provider_message_id:
            self.discord_client.set_message_reaction(channel_id=presence.address.channel_target, message_id=presence.provider_message_id, emoji=emoji)


def build_discord_runtime(**kwargs: Any) -> DiscordIntegrationRuntime:
    return DiscordIntegrationRuntime(**kwargs)


def normalize_discord_config(record: IntegrationConfigRecord) -> dict[str, Any]:
    config = dict(record.config or {})
    return {"allowed_guild_ids": _values(config.get("allowed_guild_ids")), "allowed_channel_ids": _values(config.get("allowed_channel_ids")), "owner_user_id": str(config.get("owner_user_id") or "local").strip() or "local", "presence_enabled": _bool(config.get("presence_enabled"), True), "outbox_poll_interval_sec": _positive(config.get("outbox_poll_interval_sec"), 0.5)}


def _values(value: Any) -> set[str]:
    return {str(item).strip() for item in (value if isinstance(value, (list, tuple, set)) else str(value or "").split(",")) if str(item).strip()}
def _bool(value: Any, default: bool) -> bool: return default if value is None else (value if isinstance(value, bool) else str(value).strip().lower() not in {"", "0", "false", "no", "off"})
def _positive(value: Any, default: float) -> float:
    try: result = float(value)
    except (TypeError, ValueError): return default
    return result if result > 0 else default
def _stats(stats: DiscordRuntimeStats, **updates: int) -> DiscordRuntimeStats:
    values = stats.__dict__.copy(); values.update(updates); return DiscordRuntimeStats(**values)
