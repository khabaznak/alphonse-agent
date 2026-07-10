"""Channel identity and outbound I/O helpers for Alphonse v2."""

from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.channels import channel_address_from_metadata
from alphonse.agent_v2.core.io.channels import channel_metadata
from alphonse.agent_v2.core.io.identity import IntegrationIdentity
from alphonse.agent_v2.core.io.identity import V2IdentityResolver
from alphonse.agent_v2.core.io.identity import resolve_provider_user_mapping
from alphonse.agent_v2.core.io.identity import upsert_provider_user_mapping
from alphonse.agent_v2.core.io.outbox import OutboundMessage
from alphonse.agent_v2.core.io.outbox import OutboundSelector
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.io.outbox import build_outbox_delivery_sink
from alphonse.agent_v2.core.io.outbox import project_snapshot_to_outbox

__all__ = [
    "ChannelAddress",
    "IntegrationIdentity",
    "OutboundMessage",
    "OutboundSelector",
    "SQLiteOutboundStore",
    "V2IdentityResolver",
    "build_outbox_delivery_sink",
    "channel_address_from_metadata",
    "channel_metadata",
    "project_snapshot_to_outbox",
    "resolve_provider_user_mapping",
    "upsert_provider_user_mapping",
]
