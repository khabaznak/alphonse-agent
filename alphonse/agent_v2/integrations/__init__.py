"""Optional v2 integrations.

Integrations live outside the core CAPD loop. They translate provider payloads
to canonical queue messages and consume outbound messages addressed to their
integration id.
"""

from alphonse.agent_v2.integrations.registry import IntegrationDescriptor
from alphonse.agent_v2.integrations.registry import IntegrationRegistry
from alphonse.agent_v2.integrations.registry import build_default_integration_registry
from alphonse.agent_v2.integrations.store import IntegrationConfigRecord
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore

__all__ = [
    "IntegrationConfigRecord",
    "IntegrationDescriptor",
    "IntegrationRegistry",
    "SQLiteIntegrationStore",
    "build_default_integration_registry",
]
