"""Native tools package for Alphonse agent v2."""

from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
from alphonse.agent_v2.core.tools.registry.native.respond import RESPOND_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.respond import RESPOND_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import execute_respond


def build_native_tool_registry() -> InMemoryToolRegistry:
    """Build the default v2-native tool registry."""
    registry = InMemoryToolRegistry()
    registry.register(build_respond_tool_definition())
    registry.register(build_bash_tool_definition())
    return registry


__all__ = [
    "BASH_TOOL_ID",
    "BASH_TOOL_NAME",
    "RESPOND_TOOL_ID",
    "RESPOND_TOOL_NAME",
    "build_bash_tool_definition",
    "build_native_tool_registry",
    "build_respond_tool_definition",
    "execute_bash",
    "execute_respond",
]
