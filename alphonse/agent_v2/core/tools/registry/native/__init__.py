"""Native tools package for Alphonse agent v2."""

from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash


def build_native_tool_registry() -> InMemoryToolRegistry:
    """Build the default v2-native tool registry."""
    registry = InMemoryToolRegistry()
    registry.register(build_bash_tool_definition())
    return registry


__all__ = [
    "BASH_TOOL_ID",
    "BASH_TOOL_NAME",
    "build_bash_tool_definition",
    "build_native_tool_registry",
    "execute_bash",
]
