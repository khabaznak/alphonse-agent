"""Native tools package for Alphonse agent v2."""

from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.ask_question import ASK_QUESTION_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.ask_question import ASK_QUESTION_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
from alphonse.agent_v2.core.tools.registry.native.ask_question import execute_ask_question
from alphonse.agent_v2.core.tools.registry.native.artifact_registration import build_artifact_registration_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
from alphonse.agent_v2.core.tools.registry.native.deliver_message import DELIVER_MESSAGE_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.deliver_message import DELIVER_MESSAGE_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.deliver_message import build_deliver_message_tool_definition
from alphonse.agent_v2.core.tools.registry.native.deliver_message import execute_deliver_message
from alphonse.agent_v2.core.tools.registry.native.send_attachment import build_send_attachment_tool_definition
from alphonse.agent_v2.core.tools.registry.native.media import ANALYZE_IMAGE_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.media import build_analyze_image_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import RESPOND_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.respond import RESPOND_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import execute_respond
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import SCHEDULED_TASK_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import SCHEDULED_TASK_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import build_scheduled_task_tool_definition
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import execute_scheduled_task
from alphonse.agent_v2.core.tools.registry.native.web import build_web_fetch_tool_definition
from alphonse.agent_v2.core.tools.registry.native.web import build_web_search_tool_definition
from alphonse.agent_v2.web_tools_settings import WebToolsSettings
from alphonse.agent_v2.media_tools_settings import MediaToolsSettings


def build_native_tool_registry(web_tools_settings: WebToolsSettings | None = None, asset_store: object | None = None, media_tools_settings: MediaToolsSettings | None = None, artifact_store: object | None = None, on_artifact_changed: object | None = None) -> InMemoryToolRegistry:
    """Build the default v2-native tool registry."""
    registry = InMemoryToolRegistry()
    registry.register(build_respond_tool_definition())
    registry.register(build_bash_tool_definition())
    registry.register(build_deliver_message_tool_definition())
    registry.register(build_send_attachment_tool_definition(asset_store))
    registry.register(build_ask_question_tool_definition())
    registry.register(build_scheduled_task_tool_definition())
    if artifact_store is not None:
        registry.register(build_artifact_registration_tool_definition(artifact_store, on_artifact_changed if callable(on_artifact_changed) else None))
    media = media_tools_settings or MediaToolsSettings()
    registry.register(build_analyze_image_tool_definition(media.ocr, asset_store))
    settings = web_tools_settings or WebToolsSettings()
    registry.register(build_web_search_tool_definition(settings))
    registry.register(build_web_fetch_tool_definition(settings))
    return registry


__all__ = [
    "ASK_QUESTION_TOOL_ID",
    "ANALYZE_IMAGE_TOOL_ID",
    "ASK_QUESTION_TOOL_NAME",
    "BASH_TOOL_ID",
    "BASH_TOOL_NAME",
    "DELIVER_MESSAGE_TOOL_ID",
    "DELIVER_MESSAGE_TOOL_NAME",
    "RESPOND_TOOL_ID",
    "RESPOND_TOOL_NAME",
    "SCHEDULED_TASK_TOOL_ID",
    "SCHEDULED_TASK_TOOL_NAME",
    "build_ask_question_tool_definition",
    "build_bash_tool_definition",
    "build_deliver_message_tool_definition",
    "build_native_tool_registry",
    "build_respond_tool_definition",
    "build_scheduled_task_tool_definition",
    "execute_ask_question",
    "execute_bash",
    "execute_deliver_message",
    "execute_respond",
    "execute_scheduled_task",
]
