"""Daemon-owned v2 runtime construction and lifecycle primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.core import IntelligenceProcessor
from alphonse.agent_v2.core.core import MemoryRecord
from alphonse.agent_v2.core.core import PromptFile
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolRegistry
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.intelligence import PDCAIntelligenceProcessor
from alphonse.agent_v2.core.io import IntegrationIdentity
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.io import V2IdentityResolver
from alphonse.agent_v2.core.io import CommunicationRouter
from alphonse.agent_v2.core.io import SQLiteCommunicationThreadStore
from alphonse.agent_v2.core.io import build_outbox_delivery_sink
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.programs import ProgramRunner
from alphonse.agent_v2.integrations import IntegrationRegistry
from alphonse.agent_v2.integrations import SQLiteIntegrationStore
from alphonse.agent_v2.integrations import build_default_integration_registry
from alphonse.agent_v2.integrations.presence import PresenceProjector
from alphonse.agent_v2.integrations.presence import TuiPresenceAdapter
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord
from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
from alphonse.agent_v2.inference_settings import build_inference_router_from_settings
from alphonse.agent_v2.agent_config import AgentConfigPromptLoader
from alphonse.agent_v2.agent_config import AgentConfigStore
from alphonse.agent_v2.agent_config import packaged_agent_config_dir
from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore
from alphonse.agent_v2.code_mode_settings import SQLiteCodeModeSettingsStore
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
from alphonse.agent_v2.assets import SQLiteAssetStore
from alphonse.agent_v2.artifacts import SQLiteArtifactStore
from alphonse.agent_v2.artifacts import build_artifact_tool_definitions
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore
from alphonse.agent_v2.core.memory import LedgerMemory
from alphonse.agent_v2.conversations import SQLiteConversationStore


@dataclass
class InMemoryInternalState:
    value: StateSnapshot = field(default_factory=StateSnapshot)

    def update(self, snapshot: StateSnapshot) -> None:
        self.value = snapshot

    def snapshot(self) -> StateSnapshot:
        return self.value


class NullPromptLoader:
    def load(self, name: str) -> PromptFile:
        return PromptFile(name=name, content="")


class NullMemory:
    def write(self, record: MemoryRecord) -> None:
        _ = record

    def read(self, path: str) -> MemoryRecord | None:
        _ = path
        return None


@dataclass
class V2RuntimeHost:
    """Long-lived runtime shared by daemon and interface clients."""

    user: str
    queue: Any
    channel: CommunicationChannel
    visible_state: InMemoryInternalState
    processor: IntelligenceProcessor
    core: AlphonseCore
    question_store: SQLiteQuestionStore
    project_store: ProjectStore
    schedule_store: ScheduledTaskStore
    outbox: SQLiteOutboundStore
    identity_resolver: V2IdentityResolver
    integration_store: SQLiteIntegrationStore
    integration_registry: IntegrationRegistry
    presence_projector: PresenceProjector
    inference_settings_store: SQLiteInferenceSettingsStore
    agent_config_store: AgentConfigStore
    project_session_store: SQLiteProjectSessionStore
    inbound_router: ProjectInboundRouter
    user_store: V2UserStore
    web_tools_settings_store: SQLiteWebToolsSettingsStore
    code_mode_settings_store: SQLiteCodeModeSettingsStore
    media_tools_settings_store: SQLiteMediaToolsSettingsStore
    asset_store: SQLiteAssetStore
    artifact_store: SQLiteArtifactStore
    memory_settings_store: SQLiteMemorySettingsStore
    communication_router: CommunicationRouter
    conversation_store: SQLiteConversationStore
    integration_runtimes: list[Any] = field(default_factory=list)
    active_project_id: str = ""
    ui_events: list[CoreUiEvent] = field(default_factory=list)
    activity_events: list[CoreActivityEvent] = field(default_factory=list)


def build_runtime_host(
    *,
    user: str = "local",
    user_store: V2UserStore | None = None,
    inference: InferenceRouter | None = None,
    tools: ToolRegistry | None = None,
    processor: IntelligenceProcessor | None = None,
    question_store: SQLiteQuestionStore | None = None,
    project_store: ProjectStore | None = None,
    schedule_store: ScheduledTaskStore | None = None,
    outbox: SQLiteOutboundStore | None = None,
    identity_resolver: V2IdentityResolver | None = None,
    integration_store: SQLiteIntegrationStore | None = None,
    integration_registry: IntegrationRegistry | None = None,
    inference_settings_store: SQLiteInferenceSettingsStore | None = None,
    agent_config_store: AgentConfigStore | None = None,
    project_session_store: SQLiteProjectSessionStore | None = None,
    messages: Any | None = None,
    web_tools_settings_store: SQLiteWebToolsSettingsStore | None = None,
    code_mode_settings_store: SQLiteCodeModeSettingsStore | None = None,
    media_tools_settings_store: SQLiteMediaToolsSettingsStore | None = None,
    asset_store: SQLiteAssetStore | None = None,
    artifact_store: SQLiteArtifactStore | None = None,
    memory_settings_store: SQLiteMemorySettingsStore | None = None,
    communication_thread_store: SQLiteCommunicationThreadStore | None = None,
    conversation_store: SQLiteConversationStore | None = None,
) -> V2RuntimeHost:
    reset_state()
    provided_tools = tools is not None
    queue = messages or InMemoryMessageQueue()
    conversation_store = conversation_store or SQLiteConversationStore()
    channel = CommunicationChannel(queue, conversation_store=conversation_store)
    visible_state = InMemoryInternalState()
    processor = processor or PDCAIntelligenceProcessor()
    web_tools_settings_store = web_tools_settings_store or SQLiteWebToolsSettingsStore()
    code_mode_settings_store = code_mode_settings_store or SQLiteCodeModeSettingsStore()
    media_tools_settings_store = media_tools_settings_store or SQLiteMediaToolsSettingsStore()
    user_store = user_store or V2UserStore()
    asset_store = asset_store or SQLiteAssetStore(users_root=user_store.users_root)
    artifact_store = artifact_store or SQLiteArtifactStore()
    # Generic embedded/test hosts are intentionally ephemeral. The daemon
    # injects the durable store explicitly.
    memory_settings_store = memory_settings_store or SQLiteMemorySettingsStore()
    tools = tools or build_native_tool_registry(web_tools_settings_store.get(), asset_store, media_tools_settings_store.get(), artifact_store)
    inference_settings_store = inference_settings_store or SQLiteInferenceSettingsStore()
    # Persistent daemon/TUI constructors pass `AgentConfigStore.default()`.
    # Generic test and helper runtimes only need the package defaults.
    agent_config_store = agent_config_store or AgentConfigStore(packaged_agent_config_dir())
    project_session_store = project_session_store or SQLiteProjectSessionStore()
    inference = inference or build_inference_router_from_settings(inference_settings_store.get())
    question_store = question_store or SQLiteQuestionStore()
    project_store = project_store or ProjectStore()
    if tools is not None:
        for definition in build_artifact_tool_definitions(artifact_store, project_store):
            tools.register(definition)
    schedule_store = schedule_store or ScheduledTaskStore()
    outbox = outbox or SQLiteOutboundStore()
    integration_store = integration_store or SQLiteIntegrationStore()
    communication_thread_store = communication_thread_store or SQLiteCommunicationThreadStore()
    def _compact_memory(source: str) -> str:
        from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
        result = inference.generate_markdown(InferenceRequest(
            prompt="Summarize this conversation ledger for the next ledger. Preserve active context, decisions, commitments, and unresolved work. Return concise Markdown only.\n\n" + source,
            purpose=InferencePurpose.MEMORY_COMPACTION,
        ))
        return str(result.content or "")
    memory = LedgerMemory(users_root=user_store.users_root, settings_store=memory_settings_store, summarizer=_compact_memory)
    integration_registry = integration_registry or build_default_integration_registry()
    presence_projector = PresenceProjector()
    presence_projector.register("tui", TuiPresenceAdapter())
    identity_resolver = identity_resolver or build_identity_resolver(integration_store, user_store=user_store)
    communication_router = CommunicationRouter(users=user_store, resolver=identity_resolver, outbox=outbox, threads=communication_thread_store)
    delivery_sink = build_outbox_delivery_sink(outbox=outbox, identity_resolver=identity_resolver, communication_router=communication_router)
    inbound_router = ProjectInboundRouter(
        channel=channel,
        outbox=outbox,
        projects=project_store,
        sessions=project_session_store,
        is_admin=user_store.is_admin,
        managed_root=user_store.managed_project_root,
        communication_router=communication_router,
    )
    ui_events: list[CoreUiEvent] = []
    activity_events: list[CoreActivityEvent] = []

    def _activity_sink(event: CoreActivityEvent) -> None:
        presence_projector.on_activity(event)
        activity_events.append(event)

    core = AlphonseCore(
        intelligence=processor,
        messages=queue,
        tools=tools,
        prompts=AgentConfigPromptLoader.from_store(agent_config_store),
        state=visible_state,
        memory=memory,
        inference=inference,
        ui_event_sink=ui_events.append,
        question_store=question_store,
        project_store=project_store,
        schedule_store=schedule_store,
        delivery_sink=delivery_sink,
        user_context_provider=user_store.read_user_context,
        user_timezone_provider=lambda _user_id: user_store.timezone(),
        program_runner=ProgramRunner(settings_provider=code_mode_settings_store.get),
        activity_sink=_activity_sink,
    )
    runtime = V2RuntimeHost(
        user=str(user or (user_store.admin_user().user_id if user_store.admin_user() else "local")).strip() or "local",
        queue=queue,
        channel=channel,
        visible_state=visible_state,
        processor=processor,
        core=core,
        question_store=question_store,
        project_store=project_store,
        schedule_store=schedule_store,
        outbox=outbox,
        identity_resolver=identity_resolver,
        integration_store=integration_store,
        integration_registry=integration_registry,
        presence_projector=presence_projector,
        inference_settings_store=inference_settings_store,
        agent_config_store=agent_config_store,
        project_session_store=project_session_store,
        inbound_router=inbound_router,
        user_store=user_store,
        web_tools_settings_store=web_tools_settings_store,
        code_mode_settings_store=code_mode_settings_store,
        media_tools_settings_store=media_tools_settings_store,
        asset_store=asset_store,
        artifact_store=artifact_store,
        memory_settings_store=memory_settings_store,
        communication_router=communication_router,
        conversation_store=conversation_store,
        ui_events=ui_events,
        activity_events=activity_events,
    )
    if not provided_tools:
        refresh_runtime_artifacts(runtime)
    return runtime


def refresh_runtime_web_tools(runtime: V2RuntimeHost) -> None:
    """Apply Web Tools settings to tasks started after this call."""
    refresh_runtime_artifacts(runtime)


def refresh_runtime_media_tools(runtime: V2RuntimeHost) -> None:
    """Apply verified local media-tool settings to tasks started after this call."""
    refresh_runtime_artifacts(runtime)


def refresh_runtime_artifacts(runtime: V2RuntimeHost) -> None:
    """Rebuild native and enabled artifact definitions for later PDCA planning."""
    registry = build_native_tool_registry(runtime.web_tools_settings_store.get(), runtime.asset_store, runtime.media_tools_settings_store.get(), runtime.artifact_store, lambda: refresh_runtime_artifacts(runtime))
    for definition in build_artifact_tool_definitions(runtime.artifact_store, runtime.project_store):
        registry.register(definition)
    runtime.core.tools = registry


def build_default_runtime_inference_router() -> InferenceRouter:
    return build_inference_router_from_settings(SQLiteInferenceSettingsStore().get())


def refresh_runtime_inference(runtime: V2RuntimeHost, settings: InferenceSettingsRecord | None = None) -> InferenceSettingsRecord:
    """Apply saved settings to subsequently started core tasks.

    `AlphonseCore.step` captures its router in its loop context, so replacing the
    router here cannot alter a task already being processed.
    """
    selected = settings or runtime.inference_settings_store.get()
    runtime.core.inference = build_inference_router_from_settings(selected)
    return selected


def build_identity_resolver(store: SQLiteIntegrationStore, *, user_store: V2UserStore | None = None) -> V2IdentityResolver:
    # Native clients use separate integration ids so their durable outbox
    # deliveries cannot be consumed by another interface. Both are local
    # first-party clients and therefore share the lightweight ``tui`` provider
    # identity semantics.
    identities = [IntegrationIdentity("tui", "tui"), IntegrationIdentity("desktop", "tui")]
    identities.extend(
        IntegrationIdentity(record.integration_id, record.provider_key)
        for record in store.list_enabled()
    )
    return V2IdentityResolver(tuple(identities), user_store=user_store)


def refresh_runtime_identity_resolver(runtime: V2RuntimeHost) -> None:
    runtime.identity_resolver = build_identity_resolver(runtime.integration_store, user_store=runtime.user_store)
    runtime.communication_router.resolver = runtime.identity_resolver
    runtime.core.delivery_sink = build_outbox_delivery_sink(
        outbox=runtime.outbox,
        identity_resolver=runtime.identity_resolver,
        communication_router=runtime.communication_router,
    )


def start_runtime_integrations(
    runtime: V2RuntimeHost,
    *,
    on_message_queued: Any | None = None,
    on_outbox_delivered: Any | None = None,
    on_outbox_failed: Any | None = None,
) -> list[Any]:
    stop_runtime_integrations(runtime)
    refresh_runtime_identity_resolver(runtime)
    started: list[Any] = []
    for record in runtime.integration_store.list_enabled():
        descriptor = runtime.integration_registry.get(record.provider_key)
        if descriptor is None or descriptor.runtime_factory is None:
            continue
        try:
            integration_runtime = descriptor.runtime_factory(
                record=record,
                channel=runtime.channel,
                inbound_router=runtime.inbound_router,
                outbox=runtime.outbox,
                identity_resolver=runtime.identity_resolver,
                owner_user_id=runtime.user,
                on_message_queued=on_message_queued,
                on_outbox_delivered=on_outbox_delivered,
                on_outbox_failed=on_outbox_failed,
                presence_projector=runtime.presence_projector,
                access_request_store=runtime.user_store,
                asset_store=runtime.asset_store,
                stt_settings_provider=lambda: runtime.media_tools_settings_store.get().stt,
            )
            integration_runtime.start()
        except Exception:
            continue
        presence_adapter = getattr(integration_runtime, "presence_adapter", None)
        if presence_adapter is not None:
            runtime.presence_projector.register(record.integration_id, presence_adapter)
        started.append(integration_runtime)
    runtime.integration_runtimes = started
    return started


def stop_runtime_integrations(runtime: V2RuntimeHost) -> None:
    for integration_runtime in list(runtime.integration_runtimes):
        integration_id = str(getattr(integration_runtime, "integration_id", "") or "").strip()
        if integration_id:
            runtime.presence_projector.unregister(integration_id)
        stop = getattr(integration_runtime, "stop", None)
        if callable(stop):
            stop()
    runtime.integration_runtimes = []
