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
from alphonse.agent_v2.core.io import build_outbox_delivery_sink
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.integrations import IntegrationRegistry
from alphonse.agent_v2.integrations import SQLiteIntegrationStore
from alphonse.agent_v2.integrations import build_default_integration_registry
from alphonse.agent_v2.integrations.presence import PresenceProjector
from alphonse.agent_v2.integrations.presence import TuiPresenceAdapter
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord
from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
from alphonse.agent_v2.inference_settings import build_inference_router_from_settings


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
    integration_runtimes: list[Any] = field(default_factory=list)
    active_project_id: str = ""
    ui_events: list[CoreUiEvent] = field(default_factory=list)
    activity_events: list[CoreActivityEvent] = field(default_factory=list)


def build_runtime_host(
    *,
    user: str = "local",
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
    messages: Any | None = None,
) -> V2RuntimeHost:
    reset_state()
    queue = messages or InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    visible_state = InMemoryInternalState()
    processor = processor or PDCAIntelligenceProcessor()
    tools = tools or build_native_tool_registry()
    inference_settings_store = inference_settings_store or SQLiteInferenceSettingsStore()
    inference = inference or build_inference_router_from_settings(inference_settings_store.get())
    question_store = question_store or SQLiteQuestionStore()
    project_store = project_store or ProjectStore()
    schedule_store = schedule_store or ScheduledTaskStore()
    outbox = outbox or SQLiteOutboundStore()
    integration_store = integration_store or SQLiteIntegrationStore()
    integration_registry = integration_registry or build_default_integration_registry()
    presence_projector = PresenceProjector()
    presence_projector.register("tui", TuiPresenceAdapter())
    identity_resolver = identity_resolver or build_identity_resolver(integration_store)
    delivery_sink = build_outbox_delivery_sink(outbox=outbox, identity_resolver=identity_resolver)
    ui_events: list[CoreUiEvent] = []
    activity_events: list[CoreActivityEvent] = []

    def _activity_sink(event: CoreActivityEvent) -> None:
        presence_projector.on_activity(event)
        activity_events.append(event)

    core = AlphonseCore(
        intelligence=processor,
        messages=queue,
        tools=tools,
        prompts=NullPromptLoader(),
        state=visible_state,
        memory=NullMemory(),
        inference=inference,
        ui_event_sink=ui_events.append,
        question_store=question_store,
        project_store=project_store,
        schedule_store=schedule_store,
        delivery_sink=delivery_sink,
        activity_sink=_activity_sink,
    )
    return V2RuntimeHost(
        user=str(user or "local").strip() or "local",
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
        ui_events=ui_events,
        activity_events=activity_events,
    )


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


def build_identity_resolver(store: SQLiteIntegrationStore) -> V2IdentityResolver:
    identities = [IntegrationIdentity("tui", "tui")]
    identities.extend(
        IntegrationIdentity(record.integration_id, record.provider_key)
        for record in store.list_enabled()
    )
    return V2IdentityResolver(tuple(identities))


def refresh_runtime_identity_resolver(runtime: V2RuntimeHost) -> None:
    runtime.identity_resolver = build_identity_resolver(runtime.integration_store)
    runtime.core.delivery_sink = build_outbox_delivery_sink(
        outbox=runtime.outbox,
        identity_resolver=runtime.identity_resolver,
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
                outbox=runtime.outbox,
                identity_resolver=runtime.identity_resolver,
                owner_user_id=runtime.user,
                on_message_queued=on_message_queued,
                on_outbox_delivered=on_outbox_delivered,
                on_outbox_failed=on_outbox_failed,
                presence_projector=runtime.presence_projector,
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
