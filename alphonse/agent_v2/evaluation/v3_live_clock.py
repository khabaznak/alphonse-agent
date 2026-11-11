"""Explicitly enabled V3 wiring for a synthetic clock experiment.

Nothing here creates providers, credentials, or an isolation boundary. The
caller must separately obtain the approved provider/Jev scope and establish
isolation, cost control and model identity. A local switch is not approval.
The active process must have the intended projection configuration; concurrent
arms belong in separate processes. No real household data or live effects.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import math
from typing import Any, Callable

from .candidate_clock_responses import _GOAL
from .prescription_http_processors import LocalSource
from .responses_pilot_transport import ResponsesPilotTransport
from .simplified_loop import Run
from .v3_responses_adapter import OfflineV3ResponsesProvider


class V3PilotResponsesProvider(OfflineV3ResponsesProvider):
    """Adapt actual Responses transport without claiming it is offline.

    Enabling this object does not establish approval or budget enforcement.
    A provider-enforced framed input bound and non-bypassable spend control
    remain independent prerequisites, and usage can remain unknown.
    """
    def __init__(self, transport: ResponsesPilotTransport, *, max_input_tokens: int,
                 max_output_tokens: int, emit: Callable[[dict], None] | None = None,
                 dispatch_enabled: bool = False):
        if not isinstance(transport, ResponsesPilotTransport):
            raise ValueError('pilot_responses_transport_required')
        if any(type(value) is not int or value <= 0 for value in (max_input_tokens, max_output_tokens)):
            raise ValueError('pilot_invalid_inference_limit')
        if type(dispatch_enabled) is not bool:
            raise ValueError('pilot_invalid_dispatch_flag')
        self.transport, self.max_input_tokens = transport, max_input_tokens
        self.max_output_tokens, self.emit, self.dispatch_enabled = max_output_tokens, emit, dispatch_enabled

    def _call(self, request, kind):
        if not self.dispatch_enabled:
            raise ValueError('pilot_v3_dispatch_disabled')
        return super()._call(request, kind)


def run_v3_live_clock(*, base_url: str, provider: V3PilotResponsesProvider,
                      system_one: Any, projection_enabled: bool,
                      emit: Callable[[dict], None], dispatch_enabled: bool = False,
                      timeout_seconds: float = 100, cancellation_checker: Callable[[], bool] | None = None) -> Run:
    """Run the V3 processor on the one-source synthetic clock fixture.

    This does not itself launch a process or grade the final response. The
    caller supplies a separately authorized Jev provider wired through the
    dedicated attempt meter and fixed-endpoint transport; the runner checks the
    local wiring, a fresh ledger and at least one observed attempt. Those checks
    do not establish interpreter integrity, isolation, billing or approval. The
    monotonic deadline is cooperative: it cannot interrupt an in-flight call
    and is not atomic with dispatch. An external process deadline is still
    needed. Output and events can include model text: retain them privately.
    """
    if type(dispatch_enabled) is not bool or not dispatch_enabled:
        raise ValueError('pilot_v3_dispatch_disabled')
    if (not isinstance(provider, V3PilotResponsesProvider) or not provider.dispatch_enabled
            or provider.transport._enabled is not True
            or system_one is None or type(projection_enabled) is not bool or not callable(emit)):
        raise ValueError('pilot_v3_configuration_invalid')
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0
            or (cancellation_checker is not None and not callable(cancellation_checker))):
        raise ValueError('pilot_v3_invalid_timeout')
    deadline = monotonic() + timeout_seconds

    def cancelled():
        try:
            if monotonic() >= deadline:
                return True
            if cancellation_checker is None:
                return False
            state = cancellation_checker()
            return state if type(state) is bool else True
        except Exception:
            return True  # Failure of the checker must never authorize work.
    from alphonse.agent_v2.evaluation.jev_attempt_meter import JevAttemptLedger, MeteredJevTransport
    from alphonse.agent_v2.evaluation.jev_pilot_http import JevPilotHTTPTransport
    from alphonse.agent_v2.system_one import (DEFAULT_SYSTEM_ONE_URL, JevCriterionDecisionProvider,
                                             SystemOneSettings, TypeSafeSystemOneClient)

    # A live binding must go through the per-attempt ledger and the dedicated
    # single-request transport, not a Jev object or transport that merely claims
    # to be metered. These object checks are not a process/network sandbox and
    # do not establish user approval, provider prices or an output-cost cap.
    client = getattr(system_one, 'client', None)
    metered = getattr(client, 'transport', None)
    http = getattr(metered, 'transport', None)
    if (type(system_one) is not JevCriterionDecisionProvider or
            set(vars(system_one)) != {'settings', 'client'} or
            type(system_one.settings) is not SystemOneSettings or
            type(client) is not TypeSafeSystemOneClient or
            set(vars(client)) != {'api_url', 'api_key', 'model', 'timeout_seconds', 'transport'} or
            type(metered) is not MeteredJevTransport or
            set(vars(metered)) != {'transport', 'ledger', 'cancellation_checker'} or
            (metered.cancellation_checker is not None and not callable(metered.cancellation_checker)) or
            type(metered.ledger) is not JevAttemptLedger or
            set(vars(metered.ledger)) != {'path', 'max_attempts', 'max_payload_bytes'} or
            type(http) is not JevPilotHTTPTransport or http.enabled is not True or
            set(vars(http)) != {'enabled', 'max_response_bytes'} or
            getattr(client, 'api_url', None) != DEFAULT_SYSTEM_ONE_URL or
            not getattr(client, 'api_key', None) or not getattr(client, 'model', None) or
            system_one.settings.api_url != client.api_url or
            system_one.settings.api_key != client.api_key or system_one.settings.model != client.model):
        raise ValueError('pilot_v3_metered_jev_required')
    if metered.ledger.snapshot():
        raise ValueError('pilot_v3_fresh_jev_ledger_required')
    previous_jev_cancellation = metered.cancellation_checker

    def cancel_jev_attempt():
        if cancelled():
            return True
        if previous_jev_cancellation is None:
            return False
        try:
            result = previous_jev_cancellation()
            return result if type(result) is bool else True
        except Exception:
            return True
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.inference import InferenceRouter, ModelProfile
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import ToolDefinition
    from .deterministic_adapters import _Action, _RunState, _Scenario, _registry

    # Never mutate the process environment to switch an arm mid-run.
    if enabled() is not projection_enabled:
        raise ValueError('pilot_v3_projection_process_mismatch')
    source = LocalSource(base_url, allowed_source_names=('index',))
    with TemporaryDirectory(prefix='v3-live-clock-fixture-') as directory:
        action = _Action('native.fixture_read', 'project_record_search', {}, role='read_clock', behavior='read')
        state = _RunState(_Scenario((action,), 'Answer from the synthetic clock.', ''),
                          Path(directory), 'hierarchical_v3')
        registry = _registry(state)
        descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
            description='Read the synthetic clock source.',
            argument_schema={'type': 'object', 'properties': {'source': {'const': 'index'}},
                             'required': ['source'], 'additionalProperties': False},
            metadata={'v3_capabilities': ['project_record_search']}, read_only=True)

        def read(args):
            if cancelled():
                raise ValueError('pilot_v3_cancelled_before_read')
            emit({'kind': 'tool', 'tool': 'native.fixture_read'})
            if cancelled():
                raise ValueError('pilot_v3_cancelled_before_read')
            result = source.read(args)
            if cancelled():
                raise ValueError('pilot_v3_cancelled_after_read')
            emit({'kind': 'source_read', 'source': 'index'})
            return result

        registry.register(ToolDefinition(descriptor, read, descriptor.argument_schema,
                                         enabled=True, accepts_context=False))
        router = InferenceRouter(provider=provider, default_profile=ModelProfile(
            provider='pilot', model='configured', profile_id='synthetic-clock'))
        task = TaskState(task_id='pilot-v3-clock', message_id='pilot-v3-clock-message',
                         user='synthetic-evaluation-user', project_id='synthetic-clock', goal=_GOAL,
                         intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
                                  inference=router, system_one=system_one,
                                  cancellation_checker=cancelled)
        # Meter checks the same deadline/cancellation before each client retry
        # and immediately before HTTP. Preserve any independent meter checker.
        # Do not share this meter with another worker while this run is active.
        metered.cancellation_checker = cancel_jev_attempt
        try:
            outcome = HierarchicalCAPDProcessor().process(task, context)
        finally:
            metered.cancellation_checker = previous_jev_cancellation
        # A purported model-backed V3 run that never traversed its Jev meter
        # cannot be mistaken for measured Jev evidence. This does not prove
        # isolation or defend against an altered interpreter/class definition.
        if not metered.ledger.snapshot():
            raise ValueError('pilot_v3_no_jev_attempt_observed')
        if cancelled():
            return Run(status='cancelled', reason='pilot_v3_deadline_or_cancelled')
        final = TaskState.from_dict(outcome.snapshot.metadata.get('task_state') or task.to_dict())
        response = final.metadata.get('prepared_user_response') or {}
        return Run(status=final.status, response=str(response.get('message') or ''),
                   reason=str((final.outcome or {}).get('reason') or '') or None)
