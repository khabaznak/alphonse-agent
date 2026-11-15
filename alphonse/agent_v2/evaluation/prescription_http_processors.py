"""Offline-injected real processors over a common synthetic HTTP source.

Both arms get the same read-only source names and capability URL, not an oracle.
Injectable decisions/providers are restricted to declared offline doubles; that
declaration is not an isolation boundary. The adapter cannot authorize a live
call, prove delivery, verify production permissions, or grade the returned answer.
Use independent `loopback_prescription_pilot.evaluate` after fixture shutdown.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from .simplified_loop import Run

_SOURCES = {'index': '', 'conversation': 'conversation',
            'prescription': 'prescription', 'treatment-status': 'treatment-status',
            'calendar': 'calendar', 'contacts': 'contacts', 'household-note': 'household-note'}
_DEFAULT_ALLOWED = ('index', 'conversation', 'prescription', 'treatment-status')
_ACTOR = 'synthetic-evaluation-user'


def _allowed_sources(value):
    if (not isinstance(value, tuple) or not value
            or any(not isinstance(name, str) or name not in _SOURCES for name in value)
            or len(set(value)) != len(value)):
        raise ValueError('invalid_fixture_sources')
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError('fixture_redirect_denied')


class LocalSource:
    """Read explicitly allowed capability-scoped paths from a numeric loopback URL."""
    def __init__(self, base_url, *, allowed_source_names=_DEFAULT_ALLOWED):
        self._allowed = frozenset(_allowed_sources(allowed_source_names))
        if not isinstance(base_url, str):
            raise ValueError('invalid_fixture_url')
        try:
            parts = urlsplit(base_url)
            port = parts.port
        except (TypeError, ValueError):
            parts = None
            port = None
        if (parts is None or parts.scheme != 'http' or parts.hostname != '127.0.0.1' or not port
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or not parts.path.endswith('/')
                or not re.fullmatch(r'/[A-Za-z0-9_-]{43}/', parts.path)):
            raise ValueError('invalid_fixture_url')
        self._base_url = base_url
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def read(self, args):
        if (not isinstance(args, dict) or set(args) != {'source'}
                or not isinstance(args['source'], str) or args['source'] not in self._allowed):
            raise PermissionError('fixture_source_denied')
        source = args['source']
        transport_failed = False
        try:
            with self._opener.open(self._base_url + _SOURCES[source], timeout=2) as response:
                if response.status != 200 or response.headers.get_content_type() != 'application/json':
                    raise ValueError('invalid_fixture_response')
                raw = response.read(32769)
        except (OSError, ValueError):
            # Raise outside the handler, so neither exception nor implicit
            # exception context retains a bearer URL from the HTTP library.
            transport_failed = True
        if transport_failed:
            raise ValueError('fixture_transport_failed')
        if len(raw) > 32768:
            raise ValueError('fixture_response_too_large')
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            value = None
        if not isinstance(value, dict):
            raise ValueError('invalid_fixture_response')
        return {'source': source, 'content': value, 'candidate_complete': True}


def run_candidate_http(*, base_url, goal, chooser, emit, allowed_source_names=_DEFAULT_ALLOWED):
    """Run actual read-only single-loop candidate with an injected chooser."""
    if getattr(chooser, 'offline', None) is not True:
        raise ValueError('offline_chooser_required')
    allowed_names = _allowed_sources(allowed_source_names)
    # Prompt guidance does not grant access: a configured bridge must match the
    # independently supplied host capability set, and both are checked here.
    if (getattr(chooser, 'fixture_argument', None) == 'source'
            and getattr(chooser, 'allowed_source_names', None) != allowed_names):
        raise ValueError('fixture_source_configuration_mismatch')
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition

    source = LocalSource(base_url, allowed_source_names=allowed_names)
    evidence = {'goal': goal}
    descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
        argument_schema={'type': 'object', 'properties': {'source': {'type': 'string'}},
                         'required': ['source'], 'additionalProperties': False}, read_only=True)
    registry = InMemoryToolRegistry()
    def allowed(actor, tool, args):
        return (actor == _ACTOR and tool == 'native.fixture_read' and isinstance(args, dict)
                and set(args) == {'source'} and isinstance(args['source'], str) and args['source'] in allowed_names)
    def read(args):
        emit({'kind': 'tool', 'tool': 'native.fixture_read', 'arguments': dict(args)})
        value = source.read(args)
        emit({'kind': 'source_read', 'source': args['source']})
        return value
    registry.register(ToolDefinition(descriptor, read))
    task = TaskState(task_id='http-prescription', user=_ACTOR)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
    def record(actor, result):
        payload = result['result']
        ref = 'read:' + payload['source']
        evidence[ref] = json.dumps(payload, ensure_ascii=False)
        return ref
    def verify(actor, text, refs):
        return (actor == _ACTOR and bool(text.strip()) and bool(refs)
                and all(ref in evidence for ref in refs))
    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}
    with TemporaryDirectory(prefix='http-prescription-candidate-') as directory:
        store = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.db'))
        store.create(task.task_id, TaskBudget(BudgetLimits(units(6, 8, 40000, 4096),
                     units(0, 1, 200, 100), 100000, 1000), started_at_ms=0))
        host = CandidateHost(task, context, store, lambda _snapshot: None, chooser,
            lambda actor, ref: actor == _ACTOR and ref in evidence,
            lambda actor, ref: evidence[ref], allowed, record, verify, lambda: 100,
            input_reservation=4096, output_reservation=512, projection_budget=4096)
        state = run_candidate(CandidateState(task.task_id, _ACTOR, 'goal'), host)
    return Run(status=state.status, response=state.response, reason=state.blocker or None)
