"""Run the real experimental candidate against synthetic follow-through fixtures.

This is an offline development adapter with an injected scripted chooser. The
candidate has no delivery integration: the optional bridge only simulates a
message sink *after* the processor returns. A passing simulated fixture is not
user-visible delivery, autonomous recovery, or production safety evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .decision_followthrough_development import Case, FollowThroughFixture, evaluate


def run_candidate_fixture(case: Case, chooser, *, simulate_delivery: bool = False) -> dict:
    """Execute candidate and grade observable fixture behavior without live calls.

    ``chooser`` is a local scripted double taking projected text and returning
    decisions; this code does not enforce network isolation. Independent expected
    outcomes stay inside the evaluator. The optional bridge models a hypothetical
    delivery component; it does not exist in the candidate runtime.
    """
    if type(simulate_delivery) is not bool:
        raise ValueError('simulate_delivery_must_be_boolean')
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost, CandidateState, run_candidate
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition

    actor = 'synthetic-evaluation-user'
    fixture = FollowThroughFixture(case)
    sources = {'goal': case.goal}
    events = []
    registry = InMemoryToolRegistry()
    descriptor = ToolDescriptor('native.fixture_lookup', 'fixture_lookup', ToolKind.NATIVE,
        argument_schema={'type': 'object', 'properties': {'source': {'type': 'string'}},
                         'required': ['source'], 'additionalProperties': False}, read_only=True)

    def permit(user, tool, args):
        allowed = (user == actor and tool == 'native.fixture_lookup' and isinstance(args, dict)
                   and set(args) == {'source'} and isinstance(args['source'], str)
                   and args['source'] in {'primary', 'backup'})
        if not allowed:
            source = args.get('source') if isinstance(args, dict) else None
            fixture.record_denied_read(source if isinstance(source, str) else '<invalid>')
        return allowed

    def lookup(args):
        # An unavailable logical source is a successful tool invocation carrying
        # a negative observation; it is not a successful task outcome.
        if not permit(actor, 'native.fixture_lookup', args):
            raise PermissionError('fixture_lookup_denied')
        events.append({'kind': 'tool', 'source': args['source']})
        observation = fixture.read(args['source'])
        return {'candidate_complete': True, 'source': args['source'], 'observation': observation}

    registry.register(ToolDefinition(descriptor, lookup))
    task = TaskState(task_id='followthrough-development', user=actor)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
                              cancellation_checker=lambda: case.name == 'cancelled')

    def choose(text, cap):
        decision = chooser(text)
        if not isinstance(decision, dict):
            return decision
        decision = dict(decision)
        output_units = len(json.dumps(decision, ensure_ascii=False).encode())
        if output_units > cap:
            raise ValueError('fixture_output_exceeds_cap')
        events.append({'kind': 'system_two'})
        decision['usage'] = {'input_tokens': len(text.encode()), 'output_tokens': output_units}
        return decision

    def record(user, result):
        payload = result['result']
        ref = 'read:' + payload['source']
        sources[ref] = json.dumps(payload, ensure_ascii=False)
        return ref

    def verify(user, text, refs):
        # This deliberately verifies only that observation references exist;
        # correctness is judged separately by the fixture oracle.
        return user == actor and bool(refs) and all(ref in sources for ref in refs) and bool(text.strip())

    def units(actions=0, calls=0, inp=0, out=0):
        return {'actions': actions, 'model_calls': calls, 'input_tokens': inp, 'output_tokens': out}

    with TemporaryDirectory(prefix='followthrough-candidate-') as directory:
        store = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.db'))
        store.create(task.task_id, TaskBudget(BudgetLimits(units(4, 6, 30000, 3072),
                     units(0, 1, 200, 100), 100000, 1000), started_at_ms=0))
        checkpoint_path = Path(directory) / 'checkpoint.json'
        host = CandidateHost(task, context, store,
            lambda snapshot: checkpoint_path.write_text(json.dumps(snapshot)), choose,
            lambda user, ref: user == actor and ref in sources,
            lambda user, ref: sources[ref], permit, record, verify, lambda: 100,
            input_reservation=4096, output_reservation=512, projection_budget=4096)
        state = run_candidate(CandidateState(task.task_id, actor, 'goal'), host)

    bridge = 'not_requested'
    if simulate_delivery and state.status in {'completed', 'waiting_user'}:
        try:
            message = json.loads(state.response)
        except (ValueError, TypeError):
            bridge = 'invalid_structured_response'
        else:
            bridge = fixture.send('actor_private', message)
    return {'processor_status': state.status, 'processor_reason': state.blocker or None,
            'processor_response': state.response, 'events': events,
            'simulated_bridge': bridge, 'fixture_audit': fixture.audit(),
            'fixture_outcome': evaluate(case, fixture,
                processor_status=state.status, processor_reason=state.blocker or None),
            'product_delivery': 'not_integrated', 'decision_qualified': False,
            'real_time_deadline': 'unassessed',
            'limitations': ['Scripted chooser; no autonomous model.',
                            'Fixture delivery is synthetic and outside the processor.',
                            'Synthetic fixture time is static; real latency is unmeasured.',
                            'No production authorization, restart or notification guarantee.']}
