"""Disposable Bash Dojo: actual simple-loop candidate, not the owner daemon."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.simple_loop.router_candidate_chooser import RouterCandidateChooser
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateHost, CandidateState, run_candidate
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.evaluation.dojo_bash import DojoBash
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings

GOAL = 'Find the numbers in /workspace/notes/numbers.txt and add them. Reply with only the total.'
FIXTURE = (3, 5, 8, 13)


def vector(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


def run(*, image_id, socket_path, router, emit):
    """Run one attempt; emit records immediately, no automatic retries."""
    started = time.monotonic()
    events, sources = [], {'goal': GOAL}
    with TemporaryDirectory(prefix='dojo-budget-') as directory:
        budgets = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.sqlite'))
        now = int(time.monotonic() * 1000)
        budgets.create('dojo', TaskBudget(BudgetLimits(vector(5, 8, 524288, 131072),
            vector(0, 1, 65536, 16384), 120000, 1000), started_at_ms=now))
        actor = 'synthetic-owner'
        task = TaskState(task_id='dojo', user=actor)
        shell_def = build_bash_tool_definition()
        schema = {'type': 'object', 'properties': {'command': {'type': 'string'}},
                  'required': ['command'], 'additionalProperties': False}
        descriptor = replace(shell_def.descriptor, argument_schema=schema,
            description='Execute a Bash command in the disposable environment. The working directory is /workspace. '
                        'Commands may read and write its temporary files; there is no network.')
        registry = InMemoryToolRegistry()
        def forbidden(*_args, **_kwargs):
            raise RuntimeError('dojo_direct_registry_dispatch_forbidden')
        registry.register(ToolDefinition(descriptor, forbidden, argument_schema=schema))
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry)
        chooser = RouterCandidateChooser(router=router, task=task, tools=(descriptor,),
            max_input_tokens=65536, allow_disposable_bash=True)
        def choose(projection, limit):
            decision = chooser(projection, limit)
            emit({'event': 'model_decision', 'kind': decision.get('kind'),
                  'tool_id': decision.get('tool_id'), 'usage': decision.get('usage')})
            return decision
        def authorize(who, tool, args):
            return (who == actor and tool == 'native.bash' and isinstance(args, dict)
                    and set(args) == {'command'} and isinstance(args['command'], str)
                    and 0 < len(args['command'].encode()) <= 16384)
        def record(_actor, result):
            ref = 'observation:' + str(len(sources))
            sources[ref] = json.dumps(result['result'])
            return ref
        def checkpoint(snapshot):
            emit({'event': 'checkpoint', 'state': snapshot})
        with DojoBash(image_id=image_id, docker_socket=socket_path) as shell:
            # The fixture setup is evaluator work, never an agent tool call.
            contents = '\\n'.join(str(n) for n in FIXTURE) + '\\n'
            setup = shell.execute("mkdir -p /workspace/notes && printf '" + contents + "' > /workspace/notes/numbers.txt")
            if setup['status'] != 'completed':
                raise RuntimeError('dojo_fixture_setup_failed')
            def execute(_actor, tool, args):
                emit({'event': 'tool_started', 'tool': tool, 'arguments': args})
                result = shell.execute(args['command'], cancelled=context.is_cancelled)
                event = {'event': 'tool_finished', 'tool': tool, **result}
                events.append(event)
                emit(event)
                if result['status'] not in {'completed', 'failed'}:
                    raise RuntimeError('dojo_effect_unconfirmed')
                # A nonzero command is a completed observation, not task success.
                return {'status': 'success', 'result': {'candidate_complete': True, **result}}
            host = CandidateHost(task, context, budgets, checkpoint, choose,
                lambda who, ref: who == actor and ref in sources, lambda who, ref: sources[ref],
                lambda *_: False, record, lambda who, text, refs: who == actor and bool(text),
                lambda: int(time.monotonic() * 1000), input_reservation=65536,
                output_reservation=16384, projection_budget=65536,
                authorize_disposable_tool=authorize, execute_disposable_tool=execute)
            result = run_candidate(CandidateState('dojo', actor, 'goal'), host)
        expected = str(sum(FIXTURE))
        correct = result.status == 'completed' and result.response.strip() == expected
        observed = any(event.get('status') == 'completed' for event in events)
        unassessed = result.blocker in {'candidate_model_usage_unknown_or_overrun', 'candidate_model_usage_overrun', 'candidate_model_unavailable',
                                       'candidate_disposable_effect_unconfirmed'}
        outcome = ('infrastructure_error' if not shell.cleanup_verified else
                   'cancelled' if result.status == 'cancelled' else
                   'timeout' if result.blocker == 'candidate_deadline_exceeded' else
                   'unassessed' if unassessed else 'passed' if correct else 'failed')
        return {'status': result.status, 'blocker': result.blocker, 'answer': result.response,
                'outcome': outcome, 'passed': correct if outcome in {'passed', 'failed'} else None,
                'task_answer_correct': correct if result.status == 'completed' else None,
                'successful_bash_observed': observed, 'grounding': 'unassessed',
                'shell_calls': len(events), 'cleanup_verified': shell.cleanup_verified,
                'elapsed_seconds': round(time.monotonic() - started, 3),
                'pillars': {'resourcefulness_improvisation': 'shell_attempt_observed' if events else 'unassessed',
                            'resilience': 'unassessed', 'accountability': 'partial_action_record',
                            'human_understanding': 'one_request_only', 'trustworthiness': 'unassessed'},
                'limitations': ['Isolated candidate path with Bash only, not the owner daemon or Desktop.',
                                'Provider-side tool activity is not captured.',
                                'Correct answer and shell use do not prove grounding.',
                                'Container is not a complete security boundary.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    args = parser.parse_args()
    router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
    router.provider.config = replace(router.provider.config, timeout_seconds=40)
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*.py')):
        digest.update(str(path.relative_to(source)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(record):
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': 'synthetic-number-total', 'model': args.model,
              'image_id': args.image_id, 'source_tree_sha256': digest.hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'disposable candidate and Bash container; not owner daemon or installed Desktop'})
        fd = os.open(args.journal.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            result = run(image_id=args.image_id, socket_path=args.docker_socket, router=router, emit=emit)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'passed': None,
                  'error_type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
