"""Disposable, independently checked household shopping-list Dojo challenge.

This uses the simple-loop candidate in a Bash-only disposable host, not Desktop
or the owner daemon. It requires a trusted local Docker runtime and image.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
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
from alphonse.agent_v2.evaluation.dojo_bash_candidate import vector
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings

GOAL = ('Please prepare a grocery list from the pantry and meal-plan files in '
        '/workspace/household. Write /workspace/shopping-list.json as a JSON object '
        'mapping item names to integer amounts we still need, omitting items already '
        'covered by the pantry. Keep the original files and tell me when finished.')
PANTRY = {'rice': 1, 'beans': 2, 'tomatoes': 1}
PLAN = {'rice': 3, 'beans': 2, 'tomatoes': 4}
FILES = {'pantry.json': PANTRY, 'meal-plan.json': PLAN}


def fixture_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def grade(readback):
    """Score final observed state against separately computed fixture facts."""
    if not isinstance(readback, dict) or readback.get('status') != 'observed':
        return {'effect_passed': None, 'reason': 'readback_unavailable'}
    expected = {name: amount - PANTRY.get(name, 0) for name, amount in PLAN.items()
                if amount > PANTRY.get(name, 0)}
    output = readback.get('output')
    output_valid = (isinstance(output, dict) and all(isinstance(k, str) and type(v) is int
                    for k, v in output.items()))
    correct = output_valid and output == expected
    expected_hashes = {name: hashlib.sha256(fixture_bytes(value)).hexdigest() for name, value in FILES.items()}
    checks = {'shopping_list_correct': bool(correct),
              'source_files_unchanged': readback.get('source_hashes') == expected_hashes}
    alternatives = readback.get('other_json_files', {})
    matches_elsewhere = (sorted(path for path, value in alternatives.items()
                                if isinstance(path, str) and isinstance(value, dict)
                                and all(type(v) is int for v in value.values()) and value == expected)
                         if isinstance(alternatives, dict) else [])
    return {'effect_passed': all(checks.values()), 'checks': checks,
            'requested_target_status': readback.get('target_status', 'unknown'),
            'matching_alternative_paths': matches_elsewhere,
            'alternative_scan_complete': readback.get('alternative_scan_complete', False),
            'response_semantics': 'unassessed', 'expected_item_count': len(expected)}


def _seed(shell):
    # Fixed disposable synthetic input; this setup is not an agent action.
    script = ('import json,pathlib; p=pathlib.Path("/workspace/household"); p.mkdir(); '
              + '; '.join('p.joinpath(%r).write_text(%r)' % (name, fixture_bytes(value).decode())
                          for name, value in FILES.items()))
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(script))
    if result['status'] != 'completed':
        raise RuntimeError('dojo_fixture_setup_failed')


def _readback(shell):
    # Run independently after the agent stops. Absolute interpreter and isolated
    # mode avoid imports from the mutable workspace. Reject symlinked outputs.
    script = '''import hashlib,json,os,pathlib,stat
hashes={}
try:
 for name in ('pantry.json','meal-plan.json'):
  f=pathlib.Path('/workspace/household')/name
  s=f.lstat()
  if not stat.S_ISREG(s.st_mode) or s.st_size > 16384: raise ValueError()
  hashes[name]=hashlib.sha256(f.read_bytes()).hexdigest()
except (OSError,ValueError):
 hashes=None
def object_pairs(pairs):
 obj={}
 for k,v in pairs:
  if k in obj: raise ValueError()
  obj[k]=v
 return obj
def load_json(p):
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_size > 16384: raise ValueError()
 return json.loads(p.read_text(),object_pairs_hook=object_pairs)
p=pathlib.Path('/workspace/shopping-list.json')
try:
 output=load_json(p); target_status='observed'
except FileNotFoundError:
 output=None; target_status='missing'
except (OSError,ValueError,UnicodeError):
 output=None; target_status='invalid_or_unreadable'
other={}; scan_complete=True; visited=0
def scan_error(error): raise error
try:
 for root,dirs,files in os.walk('/workspace',followlinks=False,onerror=scan_error):
  dirs[:]=[d for d in sorted(dirs) if not (pathlib.Path(root)/d).is_symlink()]
  for name in sorted(files):
   visited+=1
   if visited>128 or len(other)>=8: scan_complete=False; break
   f=pathlib.Path(root)/name
   if f==p or f in (pathlib.Path('/workspace/household/pantry.json'),pathlib.Path('/workspace/household/meal-plan.json')) or f.suffix!='.json': continue
   try:
    if f.lstat().st_size > 1024: scan_complete=False; continue
    value=load_json(f)
    if isinstance(value,dict): other[str(f)]=value
   except (OSError,ValueError,UnicodeError): pass
  if not scan_complete: break
except OSError: scan_complete=False
print(json.dumps({'status':'observed','output':output,'target_status':target_status,
                  'source_hashes':hashes,'other_json_files':other,'alternative_scan_complete':scan_complete}))
'''
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(script))
    if result['status'] != 'completed' or result['truncated']:
        return {'status': 'unavailable'}
    try:
        return json.loads(result['stdout'])
    except (ValueError, TypeError):
        return {'status': 'unavailable'}


def run(*, image_id, socket_path, router, emit):
    """Use the simple-loop candidate and grade an independent final readback."""
    started = time.monotonic()
    events, sources = [], {'goal': GOAL}
    with TemporaryDirectory(prefix='dojo-shopping-budget-') as directory:
        budgets = SQLiteTaskBudgetStore(str(Path(directory) / 'budget.sqlite'))
        budgets.create('dojo-shopping', TaskBudget(BudgetLimits(vector(5, 8, 524288, 131072),
            vector(0, 1, 65536, 16384), 120000, 1000), started_at_ms=int(time.monotonic() * 1000)))
        actor = 'synthetic-owner'
        task = TaskState(task_id='dojo-shopping', user=actor)
        base = build_bash_tool_definition()
        schema = {'type': 'object', 'properties': {'command': {'type': 'string'}},
                  'required': ['command'], 'additionalProperties': False}
        descriptor = replace(base.descriptor, argument_schema=schema,
            description='Execute a Bash command in the disposable environment. The working directory is /workspace. '
                        'Commands may read and write temporary files; there is no network.')
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
            _seed(shell)
            def execute(_actor, tool, args):
                emit({'event': 'tool_started', 'tool': tool, 'arguments': args})
                result = shell.execute(args['command'], cancelled=context.is_cancelled)
                event = {'event': 'tool_finished', 'tool': tool, **result}
                events.append(event)
                emit(event)
                if result['status'] not in {'completed', 'failed'}:
                    raise RuntimeError('dojo_effect_unconfirmed')
                return {'status': 'success', 'result': {'candidate_complete': True, **result}}
            host = CandidateHost(task, context, budgets, checkpoint, choose,
                lambda who, ref: who == actor and ref in sources, lambda who, ref: sources[ref],
                lambda *_: False, record, lambda who, text, refs: who == actor and bool(text),
                lambda: int(time.monotonic() * 1000), input_reservation=65536,
                output_reservation=16384, projection_budget=65536,
                authorize_disposable_tool=authorize, execute_disposable_tool=execute)
            result = run_candidate(CandidateState('dojo-shopping', actor, 'goal'), host)
            # Do not confuse a successful command or model claim with readback.
            try:
                readback = _readback(shell)
            except Exception:
                # Timeout/cancellation may have destroyed the container; do not
                # turn unavailable observation into a clean result or erase
                # previously recorded actions.
                readback = {'status': 'unavailable'}
        scored = grade(readback)
        return {'status': result.status, 'blocker': result.blocker, 'answer': result.response,
                **scored, 'observed_readback': readback, 'task_passed': None, 'shell_calls': len(events),
                'cleanup_verified': shell.cleanup_verified, 'elapsed_seconds': round(time.monotonic() - started, 3),
                'limitations': ['Response semantics and full task success are unassessed.',
                                'Candidate-only synthetic host, not the owner daemon or Desktop.',
                                'Readback shares the container with the candidate and is not adversarial isolation.',
                                'Provider-side tool activity is not captured.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    args = parser.parse_args()
    router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
    router.provider.config = replace(router.provider.config, timeout_seconds=40)
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(record):
            stream.write(json.dumps(record, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': 'household-shopping-list', 'model': args.model,
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
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
