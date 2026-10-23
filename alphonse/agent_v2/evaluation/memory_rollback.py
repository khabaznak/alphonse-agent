"""Disposable DC-5 crash/restart drill through PhaseExecutor and real stores.

Scripted decisions qualify deterministic recovery only, never live usefulness.
Every subprocess opens the same fixture database and memory roots; no restore.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import (
    CompletionCondition, MutationScope, PhaseExecutor, PhasePlan, PhaseSubgoal,
    SideEffectClass, TacticalState, new_tactical_state,
)
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
from alphonse.agent_v2.core.memory.knowledge import KnowledgeMemory, KnowledgeStore
from alphonse.agent_v2.core.memory.ledger import LedgerMemory
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore
from alphonse.agent_v2.system_one import SystemOneTacticalReview
from .context_pilot import projection_flag

OWNER = 'rollback-fixture'
CONSTRAINT = 'Only update fixture.txt; preserve the original and do not send messages.'


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def stores(root):
    path = root / 'state.sqlite3'
    conversations = SQLiteConversationStore(path)
    knowledge = KnowledgeMemory(store=KnowledgeStore(str(path)), conversations=conversations,
                               extractor=None, reviewer=None)
    memory = LedgerMemory(users_root=lambda: root / 'users',
                          settings_store=SQLiteMemorySettingsStore(path), knowledge=knowledge)
    return conversations, knowledge, memory, SQLiteQuestionStore(path)


def write_compatible(root, tag):
    conversations, knowledge, memory, questions = stores(root)
    text = f'The household storage label for {tag} is R-{tag}.'
    event = conversations.record(owner_user_id=OWNER, content=text, role='user', source='synthetic',
        source_message_id=tag, created_at='2026-10-22T12:00:00+00:00')
    # Explicit synthetic extraction and review; no providers or background workers.
    knowledge.extractor = lambda *args, **kwargs: {'candidates': [dict(content=text, kind='fact',
        subjects=[OWNER, tag], claim_key=f'{OWNER}.{tag}', quote=text)]}
    knowledge.reviewer = lambda payload, candidates: [True]*len(candidates)
    knowledge.support_selector = lambda source, existing: []
    knowledge.comparison_reviewer = lambda pairs: ['independent']*len(pairs)
    knowledge.process_event(event, OWNER)
    return event


def initialize(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / 'rollback-fixture.json').write_text('{"synthetic": true, "version": 1}\n')
    conversations, knowledge, memory, questions = stores(root)
    task = TaskState(task_id='rollback-task', message_id='rollback-request', user=OWNER,
                     goal='Update two fixture fields and verify them.', acceptance_criteria_md=CONSTRAINT)
    task.append_conversation_message(OWNER, CONSTRAINT)
    task.memory_context_md = memory.start_task(task)
    write_compatible(root, 'original')
    with sqlite3.connect(root / 'state.sqlite3') as conn:
        conn.execute('CREATE TABLE fixture_effects (step TEXT, path TEXT, value TEXT)')
    phase = PhasePlan(phase_id='rollback-phase', objective=task.goal,
        authorized_capabilities=('native.exact_text_edit','native.read'),
        mutation_scope=MutationScope(('fixture.txt',)),
        subgoals=tuple(PhaseSubgoal(step, 'Update fixture '+step, 'verified_'+step,
            depends_on=(('first',) if step=='second' else ()),
            allowed_capabilities=('native.exact_text_edit',),
            allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,),
            completion=CompletionCondition('field_equals', field='verification.status', expected='verified'))
            for step in ('first','second')) + (PhaseSubgoal('verify','Read both fields','observation',
                depends_on=('second',),allowed_capabilities=('native.read',),
                completion=CompletionCondition('field_equals', field='verified', expected=True)),))
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ['native.exact_text_edit','native.read']
    task.hierarchical_state = state.to_dict()
    questions.create_question(task=task, question='Which fixture file should I update?', kind='open_text')
    questions.save_task_checkpoint(task)


def worker(root, stage):
    if (root / 'rollback-fixture.json').read_text() != '{"synthetic": true, "version": 1}\n':
        raise ValueError('not_a_rollback_fixture')
    conversations, knowledge, memory, questions = stores(root)
    task = questions.load_task_checkpoint('rollback-task')
    if pending := task.metadata.get('pending_question_id'):
        task = questions.route_answer(respondent_user_id=OWNER, text='fixture.txt; keep both constraints.',
                                     question_id=pending).resumed_task
    state = TacticalState.from_dict(task.hierarchical_state)
    registry = InMemoryToolRegistry()
    def edit(arguments):
        if arguments['path'] != 'fixture.txt':
            raise PermissionError('outside_fixture_scope')
        with sqlite3.connect(root / 'state.sqlite3') as conn:
            conn.execute('INSERT INTO fixture_effects VALUES (?,?,?)', (arguments['step'],arguments['path'],'done'))
        return {'affected_paths': ['fixture.txt'], 'verification': {'status':'verified'},
                'source_ref':'fixture:effect:'+arguments['step']}
    def read(arguments):
        with sqlite3.connect(root / 'state.sqlite3') as conn:
            values = conn.execute('SELECT step, path, value FROM fixture_effects ORDER BY rowid').fetchall()
        return {'verified': values == [('first','fixture.txt','done'),('second','fixture.txt','done')],
                'values':values,'source_ref':'fixture:readback'}
    for identity, callback, readonly in (('native.exact_text_edit',edit,False),('native.read',read,True)):
        registry.register(ToolDefinition(ToolDescriptor(tool_id=identity,name=identity,kind=ToolKind.NATIVE,read_only=readonly), callable=callback))
    jev = SimpleNamespace(evaluate_tactical_progress=lambda **kwargs: SystemOneTacticalReview(
        complete=kwargs['action'].get('status')=='success', confidence=.99, confident=True))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, question_store=questions,
                              conversation_store=conversations, memory=memory, system_one=jev)
    # Compatible writes made while projections are on must survive flag-off.
    write_compatible(root, stage)
    def select(tactical, subgoal, tools):
        rendered = task_context_for_decision(task, context, purpose='execution')
        if CONSTRAINT not in rendered:
            raise AssertionError('lost_constraint')
        if (stage=='interrupted' and subgoal.subgoal_id=='second' or
                stage=='resumed' and subgoal.subgoal_id=='verify'):
            # Executor already committed the previous effect and checkpoint.
            # Deliberately skip cleanup, simulating process loss.
            os._exit(73)
        return {'tool_id':'native.read' if subgoal.subgoal_id=='verify' else 'native.exact_text_edit',
                'arguments': {'path':'fixture.txt', **({'step':subgoal.subgoal_id} if subgoal.subgoal_id!='verify' else {})}}
    with projection_flag(stage!='off'):
        if stage=='off':
            # Exercise the actual legacy ledger/memory assembly on current storage.
            task.memory_context_md = memory.start_task(task)
            rendered = task_context_for_decision(task, context, purpose='execution')
            if CONSTRAINT not in rendered or 'Durable household memory' not in rendered or not memory.latest_content(user_id=OWNER):
                raise AssertionError('legacy_path_unavailable')
        outcome = PhaseExecutor(action_selector=select).run(task,state,context)
    if stage=='off' and outcome.status.value!='phase_complete':
        raise AssertionError('resume_not_complete:'+outcome.status.value)


def snapshot(root):
    conversations, knowledge, memory, questions = stores(root)
    task = questions.load_task_checkpoint('rollback-task')
    with sqlite3.connect(root / 'state.sqlite3') as conn:
        effects = conn.execute('SELECT step,path,value FROM fixture_effects ORDER BY rowid').fetchall()
        integrity = conn.execute('PRAGMA integrity_check').fetchone()[0]
    return {'events': [e.to_dict() for e in conversations.list_all(owner_user_id=OWNER)],
            'knowledge': [k.to_dict() for k in knowledge.recall(OWNER,historical=True)],
            'task':task.to_checkpoint_dict(),'effects':effects,'integrity':integrity}


def drill():
    observations=[]
    for resume_before_off in (False, True):
        with tempfile.TemporaryDirectory(prefix='memory-rollback-') as temporary:
            root=Path(temporary)/'fixture'
            initialize(root)
            originals=snapshot(root)
            files={str(p.relative_to(root/'users')): p.read_bytes() for p in (root/'users').rglob('*') if p.is_file()}
            stages=['interrupted']+(['resumed'] if resume_before_off else [])+['off']
            checkpoints=[]
            for stage in stages:
                result=subprocess.run([sys.executable,'-m','alphonse.agent_v2.evaluation.memory_rollback','--worker',str(root),'--stage',stage],capture_output=True,text=True,timeout=30)
                expected=0 if stage=='off' else 73
                if result.returncode!=expected:
                    raise RuntimeError(f'{stage}: expected {expected}, got {result.returncode}\n{result.stderr}')
                checkpoints.append({'stage':stage,'returncode':result.returncode,'readback':snapshot(root)})
            final=checkpoints[-1]['readback']
            expected_steps=[('first','fixture.txt','done'),('second','fixture.txt','done')]
            checks = {
                'exactly_once_completed_effects': final['effects']==expected_steps,
                'integrity': final['integrity']=='ok',
                'original_events_preserved': all(e in final['events'] for e in originals['events']),
                'original_knowledge_preserved': all(k in final['knowledge'] for k in originals['knowledge']),
            }
            for checkpoint in checkpoints:
                checks['events_preserved:'+checkpoint['stage']] = all(e in final['events'] for e in checkpoint['readback']['events'])
                checks['knowledge_preserved:'+checkpoint['stage']] = all(k in final['knowledge'] for k in checkpoint['readback']['knowledge'])
            event_refs = {'conversation:'+e['event_id'] for e in final['events']}
            checks['provenance_resolves'] = all(source['source_id'] in event_refs for k in final['knowledge'] for source in k['sources'])
            checks['compatible_writes_preserved'] = len(final['knowledge'])==len(final['events'])==len(stages)+1
            checks['constraints_preserved'] = final['task']['acceptance_criteria_md']==CONSTRAINT
            checks['all_subgoals_completed'] = final['task']['hierarchical_state']['completed_subgoal_ids']==['first','second','verify']
            _, _, _, questions = stores(root)
            question = questions.get_question(originals['task']['metadata']['pending_question_id'])
            checks['answered_question_preserved'] = question.status=='answered' and 'fixture.txt' in str(final['task'])
            # Legacy audit may append; retain its original prefix independently.
            checks['original_ledger_prefixes_preserved'] = all((root/'users'/p).is_file() and (root/'users'/p).read_bytes().startswith(original) for p, original in files.items())
            checks['no_original_file_deletion'] = set(files)<=set(inventory(root/'users'))
            if not all(checks.values()):
                raise RuntimeError('rollback_invariants_failed:'+','.join(k for k,v in checks.items() if not v))
            observations.append({'resume_before_off':resume_before_off,'checkpoints':checkpoints,
                'checks':{**checks, 'legacy_path_available':True,'same_fixture_store':True}})
    return {'schema_version':1,'date':'2026-10-22','kind':'deterministic_crash_restart',
            'result':'pass','observations':observations,'DC-5':'unassessed',
            'limits':['Scripted System One and action selection; no live inference or independent review.',
                      'Crash after committed checkpoint only. Crash between effect and checkpoint requires reconciliation; no exactly-once claim for that window.',
                      'Exercises PhaseExecutor and stored question-answer recovery, not daemon transport restart.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--stage',choices=('interrupted','resumed','off'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.stage)
    else:
        report=drill()
        if args.output:
            with args.output.open('x') as stream:
                stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        else:
            print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
