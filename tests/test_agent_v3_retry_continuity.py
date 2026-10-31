"""Synthetic approved-payload retries: no household records or live providers."""
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseStatus, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.outer import V3OuterController, StrategicAction
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase, V3PhasePlanValidationError
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.memory.retry_context import prepare_retry_continuity, retry_continuity_context
from alphonse.agent_v2.core.memory.transcript import conversation_transcript
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.system_one import (
    JevCriterionDecisionProvider, SystemOneSettings, SystemOneToolRegistrySelection,
    SystemOneTacticalReview, SystemOneReviewResult, SystemOneActRecommendation,
)


@pytest.fixture
def history(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'conversation.sqlite')
    sequence = iter(range(1000))
    def record(role, text, date, owner='fixture-owner'):
        return store.record(owner_user_id=owner, role=role, content=text, source='desktop', created_at=date, source_message_id='message-' + str(next(sequence)))
    old = record('user', 'Record OLD_SEPTEMBER_PAYLOAD: FixtureAlpha, prescription 2026-09-11.', '2026-09-12T10:00:00+00:00')
    record('assistant', 'OLD_SEPTEMBER_PAYLOAD is ready for review.', '2026-09-12T10:01:00+00:00')
    old_approval = record('user', 'OLD_APPROVAL: save FixtureAlpha.', '2026-09-12T10:02:00+00:00')
    record('assistant', 'The old record was saved.', '2026-09-12T10:03:00+00:00')
    record('user', 'An unrelated household question.', '2026-10-31T16:00:00+00:00')
    record('assistant', 'Unrelated answer.', '2026-10-31T16:01:00+00:00')
    request = record('user', 'Please record this new prescription for Test Child.', '2026-10-31T17:04:00+00:00')
    record('assistant', 'Please provide the new source clearly.', '2026-10-31T17:04:20+00:00')
    payload = record('assistant', 'NEW_OCTOBER_PAYLOAD: prescription 2026-10-30; Test Child; medication FixtureBeta; topical_name uncertain. This is a reviewed extraction, not a persisted database update.', '2026-10-31T17:05:00+00:00')
    approval = record('user', 'NEW_APPROVAL: this looks correct; save the reviewed October prescription, preserving its flagged uncertainty.', '2026-10-31T17:06:00+00:00')
    record('assistant', 'I will save the reviewed source.', '2026-10-31T17:06:20+00:00')
    record('assistant', 'The write failed before execution.', '2026-10-31T17:07:00+00:00')
    for i in range(40):
        record('user', f'Retry {i}: please try recording the new treatment again.', '2026-10-31T18:00:00+00:00')
        record('assistant', 'This retry failed; the database update is still unverified.', '2026-10-31T18:00:01+00:00')
    private = record('user', 'PRIVATE_OTHER_HOUSEHOLD_PAYLOAD', '2026-10-31T19:00:00+00:00', owner='other')
    return store, request, payload, approval, old, old_approval, private


class ContinuityJev:
    def __init__(self):
        self.selections = 0

    def judge_recall(self, **kwargs):
        return True if not kwargs['candidates'] else [h['source_id'] for h in kwargs['candidates']]

    def select_retry_context(self, *, request, candidates):
        self.selections += 1
        assert 'PRIVATE_OTHER_HOUSEHOLD_PAYLOAD' not in str(candidates)
        assert len(candidates) <= 64 and sum(len(c['content']) for c in candidates) <= 48000
        return tuple(c['id'] for c in candidates if 'NEW_OCTOBER_PAYLOAD' in c['content'] and 'NEW_APPROVAL' in c['content'])

    def curate_request_tools(self, *, task_context, **kwargs):
        assert 'NEW_OCTOBER_PAYLOAD' in task_context and 'NEW_APPROVAL' in task_context
        return SystemOneToolRegistrySelection(selected_tool_ids=('native.bash', 'native.respond'))

    def select_plan_tools(self, **kwargs):
        return SystemOneToolRegistrySelection(selected_tool_ids=('native.bash', 'native.respond'))

    def evaluate_tactical_progress(self, **kwargs):
        return SystemOneTacticalReview(True, .99, True)

    def evaluate(self, **kwargs):
        verified = [e for e in kwargs['evidence']['entries'] if e.get('tool_id') == 'native.bash'
                    and e.get('status') == 'success' and '2026-10-30|FixtureBeta' in (e.get('result') or {}).get('stdout', '')]
        return SystemOneReviewResult(tuple({'criterion_id': c['id'], 'status': 'satisfied', 'evidence_refs': [verified[-1]['evidence_ref']]}
                                          for c in kwargs['contract']['criteria']) if verified else (), ())

    def recommend_act(self, **kwargs):
        return SystemOneActRecommendation('complete', .99, True, 'Exact requested payload persisted and read back', answers={
            'mission_outcome_is_fulfilled': .99, 'mission_requires_persisted_effect': .99,
        })


def retry_task():
    task = TaskState(task_id='latest-retry', message_id='latest-retry', user='fixture-owner',
        goal='New attempt: register the new treatment for Test Child', intelligence_engine='hierarchical_v3', intelligence_schema_version=3)
    task.append_conversation_message(task.user, task.goal)
    return task


@pytest.mark.parametrize('projected', [False, True])
@pytest.mark.parametrize('dependency_error', [False, True])
def test_approved_october_payload_survives_many_failed_retries_and_is_persisted(history, tmp_path, monkeypatch, projected, dependency_error):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', str(int(projected)))
    store, request, payload, approval, *_ = history
    task = retry_task()
    assert 'NEW_OCTOBER_PAYLOAD' not in conversation_transcript(task, store)
    database = tmp_path / 'records.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE prescriptions(date TEXT PRIMARY KEY, medication TEXT, provenance TEXT)')
        db.execute("INSERT INTO prescriptions VALUES('2026-09-11','FixtureAlpha','old record')")
    phase = dict(schema_version=3, phase_id='record-october', objective='Persist the reviewed October 30 prescription',
        acceptance_criteria=['The reviewed October 30 prescription is persisted and verified by fresh read-back'], criterion_ids=['ac-1'],
        authorized_capabilities=['local_shell', 'user_response'], mutation_scope={'allowed_paths': ['records.sqlite'], 'allow_external_effects': False},
        subgoals=[dict(subgoal_id='record_or_ask_for_missing_details', objective='Record the reviewed October 30 FixtureBeta payload with its source references and flagged uncertainty',
            required_output_type='verified_record', allowed_capabilities=['local_shell'], allowed_side_effects=['project_mutation'], completion={'kind': 'output_present'}),
            dict(subgoal_id='report_outcome', objective='Report the verified October update', required_output_type='response',
                 depends_on=['record_or_ask_for_missing_details'], allowed_capabilities=['user_response'], allowed_side_effects=['user_response'], completion={'kind': 'output_present'})])
    requests = []
    def generate(inference):
        requests.append(inference)
        assert 'NEW_OCTOBER_PAYLOAD' in inference.prompt and 'NEW_APPROVAL' in inference.prompt
        assert 'topical_name uncertain' in inference.prompt
        for event in (request, payload, approval):
            assert any(r['source_id'] == 'conversation:' + event.event_id and r['status'] == 'included'
                       for manifest in inference.context_manifests for r in manifest['records'])
        candidate = deepcopy(phase)
        if dependency_error and len(requests) == 1:
            candidate['subgoals'][-1]['depends_on'] = ['record-or-ask-for-missing-details']
        if len(requests) == 2:
            assert 'invalid subgoal dependency' in inference.prompt
            assert 'do not silently remove prerequisites' in inference.prompt
            with sqlite3.connect(database) as db:
                assert db.execute('SELECT count(*) FROM prescriptions').fetchone()[0] == 1
        return SimpleNamespace(json_value=candidate)
    tools = InMemoryToolRegistry()
    tools.register(build_bash_tool_definition()); tools.register(build_respond_tool_definition())
    jev = ContinuityJev()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev, tools=tools,
                              inference=SimpleNamespace(generate_json=generate))
    planned = plan_phase(task, context)
    assert len(requests) == (2 if dependency_error else 1)
    assert jev.selections == 1
    assert 'NEW_OCTOBER_PAYLOAD' not in str(task.metadata['v3_retry_continuity'])  # IDs only, no cached payload.
    source_ids = task.metadata['v3_retry_continuity']['source_ids']
    script = "import sqlite3; c=sqlite3.connect('records.sqlite'); c.execute('BEGIN'); c.execute('INSERT INTO prescriptions VALUES(?,?,?)',('2026-10-30','FixtureBeta'," + repr(json.dumps(source_ids)) + ")); c.commit(); c.close(); r=sqlite3.connect('file:records.sqlite?mode=ro',uri=True); row=r.execute(\"SELECT date,medication FROM prescriptions WHERE date='2026-10-30'\").fetchone(); print(row[0]+'|'+row[1])"
    actions = iter([{'tool_id': 'native.bash', 'arguments': {'command': shlex.quote(sys.executable) + ' -c ' + shlex.quote(script), 'cwd': str(tmp_path)}},
                    {'tool_id': 'native.respond', 'arguments': {'message': 'October prescription saved and read back.'}}])
    state = new_tactical_state(planned)
    outcome = PhaseExecutor(action_selector=lambda *_: next(actions), reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    _, decision = V3OuterController().review_and_route(task, state, outcome, context)
    assert decision.action == StrategicAction.COMPLETE
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT date,medication FROM prescriptions ORDER BY date').fetchall() == [('2026-09-11', 'FixtureAlpha'), ('2026-10-30', 'FixtureBeta')]


def test_cached_binding_rechecks_access_and_never_leaves_confirmation_without_payload(history):
    store, _, payload, _, *_ = history
    task = retry_task(); jev = ContinuityJev()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev)
    prepare_retry_continuity(task, context)
    restored = TaskState.from_dict(task.to_checkpoint_dict())
    assert 'NEW_OCTOBER_PAYLOAD' in retry_continuity_context(restored, context)
    prepare_retry_continuity(restored, context)
    assert jev.selections == 1
    store.delete(event_id=payload.event_id, owner_user_id='fixture-owner')
    rendered = retry_continuity_context(restored, context)
    assert 'NEW_OCTOBER_PAYLOAD' not in rendered and 'NEW_APPROVAL' not in rendered
    assert 'could not be resolved completely' in rendered
    restored.user = 'other'
    assert 'NEW_APPROVAL' not in retry_continuity_context(restored, context)


def test_distinct_episodes_cannot_be_combined_or_guessed_from_lexical_similarity(history):
    store, _, _, approval, _, old_approval, _ = history
    jev = ContinuityJev()
    jev.select_retry_context = lambda **kwargs: ('conversation:' + old_approval.event_id, 'conversation:' + approval.event_id)
    task = retry_task()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev)
    prepare_retry_continuity(task, context)
    assert task.metadata['v3_retry_continuity']['status'] == 'ambiguous'
    rendered = retry_continuity_context(task, context)
    assert 'NEW_OCTOBER_PAYLOAD' not in rendered and 'OLD_SEPTEMBER_PAYLOAD' not in rendered
    assert 'ask which exact source/change' in rendered


def test_provider_selects_only_bounded_candidate_ids_and_rejects_invalid_judgments():
    seen = []
    def transport(url, key, payload, timeout):
        seen.append(payload)
        return {'answers': {q: {'type': 'noul', 'noul': .99 if q == 'retry_origin_1' else .01} for q in payload['questions']}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    candidates = [{'id': 'old', 'content': 'Older similar record'}, {'id': 'approved', 'content': 'Reviewed payload and user confirmation'}]
    assert provider.select_retry_context(request='Try saving the new prescription again', candidates=candidates) == ('approved',)
    assert all(q['type'] == 'noul' for q in seen[0]['questions'].values())
    assert 'never permission' in seen[0]['questions']['retry_origin_1']['instructions']
    provider.client.transport = lambda *args: {'answers': {'retry_origin_0': {'type': 'noul', 'noul': True}}}
    with pytest.raises(ValueError, match='retry_selection_invalid'):
        provider.select_retry_context(request='Again', candidates=candidates[:1])


@pytest.mark.parametrize('kind', ['missing', 'forward', 'self', 'cycle'])
def test_invalid_dependencies_get_one_correction_without_executing_rejected_plan(kind):
    base = dict(schema_version=3, phase_id='dependency-test', objective='Inspect then update fixture',
        acceptance_criteria=['Fixture is updated and verified'], criterion_ids=['ac-1'], authorized_capabilities=['local_shell'],
        mutation_scope={'allowed_paths': ['records.sqlite'], 'allow_external_effects': False},
        subgoals=[dict(subgoal_id='inspect', objective='Inspect fixture', required_output_type='record', allowed_capabilities=['local_shell'], completion={'kind': 'output_present'}),
                  dict(subgoal_id='update', objective='Update and read back', required_output_type='verified_record', depends_on=['inspect'],
                       allowed_capabilities=['local_shell'], allowed_side_effects=['project_mutation'], completion={'kind': 'output_present'})])
    invalid = deepcopy(base)
    if kind == 'missing': invalid['subgoals'][1]['depends_on'] = ['typo-inspect']
    elif kind == 'forward': invalid['subgoals'][0]['depends_on'] = ['update']; invalid['subgoals'][1]['depends_on'] = []
    elif kind == 'self': invalid['subgoals'][0]['depends_on'] = ['inspect']
    else: invalid['subgoals'][0]['depends_on'] = ['update']
    calls = []
    def generate(request):
        calls.append(request)
        return SimpleNamespace(json_value=deepcopy(invalid if len(calls) == 1 else base))
    task = retry_task(); task.set_acceptance_contract_from_markdown('1.- [ ] Original requested fixture effect remains required')
    original = deepcopy(task.acceptance_contract)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=generate),
                              system_one=SimpleNamespace(select_plan_tools=lambda **kwargs: SystemOneToolRegistrySelection()))
    planned = plan_phase(task, context)
    assert len(calls) == 2 and planned.subgoals[1].depends_on == ('inspect',)
    assert task.acceptance_contract == original
    assert not task.hierarchical_state
    context.inference.generate_json = lambda request: SimpleNamespace(json_value=deepcopy(invalid))
    with pytest.raises(V3PhasePlanValidationError, match='dependency_invalid|self_dependency'):
        plan_phase(task, context)


def test_new_attachment_or_self_contained_request_does_not_borrow_an_old_payload(history):
    store, *_ = history
    jev = ContinuityJev()
    jev.judge_recall = lambda **kwargs: False
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev)
    task = retry_task()
    task.goal = 'Create a standalone fixture record with value 17'
    prepare_retry_continuity(task, context)
    assert task.metadata['v3_retry_continuity']['status'] == 'not_needed'
    assert not retry_continuity_context(task, context)
    assert jev.selections == 0
    attached = retry_task(); attached.metadata['attachments'] = [{'asset_id': 'new-current-source'}]
    jev.judge_recall = lambda **kwargs: pytest.fail('A new current attachment must not borrow an older payload')
    prepare_retry_continuity(attached, context)
    assert not retry_continuity_context(attached, context)


def test_new_current_steering_invalidates_source_selection(history):
    store, *_ = history
    jev = ContinuityJev(); task = retry_task()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev)
    prepare_retry_continuity(task, context)
    before = task.metadata['v3_retry_continuity']['request_signature']
    original = jev.select_retry_context
    def select(**kwargs):
        assert 'Preserve the uncertainty in the ointment name' in kwargs['request']
        return original(**kwargs)
    jev.select_retry_context = select
    task.append_conversation_message(task.user, 'Preserve the uncertainty in the ointment name')
    prepare_retry_continuity(task, context)
    assert jev.selections == 2 and task.metadata['v3_retry_continuity']['request_signature'] != before


def test_access_filtering_precedes_candidate_previews_even_for_an_unsafe_adapter(history):
    store, *_ = history
    proxy = SimpleNamespace(list_recent=lambda **kwargs: store.list_recent(**kwargs) + store.list_recent(owner_user_id='other'), read=store.read)
    task = retry_task()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=proxy, system_one=ContinuityJev())
    prepare_retry_continuity(task, context)
    assert task.metadata['v3_retry_continuity']['status'] == 'selected'
    assert 'PRIVATE_OTHER_HOUSEHOLD_PAYLOAD' not in retry_continuity_context(task, context)


def test_over_budget_selected_payload_is_withheld_whole_instead_of_clipped(history):
    store, *_ = history
    store.record(owner_user_id='fixture-owner', role='user', content='Review a new large source', source='desktop', source_message_id='large-request')
    store.record(owner_user_id='fixture-owner', role='assistant', content='OVERSIZED_SOURCE ' * 2000, source='desktop', source_message_id='large-extraction')
    approval = store.record(owner_user_id='fixture-owner', role='user', content='Approve that entire large source', source='desktop', source_message_id='large-approval')
    jev = ContinuityJev(); jev.select_retry_context = lambda **kwargs: ('conversation:' + approval.event_id,)
    task = retry_task(); context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev)
    prepare_retry_continuity(task, context)
    assert task.metadata['v3_retry_continuity']['status'] == 'over_budget'
    rendered = retry_continuity_context(task, context)
    assert 'OVERSIZED_SOURCE' not in rendered and 'Approve that entire large source' not in rendered
    assert 'could not be resolved completely' in rendered


def test_cancellation_between_invalid_dependency_and_correction_prevents_second_call():
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    cancelled = [False]; calls = []
    invalid = dict(schema_version=3, phase_id='invalid', objective='Update fixture', subgoals=[dict(
        subgoal_id='write', objective='Write fixture', required_output_type='record', depends_on=['missing'], completion={'kind': 'output_present'})])
    def generate(request):
        calls.append(request); cancelled[0] = True
        return SimpleNamespace(json_value=invalid)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=generate), cancellation_checker=lambda: cancelled[0])
    task = retry_task()
    result = HierarchicalCAPDProcessor().process(task, context)
    assert result.status.value == 'cancelled' and len(calls) == 1
    assert not task.hierarchical_state


def test_selection_previews_preserve_origin_amid_sixty_retry_windows(history):
    store, _, payload, approval, *_ = history
    for i in range(20):
        store.record(owner_user_id='fixture-owner', role='user', content='Please retry the update again', source='desktop', source_message_id=f'extra-retry-{i}')
        store.record(owner_user_id='fixture-owner', role='assistant', content='Another attempt failed before the record update was verified.', source='desktop', source_message_id=f'extra-failure-{i}')
    jev = ContinuityJev(); task = retry_task()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev)
    prepare_retry_continuity(task, context)
    assert task.metadata['v3_retry_continuity']['status'] == 'selected'
    assert {'conversation:' + payload.event_id, 'conversation:' + approval.event_id} <= set(task.metadata['v3_retry_continuity']['source_ids'])
    assert 'NEW_OCTOBER_PAYLOAD' in retry_continuity_context(task, context)


@pytest.mark.parametrize('projected', [False, True])
def test_tactical_inference_receives_full_retry_payload_and_canonical_source_receipts(history, monkeypatch, projected):
    from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan, PhaseSubgoal
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', str(int(projected)))
    store, _, payload, approval, *_ = history
    task = retry_task(); requests = []
    def generate(request):
        requests.append(request)
        assert 'NEW_OCTOBER_PAYLOAD' in request.prompt and 'NEW_APPROVAL' in request.prompt
        for event in (payload, approval):
            assert any(r['source_id'] == 'conversation:' + event.event_id and r['status'] == 'included'
                       for manifest in request.context_manifests for r in manifest['records'])
        return SimpleNamespace(json_value={'tool_id': 'native.bash', 'arguments': {'command': 'true'}})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=ContinuityJev(), inference=SimpleNamespace(generate_json=generate))
    prepare_retry_continuity(task, context)
    phase = PhasePlan('act', 'Use the approved October prescription', (PhaseSubgoal(
        'act', 'Persist exact reviewed October payload', 'record', allowed_capabilities=('local_shell',)),), authorized_capabilities=('local_shell',))
    state = new_tactical_state(phase)
    PhaseExecutor()._select_action(task, state, phase.subgoals[0], (), context)
    assert len(requests) == 1


def test_stale_binding_is_not_rendered_after_current_steering_changes(history):
    store, *_ = history
    task = retry_task(); context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=ContinuityJev())
    prepare_retry_continuity(task, context)
    task.append_conversation_message(task.user, 'Use a different source instead; its details are pending')
    assert 'NEW_OCTOBER_PAYLOAD' not in retry_continuity_context(task, context)


def test_unresolved_origin_rejects_mutation_plan_and_parks_with_a_source_question(history):
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
    store, _, _, approval, _, old_approval, _ = history
    task = retry_task(); task.set_acceptance_contract_from_markdown('1.- [ ] The intended reviewed source is persisted and read back')
    original = deepcopy(task.acceptance_contract)
    jev = ContinuityJev()
    jev.select_retry_context = lambda **kwargs: ('conversation:' + old_approval.event_id, 'conversation:' + approval.event_id)
    jev.curate_request_tools = lambda **kwargs: SystemOneToolRegistrySelection(selected_tool_ids=('native.bash',))
    jev.select_plan_tools = lambda **kwargs: SystemOneToolRegistrySelection(selected_tool_ids=('native.bash',))
    tools = InMemoryToolRegistry(); tools.register(build_bash_tool_definition()); tools.register(build_ask_question_tool_definition())
    wrong = dict(schema_version=3, phase_id='guess-old', objective='Write guessed older record', acceptance_criteria=[],
        criterion_ids=['ac-1'], authorized_capabilities=['local_shell'], mutation_scope={'allowed_paths': ['records.sqlite']},
        subgoals=[dict(subgoal_id='write', objective='Write old record', required_output_type='record', allowed_capabilities=['local_shell'],
                       allowed_side_effects=['project_mutation'], completion={'kind': 'output_present'})])
    question = dict(schema_version=3, phase_id='ask-source', objective='Resolve exact source', acceptance_criteria=[], criterion_ids=['ac-1'],
        authorized_capabilities=['user_interaction'], mutation_scope={'allowed_paths': []}, subgoals=[dict(subgoal_id='ask', objective='Ask which exact source is intended',
        required_output_type='question', allowed_capabilities=['user_interaction'], allowed_side_effects=['user_response'], completion={'kind': 'output_present'})])
    plans = iter([wrong, question]); questions = SQLiteQuestionStore(':memory:')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=jev, tools=tools, question_store=questions,
                              inference=SimpleNamespace(generate_json=lambda request: SimpleNamespace(json_value=next(plans))))
    planned = plan_phase(task, context)
    assert planned.phase_id == 'ask-source' and 'native.ask_question' in planned.curated_tool_ids
    state = new_tactical_state(planned)
    outcome = PhaseExecutor(action_selector=lambda *_: {'tool_id': 'native.ask_question', 'arguments': {
        'question': 'Which prescription date should I use for this update? The available episodes do not resolve the intended source.', 'question_kind': 'open_text'}},
        reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.WAITING_USER
    assert [action.tool_id for action in state.actions] == ['native.ask_question']
    assert task.acceptance_contract == original and not task.acceptance_criteria_all_complete()


def test_revoked_origin_blocks_an_already_planned_operation_before_action_selection(history):
    from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan, PhaseSubgoal, SideEffectClass
    store, _, payload, *_ = history
    task = retry_task(); context = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store, system_one=ContinuityJev())
    prepare_retry_continuity(task, context)
    assert 'NEW_OCTOBER_PAYLOAD' in retry_continuity_context(task, context)
    phase = PhasePlan('write', 'Write requested source', (PhaseSubgoal('write', 'Persist and verify', 'record',
        allowed_capabilities=('local_shell',), allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,)),), authorized_capabilities=('local_shell',))
    store.delete(event_id=payload.event_id, owner_user_id='fixture-owner')
    state = new_tactical_state(phase)
    outcome = PhaseExecutor(action_selector=lambda *_: pytest.fail('Revoked source must block before selecting or executing a command')).run(task, state, context)
    assert outcome.blockers == ('retry_source_requires_user_input',)
    assert not state.actions
