"""Question-answer interpretation belongs to System Two; fixtures stay synthetic."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.memory.clarification import authenticated_answers
from alphonse.agent_v2.core.memory.retry_context import (
    prepare_retry_continuity, retry_continuity_context, retry_source_unresolved,
)
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection


def receipt(task, number, question, answer):
    return {'evidence_ref': 'question-answer:' + number, 'task_id': task.task_id,
            'tool_id': 'native.ask_question', 'kind': 'authenticated_question_answer', 'status': 'success',
            'result': {'question_id': number, 'question': question, 'answer': {'text': answer}, 'answered_by': task.user}}


def setup(answer='Use fixture invoice INV-17, amount 42, date 2026-10-30.'):
    task = TaskState(task_id='fixture-task', user='fixture-owner', goal='Remember that invoice and update its record')
    task.metadata['v3_retry_continuity'] = {'status': 'ambiguous', 'source_ids': [], 'request_signature': 'old'}
    task.metadata['v3_resumption_evidence'] = [receipt(task, 'q1', 'Which invoice and exact values?', answer)]
    return task


def judgment(resolution='resolved', refs=None, missing=None):
    return {'resolution': resolution, 'supporting_answer_refs': refs if refs is not None else ['question-answer:q1'],
            'remaining_information': missing or [], 'reason': 'The current answer supplies the requested invoice values.'}


def context_with(value):
    calls = []
    def infer(request):
        calls.append(request)
        assert request.tools == () and request.metadata['reasoning_stage'] == 'clarification_resolution'
        return SimpleNamespace(json_value=deepcopy(value))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=infer),
        system_one=SimpleNamespace(judge_recall=lambda **kw: pytest.fail('Jev must not overrule resolved current input')))
    return context, calls


@pytest.mark.parametrize('answer', [
    'Use fixture invoice INV-17, amount 42, date 2026-10-30.',
    'The event is Fixture Lunch, 12:30 on 2026-11-03 in Room A.',
    'The prescription source is 2026-10-30, Test Child, FixtureBeta. Preserve the uncertain topical name.',
])
def test_system_two_can_resolve_current_answers_across_domains_and_restart(answer):
    task = setup(answer); context, calls = context_with(judgment())
    prepare_retry_continuity(task, context)
    assert task.metadata['v3_retry_continuity']['status'] == 'current_input'
    assert not retry_source_unresolved(task) and answer in retry_continuity_context(task, context)
    assert 'question-answer:q1' in str(calls[0].input_sources)
    assert '"answer_ref": "question-answer:q1"' in calls[0].prompt
    assert task.acceptance_contract == {} and not task.hierarchical_state
    restored = TaskState.from_dict(task.to_checkpoint_dict())
    prepare_retry_continuity(restored, context)
    assert len(calls) == 1 and answer in retry_continuity_context(restored, context)


@pytest.mark.parametrize('answer,missing', [('The same invoice as last time', 'Exact source values are missing'),
    ('Either amount 42 or 84, I am unsure', 'Which amount is intended?'), ('Thank you', 'Invoice and values remain missing')])
def test_partial_conflicting_or_nonresponsive_answer_requires_specific_information(answer, missing):
    task = setup(answer); context, _ = context_with(judgment('requires_clarification', missing=[missing]))
    prepare_retry_continuity(task, context)
    assert retry_source_unresolved(task)
    assert missing in retry_continuity_context(task, context)


def test_current_answer_can_still_require_history_without_silently_clearing_gap():
    task = setup('Use the invoice I supplied yesterday')
    context, calls = context_with(judgment('requires_history', missing=['Original invoice contents']))
    seen = []
    context.system_one.judge_recall = lambda **kw: seen.append(kw) or True
    context.system_one.select_retry_context = lambda **kw: ()
    context.conversation_store = SimpleNamespace(list_recent=lambda **kw: [], read=lambda **kw: None)
    prepare_retry_continuity(task, context)
    assert len(calls) == len(seen) == 1 and retry_source_unresolved(task)


@pytest.mark.parametrize('change', [
    lambda value: value.update(supporting_answer_refs=['invented-ref']),
    lambda value: value.update(supporting_answer_refs=[]),
    lambda value: value.update(remaining_information=['Missing amount']),
    lambda value: value.update(resolution='probably'),
])
def test_invalid_system_two_assessment_cannot_clear_blocker(change):
    task = setup(); value = judgment(); change(value); context, _ = context_with(value)
    prepare_retry_continuity(task, context)
    assert retry_source_unresolved(task) and task.metadata['v3_retry_continuity']['status'] == 'unavailable'


@pytest.mark.parametrize('mutation', [
    lambda r: r.update(task_id='other-task'),
    lambda r: r['result'].update(answered_by='another-person'),
    lambda r: r.update(kind='historical_quote'),
    lambda r: r.update(status='waiting'),
    lambda r: r.update(evidence_ref='wrong-ref'),
    lambda r: r.update(result=[]),
])
def test_untrusted_or_other_task_answers_are_not_current_user_evidence(mutation):
    task = setup(); mutation(task.metadata['v3_resumption_evidence'][0])
    assert not authenticated_answers(task)


def test_later_correction_invalidates_assessment_and_includes_both_questions():
    task = setup(); context, calls = context_with(judgment()); prepare_retry_continuity(task, context)
    task.metadata['v3_resumption_evidence'].append(receipt(task, 'q2', 'Confirm amount?', 'Correction: amount 84'))
    assert retry_source_unresolved(task)
    assert 'amount 42' not in retry_continuity_context(task, context)
    context.inference.generate_json = lambda request: calls.append(request) or SimpleNamespace(
        json_value=judgment(refs=['question-answer:q1', 'question-answer:q2']))
    prepare_retry_continuity(task, context)
    assert not retry_source_unresolved(task) and 'Correction: amount 84' in calls[-1].prompt
    assert 'Which invoice and exact values?' in calls[-1].prompt


def test_complete_answer_releases_planner_without_bypassing_scope_validation():
    task = setup(); context, calls = context_with(judgment()); prepare_retry_continuity(task, context)
    phase = dict(schema_version=3, phase_id='update', objective='Update fixture and read back',
        acceptance_criteria=['Requested invoice values are saved and read back'], criterion_ids=['ac-1'],
        authorized_capabilities=['local_shell'], mutation_scope={'allowed_paths':['fixture.sqlite'],'allow_external_effects':False},
        subgoals=[dict(subgoal_id='write', objective='Update fixture and verify', required_output_type='verified_record',
            allowed_capabilities=['local_shell'], allowed_side_effects=['project_mutation'], completion={'kind':'output_present'})])
    context.inference.generate_json = lambda request: SimpleNamespace(json_value=deepcopy(phase))
    context.system_one.select_plan_tools = lambda **kw: SystemOneToolRegistrySelection()
    result = plan_phase(task, context)
    assert result.phase_id == 'update' and result.mutation_scope.allowed_paths == ('fixture.sqlite',)
    assert not task.hierarchical_state  # Planning did not execute an update.


def test_oversized_current_answer_is_withheld_without_a_provider_call():
    task = setup('X' * 25000); context, calls = context_with(judgment())
    prepare_retry_continuity(task, context)
    assert retry_source_unresolved(task) and not calls


def test_provider_failure_keeps_blocker_without_falling_back_to_jev():
    task = setup(); context, _ = context_with(judgment())
    def fail(request):
        raise RuntimeError('fixture_service_unavailable')
    context.inference.generate_json = fail
    prepare_retry_continuity(task, context)
    assert retry_source_unresolved(task)


def test_cancellation_after_assessment_cannot_clear_blocker():
    from alphonse.agent_v2.system_one import SystemOneUnavailableError
    task = setup(); context, _ = context_with(judgment()); cancelled = [False]
    context.cancellation_checker = lambda: cancelled[0]
    def infer(request):
        cancelled[0] = True
        return SimpleNamespace(json_value=judgment())
    context.inference.generate_json = infer
    with pytest.raises(SystemOneUnavailableError, match='cancelled'):
        prepare_retry_continuity(task, context)
    assert task.metadata['v3_retry_continuity']['status'] != 'current_input'


def test_cached_assessment_cannot_render_or_execute_with_invalid_answer_reference():
    task = setup(); context, _ = context_with(judgment()); prepare_retry_continuity(task, context)
    task.metadata['v3_retry_continuity']['answer_refs'] = ['question-answer:unavailable']
    rendered = retry_continuity_context(task, context)
    assert 'INV-17' not in rendered and retry_source_unresolved(task)
