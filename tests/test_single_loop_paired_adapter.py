"""One paired synthetic case, not an autonomous/live model qualification."""
from pathlib import Path

from alphonse.agent_v2.evaluation.candidate_comparison import SingleLoopFixtureAdapter, V3ReadFixtureAdapter
from alphonse.agent_v2.evaluation.simplified_loop import compare

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only.json'
EXPANDED = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only_v2.json'
HOLDOUT = Path(__file__).parent / 'fixtures/single_loop_comparison/heldout_read_only_v1.json'
LIFECYCLE = Path(__file__).parent / 'fixtures/single_loop_comparison/lifecycle_v1.json'
LIFECYCLE_V2 = Path(__file__).parent / 'fixtures/single_loop_comparison/lifecycle_v2.json'
V3_SEEDED = Path(__file__).parent / 'fixtures/single_loop_comparison/v3_seeded_resume_v1.json'


def test_real_prototype_and_v3_run_same_frozen_read_only_fixture(monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.intelligence.v3 import single_loop_candidate
    invoked = []
    real_v3 = HierarchicalCAPDProcessor.process
    real_candidate = single_loop_candidate.run_candidate
    def v3(self, task, context):
        invoked.append('v3')
        return real_v3(self, task, context)
    def candidate(state, host):
        invoked.append('candidate')
        return real_candidate(state, host)
    monkeypatch.setattr(HierarchicalCAPDProcessor, 'process', v3)
    monkeypatch.setattr(single_loop_candidate, 'run_candidate', candidate)
    report = compare(CORPUS, baseline=V3ReadFixtureAdapter(), candidate=SingleLoopFixtureAdapter())
    row, unsupported = report['cases']
    assert invoked == ['v3', 'candidate']
    assert report['simulation'] is True
    for side in ('baseline', 'candidate'):
        result = row[side]
        assert result['availability'] == 'available'
        assert result['status'] == 'completed' and result['passed'] is True
        assert result['oracle']['observed_effects'] == {}
        assert 'amoxicillin' in result['response']
        assert result['metrics']['provider_input_tokens'] is None
        assert result['metrics']['provider_output_tokens'] is None
        assert result['metrics']['system_two_calls'] > 0
        assert result['metrics']['tool_calls'] > 0
        assert result['metrics']['questions'] == 0
        assert 'scripted-fixture' in result['revision']
        assert unsupported[side]['availability'] == 'unavailable'
        assert unsupported[side]['passed'] is None
    assert row['candidate']['metrics']['jev_calls'] == 0
    assert row['baseline']['metrics']['jev_calls'] > 0


def test_real_engines_retained_denial_and_cancellation_failures():
    report = compare(EXPANDED, baseline=V3ReadFixtureAdapter(), candidate=SingleLoopFixtureAdapter())
    normal, denied, cancelled = report['cases']
    assert normal['baseline']['passed'] is True and normal['candidate']['passed'] is True
    for row in (denied, cancelled):
        assert row['candidate']['passed'] is True
        for side in ('baseline', 'candidate'):
            assert row[side]['oracle']['checks']['forbidden_accesses'] is True
            assert row[side]['oracle']['observed_effects'] == {}
            assert 'amoxicillin' not in row[side]['response']
    assert denied['baseline']['reason'].startswith('tactical_planning_error_repeated')
    assert denied['baseline']['passed'] is False  # The one-action provider cannot plan a fallback.
    assert denied['candidate']['reason'] == 'candidate_read_denied'
    assert cancelled['baseline']['reason'] == 'Task cancelled.'
    assert cancelled['baseline']['passed'] is True
    assert cancelled['candidate']['reason'] == 'cancelled'


def test_heldout_note_uses_new_content_and_frozen_source():
    report = compare(HOLDOUT, baseline=V3ReadFixtureAdapter(), candidate=SingleLoopFixtureAdapter())
    assert report['split'] == 'holdout'
    row = report['cases'][0]
    for side in ('baseline', 'candidate'):
        assert row[side]['passed'] is True
        assert 'cetirizine' in row[side]['response']
        assert row[side]['oracle']['observed_effects'] == {}
        assert row[side]['metrics']['provider_input_tokens'] is None


def test_restart_after_read_before_evidence_does_not_replay_or_claim_success():
    report = compare(LIFECYCLE, baseline=V3ReadFixtureAdapter(), candidate=SingleLoopFixtureAdapter())
    row = report['cases'][0]
    assert row['baseline']['availability'] == 'unavailable'
    assert row['baseline']['passed'] is None
    candidate = row['candidate']
    assert candidate['passed'] is True and candidate['status'] == 'blocked'
    assert candidate['reason'] == 'candidate_unresolved_inflight'
    assert candidate['oracle']['source_read_counts'] == {'Receta.md': 1}
    assert candidate['oracle']['restart_count'] == 1
    assert candidate['oracle']['observed_effects'] == {}
    assert candidate['response'] == ''


def test_missing_information_does_not_claim_question_was_delivered():
    report = compare(LIFECYCLE_V2, baseline=V3ReadFixtureAdapter(), candidate=SingleLoopFixtureAdapter())
    restart, question = report['cases']
    assert restart['candidate']['passed'] is True
    assert question['baseline']['availability'] == 'unavailable'
    assert question['candidate']['passed'] is True
    assert question['candidate']['status'] == 'waiting_user'
    assert question['candidate']['reason'] == 'question_delivery_not_integrated'
    assert question['candidate']['metrics']['questions'] == 1  # attempted, not delivered
    assert question['candidate']['oracle']['source_read_counts'] == {}


def test_v3_seeded_resume_records_verified_file_effect_without_repeating_lookup(monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.evaluation.seeded_v3_resume import V3SeededResumeAdapter
    calls = []
    real_process = HierarchicalCAPDProcessor.process
    def observed(self, task, context):
        calls.append(task.task_id)
        return real_process(self, task, context)
    monkeypatch.setattr(HierarchicalCAPDProcessor, 'process', observed)
    row = compare(V3_SEEDED, baseline=V3SeededResumeAdapter(), candidate=None)['cases'][0]
    baseline = row['baseline']
    assert calls == ['v3-resume-fixture']
    assert baseline['passed'] is True and baseline['status'] == 'completed'
    assert baseline['oracle']['observed_effects'] == {'records/item.md': 'Status: complete\n'}
    assert baseline['oracle']['forbidden_tools'] == []
    assert baseline['oracle']['restart_count'] == 1
    assert baseline['oracle']['source_read_counts'] == {'records/item.md': 1}
    assert baseline['metrics']['jev_calls'] is None
    assert row['candidate']['availability'] == 'unavailable' and row['candidate']['passed'] is None


def test_seeded_resume_cli_cannot_present_unavailable_candidate_as_success(tmp_path):
    import json
    from alphonse.agent_v2.evaluation.simplified_loop import main
    receipt = tmp_path / 'seeded.json'
    assert main([str(V3_SEEDED), str(receipt), '--baseline-v3-seeded-resume']) == 1
    row = json.loads(receipt.read_text())['cases'][0]
    assert row['baseline']['passed'] is True
    assert row['candidate']['availability'] == 'unavailable'
    assert row['candidate']['passed'] is None


def test_paired_cli_retains_failed_runs_and_refuses_overwrite(tmp_path):
    import json
    import pytest
    from alphonse.agent_v2.evaluation.simplified_loop import main
    output = tmp_path / 'receipt.json'
    assert main([str(EXPANDED), str(output), '--paired-read-only']) == 1
    receipt = json.loads(output.read_text())
    assert receipt['cases'][1]['baseline']['passed'] is False
    assert receipt['cases'][1]['candidate']['passed'] is True
    with pytest.raises(FileExistsError):
        main([str(EXPANDED), str(output), '--paired-read-only'])


def test_cancelled_processor_exception_is_not_a_failure_or_leaked(monkeypatch):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    def interrupted(*args):
        raise ValueError('sensitive fixture detail')
    monkeypatch.setattr(HierarchicalCAPDProcessor, '_process', interrupted)
    task = TaskState(task_id='cancelled-fixture', user='fixture-user')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), cancellation_checker=lambda: True)
    result = HierarchicalCAPDProcessor().process(task, context)
    snapshot = result.snapshot.metadata['task_state']
    assert snapshot['status'] == 'cancelled'
    assert snapshot['outcome'] == {'status': 'cancelled', 'reason': 'Task cancelled.'}
    assert 'v3_terminal_error_type' not in snapshot['metadata']
    assert 'sensitive fixture detail' not in str(snapshot)


def test_uncancelled_processor_exception_remains_a_sanitized_failure(monkeypatch):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    def broken(*args):
        raise ValueError('sensitive fixture detail')
    monkeypatch.setattr(HierarchicalCAPDProcessor, '_process', broken)
    task = TaskState(task_id='failed-fixture', user='fixture-user')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), cancellation_checker=lambda: False)
    snapshot = HierarchicalCAPDProcessor().process(task, context).snapshot.metadata['task_state']
    assert snapshot['status'] == 'failed'
    assert snapshot['outcome']['reason'] == 'v3_runtime_error:ValueError'
    assert snapshot['metadata']['v3_terminal_error_type'] == 'ValueError'
    assert 'sensitive fixture detail' not in str(snapshot)
