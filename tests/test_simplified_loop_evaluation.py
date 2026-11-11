from pathlib import Path
import json
import hashlib

import pytest

from alphonse.agent_v2.evaluation.simplified_loop import AdapterUnavailable, Run, compare, load_frozen

CORPUS = Path(__file__).parent / 'fixtures/simplified_loop/representative.json'
HOLDOUT = Path(__file__).parent / 'fixtures/simplified_loop/holdout.json'


def frozen(tmp_path, cases):
    target = tmp_path / 'cases.json'
    raw = json.dumps({'schema_version': 1, 'split': 'fixture', 'cases': cases}).encode()
    target.write_bytes(raw)
    target.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest())
    return target


def case(**oracle):
    return {'id': 'one', 'goal': 'fixture', 'fixture': {'files': {'record.txt': 'old'}},
            'provider_script': [{'fixture': 'no network'}], 'oracle': oracle}


class FixtureAdapter:
    offline = True
    revision = 'fixture-v1'

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, case, root, emit):
        return self.fn(case, root, emit)


def test_corpora_are_frozen_and_disjoint():
    development, _ = load_frozen(CORPUS)
    holdout, _ = load_frozen(HOLDOUT)
    assert len(development['cases']) >= 10
    assert set(c['id'] for c in development['cases']).isdisjoint(c['id'] for c in holdout['cases'])
    assert all(c['provider_script'] and 'permissions' in c['fixture'] and c['oracle'] for c in development['cases'])


def test_frozen_integrity_is_enforced(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['completed'])])
    path.write_text('{}')
    with pytest.raises(ValueError, match='frozen_corpus_digest_mismatch'):
        load_frozen(path)


def test_paired_runs_have_isolated_state_and_safe_failure_is_correct(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['failed'], effects={'record.txt': 'new'}, max_questions=0)])
    roots = []
    def run(c, root, emit):
        roots.append(root)
        assert 'oracle' not in c
        assert (root / 'record.txt').read_text() == 'old'
        c['fixture']['permissions'] = ['changed locally']
        (root / 'record.txt').write_text('new')
        return Run(status='failed', measured_kinds=('question', 'tool'))
    adapter = FixtureAdapter(run)
    report = compare(path, baseline=adapter, candidate=adapter)
    row = report['cases'][0]
    assert row['baseline']['passed'] is True and row['candidate']['passed'] is True
    assert len(set(roots)) == 2 and all(not root.exists() for root in roots)
    assert row['baseline']['oracle']['false_success'] is False
    assert report['simulation'] is True


def test_oracle_detects_false_success_and_preserves_partial_usage(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['failed'], effects={}, forbidden_tools=['write'], max_questions=0)])
    def run(c, root, emit):
        emit({'kind':'system_two', 'provider_usage': {'input_tokens': 7, 'output_tokens': 2},
              'estimated_usage': {'input_tokens': 9, 'output_tokens': 3}})
        emit({'kind':'jev', 'estimated_usage': {'input_tokens': 4, 'output_tokens': 1}})
        emit({'kind':'tool', 'tool':'write', 'arguments':{'value':'x'}})
        emit({'kind':'tool', 'tool':'write', 'arguments':{'value':'x'}})
        emit({'kind':'question'})
        (root / 'record.txt').write_text('unauthorized')
        return Run(status='completed', measured_kinds=('system_two', 'jev', 'tool', 'question'))
    row = compare(path, baseline=FixtureAdapter(run), candidate=None)['cases'][0]
    result = row['baseline']
    assert result['passed'] is False and result['oracle']['false_success'] is True
    assert result['oracle']['checks']['effects'] is False
    assert result['oracle']['unnecessary_questions'] == 1
    assert result['metrics']['repeated_actions'] == 1
    assert result['metrics']['provider_input_tokens'] is None
    assert result['metrics']['estimated_input_tokens'] == 13
    assert row['candidate']['availability'] == 'unavailable' and row['candidate']['passed'] is None


def test_unsupported_and_exception_runs_are_not_hidden(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['failed'])])
    def unavailable(c, root, emit):
        raise AdapterUnavailable('not yet supported')
    def broken(c, root, emit):
        emit({'kind':'tool', 'tool':'read'})
        raise RuntimeError('fixture error')
    row = compare(path, baseline=FixtureAdapter(unavailable), candidate=FixtureAdapter(broken))['cases'][0]
    assert row['baseline']['reason'] == 'not yet supported'
    assert row['candidate']['error'] == 'RuntimeError: fixture error'
    assert row['candidate']['events'][0]['tool'] == 'read'
    assert row['candidate']['passed'] is False


def test_uninstrumented_oracle_checks_do_not_pass(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['completed'], forbidden_tools=['write'], max_questions=0)])
    adapter = FixtureAdapter(lambda c, root, emit: Run(status='completed'))
    run = compare(path, baseline=adapter, candidate=None)['cases'][0]['baseline']
    assert run['passed'] is False
    assert run['oracle']['checks']['questions'] is None
    assert run['oracle']['checks']['forbidden_tools'] is None
    assert run['metrics']['jev_calls'] is None


def test_safe_failure_requires_correct_diagnostic_and_measured_denied_access(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['failed'], reason_contains=['denied'],
                                  forbidden_accesses=['record.txt'])])
    def run(reason, measured, read=False):
        def invoke(c, root, emit):
            if read:
                emit({'kind': 'source_read', 'source': 'record.txt'})
            return Run(status='failed', reason=reason, measured_kinds=measured)
        return compare(path, baseline=FixtureAdapter(invoke), candidate=None)['cases'][0]['baseline']
    assert run('denied by fixture policy', ('source_read',))['passed'] is True
    assert run('unrelated runtime error', ('source_read',))['passed'] is False
    assert run('denied by fixture policy', ())['oracle']['checks']['forbidden_accesses'] is None
    assert run('denied by fixture policy', ('source_read',), read=True)['passed'] is False


def test_adapter_cannot_hide_activity_or_effects_behind_unavailable(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['blocked'], effects={})])
    def attempted(c, root, emit):
        (root / 'record.txt').write_text('unintended')
        raise AdapterUnavailable('fixture unsupported')
    result = compare(path, baseline=FixtureAdapter(attempted), candidate=None)['cases'][0]['baseline']
    assert result['availability'] == 'available'
    assert result['passed'] is False
    assert result['oracle']['observed_effects'] == {'record.txt': 'unintended'}
    assert 'after activity' in result['error']


def test_restart_oracle_requires_instrumentation_and_exact_reads(tmp_path):
    path = frozen(tmp_path, [case(allowed_statuses=['blocked'], source_read_counts={'record.txt': 1}, restart_count=1)])
    def fixture(c, root, emit):
        emit({'kind': 'source_read', 'source': 'record.txt'})
        emit({'kind': 'source_read', 'source': 'record.txt'})
        emit({'kind': 'restart'})
        return Run(status='blocked', measured_kinds=('source_read', 'restart'))
    result = compare(path, baseline=FixtureAdapter(fixture), candidate=None)['cases'][0]['baseline']
    assert result['passed'] is False
    assert result['oracle']['checks']['source_read_counts'] is False
    assert result['oracle']['checks']['restarts'] is True
    unknown = FixtureAdapter(lambda c, root, emit: Run(status='blocked'))
    missing = compare(path, baseline=unknown, candidate=None)['cases'][0]['baseline']
    assert missing['passed'] is False
    assert missing['oracle']['checks']['source_read_counts'] is None
    assert missing['oracle']['checks']['restarts'] is None


def test_baseline_runs_real_v3_processor_and_exposes_unsupported_cases(monkeypatch):
    from alphonse.agent_v2.evaluation.simplified_loop import ExistingV3ReplayAdapter
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    original = HierarchicalCAPDProcessor.process
    invoked = []
    def observed(self, task, context):
        invoked.append(task.goal)
        return original(self, task, context)
    monkeypatch.setattr(HierarchicalCAPDProcessor, 'process', observed)
    report = compare(CORPUS, baseline=ExistingV3ReplayAdapter(), candidate=None)
    row = next(item for item in report['cases'] if item['case_id'] == 'solar-v3-baseline')
    assert invoked == ['Mark the solar-cell project complete']
    assert row['baseline']['availability'] == 'available'
    assert row['baseline']['status'] in {'completed', 'failed', 'cancelled', 'waiting_user'}
    assert row['baseline']['oracle']['observed_effects'] == {'mejoras_hogar/backlog.md': 'Solar: complete\n'}
    assert row['baseline']['metrics']['system_two_calls'] > 0
    assert row['baseline']['metrics']['jev_calls'] > 0
    assert row['baseline']['metrics']['tool_calls'] is None
    assert row['candidate']['availability'] == 'unavailable'
    assert next(item for item in report['cases'] if item['case_id'] == 'gmail_es')['baseline']['availability'] == 'unavailable'
