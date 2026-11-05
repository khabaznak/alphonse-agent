"""Held-out harness tests are deterministic diagnostics, never measured trials."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation.heldout_memory import load_heldout, assess, CATEGORIES, fraction
from alphonse.agent_v2.evaluation.paired_fixture import BoundaryFixture
from alphonse.agent_v2.evaluation.paired_memory import freeze, validate_manifest, report
from alphonse.agent_v2.evaluation.paired_oracles import score_outcome
from alphonse.agent_v2.evaluation.memory_rollback import drill

CORPUS = load_heldout()


def test_frozen_coverage_and_equal_predeclared_pairs():
    manifest = freeze({}, corpus_kind='heldout')
    validate_manifest(manifest, {})
    assert len(CORPUS['cases']) == 100
    assert {c['category'] for c in CORPUS['cases']} == CATEGORIES
    assert len(manifest['schedule']) == 4200
    assert all(sum(t['case_id'] == c['id'] and t['arm'] == arm and not t['warmup'] for t in manifest['schedule']) == 20
               for c in CORPUS['cases'] for arm in ('legacy','candidate'))
    assert all(c['state_oracle'].get('schedule') or c['state_oracle']['required_reads'] for c in CORPUS['cases'])
    assert report(manifest, [])['gates']['DC-5'] == 'invalid'


@pytest.mark.parametrize('case', CORPUS['cases'], ids=lambda c: c['id'])
def test_fixture_sources_round_trip_and_success_sentence_cannot_replace_effects(tmp_path, case):
    fixture = BoundaryFixture(tmp_path,case,CORPUS,SimpleNamespace(calls=[]),None)
    state = fixture.readback()
    score = score_outcome(case,state,'I completed every requested action.', 'completed')
    assert score['checks']['fixture_sources_preserved']
    assert not score['effect_pass']
    if case['sources']:
        state['source_readback'].pop(case['sources'][0]['id'])
        assert not score_outcome(case,state,'Done','completed')['checks']['fixture_sources_preserved']


def test_sparse_or_missing_counts_are_not_perfect_recall():
    result = assess(freeze({},corpus_kind='heldout',repetitions=1),[])
    assert result['arms']['candidate']['source_recall']['rate'] is None
    assert result['arms']['candidate']['task_outcome_diagnostic']['passed'] == 0
    assert result['performance_assessment'] == 'unassessed'
    assert fraction(0,0)['wilson95'] is None
    assert fraction(98,100)['wilson95'][0] < .98


def synthetic_rows(manifest):
    cases = {c['id']: c for c in manifest['corpus']['cases']}
    return [{**trial,'manifest_sha256':manifest['manifest_sha256'],'status':'returned',
        'outcome':{'effect_pass':True,'response_diagnostic_pass':True},
        'delivery':{'delivery_pass':True},
        'latency_ms':100,
        'model_calls':[], 'jev_calls':[], 'tool_calls':[],
        'decisions':[{'purpose':purpose,'stages':[purpose],'delivered':[{'source_id':o['source_id'],
            'start_char':o['start_char'],'end_char':o['end_char']} for o in cases[trial['case_id']]['delivery_oracles']]}
            for purpose in ('phase_planning','tactical_action','response_composition')]}
        for trial in manifest['schedule']]


def test_mechanical_success_keeps_semantic_review_and_rollout_unassessed():
    manifest = freeze({},corpus_kind='heldout',repetitions=1)
    result = report(manifest,synthetic_rows(manifest))
    assert result['gates']['DC-5'] == 'unassessed'
    assert result['gates']['DC-2'] == 'unassessed'
    assert result['heldout']['arms']['candidate']['task_outcome_diagnostic']['passed'] == 100
    assert not result['rollout_accepted']
    assert result['heldout']['semantic_review'] == 'unassessed'


def test_task_requires_every_repetition_and_category_recall_cannot_hide_in_pool():
    manifest = freeze({},corpus_kind='heldout',repetitions=2)
    rows = synthetic_rows(manifest)
    target = next(r for r in rows if not r['warmup'] and r['arm']=='candidate' and 'explicit_recall' in r['case_id'])
    target['decisions'] = []
    result = report(manifest, rows)
    assert result['gates']['DC-5'] == 'fail'
    assert 'recall:explicit_recall' in result['heldout']['threshold_breaches']
    # Task outcome is a verified-state diagnostic; source recall is independently scored.
    target['outcome']['effect_pass'] = False
    result = report(manifest, rows)
    assert result['heldout']['arms']['candidate']['task_outcome_diagnostic']['passed'] == 99


def test_slow_case_and_excess_calls_are_retained():
    manifest = freeze({},corpus_kind='heldout')
    rows = synthetic_rows(manifest)
    for row in rows:
        if row['arm']=='candidate' and row['case_id']==CORPUS['cases'][0]['id']:
            row['latency_ms']=200
            row['model_calls']=[{'status':'failed'}]
    result = report(manifest,rows)
    assert 'latency:'+CORPUS['cases'][0]['id'] in result['heldout']['threshold_breaches']
    assert 'total_model_calls' in result['heldout']['threshold_breaches']
    assert result['gates']['DC-5']=='fail'


def test_smoke_cannot_qualify_heldout():
    manifest=freeze({},corpus_kind='heldout',smoke_cases=[CORPUS['cases'][0]['id']])
    assert report(manifest,synthetic_rows(manifest))['gates']['DC-5']=='unassessed'


def test_subprocess_crash_restart_flag_off_on_same_stores():
    result=drill()
    assert result['result']=='pass' and result['DC-5']=='unassessed'
    assert [o['resume_before_off'] for o in result['observations']]==[False,True]
    for observation in result['observations']:
        assert all(observation['checks'].values())
        assert observation['checkpoints'][0]['returncode']==73
        assert observation['checkpoints'][-1]['returncode']==0
        assert len(observation['checkpoints'][-1]['readback']['effects'])==2


def test_missing_instrumentation_is_unavailable_and_partial_trials_invalid():
    manifest=freeze({},corpus_kind='heldout')
    rows=synthetic_rows(manifest)
    target=next(r for r in rows if not r['warmup'] and r['arm']=='candidate')
    target.pop('latency_ms')
    target.pop('model_calls')
    result=report(manifest,rows)
    assert result['heldout']['performance_assessment']=='unassessed'
    assert result['heldout']['arms']['candidate']['calls']['model_calls'] is None
    assert report(manifest,rows[:-1])['gates']['DC-5']=='invalid'
    assert report(manifest,rows+[rows[0]])['gates']['DC-5']=='invalid'


@pytest.mark.parametrize('field', ['duplicate_source_ids','forbidden_source_delivery'])
def test_hard_source_violation_fails_even_in_legacy_arm_and_partial_run(field):
    manifest=freeze({},corpus_kind='heldout')
    row=synthetic_rows(manifest)[0]
    row['delivery'][field]=['synthetic-violation']
    assert report(manifest,[row])['gates']['DC-5']=='fail'


@pytest.mark.parametrize('transition',['delete','revoke'])
def test_boundary_access_transition_readback_remains_available(tmp_path,transition):
    from alphonse.agent_v2.evaluation.paired_memory import load_boundaries
    corpus=load_boundaries()
    case=next(c for c in corpus['cases'] if c['id']==transition+'-between-decisions')
    fixture=BoundaryFixture(tmp_path,case,corpus,SimpleNamespace(calls=[]),None)
    fixture._transition()
    assert 'source_readback' not in fixture.readback()
