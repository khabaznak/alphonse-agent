"""Harness qualification tests; scripted providers here are never live evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.evaluation.paired_fixture import BoundaryFixture, FixturePrivacyViolation
from alphonse.agent_v2.evaluation.paired_memory import load_boundaries, freeze, validate_manifest, report, execute
from alphonse.agent_v2.evaluation.paired_oracles import delivered_ranges, score_delivery, score_outcome


CORPUS = load_boundaries()


def fixture(tmp_path, case_id='explicit-recall'):
    case = next(c for c in CORPUS['cases'] if c['id'] == case_id)
    return BoundaryFixture(tmp_path, case, CORPUS, SimpleNamespace(calls=[]), None)


@pytest.mark.parametrize('case', CORPUS['cases'], ids=lambda c: c['id'])
def test_every_case_builds_only_disposable_stores(tmp_path, case, monkeypatch):
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', '/does-not-exist/household.sqlite')
    f = BoundaryFixture(tmp_path, case, CORPUS, SimpleNamespace(calls=[]), None)
    assert f.context.memory.knowledge.extractor is None
    assert all(not t.tool_id.startswith('artifact.') for t in f.registry.list())
    assert {t.tool_id for t in f.registry.list()} <= {'native.search_memory', 'native.read_memory_source', 'native.respond', 'native.ask_question', 'native.scheduled_task', 'native.bash'}
    assert f.context.project_store is None or case.get('scenario') == 'skill'


def test_closed_shell_and_fixture_state_readback(tmp_path):
    f = fixture(tmp_path, 'current-state')
    invoke = ToolInvocationService(context=f.context, task=f.task)
    with pytest.raises(PermissionError):
        invoke.execute_or_raise('native.bash', {'command': 'cat ~/.ssh/id_ed25519'})
    result = invoke.execute_or_raise('native.bash', {'command': 'fixture-host read sensor.study_temp'})
    assert json.loads(result['stdout'])['temperature_c'] == 24
    assert f.readback()['reads'] == ['sensor.study_temp']
    assert f.tool_trace[-1]['status'] == 'returned'
    assert not score_outcome(f.case, f.readback(), 'read', 'completed')['effect_pass']


@pytest.mark.parametrize('transition', ['delete', 'revoke'])
def test_access_changes_between_retrieval_and_next_input(tmp_path, transition):
    f = fixture(tmp_path, transition + '-between-decisions')
    invoke = ToolInvocationService(context=f.context, task=f.task)
    result = invoke.execute_or_raise('native.search_memory', {'query': 'meeting room'})
    assert result['results'] and f.transitioned
    with pytest.raises(FixturePrivacyViolation):
        f.guard({'prompt': 'The shared meeting room is Cedar.'})
    result = invoke.execute_or_raise('native.read_memory_source', {'source_id': f.sources['shared']['runtime_id']})
    assert result['status'] == 'not_found'


def test_privacy_guard_precedes_provider(tmp_path):
    f = fixture(tmp_path)
    with pytest.raises(FixturePrivacyViolation):
        f.guard({'prompt': next(s['content'] for s in f.case['sources'] if s['id'] == 'private-b')})
    assert f.context.inference.provider.calls == []


def test_source_id_and_qualifying_range_must_reach_exact_decision():
    sources = {'long': {'runtime_id': 'conversation:123', 'content': 'Header. Qualifying clause.'}}
    prefix = {'source_id': 'conversation:123', 'excerpt': 'Header.', 'start_char': 0, 'end_char': 7}
    ranges = delivered_ranges({'prompt': json.dumps(prefix)}, {}, sources)
    assert ranges == [{'source_id': 'long', 'start_char': 0, 'end_char': 7}]
    case = {'delivery_oracles': [{'source_id': 'long', 'start_char': 8, 'end_char': 26, 'purposes': ['final_response']}], 'necessary_by_purpose': {'final_response': ['long']}}
    decisions = [{'purpose': 'final_response', 'delivered': ranges}]
    assert not score_delivery(case, decisions)['necessary_delivery_pass']
    full = delivered_ranges({'prompt': json.dumps({'source_id': 'conversation:123', 'content': sources['long']['content']})}, {}, sources)
    assert not score_delivery(case, decisions + [{'purpose': 'tactical_action', 'delivered': full}])['necessary_delivery_pass']
    assert score_delivery(case, [{'purpose': 'final_response', 'delivered': full}])['necessary_delivery_pass']
    assert not score_delivery(case, [])['necessary_delivery_pass']


def test_forged_id_range_and_broken_clipped_envelope_get_no_credit():
    sources = {'s': {'runtime_id': 'conversation:123', 'content': 'Header. Clause.'}}
    for value in ({'source_id': 'wrong', 'content': 'Header. Clause.'},
                  {'source_id': 'conversation:123', 'excerpt': 'Header.', 'start_char': 8, 'end_char': 15}):
        assert delivered_ranges({'prompt': json.dumps(value)}, {}, sources) == []
    assert delivered_ranges({'prompt': '{"source_id":"conversation:123","content":"Header.'}, {}, sources) == []


def test_duplicate_instances_count_once_per_decision():
    case = {'delivery_oracles': [], 'necessary_by_purpose': {'final_response': ['a']}}
    row = {'purpose': 'final_response', 'delivered': [{'source_id': 'a'}, {'source_id': 'a'}, {'source_id': 'b'}]}
    score = score_delivery(case, [row, row])
    assert (score['historical_instances'], score['irrelevant_instances']) == (4, 2)


def test_manifest_freezes_all_originals_and_order_before_results():
    manifest = freeze({'model': 'test'}, repetitions=2)
    assert len(manifest['corpus']['cases']) == 41
    assert len(manifest['schedule']) == 41 * 3 * 2
    assert [t['arm'] for t in manifest['schedule'][:6]] == ['legacy', 'candidate', 'candidate', 'legacy', 'legacy', 'candidate']
    validate_manifest(manifest, {'model': 'test'})
    with pytest.raises(ValueError, match='configuration_changed'):
        validate_manifest(manifest, {'model': 'other'})
    tampered = deepcopy(manifest)
    tampered['corpus']['cases'].pop()
    with pytest.raises(ValueError, match='hash_mismatch'):
        validate_manifest(tampered, {'model': 'test'})


def test_incomplete_duplicate_and_missing_review_cannot_pass_gates():
    manifest = freeze({}, repetitions=1)
    assert report(manifest, [])['gates']['DC-2'] == 'invalid'
    rows = [{**trial, 'manifest_sha256': manifest['manifest_sha256'], 'status': 'returned',
             'delivery': {'necessary_delivery_pass': True},
             'outcome': {'effect_pass': True, 'response_diagnostic_pass': True}} for trial in manifest['schedule']]
    assert report(manifest, rows)['gates']['DC-2'] == 'unassessed'
    assert report(manifest, rows + [rows[0]])['gates']['DC-0'] == 'invalid'
    rows[1]['runtime_error'] = 'provider_failure'
    assert report(manifest, rows)['gates']['DC-2'] == 'fail'


def test_abort_is_retained_and_next_trial_is_not_a_retry(tmp_path, monkeypatch):
    manifest = freeze({}, repetitions=1)
    # Small test schedule is independently rehashed; never a qualifying corpus.
    manifest['corpus']['cases'] = manifest['corpus']['cases'][:1]
    monkeypatch.setattr('alphonse.agent_v2.evaluation.paired_memory.load_boundaries', lambda: manifest['corpus'])
    from alphonse.agent_v2.evaluation.paired_memory import trial_schedule
    manifest['schedule'] = trial_schedule(manifest['corpus'], manifest['repetitions'])
    from alphonse.agent_v2.evaluation.paired_memory import digest
    manifest['manifest_sha256'] = digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    def unavailable():
        raise OSError('provider unavailable')
    execute(manifest, tmp_path / 'evidence', unavailable, unavailable, configuration={})
    rows = [json.loads(line) for line in (tmp_path / 'evidence/trials.jsonl').read_text().splitlines()]
    assert len(rows) == 4 and all(r['status'] == 'aborted' for r in rows)
    assert [r['arm'] for r in rows] == ['legacy', 'candidate', 'candidate', 'legacy']
    assert json.loads((tmp_path / 'evidence/gates.json').read_text())['gates']['DC-2'] == 'fail'


@pytest.mark.parametrize('candidate', [False, True])
def test_real_admission_and_call_receipts_retain_provider_failure(tmp_path, candidate):
    from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings
    from alphonse.agent_v2.evaluation.paired_fixture import run_trial
    class FailingProvider:
        calls = []
        def generate_json(self, request):
            self.calls.append({'purpose': request.purpose.value, 'status': 'failed'})
            raise OSError('synthetic provider failure')
        generate_markdown = generate_json
        plan_tool_call = generate_json
    def transport(url, key, payload, timeout):
        return {'answers': {key: {'type': 'noul', 'noul': .99} for key in payload['questions']}}
    jev = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key='synthetic'), transport=transport)
    case = next(c for c in CORPUS['cases'] if c['id'] == 'explicit-recall')
    row = run_trial(tmp_path, case, CORPUS, FailingProvider(), jev, candidate=candidate)
    assert row['task_status'] == 'failed'
    assert row['decisions'] and row['jev_calls'] and row['model_calls']
    assert any(d['purpose'] == 'admission' for d in row['decisions'])
    assert any(d['manifest']['status'] == 'failed' for d in row['decisions'])
    assert not row['delivery']['necessary_delivery_pass']


def test_partial_search_and_pagination_preserve_real_native_results(tmp_path):
    f = fixture(tmp_path / 'partial', 'search-partial')
    result = ToolInvocationService(context=f.context, task=f.task).execute_or_raise('native.search_memory', {'query': 'receipt'})
    assert result['status'] == 'partial'
    assert result['coverage']['conversation'] == 'partial'
    f = fixture(tmp_path / 'pages', 'paginated-recall')
    invoke = ToolInvocationService(context=f.context, task=f.task)
    first = invoke.execute_or_raise('native.search_memory', {'query': 'archived filter'})
    assert len(first['results']) == 1 and first['next_offset'] == 1
    assert f.tool_trace[-1]['executed_arguments']['limit'] == 1
    second = invoke.execute_or_raise('native.search_memory', {'query': 'archived filter', 'offset': 1})
    assert first['results'][0]['source_id'] != second['results'][0]['source_id']
    assert 'search_next_page' in f.state['reads']


def test_operating_tail_is_bound_to_actual_selected_guidance(tmp_path):
    from alphonse.agent_v2.evaluation.context_pilot import projection_flag
    from alphonse.agent_v2.core.context_inventory import receipt
    from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
    f = fixture(tmp_path, 'long-artifact')
    with projection_flag(True):
        prompt = task_context_for_decision(f.task, f.context, purpose='execution')
    assert 'PRIVATE_SENTINEL' not in prompt
    manifest = {'projections': [receipt(prompt, input_section='prompt')]}
    ranges = delivered_ranges({'prompt': prompt}, manifest, f.sources)
    assert any(r['source_id'] == 'operating-tail' for r in ranges)
    assert all('artifact.' not in descriptor.tool_id for descriptor in f.registry.list())


def test_overflow_accepts_only_exact_preinference_budget_failure():
    case = next(c for c in CORPUS['cases'] if c['id'] == 'required-overflow')
    state = {'label_writes': 0, 'forbidden_attempts': [], 'reads': [], 'schedules': [],
             'terminal_reason': 'v3_required_context_exceeds_token_budget', 'inference_calls': 0}
    assert score_outcome(case, state, '', 'failed')['effect_pass']
    assert score_outcome(case, state, '', 'failed')['response_diagnostic_pass']
    state['inference_calls'] = 1
    assert not score_outcome(case, state, '', 'failed')['effect_pass']
    state.update(inference_calls=0, terminal_reason='provider_error')
    assert not score_outcome(case, state, '', 'failed')['effect_pass']


def test_zero_history_cannot_hide_behind_aggregate_noise_limit():
    case = next(c for c in CORPUS['cases'] if c['id'] == 'current-state')
    decisions = [{'purpose': 'admission', 'delivered': [{'source_id': 'stale-temperature'}]}]
    assert not score_delivery(case, decisions)['delivery_pass']


def test_acknowledgement_does_not_replace_tool_response_composition(tmp_path):
    from alphonse.agent_v2.core.context_manifests import manifest_scope
    from alphonse.agent_v2.core.inference import InferenceRequest, InferencePurpose, InferenceResult
    class Provider:
        calls = []
        def generate_markdown(self, request):
            return InferenceResult(content='I will check.')
        def generate_json(self, request):
            return InferenceResult(json_value={'tool_id': 'native.respond', 'arguments': {'message': 'FX-204-B'}})
    f = fixture(tmp_path)
    f.context.inference.provider = Provider()
    item = f.sources['filter']
    with f.observe(), manifest_scope(f.task, f.context):
        f.context.inference.generate_markdown(InferenceRequest(prompt='Acknowledge the request.', purpose=InferencePurpose.FINAL_RESPONSE))
        result = f.context.inference.generate_json(InferenceRequest(
            prompt=json.dumps({'source_id': item['runtime_id'], 'content': item['content']}), purpose=InferencePurpose.TACTICAL_ACTION))
        # Merely seeing evidence at another action is insufficient until the action
        # actually composing the answer is identified from native execution.
        assert not score_delivery(f.case, f.decisions)['necessary_delivery_pass']
        ToolInvocationService(context=f.context, task=f.task).execute_or_raise(result.json_value['tool_id'], result.json_value['arguments'])
    assert score_delivery(f.case, f.decisions)['necessary_delivery_pass']
    assert f.decisions[0]['stages'] == ['final_response']
    assert f.decisions[1]['stages'] == ['tactical_action', 'response_composition']


def test_paired_initial_history_has_identical_ingestion_time(tmp_path):
    from alphonse.agent_v2.evaluation.paired_fixture import fixture_clock
    snapshots = []
    for arm in ('legacy', 'candidate'):
        with fixture_clock(CORPUS):
            f = fixture(tmp_path / arm)
        snapshots.append([(i.content, i.source.recorded_at, i.source.observed_at) for i in f.store.recall_items(f.task.user)])
    assert snapshots[0] == snapshots[1]
    assert all('2026-02-11T14:00:00' in row[1] for row in snapshots[0])


def test_smoke_selection_is_frozen_without_replacing_boundary_corpus():
    manifest = freeze({}, smoke_cases=['current-state', 'older-explicit-recall'])
    assert len(manifest['corpus']['cases']) == 41
    assert len(manifest['schedule']) == 4
    assert manifest['experiment_kind'] == 'harness_smoke'
    validate_manifest(manifest, {})
    rows = [{**trial, 'manifest_sha256': manifest['manifest_sha256'], 'status': 'returned',
             'delivery': {'delivery_pass': True}, 'outcome': {'effect_pass': True, 'response_diagnostic_pass': True}}
            for trial in manifest['schedule']]
    assert report(manifest, rows)['gates']['DC-2'] == 'unassessed'
    with pytest.raises(ValueError, match='invalid_smoke_cases'):
        freeze({}, smoke_cases=['unknown'])


def test_interrupt_retains_partial_decisions_and_provider_attempts(tmp_path, monkeypatch):
    from alphonse.agent_v2.evaluation.paired_fixture import run_trial
    from alphonse.agent_v2.core import context_manifests
    def interrupted(self, task, context):
        context.inference.provider.calls.append({'purpose': 'phase_planning', 'status': 'started'})
        from alphonse.agent_v2.core.context_manifests import manifest_scope
        with manifest_scope(task, context):
            context_manifests.begin_decision(purpose='phase_planning', sections={'prompt': task.goal})
            raise KeyboardInterrupt()
    monkeypatch.setattr('alphonse.agent_v2.evaluation.paired_fixture.HierarchicalCAPDProcessor.process', interrupted)
    case = next(c for c in CORPUS['cases'] if c['id'] == 'current-state')
    with pytest.raises(KeyboardInterrupt) as caught:
        run_trial(tmp_path, case, CORPUS, SimpleNamespace(calls=[]), None, candidate=True)
    trace = caught.value.evaluation_trace
    assert trace['runtime_error'] == 'KeyboardInterrupt'
    assert trace['decisions'][0]['purpose'] == 'phase_planning'
    assert trace['model_calls'][0]['status'] == 'started'
    assert not trace['outcome']['effect_pass']


def test_clipping_fixture_cannot_deliver_qualifying_clause_from_recent_history(tmp_path):
    from alphonse.agent_v2.evaluation.context_pilot import projection_flag
    from alphonse.agent_v2.core.context_inventory import receipt
    from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
    from alphonse.agent_v2.evaluation.paired_oracles import covers
    f = fixture(tmp_path, 'clipped-source-expansion')
    oracle = f.case['delivery_oracles'][0]
    with projection_flag(True):
        prompt = task_context_for_decision(f.task, f.context, purpose='execution')
    ranges = delivered_ranges({'prompt': prompt}, {'projections': [receipt(prompt, input_section='prompt')]}, f.sources)
    assert not covers(ranges, oracle['source_id'], oracle['start_char'], oracle['end_char'])
    result = ToolInvocationService(context=f.context, task=f.task).execute_or_raise('native.read_memory_source',
        {'source_id': f.sources['long']['runtime_id'], 'max_chars': 12000})
    expanded = delivered_ranges({'tool_result': result}, {}, f.sources)
    assert covers(expanded, oracle['source_id'], oracle['start_char'], oracle['end_char'])


def test_followup_fixture_requires_a_different_query(tmp_path):
    f = fixture(tmp_path, 'follow-up-query')
    invoke = ToolInvocationService(context=f.context, task=f.task)
    bootstrap = invoke.execute_or_raise('native.search_memory', {'query': f.task.goal})
    identity = f.sources['replacement']['runtime_id']
    assert identity not in {r['source_id'] for r in bootstrap['results']}
    recent = f.store.list_recent(owner_user_id=f.task.user, limit=3)
    assert any(e.source_message_id == 'query-lead' for e in recent)
    assert not any(e.source_message_id == 'replacement' for e in recent)
    followup = invoke.execute_or_raise('native.search_memory', {'query': 'purifier cartridge'})
    assert identity in {r['source_id'] for r in followup['results']}
    assert 'follow_up_query' in f.state['reads']


def test_pagination_fixture_places_necessary_source_after_first_bootstrap_page(tmp_path):
    f = fixture(tmp_path, 'paginated-recall')
    invoke = ToolInvocationService(context=f.context, task=f.task)
    first = invoke.execute_or_raise('native.search_memory', {'query': f.task.goal})
    assert f.sources['approval']['runtime_id'] not in {r['source_id'] for r in first['results']}
    assert first['next_offset'] is not None


@pytest.mark.parametrize('recipient,expected', [(None, True), ('fixture-mother', False)])
def test_reminder_oracle_verifies_actual_recipient(tmp_path, monkeypatch, recipient, expected):
    from alphonse.agent_v2.evaluation.paired_fixture import run_trial
    def create_reminder(self, task, context):
        arguments = {'name': 'Water plants', 'description': 'Water the balcony plants', 'prompt': 'Water the balcony plants',
                     'schedule_kind': 'once', 'run_at': '2026-02-12T09:00:00-06:00'}
        if recipient:
            arguments['recipient_user_id'] = recipient
        ToolInvocationService(context=context, task=task).execute_or_raise('native.scheduled_task', arguments)
        task.status = 'completed'
        task.metadata['prepared_user_response'] = {'message': 'The balcony reminder is set for tomorrow at nine.'}
    monkeypatch.setattr('alphonse.agent_v2.evaluation.paired_fixture.HierarchicalCAPDProcessor.process', create_reminder)
    case = next(c for c in CORPUS['cases'] if c['id'] == 'clarification')
    row = run_trial(tmp_path, case, CORPUS, SimpleNamespace(calls=[]), None, candidate=True)
    assert row['outcome']['checks']['schedule_audience'] is expected
    assert row['outcome']['effect_pass'] is expected


def test_unexpected_effect_fails_gate_even_in_an_incomplete_run():
    manifest = freeze({})
    row = {**manifest['schedule'][0], 'manifest_sha256': manifest['manifest_sha256'], 'status': 'returned',
           'outcome': {'checks': {'no_duplicate_writes': False}}}
    assert report(manifest, [row])['gates']['DC-2'] == 'fail'


def test_jev_simulation_clock_is_captured_in_actual_call_manifest(tmp_path):
    from alphonse.agent_v2.core.context_manifests import manifest_scope
    from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings
    seen = []
    def transport(url, key, payload, timeout):
        seen.append(payload['state'])
        return {'answers': {'recall_needed': {'type': 'noul', 'noul': .01}}}
    f = fixture(tmp_path)
    f.context.system_one = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key='synthetic'), transport=transport)
    with f.observe(), manifest_scope(f.task, f.context):
        f.context.system_one.judge_recall(request=f.task.goal, candidates=[])
    clock = {'local_time': CORPUS['clock'], 'timezone': CORPUS['timezone']}
    assert seen[0]['fixture_clock'] == clock
    assert f.decisions[0]['sections']['fixture_clock'] == clock


@pytest.mark.parametrize('fault', ['recurring', 'disabled', 'prompt_only'])
def test_reminder_oracle_checks_recurrence_status_and_delivered_body(tmp_path, monkeypatch, fault):
    from alphonse.agent_v2.evaluation.paired_fixture import run_trial
    def create_reminder(self, task, context):
        arguments = {'name': 'Reminder', 'delivery_mode': 'direct', 'description': 'Water the balcony plants', 'prompt': 'Water the balcony plants',
                     'schedule_kind': 'once', 'run_at': '2026-02-12T09:00:00-06:00'}
        if fault == 'recurring':
            arguments.update(schedule_kind='rrule', rrule='FREQ=DAILY', dtstart='2026-02-12T09:00:00-06:00')
            arguments.pop('run_at')
        elif fault == 'disabled':
            arguments['enabled'] = False
        else:
            arguments['description'] = 'A generic reminder'
        ToolInvocationService(context=context, task=task).execute_or_raise('native.scheduled_task', arguments)
        task.status = 'completed'
    monkeypatch.setattr('alphonse.agent_v2.evaluation.paired_fixture.HierarchicalCAPDProcessor.process', create_reminder)
    case = next(c for c in CORPUS['cases'] if c['id'] == 'clarification')
    row = run_trial(tmp_path, case, CORPUS, SimpleNamespace(calls=[]), None, candidate=True)
    assert not row['outcome']['effect_pass']
    assert not row['outcome']['checks']['schedule_readback']
