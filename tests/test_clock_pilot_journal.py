"""Synthetic clock integration; no live provider or real data."""
import json
import hashlib

import pytest

from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.evaluation.clock_pilot_journal import run_offline_clock_journal
from alphonse.agent_v2.evaluation.offline_run_journal import inspect_journal


CONFIG = {'source_instant': '2026-04-12T14:00:00+00:00',
          'source_timezone': 'America/Mexico_City',
          'expected_instant': '2026-04-12T14:00:00+00:00',
          'expected_timezone': 'America/Mexico_City'}


class FakeProvider:
    offline = True

    def __init__(self):
        self.calls = []
        self.decisions = iter([
            {'kind': 'tool', 'tool_id': 'native.fixture_read', 'arguments': {'source': 'index'}},
            {'kind': 'answer', 'text': json.dumps({'local_time': '08:00', 'local_date': '2026-04-12',
                                                   'timezone': 'America/Mexico_City'})},
        ])

    def generate_json(self, request, *, max_input_tokens, max_output_tokens):
        self.calls.append(request.prompt)
        return InferenceResult(json_value=next(self.decisions),
                               usage={'input_tokens': 100, 'output_tokens': 30})


def test_default_records_unavailable_without_call_or_socket(tmp_path):
    provider = FakeProvider()
    receipt = run_offline_clock_journal(tmp_path / 'run.jsonl', transport=provider, **CONFIG)
    assert receipt['complete'] is True and provider.calls == []
    result = next(iter(receipt['outcomes'].values()))
    assert result['availability'] == 'unavailable' and result['passed'] is None
    assert inspect_journal(tmp_path / 'run.jsonl')['qualification'] == 'unassessed'


def test_unavailable_transport_cannot_be_used_for_test_dispatch(tmp_path):
    with pytest.raises(ValueError, match='offline_transport_required'):
        run_offline_clock_journal(tmp_path / 'run.jsonl', transport=object(),
                                  enable_offline_test_dispatch=True, **CONFIG)
    assert not (tmp_path / 'run.jsonl').exists()


def test_candidate_fixture_oracle_and_durable_records(tmp_path):
    provider = FakeProvider()
    receipt = run_offline_clock_journal(tmp_path / 'run.jsonl', transport=provider,
                                        enable_offline_test_dispatch=True, **CONFIG)
    assert receipt['complete'] is True and len(provider.calls) == 2
    result = next(iter(receipt['outcomes'].values()))
    assert result['status'] == 'completed' and result['passed'] is True
    assert result['details']['portal_oracle']['clock_source_matches'] is True
    assert result['details']['portal_oracle']['decision_qualified'] is False
    assert result['details']['delivery'] == 'unassessed'
    assert result['events'][0]['provider_usage'] == {'input_tokens': 100, 'output_tokens': 30}
    assert result['events'][0]['prompt_sha256'] == hashlib.sha256(provider.calls[0].encode()).hexdigest()
    assert '2026-04-12T14:00:00+00:00' in provider.calls[-1]
    assert '127.0.0.1' not in (tmp_path / 'run.jsonl').read_text()
    with pytest.raises(FileExistsError):
        run_offline_clock_journal(tmp_path / 'run.jsonl', transport=FakeProvider(),
                                  enable_offline_test_dispatch=True, **CONFIG)
