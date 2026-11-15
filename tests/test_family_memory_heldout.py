"""Check the frozen held-out oracle without giving its expectations to a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from alphonse.agent_v2.evaluation.family_memory_oracle import score_heldout


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/family_memory/heldout-v1.json'
EVIDENCE = ROOT / 'docs/agent-loop/evidence/family-memory-heldout-2026-11-15'
FIXTURE_SHA256 = 'a4e67c005fcf09bb5f7bed4899664d2d53ea8008fde94705b5d0c5f4c4706bfe'


@pytest.mark.parametrize('name', ['members', 'appliance'])
def test_recorded_heldout_result_matches_frozen_independent_oracle(name):
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    case = json.loads(raw)['cases'][name]
    result = json.loads((EVIDENCE / f'{name}.json').read_text())
    assert result['fixture_sha256'] == FIXTURE_SHA256
    assert result['source_changed_during_attempt'] is False
    score = score_heldout(case, result['answer'],
                          full_source_text_visible=result['full_source_text_visible'],
                          explicitly_read=result['explicitly_read'])
    for key, value in score.items():
        recorded = result['checks'][key] if key in result['checks'] else result[key]
        assert value == recorded


def test_oracle_distinguishes_answer_relevance_from_evidence_coverage():
    case = json.loads(FIXTURE.read_text())['cases']['members']
    answer = 'Lena and Idris share a home with their cat Juniper.'
    source_text = {'people': True, 'pet': True, 'event': True}
    score = score_heldout(case, answer, full_source_text_visible=source_text, explicitly_read=[])
    assert score['answer_and_full_source_text_check_pass']
    assert not score['strict_explicit_read_check_pass']
    polluted = score_heldout(case, answer + ' The blender needs a replacement lid.',
                             full_source_text_visible=source_text, explicitly_read=['people', 'pet'])
    assert not polluted['answer_check_pass']
    missing = score_heldout(case, answer, full_source_text_visible={'people': True}, explicitly_read=[])
    assert not missing['answer_and_full_source_text_check_pass']
