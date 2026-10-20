"""Explicit synthetic decision providers for legacy CAPD compatibility tests."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.system_one import SystemOneActRecommendation, SystemOneReviewResult


@pytest.fixture
def v2_decisions():
    def build(*, satisfy=False):
        def evaluate(*, contract, evidence, **kwargs):
            entries = evidence.get('entries', [])
            refs = [entry['evidence_ref'] for entry in entries if entry.get('status') == 'success']
            return SystemOneReviewResult(updates=tuple({
                'criterion_id': criterion['id'],
                'status': 'satisfied' if satisfy and refs else 'pending',
                'evidence_refs': refs if satisfy else [],
                'reason': 'Synthetic evidence judgment',
            } for criterion in contract['criteria']), ambiguous_criterion_ids=())
        def recommend_act(*, state):
            complete = all(c['status'] == 'satisfied' for c in state['acceptance_contract']['criteria'])
            return SystemOneActRecommendation('complete' if complete else 'continue', 1.0, True,
                                             'Synthetic route', answers={'continuation_is_worthwhile': 1.0})
        return SimpleNamespace(evaluate=evaluate, recommend_act=recommend_act)
    return build
