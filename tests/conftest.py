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


@pytest.fixture
def task_understanding_provider(monkeypatch):
    """Explicit System Two double for pre-existing phase/action script fixtures.

    Keep the real revision/validation path. Only the new interpretation inference
    is synthetic, so existing scripts still own their phase/action/review calls.
    Semantic interpretation scenarios use their own providers in dedicated tests.
    """
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
    from alphonse.agent_v2.core.intelligence.v3 import task_understanding as module
    real_ensure = module.ensure

    def ensure(task, context):
        original = context.inference

        class Provider:
            def __getattr__(self, name):
                return getattr(original, name)

            def generate_json(self, request):
                assert request.purpose == InferencePurpose.TASK_UNDERSTANDING
                return InferenceResult(json_value={
                    'requested_outcome': 'Carry out the supplied synthetic task',
                    'supplied_parameters': [], 'constraints': [],
                    'unresolved_dependencies': [], 'superseded_constraints': [],
                })

        context.inference = Provider()
        try:
            return real_ensure(task, context)
        finally:
            context.inference = original

    monkeypatch.setattr(module, 'ensure', ensure)
