"""Explicit synthetic decision providers for legacy CAPD compatibility tests."""
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.system_one import SystemOneActJudgments, SystemOneReviewResult


def act_judgments(*, effect_requested=.01, effect_supported=.99, effect_contradicted=.01,
                  reply_addresses_request=.99, reply_grounded=.99):
    return SystemOneActJudgments({
        'effect_requested': effect_requested, 'effect_supported': effect_supported,
        'effect_contradicted': effect_contradicted, 'reply_addresses_request': reply_addresses_request,
        'reply_grounded': reply_grounded,
    })


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
        return SimpleNamespace(evaluate=evaluate)
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




import pytest

@pytest.fixture
def secure_vault(tmp_path):
    """Disposable encrypted backend; never touch the host keychain."""
    import os
    from alphonse.agent_v2.integrations.credentials import CredentialVault, EncryptedFileBackend
    root = tmp_path / 'credentials'
    root.mkdir(mode=0o700)
    key = tmp_path / 'credential-key'
    key.write_bytes(os.urandom(32))
    key.chmod(0o600)
    return CredentialVault(EncryptedFileBackend(root, key))


@pytest.fixture
def pdca_decisions():
    """Explicit synthetic Check/Act answers for legacy graph mechanics tests.

    Only successful evidence references can support the configured criteria.
    Production code still owns patch validation and terminal-state invariants.
    """
    from alphonse.agent_v2.system_one import SystemOneActRecommendation, SystemOneReviewResult

    class Decisions:
        def __init__(self):
            self.satisfy = False
            self.action = None
            self.calls = []

        def recommend_act(self, *, state):
            self.calls.append(("act", state))
            criteria = state["acceptance_contract"].get("criteria", [])
            complete = bool(criteria) and all(c["status"] == "satisfied" for c in criteria)
            action = self.action or ("complete" if complete else "continue")
            return SystemOneActRecommendation(
                action, 1.0, True,
                "All acceptance criteria are complete and a user response is prepared." if action == "complete" else "Synthetic continuation judgment.",
                answers={"continuation_is_worthwhile": 1.0},
            )

        def evaluate(self, *, contract, phase, evidence):
            self.calls.append(("check", evidence))
            refs = [e["evidence_ref"] for e in evidence["entries"] if e["status"] == "success"]
            updates = tuple({"criterion_id": c["id"], "status": "satisfied", "evidence_refs": refs}
                            for c in contract["criteria"] if self.satisfy and refs)
            return SystemOneReviewResult(updates, ())

    return Decisions()
