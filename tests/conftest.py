"""Explicit synthetic decision providers for legacy CAPD compatibility tests."""

import pytest

from alphonse.agent_v2.system_one import SystemOneActJudgments


def act_judgments(*, effect_requested=.01, effect_supported=.99, effect_contradicted=.01,
                  reply_addresses_request=.99, reply_grounded=.99):
    return SystemOneActJudgments({
        'effect_requested': effect_requested, 'effect_supported': effect_supported,
        'effect_contradicted': effect_contradicted, 'reply_addresses_request': reply_addresses_request,
        'reply_grounded': reply_grounded,
    })




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
