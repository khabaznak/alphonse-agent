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
