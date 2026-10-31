

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
