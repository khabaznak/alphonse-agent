

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
