"""Steering accepted before a continuation retains its original claim."""
from types import SimpleNamespace

from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import (
    _decision_digest, _steering_claim_in_lineage,
)
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest


def test_steering_claim_is_preserved_across_an_approved_handoff():
    old = SimpleNamespace(message_id='m', user_id='u', lease_owner='old', claim_generation='1')
    current = SimpleNamespace(message_id='m', user_id='u', lease_owner='new', claim_generation='2')
    pause = {'claim_digest': _claim_digest(old)}
    grant = {'status': 'consumed'}
    link = {'original_claim_digest': _claim_digest(old), 'current_claim_digest': _claim_digest(current),
            'pause_sha256': _decision_digest(pause), 'grant_sha256': _decision_digest(grant)}
    metadata = {'owner_continuation_history': [{'pause': pause, 'grant': grant, 'reclaim': link}]}
    assert _steering_claim_in_lineage(metadata, current, _claim_digest(old))
    assert not _steering_claim_in_lineage(metadata, current, 'unrelated')
    link['grant_sha256'] = 'tampered'
    assert not _steering_claim_in_lineage(metadata, current, _claim_digest(old))
