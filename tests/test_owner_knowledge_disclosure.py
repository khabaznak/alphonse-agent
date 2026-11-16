from contextlib import closing
from dataclasses import replace

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop.owner_knowledge_disclosure import (
    OwnerKnowledgeDisclosureStore, ProviderIdentity, disclosure_binding_locked, disclosure_matches_locked,
)
from alphonse.agent_v2.database import connect_database
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def setup(tmp_path):
    path = str(tmp_path / 'state.sqlite')
    users = V2UserStore(path)
    owner = users.onboard(display_name='Owner', users_root=tmp_path / 'users')
    store = OwnerKnowledgeDisclosureStore(path)
    provider = ProviderIdentity('fixture', 'local fixture endpoint', 'test account', 'test model')
    return path, owner.user_id, store, provider


def read(path, owner, provider):
    with closing(connect_database(path)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        return disclosure_binding_locked(conn, owner, provider)


def test_default_deny_and_exact_provider_binding(setup):
    path, owner, store, provider = setup
    assert read(path, owner, provider) is None
    binding = store.issue(owner, provider)
    assert read(path, owner, provider) == binding
    for field in ('provider', 'endpoint', 'account', 'model'):
        assert read(path, owner, replace(provider, **{field: 'another'})) is None
    assert read(path, 'other', provider) is None


def test_revoke_and_regrant_invalidate_old_binding(setup):
    path, owner, store, provider = setup
    binding = store.issue(owner, provider)
    store.revoke(owner)
    assert read(path, owner, provider) is None
    replacement = store.issue(owner, provider)
    assert replacement != binding
    with closing(connect_database(path)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        assert not disclosure_matches_locked(conn, binding)
        assert disclosure_matches_locked(conn, replacement)


def test_owner_change_and_inactive_owner_fail_closed(setup):
    path, owner, store, provider = setup
    store.issue(owner, provider)
    with closing(connect_database(path)) as conn, conn:
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner,))
    assert read(path, owner, provider) is None
    with pytest.raises(PermissionError):
        store.issue(owner, provider)
    with pytest.raises(PermissionError):
        store.revoke(owner)


def test_requires_transaction_and_invalid_inputs_fail_closed(setup):
    path, owner, store, provider = setup
    store.issue(owner, provider)
    with closing(connect_database(path)) as conn:
        assert disclosure_binding_locked(conn, owner, provider) is None
        conn.execute('BEGIN IMMEDIATE')
        assert disclosure_binding_locked(conn, owner, object()) is None
        assert not disclosure_matches_locked(conn, object())
    with pytest.raises(ValueError):
        ProviderIdentity('fixture', '', 'account', 'model')
