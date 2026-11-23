import os
import pytest
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.owner_composer_drafts import OwnerComposerDrafts
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.conversations import SQLiteConversationStore


def test_draft_is_durable_idempotent_and_removable(tmp_path):
    path = tmp_path / 'test.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteConversationStore(path)
    active = queue.enqueue_trusted_local_owner(prompt='Initial', peer_uid=os.getuid())
    store = OwnerComposerDrafts(queue, owner.user_id, os.getuid())
    draft_id = 'a' * 32
    assert store.save(draft_id, active, 'Add apples')['status'] == 'saved'
    assert store.save(draft_id, active, 'Add apples')['status'] == 'saved'
    with pytest.raises(ValueError):
        store.save(draft_id, active, 'Changed')
    assert OwnerComposerDrafts(queue, owner.user_id, os.getuid()).list_drafts()['items'][0]['prompt'] == 'Add apples'
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0] == 1
    with pytest.raises(ValueError):
        store.delete_draft(draft_id)
    assert store.reconcile(draft_id)['resolution'] == 'not_submitted'
    from alphonse.agent_v2.owner_followups import OwnerFollowups
    with pytest.raises(ValueError, match='closed'):
        OwnerFollowups(queue, owner.user_id, os.getuid()).add(active, 'Add apples', draft_id)
    assert store.delete_draft(draft_id)['status'] == 'deleted'
    with pytest.raises(ValueError, match='closed'):
        OwnerFollowups(queue, owner.user_id, os.getuid()).add(active, 'Add apples', draft_id)
    assert store.list_drafts()['items'] == []
    with queue._connect() as conn:
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
    with pytest.raises(PermissionError):
        store.list_drafts()


def test_committed_submission_reconciles_after_reopen_without_replay(tmp_path):
    from alphonse.agent_v2.owner_followups import OwnerFollowups
    path = tmp_path / 'test.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteConversationStore(path)
    active = queue.enqueue_trusted_local_owner(prompt='Initial', peer_uid=os.getuid())
    store = OwnerComposerDrafts(queue, owner.user_id, os.getuid())
    item_id = 'b' * 32
    store.save(item_id, active, 'Add apples')
    followups = OwnerFollowups(queue, owner.user_id, os.getuid())
    assert followups.add(active, 'Add apples', item_id)['status'] == 'queued'
    reopened = OwnerComposerDrafts(queue, owner.user_id, os.getuid())
    assert reopened.reconcile(item_id)['resolution'] == 'submitted'
    assert followups.add(active, 'Add apples', item_id)['status'] == 'queued'
    assert len(followups.list()['items']) == 1
    assert reopened.delete_draft(item_id)['status'] == 'deleted'


def test_linked_conversation_deletion_requires_draft_reconciliation(tmp_path):
    path = tmp_path / 'test.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversations = SQLiteConversationStore(path)
    active = queue.enqueue_trusted_local_owner(prompt='Initial', peer_uid=os.getuid())
    store = OwnerComposerDrafts(queue, owner.user_id, os.getuid())
    draft_id = 'd' * 32
    store.save(draft_id, active, 'Private draft')
    event = conversations.event_for_source_message_id(owner_user_id=owner.user_id, source_message_id='inbound:' + active)
    with pytest.raises(ValueError, match='owner_conversation_deletion_requires_reconciliation'):
        conversations.delete(event_id=event.event_id, owner_user_id=owner.user_id)
    store.reconcile(draft_id)
    store.delete_draft(draft_id)
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_owner_composer_drafts').fetchone()[0] == 0


def test_former_member_deletion_cleans_private_drafts_and_tombstones(tmp_path):
    path = tmp_path / 'test.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Owner', users_root=tmp_path / 'users')
    former = users.create_user(display_name='Former')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteConversationStore(path)
    active = queue.enqueue_trusted_local_owner(prompt='Initial', peer_uid=os.getuid())
    store = OwnerComposerDrafts(queue, owner.user_id, os.getuid())
    store.save('e' * 32, active, 'Current private draft')
    with queue._connect() as conn:
        conn.execute('INSERT INTO v2_owner_composer_drafts VALUES (?,?,?,?,?,?)',
                     ('f' * 32, former.user_id, os.getuid(), active, 'Former private draft', 'unresolved'))
        conn.execute('INSERT INTO v2_owner_composer_tombstones VALUES (?,?,?)',
                     ('c' * 32, former.user_id, os.getuid()))
    assert users.delete_user(former.user_id)
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_owner_composer_drafts WHERE owner_id=?', (former.user_id,)).fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM v2_owner_composer_tombstones WHERE owner_id=?', (former.user_id,)).fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM v2_owner_composer_drafts WHERE owner_id=?', (owner.user_id,)).fetchone()[0] == 1
