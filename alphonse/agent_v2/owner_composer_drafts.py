"""Owner-scoped durable drafts for recovering an interrupted follow-up send.

Saving a draft does not enqueue or execute work. Draft IDs are immutable and
must be reused as follow-up request IDs. Explicit deletion erases the text.
"""
from alphonse.agent_v2.owner_followups import OwnerFollowups, _id


class OwnerComposerDrafts(OwnerFollowups):
    def _authorize(self, conn):
        super()._authorize(conn)
        conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_composer_drafts (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, peer_uid INTEGER NOT NULL,
            active_message_id TEXT NOT NULL, prompt TEXT NOT NULL,
            resolution TEXT NOT NULL DEFAULT 'unresolved' CHECK(resolution IN ('unresolved','not_submitted','submitted'))) STRICT''')
        conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_composer_tombstones (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, peer_uid INTEGER NOT NULL) STRICT''')

    def save(self, draft_id, active_message_id, prompt):
        draft_id, active_message_id = _id(draft_id), _id(active_message_id)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 16384:
            raise ValueError('owner_draft_invalid')
        prompt = prompt.strip()
        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            if conn.execute('SELECT 1 FROM v2_owner_composer_tombstones WHERE id=?', (draft_id,)).fetchone():
                raise ValueError('owner_draft_closed')
            existing = conn.execute('SELECT * FROM v2_owner_composer_drafts WHERE id=?', (draft_id,)).fetchone()
            if existing:
                if (existing['owner_id'], existing['peer_uid'], existing['active_message_id'], existing['prompt']) != (self.owner, self.peer_uid, active_message_id, prompt):
                    raise ValueError('owner_draft_invalid')
                return {'id': draft_id, 'status': 'saved'}
            if conn.execute('''SELECT 1 FROM v2_trusted_owner_ingress WHERE message_id=? AND owner_id=? AND peer_uid=?''',
                            (active_message_id, self.owner, self.peer_uid)).fetchone() is None:
                raise ValueError('owner_draft_invalid')
            count = conn.execute('SELECT count(*) FROM v2_owner_composer_drafts WHERE owner_id=?', (self.owner,)).fetchone()[0]
            if count >= 100:
                raise ValueError('owner_draft_limit')
            conn.execute('INSERT INTO v2_owner_composer_drafts (id,owner_id,peer_uid,active_message_id,prompt) VALUES (?,?,?,?,?)',
                         (draft_id, self.owner, self.peer_uid, active_message_id, prompt))
            return {'id': draft_id, 'status': 'saved'}

    def list_drafts(self):
        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            rows = conn.execute('''SELECT id,active_message_id,prompt,resolution FROM v2_owner_composer_drafts
                WHERE owner_id=? AND peer_uid=? ORDER BY rowid''', (self.owner, self.peer_uid)).fetchall()
            return {'items': [dict(row) for row in rows]}

    def reconcile(self, draft_id):
        draft_id = _id(draft_id)
        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            row = conn.execute('SELECT * FROM v2_owner_composer_drafts WHERE id=? AND owner_id=? AND peer_uid=?',
                               (draft_id, self.owner, self.peer_uid)).fetchone()
            if row is None:
                raise ValueError('owner_draft_invalid')
            found = conn.execute('SELECT status FROM v2_owner_held_followups WHERE id=? AND owner_id=? AND peer_uid=?',
                                 (draft_id, self.owner, self.peer_uid)).fetchone()
            steering_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone()
            steering = conn.execute('SELECT status FROM v2_owner_steering WHERE id=? AND owner_id=? AND peer_uid=?',
                                    (draft_id, self.owner, self.peer_uid)).fetchone() if steering_table else None
            resolution = 'submitted' if found or steering else 'not_submitted'
            conn.execute('UPDATE v2_owner_composer_drafts SET resolution=? WHERE id=?', (resolution, draft_id))
            return {'id': draft_id, 'resolution': resolution}

    def delete_draft(self, draft_id):
        draft_id = _id(draft_id)
        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            row = conn.execute('SELECT resolution FROM v2_owner_composer_drafts WHERE id=? AND owner_id=? AND peer_uid=?',
                               (draft_id, self.owner, self.peer_uid)).fetchone()
            if row is None or row['resolution'] == 'unresolved':
                raise ValueError('owner_draft_unresolved')
            conn.execute('INSERT OR IGNORE INTO v2_owner_composer_tombstones VALUES (?,?,?)',
                         (draft_id, self.owner, self.peer_uid))
            conn.execute('DELETE FROM v2_owner_composer_drafts WHERE id=? AND owner_id=? AND peer_uid=?',
                         (draft_id, self.owner, self.peer_uid))
            return {'id': draft_id, 'status': 'deleted'}
