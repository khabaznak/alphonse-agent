"""Durable, actor-scoped identities and lifecycle for coding-agent jobs.

Job records never contain briefs, provider output, package files or credentials.
Interrupted jobs require reconciliation instead of automatic re-execution.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path


class CodingJobStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS coding_jobs (
                job_id TEXT PRIMARY KEY, actor TEXT NOT NULL, task_id TEXT NOT NULL,
                request_hash TEXT NOT NULL, state TEXT NOT NULL, result_code TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def reserve(self, *, actor: str, task_id: str, operation_id: str, request_hash: str) -> dict:
        if not all(isinstance(value, str) and value.strip() for value in (actor, task_id, operation_id, request_hash)):
            raise ValueError('coding_job_identity_required')
        if any('\0' in value or len(value) > 512 for value in (actor, task_id, operation_id)):
            raise ValueError('coding_job_identity_invalid')
        if not re.fullmatch(r'[0-9a-f]{64}', request_hash):
            raise ValueError('coding_job_request_hash_invalid')
        identity = '\0'.join((actor, task_id, operation_id)).encode()
        job_id = 'coding-job:' + hashlib.sha256(identity).hexdigest()
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT OR IGNORE INTO coding_jobs(job_id, actor, task_id, request_hash, state) VALUES (?, ?, ?, ?, ?)',
                       (job_id, actor, task_id, request_hash, 'queued'))
            row = db.execute('SELECT * FROM coding_jobs WHERE job_id=? AND actor=? AND task_id=?',
                             (job_id, actor, task_id)).fetchone()
            if row['request_hash'] != request_hash:
                raise ValueError('coding_job_operation_conflict')
            return dict(row)

    def get(self, *, job_id: str, actor: str, task_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute('SELECT * FROM coding_jobs WHERE job_id=? AND actor=? AND task_id=?',
                             (job_id, actor, task_id)).fetchone()
            return dict(row) if row else None

    def transition(self, *, job_id: str, actor: str, task_id: str, expected: str, state: str,
                   result_code: str = '') -> dict:
        allowed = {
            ('queued', 'running'), ('queued', 'cancelled'),
            ('running', 'completed'), ('running', 'failed'), ('running', 'cancelled'),
            ('running', 'interrupted'),
        }
        if (expected, state) not in allowed:
            raise ValueError('coding_job_transition_invalid')
        if not isinstance(result_code, str) or len(result_code) > 128 or not all(c.isalnum() or c in '_-' for c in result_code):
            raise ValueError('coding_job_result_code_invalid')
        with self._connect() as db:
            changed = db.execute("""UPDATE coding_jobs SET state=?, result_code=?, updated_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND actor=? AND task_id=? AND state=?""",
                (state, result_code, job_id, actor, task_id, expected)).rowcount
            if changed != 1:
                raise ValueError('coding_job_state_conflict')
            row = db.execute('SELECT * FROM coding_jobs WHERE job_id=? AND actor=? AND task_id=?',
                             (job_id, actor, task_id)).fetchone()
            return dict(row)
