"""Authenticated, task-bound Gmail account-choice receipts.

The authority in this module is the server-owned challenge table plus an answered
question record, never a model-provided account identifier. Integrations must fence
private delivery and call `issue_choice` only with freshly authorized candidates.
The module alone neither authorizes Gmail access nor proves delivery is private.
"""
from __future__ import annotations

import json
import sqlite3


def _encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _options(accounts):
    choices = []
    for item in accounts:
        if not isinstance(item, dict):
            raise ValueError('gmail_choice_invalid_accounts')
        identity = item.get('connection_id')
        name = item.get('display_name')
        if not isinstance(identity, str) or not identity.strip() or not isinstance(name, str):
            raise ValueError('gmail_choice_invalid_accounts')
        choices.append({'id': identity, 'label': f'{name.strip() or "Gmail"} ({identity})'})
    choices.sort(key=lambda item: item['id'])
    if len(choices) < 2 or len({item['id'] for item in choices}) != len(choices):
        raise ValueError('gmail_choice_invalid_accounts')
    return choices


def _channel(task):
    value = task.metadata.get('channel') if isinstance(task.metadata, dict) else None
    if not isinstance(value, dict):
        raise ValueError('gmail_choice_origin_unavailable')
    keys = ('integration_id', 'provider_key', 'channel_target', 'alphonse_user_id')
    if any(not isinstance(value.get(k), str) or not value[k].strip() for k in keys):
        raise ValueError('gmail_choice_origin_unavailable')
    if value['alphonse_user_id'] != task.user:
        raise ValueError('gmail_choice_origin_mismatch')
    return {k: value[k] for k in keys}


def _message(label_id, unit):
    return f'Which Gmail account should I use to count unread {unit} in {label_id}?'


def _schema(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS gmail_account_choice_challenges (
        question_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, actor TEXT NOT NULL,
        channel_json TEXT NOT NULL, label_id TEXT NOT NULL, unit TEXT NOT NULL,
        choices_json TEXT NOT NULL) STRICT''')


def issue_choice(store, *, task, accounts, label_id, unit):
    """Park a task with a server-authored choice and persist its purpose.

    Caller must have already verified that the question and subsequent response
    are deliverable privately. Deliver the returned interrupt through the normal
    question-delivery path. A crash before challenge creation leaves an unusable
    question, never an authorized choice.
    """
    if not task.task_id or not task.user or unit not in {'messages', 'threads'} or not isinstance(label_id, str) or not label_id:
        raise ValueError('gmail_choice_invalid_request')
    channel = _channel(task)
    choices = _options(accounts)
    question = store.create_question(task=task, question=_message(label_id, unit),
        kind='single_choice', choices=choices, respondent_user_id=task.user,
        delivery_metadata={'origin_channel': channel})
    try:
        with store._connect() as conn:
            _schema(conn)
            conn.execute('INSERT INTO gmail_account_choice_challenges VALUES (?, ?, ?, ?, ?, ?, ?)',
                (question.question_id, task.task_id, task.user, _encode(channel), label_id, unit, _encode(choices)))
    except Exception:
        store.cancel_question(question.question_id)
        raise
    return question


def resolve_choice(store, *, task, question_id, accounts, label_id, unit):
    """Return selected current account, or None for any missing/untrusted proof."""
    try:
        if not isinstance(question_id, str) or not question_id.strip() or not task.task_id or not task.user:
            return None
        channel = _channel(task)
        choices = _options(accounts)
        with store._connect() as conn:
            # This is a read-only check: do not create the challenge table while
            # verifying an arbitrary model-supplied question ID.
            row = conn.execute('''SELECT c.task_id AS challenge_task, c.actor AS challenge_actor,
                c.channel_json, c.label_id, c.unit, c.choices_json AS challenge_choices,
                q.task_id, q.respondent_user_id, q.originator_user_id, q.status,
                q.kind, q.message, q.choices_json, q.answer_json, q.delivery_metadata_json
                FROM gmail_account_choice_challenges AS c JOIN v2_questions AS q
                ON q.question_id = c.question_id WHERE c.question_id = ?
                AND c.rowid = (SELECT MAX(other.rowid) FROM gmail_account_choice_challenges AS other
                    WHERE other.task_id=c.task_id AND other.actor=c.actor
                      AND other.label_id=c.label_id AND other.unit=c.unit)''', (question_id,)).fetchone()
        if row is None or row['challenge_task'] != task.task_id or row['task_id'] != task.task_id:
            return None
        if row['challenge_actor'] != task.user or row['respondent_user_id'] != task.user or row['originator_user_id'] != task.user:
            return None
        if row['channel_json'] != _encode(channel) or row['label_id'] != label_id or row['unit'] != unit:
            return None
        if row['challenge_choices'] != _encode(choices) or json.loads(row['choices_json']) != choices:
            return None
        if row['status'] != 'answered' or row['kind'] != 'single_choice' or row['message'] != _message(label_id, unit):
            return None
        delivery = json.loads(row['delivery_metadata_json'])
        if delivery.get('origin_channel') != channel:
            return None
        answer = json.loads(row['answer_json'])
        chosen = answer.get('choice_id') if isinstance(answer, dict) else None
        selected = next((choice for choice in choices if choice['id'] == chosen), None)
        if selected is None or answer.get('label') != selected['label']:
            return None
        return chosen
    except (sqlite3.Error, ValueError, TypeError, KeyError, AttributeError):
        return None


def choice_was_issued(store, *, task, label_id, unit):
    """Detect outstanding choice context so a caller cannot bypass it by omission.

    A missing table means this database has never issued a Gmail challenge.
    Other database errors fail closed by reporting that a challenge exists.
    """
    try:
        with store._connect() as conn:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gmail_account_choice_challenges'").fetchone()
            if not exists:
                return False
            row = conn.execute('''SELECT 1 FROM gmail_account_choice_challenges
                WHERE task_id=? AND actor=? AND label_id=? AND unit=? LIMIT 1''',
                (task.task_id, task.user, label_id, unit)).fetchone()
        return row is not None
    except (sqlite3.Error, ValueError, TypeError, AttributeError):
        return True
