"""Validated proactive policy authority; no scheduling, provider I/O or inference.

Trusted conversation/management code supplies AccountSession and an explicit
instruction reference. Neither is taken from model-supplied policy fields.
Consumers must call guard at dispatch, each read/page, CAPD entry and delivery.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
import hashlib
import json
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from alphonse.agent_v2.accounts import AccountSession, _deadline, _strings
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.skill_bindings import SkillBindings


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _integer(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _clock(value: Any) -> bool:
    try:
        return isinstance(value, str) and len(value) == 5 and time.fromisoformat(value).isoformat(timespec='minutes') == value
    except ValueError:
        return False


def _weekdays(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_integer(v, 0, 6) for v in value) and len(set(value)) == len(value)


class ProjectCheckService:
    """Policies live in the project database, never in project_config.json."""

    def __init__(self, projects: ProjectStore, bindings: SkillBindings):
        self.projects, self.bindings = projects, bindings
        with projects._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS v2_project_check_policies (
                policy_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                executing_user_id TEXT NOT NULL, revision INTEGER NOT NULL,
                definition TEXT NOT NULL)''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_project_checks_project ON v2_project_check_policies(project_id)')

        with bindings.integrations._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS v2_project_check_choice_revisions (
                    actor_id TEXT NOT NULL, skill_id TEXT NOT NULL, name TEXT NOT NULL,
                    revision INTEGER NOT NULL, PRIMARY KEY(actor_id, skill_id, name)
                );
                CREATE TRIGGER IF NOT EXISTS project_check_choice_selected
                AFTER INSERT ON v2_skill_binding_choices
                BEGIN
                    INSERT INTO v2_project_check_choice_revisions VALUES (NEW.actor_id, NEW.skill_id, NEW.name, 1)
                    ON CONFLICT(actor_id, skill_id, name) DO UPDATE SET revision=revision+1;
                END;
                CREATE TRIGGER IF NOT EXISTS project_check_choice_removed
                AFTER DELETE ON v2_skill_binding_choices
                BEGIN
                    INSERT INTO v2_project_check_choice_revisions VALUES (OLD.actor_id, OLD.skill_id, OLD.name, 1)
                    ON CONFLICT(actor_id, skill_id, name) DO UPDATE SET revision=revision+1;
                END;
            """)

    def _project(self, session: AccountSession, project_id: str):
        self.bindings.accounts._active(session.actor_user_id)
        project = self.projects.get_project(project_id, requester_user_id=session.actor_user_id)
        if not project or project.status != 'active':
            raise PermissionError('project_check_project_unavailable')
        return project

    def _snapshot(self, session: AccountSession, project_id: str) -> dict:
        project = self._project(session, project_id)
        config = self.projects.read_project_config(project_id, requester_user_id=session.actor_user_id)
        return dict(scope=config['scope'], goals=config['goals'], duration=config['duration'],
                    description=project.description, owner_user_id=project.owner_user_id)

    def _project_revision(self, project_id: str) -> int:
        with self.projects._connect() as conn:
            row = conn.execute('SELECT revision FROM v2_project_authority_revisions WHERE project_id=?', (project_id,)).fetchone()
        if not row:
            raise PermissionError('project_check_project_unavailable')
        return row[0]

    def _load(self, policy_id: str) -> dict:
        with self.projects._connect() as conn:
            row = conn.execute('SELECT definition FROM v2_project_check_policies WHERE policy_id=?', (policy_id,)).fetchone()
        if not row:
            raise KeyError('project_check_missing')
        return json.loads(row[0])

    def _managed(self, session: AccountSession, policy_id: str) -> dict:
        self.bindings.accounts._active(session.actor_user_id)
        value = self._load(policy_id)
        if value['executing_user_id'] != session.actor_user_id:
            raise PermissionError('project_check_owner_required')
        project = self.projects.get_project(value['project_id'], requester_user_id=session.actor_user_id, include_archived=True)
        if not project:
            raise PermissionError('project_check_project_unavailable')
        return value

    def _put(self, value: dict, expected_revision: int) -> dict:
        value = dict(value, revision=expected_revision + 1, updated_at=datetime.now(timezone.utc).isoformat())
        with self.projects._connect() as conn:
            if expected_revision == 0:
                conn.execute('INSERT INTO v2_project_check_policies VALUES (?, ?, ?, ?, ?)',
                             (value['policy_id'], value['project_id'], value['executing_user_id'], value['revision'], json.dumps(value)))
            else:
                cursor = conn.execute('UPDATE v2_project_check_policies SET revision=?, definition=? WHERE policy_id=? AND revision=?',
                                      (value['revision'], json.dumps(value), value['policy_id'], expected_revision))
                if cursor.rowcount != 1:
                    raise ValueError('project_check_revision_conflict')
        return value

    def _sources(self, session: AccountSession, value: dict, *, pinned: bool) -> list[dict]:
        result = []
        for source in value['source_bindings']:
            handle = self.bindings.resolve_account(session, skill_id=source['skill_id'], name=source['role'],
                operation=source['operation'], resources=source['resources'],
                recipient_user_ids=value['recipient_user_ids'], project_id=value['project_id'],
                expected_binding_id=source['binding_id'], expected_connection_id=source['connection_id'],
                expected_grant_id=source['grant_id'],
                expected_grant_revision=source.get('grant_revision') if pinned else None)
            binding_revision = self.bindings._get(handle.binding_id)['revision']
            with self.bindings.integrations._connect() as conn:
                choice = conn.execute('SELECT revision FROM v2_project_check_choice_revisions WHERE actor_id=? AND skill_id=? AND name=?',
                                      (session.actor_user_id, source['skill_id'], source['role'])).fetchone()
            choice_revision = choice[0] if choice else 0
            if pinned and choice_revision != source['choice_revision']:
                raise PermissionError('project_check_choice_changed')
            if pinned and (handle.connection_revision != source['connection_revision'] or binding_revision != source['binding_revision']):
                raise PermissionError('project_check_source_changed')
            result.append(dict(source, grant_revision=handle.grant_revision,
                               connection_revision=handle.connection_revision, binding_revision=binding_revision, choice_revision=choice_revision))
        return result

    def _skills(self, session: AccountSession, value: dict) -> None:
        for skill_id in value['skill_hints']:
            if not self.bindings.skills.get(skill_id) or not self.bindings.skills.can_use(skill_id, session.actor_user_id):
                raise PermissionError('project_check_skill_unavailable')

    def save(self, session: AccountSession, *, definition: dict, expected_revision: int = 0,
             policy_id: str | None = None, authorization_reference: str = '', activate: bool = False) -> dict:
        """Create/re-authorize a concrete policy from a trusted explicit instruction.

        An instruction can authorize recurring execution; no approval per run.
        Caller must present/resolve the concrete summary before activating it.
        Drafts default disabled and still require valid bounded inputs/grants.
        """
        value = self._validate(definition)
        self._project(session, value['project_id'])
        old = self._managed(session, policy_id) if policy_id else None
        if expected_revision != (old['revision'] if old else 0):
            raise ValueError('project_check_revision_conflict')
        if old and old['project_id'] != value['project_id']:
            raise ValueError('project_check_identity_immutable')
        if activate and (not isinstance(authorization_reference, str) or not authorization_reference.strip()):
            raise PermissionError('project_check_authorization_required')
        snapshot = self._snapshot(session, value['project_id'])
        value.update(schema_version=1, policy_id=policy_id or 'check_' + uuid4().hex,
                     executing_user_id=session.actor_user_id, status='active' if activate else 'disabled',
                     goal_snapshot=snapshot, goal_digest=_digest(snapshot), query_digest=_digest(value['query']),
                     project_revision=self._project_revision(value['project_id']),
                     authorized_by=session.actor_user_id if activate else '',
                     authorized_at=datetime.now(timezone.utc).isoformat() if activate else '',
                     authorization_reference=authorization_reference if activate else '',
                     created_at=old['created_at'] if old else datetime.now(timezone.utc).isoformat())
        value['source_bindings'] = self._sources(session, value, pinned=False)
        self._skills(session, value)
        self._duration(value, datetime.now(timezone.utc), require_started=False)
        return self._put(value, expected_revision)

    def get(self, session: AccountSession, policy_id: str) -> dict:
        value = self._managed(session, policy_id)
        if value['status'] == 'active':
            try:
                self.guard(policy_id, expected_revision=value['revision'])
            except (PermissionError, ValueError, KeyError, OSError):
                value = self._load(policy_id)
        return value

    def list(self, session: AccountSession, project_id: str) -> list[dict]:
        self._project(session, project_id)
        with self.projects._connect() as conn:
            ids = [r[0] for r in conn.execute('SELECT policy_id FROM v2_project_check_policies WHERE project_id=? AND executing_user_id=? ORDER BY policy_id',
                                             (project_id, session.actor_user_id))]
        return [self.get(session, policy_id) for policy_id in ids]

    def set_status(self, session: AccountSession, policy_id: str, *, expected_revision: int, status: str) -> dict:
        value = self._managed(session, policy_id)
        if value['revision'] != expected_revision:
            raise ValueError('project_check_revision_conflict')
        if status not in {'active', 'paused', 'disabled', 'ended'}:
            raise ValueError('project_check_status_invalid')
        if status == 'paused' and value['status'] not in {'active', 'paused'}:
            raise PermissionError('project_check_reauthorization_required')
        if status == 'active':
            if value['status'] != 'paused' or not value['authorized_at']:
                raise PermissionError('project_check_reauthorization_required')
            self._check_dependencies(value, check_project_revision=False)
            value['project_revision'] = self._project_revision(value['project_id'])
        return self._put(dict(value, status=status), expected_revision)

    @staticmethod
    def _duration(value: dict, now: datetime, *, require_started: bool = True) -> None:
        duration = value['duration']
        if require_started and now < _deadline(duration['starts_at']):
            raise PermissionError('project_check_not_started')
        if duration['ends_at'] and now >= _deadline(duration['ends_at']):
            raise PermissionError('project_check_outside_duration')
        project_duration = value['goal_snapshot']['duration']
        local_date = now.astimezone(ZoneInfo(value['timezone'])).date().isoformat()
        if require_started and project_duration.get('start_date') and local_date < project_duration['start_date']:
            raise PermissionError('project_check_not_started')
        if project_duration.get('end_date') and local_date > project_duration['end_date']:
            raise PermissionError('project_check_outside_duration')

    def _check_dependencies(self, value: dict, *, check_project_revision: bool = True, now: datetime | None = None) -> None:
        session = AccountSession(value['executing_user_id'])
        if _digest(self._snapshot(session, value['project_id'])) != value['goal_digest']:
            raise PermissionError('project_check_goal_changed')
        if check_project_revision and self._project_revision(value['project_id']) != value['project_revision']:
            raise PermissionError('project_check_project_changed')
        self._duration(value, now or datetime.now(timezone.utc))
        self._sources(session, value, pinned=True)
        self._skills(session, value)

    def guard(self, policy_id: str, *, expected_revision: int, cancelled: bool = False,
              now: datetime | None = None) -> dict:
        """Revision fence for queued work. This does not authorize off-window reads.

        #113 computes occurrences/windows and enforces their deadlines in addition
        to this policy fence. Cancellation must be checked again after in-flight I/O.
        Never interpret rejection as successful acquisition or delivery.
        """
        if cancelled:
            raise PermissionError('project_check_cancelled')
        if now is not None and now.tzinfo is None:
            raise ValueError('project_check_time_invalid')
        value = self._load(policy_id)
        if value['revision'] != expected_revision or value['status'] != 'active':
            raise PermissionError('project_check_stale_or_inactive')
        try:
            self._duration(value, now or datetime.now(timezone.utc))
        except PermissionError as exc:
            if str(exc) == 'project_check_outside_duration':
                self._put(dict(value, status='ended'), expected_revision)
            raise
        try:
            self._check_dependencies(value, now=now)
        except (PermissionError, ValueError, KeyError, OSError):
            # Durable suspension; even a later restore/reconnect cannot revive it.
            self._put(dict(value, status='needs_review'), expected_revision)
            raise
        if (self._project_revision(value['project_id']) != value['project_revision']
                or self._load(policy_id)['revision'] != expected_revision):
            raise PermissionError('project_check_stale_or_inactive')
        return value

    @staticmethod
    def _validate(definition: dict) -> dict:
        required = {'project_id', 'source_bindings', 'query', 'trigger', 'timezone',
                    'retrieval_window', 'duration', 'budgets'}
        optional = {'initial_lookback_days', 'missed_run', 'skill_hints', 'recipient_user_ids', 'notification_mode'}
        if not isinstance(definition, dict) or not required <= definition.keys() or definition.keys() - required - optional:
            raise ValueError('project_check_definition_invalid')
        # JSON copy detaches callers' mutable structures and rejects non-JSON data.
        value = json.loads(json.dumps(definition, allow_nan=False))
        value.setdefault('initial_lookback_days', 30)
        value.setdefault('missed_run', {'kind': 'next_allowed_window'})
        value.setdefault('skill_hints', [])
        value.setdefault('recipient_user_ids', [])
        value.setdefault('notification_mode', 'material_changes')
        try:
            ZoneInfo(value['timezone'])
            if not isinstance(value['project_id'], str) or not value['project_id'].strip():
                raise ValueError()
            query = value['query']
            if not isinstance(query, dict) or set(query) != {'text'} or not isinstance(query['text'], str) or len(query['text']) > 4096:
                raise ValueError()
            trigger, window, duration, budgets, missed = (value[k] for k in ['trigger', 'retrieval_window', 'duration', 'budgets', 'missed_run'])
            if (set(trigger) != {'kind', 'weekdays', 'local_time'} or trigger['kind'] != 'schedule'
                    or not _weekdays(trigger['weekdays']) or not _clock(trigger['local_time'])
                    or set(window) != {'weekdays', 'start', 'end'} or not _weekdays(window['weekdays'])
                    or not _clock(window['start']) or not _clock(window['end']) or window['start'] >= window['end']
                    or not set(trigger['weekdays']) <= set(window['weekdays'])
                    or not window['start'] <= trigger['local_time'] < window['end']):
                raise ValueError()
            if (set(duration) != {'ongoing', 'starts_at', 'ends_at'} or type(duration['ongoing']) is not bool
                    or (duration['ongoing'] and duration['ends_at'] is not None)
                    or (not duration['ongoing'] and (not duration['ends_at'] or _deadline(duration['ends_at']) <= _deadline(duration['starts_at'])))):
                raise ValueError()
            _deadline(duration['starts_at'])
            if (set(budgets) != {'max_pages', 'max_items', 'max_bytes', 'max_model_calls', 'max_runtime_seconds'}
                    or any(not _integer(v, 1, 10**9) for k, v in budgets.items() if k != 'max_model_calls')
                    or not _integer(budgets['max_model_calls'], 0, 10000)
                    or not _integer(value['initial_lookback_days'], 1, 365)):
                raise ValueError()
            if (not isinstance(missed, dict) or not (
                    missed == {'kind': 'next_allowed_window'} or
                    (set(missed) == {'kind', 'max_lateness_seconds'} and missed['kind'] == 'bounded_late'
                     and _integer(missed['max_lateness_seconds'], 1, 86400)))):
                raise ValueError()
            if value['notification_mode'] not in {'material_changes', 'requested_digest'}:
                raise ValueError()
            for key in ['skill_hints', 'recipient_user_ids']:
                value[key] = _strings(value[key])
            if not isinstance(value['source_bindings'], list) or not 1 <= len(value['source_bindings']) <= 32:
                raise ValueError()
            for source in value['source_bindings']:
                if set(source) != {'skill_id', 'role', 'binding_id', 'connection_id', 'grant_id', 'operation', 'resources'}:
                    raise ValueError()
                if any(not isinstance(source[k], str) or not source[k].strip() for k in source if k != 'resources'):
                    raise ValueError()
                if source['operation'] not in {'read', 'search', 'list', 'changes', 'attachments.read', 'mail_read', 'calendar_read', 'classroom_read'}:
                    raise ValueError()
                source['resources'] = _strings(source['resources'])
                if not source['resources']:
                    raise ValueError()
        except (ValueError, TypeError, KeyError, AttributeError, ZoneInfoNotFoundError):
            raise ValueError('project_check_definition_invalid') from None
        return value
