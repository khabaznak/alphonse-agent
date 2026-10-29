"""Authorized batch maintenance with staged native Bash verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
from alphonse.agent_v2.skill_maintenance import (
    draft_package, maintenance_lock, operation_path, publish_draft, save_json,
)
from alphonse.agent_v2.skill_resources import private_legacy_package, require_skill

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'skill_id': {'type': 'string'},
        'operation': {'type': 'string', 'enum': ['status', 'apply']},
        'operation_id': {'type': 'string', 'minLength': 1},
        'expected_revision': {'type': 'string'},
        'files': {'type': 'object', 'additionalProperties': {'type': 'string'},
                  'description': 'Batch of package-relative UTF-8 resources under scripts/, references/, assets/, plus updated SKILL.md. Preserve live data outside this edit set.'},
        'checks': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 20,
                   'description': 'Documented behavior checks run by native Bash in the draft root before publication. Use synthetic inputs, no service mutations or credentials; no automatic dependency installs.'},
        'runtime': {'type': 'string'},
        'timeout_seconds': {'type': 'number', 'exclusiveMinimum': 0},
    },
    'required': ['skill_id', 'operation'],
}


def build_skill_maintain_tool_definition(store, *, is_admin=None):
    descriptor = ToolDescriptor(
        'native.skill_maintain', 'skill_maintain', ToolKind.NATIVE,
        'Inspect maintenance status or repair/extend an activated skill in one authorized workflow. '
        'Requires a household administrator, skill_management and local_shell, with local_skill_installation effects. '
        'Call status for local_revision, then apply with expected_revision, a stable operation_id, all changed files '
        'and updated operating instructions. Checks run in a disposable draft through trusted host Bash; '
        'they must verify relevant behavior before publication. No extra approval per file when the requester '
        'already authorized maintenance. Bundled skills become explicit household overrides. Failed checks '
        'keep the active package and retain draft work. Reuse operation_id on resume: an uncertain/failed attempt '
        'requires reconciliation before a new operation. No artifact registration or new data/service access. '
        'After publication start a fresh conversation/skill selection to load the revised instructions.',
        SCHEMA, ('skill_management', 'local_shell'), ('native', 'skills'), read_only=False,
        metadata={'v3_capabilities': ['skill_management', 'local_shell']},
    )
    return ToolDefinition(descriptor, lambda arguments, context=None: execute_skill_maintain(
        arguments, context=context, store=store, is_admin=is_admin), SCHEMA, accepts_context=True)


def execute_skill_maintain(arguments, *, context, store, is_admin=None):
    if context is None:
        raise PermissionError('skill_maintenance_context_required')
    task = context.task
    actor = str(task.user or '')
    if not actor or not (is_admin and is_admin(actor)):
        raise PermissionError('skill_manager_required')
    if task.status == 'cancelled' or (context.cancellation_checker and context.cancellation_checker()):
        raise PermissionError('skill_maintenance_cancelled')
    skill_id = str(arguments.get('skill_id') or '')
    if skill_id not in (task.metadata.get('v3_skill_selection') or {}).get('selected_skill_ids', []):
        raise PermissionError('skill_not_selected')
    record = require_skill(task, store, skill_id)
    if private_legacy_package(record, actor, context.artifact_resources):
        raise PermissionError('skill_private_legacy_resources_denied')
    if arguments.get('operation') == 'status':
        return store.management_status(skill_id)
    if arguments.get('operation') != 'apply':
        raise ValueError('skill_maintenance_operation_invalid')
    operation_id = str(arguments.get('operation_id') or '').strip()
    checks = arguments.get('checks')
    if not operation_id or not arguments.get('expected_revision'):
        raise ValueError('skill_maintenance_identity_required')
    if not isinstance(checks, list) or not 1 <= len(checks) <= 20 or not all(isinstance(c, str) and c.strip() for c in checks):
        raise ValueError('skill_maintenance_behavior_checks_required')
    request_hash = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
    with maintenance_lock(store, record.name):
        journal = operation_path(store, record.name, actor, task.task_id, operation_id)
        if journal.is_symlink():
            raise ValueError('skill_state_symlink_not_allowed')
        if journal.exists():
            prior = json.loads(journal.read_text())
            if prior['request_hash'] != request_hash:
                raise ValueError('skill_maintenance_operation_conflict: use a new ID only after reconciliation')
            if prior.get('status') == 'published' and store._digest(Path(store.get(skill_id).directory)) == prior['result']['revision']:
                _refresh_published_pin(task, store, skill_id, prior['result'], context.artifact_resources)
                return {**prior['result'], 'replayed': True}
            raise ValueError('skill_maintenance_reconciliation_required: inspect retained drafts, recovery copies and previous effects; do not repeat checks blindly')
        if store._digest(Path(record.directory)) != arguments['expected_revision']:
            raise ValueError('skill_maintenance_conflict: inspect current status before editing')
        from alphonse.agent_v2.skill_resources import pin_skill
        pin_skill(task, record, context.artifact_resources)
        entry = {'request_hash': request_hash, 'status': 'started', 'checks': []}
        save_json(journal, entry)  # Reserve before any validation can have effects.
        publication = None
        try:
            with draft_package(store, record, arguments.get('files')) as draft:
                revision = store._digest(draft)
                for command in checks:
                    result = execute_bash({'command': command, 'cwd': str(draft),
                                           'runtime': arguments.get('runtime'),
                                           'timeout_seconds': arguments.get('timeout_seconds', 30)}, context=context)
                    entry['checks'].append(result)
                    save_json(journal, entry)
                    if result['exit_code'] != 0 or result.get('cancelled') or result.get('timed_out'):
                        raise ValueError('skill_maintenance_validation_failed: active package preserved; inspect retained draft and check evidence before retrying')
                if context.cancellation_checker and context.cancellation_checker():
                    raise ValueError('skill_maintenance_cancelled')
                if store._digest(draft) != revision:
                    raise ValueError('skill_maintenance_validation_changed_package: checks must not mutate resources being verified')
                # Access may have been revoked while checks ran.
                if not is_admin(actor) or not store.can_use(skill_id, actor):
                    raise PermissionError('skill_maintenance_access_revoked')
                publication = publish_draft(store, record, draft, arguments['expected_revision'], owner=actor)
                publication['checks'] = entry['checks']
                entry.update(status='published', result=publication)
                save_json(journal, entry)
                _refresh_published_pin(task, store, skill_id, publication, context.artifact_resources)
                return publication
        except Exception as exc:
            if publication is not None:
                # Publication already happened. Never misreport it as rolled back
                # if receipt persistence or this task's activation refresh failed.
                return {**publication, 'exit_code': 1,
                        'exception': {'code': 'skill_maintenance_receipt_reconciliation_required',
                                      'message': 'Package published; receipt/activation refresh failed. Inspect the active revision and operation journal before resuming.'}}
            entry.update(status='blocked', error=str(exc))
            save_json(journal, entry)
            return {'exit_code': 1, 'published': False, 'error': str(exc), 'checks': entry['checks'],
                    'exception': {'code': 'skill_maintenance_failed', 'message': str(exc)},
                    'recovery_directory': str(store._state_root(record.name)),
                    'recovery': 'Preserve useful work; inspect the retained draft and prior effects. Reconcile the blocker before using a new operation ID. Missing dependencies require explicit setup; maintenance grants no account/data permissions.'}


def _refresh_published_pin(task, store, skill_id, result, resources):
    """Only a verified, authorized publication can advance its own task's pin.

    Other tasks and out-of-band edits still fail closed. The next projection loads
    the complete new instructions; evidence retains the old/new activation pins.
    """
    from alphonse.agent_v2.skill_resources import pin_skill
    selection = task.metadata['v3_skill_selection']
    pins = selection.setdefault('packages', {})
    previous = pins.pop(skill_id, None)
    current = pin_skill(task, store.get(skill_id), resources)
    transition = {'skill_id': skill_id, 'from': previous, 'to': current, 'revision': result['revision']}
    history = task.metadata.setdefault('v3_skill_publications', [])
    if previous != current:
        history.append(transition)
