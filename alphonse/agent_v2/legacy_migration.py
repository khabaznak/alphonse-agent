"""Explicit, metadata-only retirement of legacy registrations.

The journal is recovery/provenance, never a script registry. No household file is
moved, rewritten, executed or deleted. Original catalog rows stay as tombstones
for private-package containment and rollback, outside normal management APIs.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import replace
from pathlib import Path

from alphonse.agent_v2.artifact_resources import contained_path


class LegacyMigration:
    def __init__(self, resources, bindings):
        self.resources, self.bindings = resources, bindings
        resources.bindings = bindings
        with resources.store._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS v2_legacy_migrations (
                artifact_id TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL)""")

    @staticmethod
    def _write_instructions(path, content):
        import os
        import tempfile
        from alphonse.agent_v2.skills import MAX_SKILL_FILE_BYTES, _parse_skill_file
        _parse_skill_file(content)
        if len(content.encode('utf-8')) > MAX_SKILL_FILE_BYTES:
            raise ValueError('legacy_instruction_budget_exceeded')
        # Same-directory atomic replacement: interruption leaves before or after,
        # never a partially rewritten definition. No program/data files touched.
        fd, name = tempfile.mkstemp(prefix='.skill-migration-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(content)
                os.fchmod(stream.fileno(), path.stat().st_mode & 0o777)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    def _journal(self, identity):
        with self.resources.store._connect() as conn:
            row = conn.execute("SELECT state, payload FROM v2_legacy_migrations WHERE artifact_id=?", (identity,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def _state(self, identity, state):
        with self.resources.store._connect() as conn:
            conn.execute("UPDATE v2_legacy_migrations SET state=? WHERE artifact_id=?", (state, identity))

    def preview(self, actor):
        self.bindings._actor(actor)
        result = []
        for record in self.resources.store.list():
            if record.owner_user_id != actor and not self.resources.is_admin(actor):
                continue
            report = self.resources.reconcile(record, actor)
            report.update(source=record.to_dict(), skill_id=record.skill_id,
                          instructions=self.resources.skills.artifact_instructions(record.skill_id, record.artifact_id) if record.skill_id else '',
                          file_moves=[], preserved_data='All files remain in their original storage, including undeclared files.')
            report['review_token'] = self._token(report)
            journal = self._journal(record.artifact_id)
            report['migration_state'] = journal[0] if journal else 'pending'
            if journal:
                report['reviewed_mapping'] = {'skill_id': journal[1]['skill_id'], 'binding_name': journal[1]['binding_id'],
                                             'selected_location': journal[1]['selected'], 'verified_paths': journal[1]['detail']}
            if report.get('invocation_prefix'):
                report['suggested_command'] = 'cd ' + shlex.quote(report['working_directory']) + ' && ' + shlex.join(report['invocation_prefix'])
            result.append(report)
        return result

    @staticmethod
    def _token(report):
        # Deliberately no file contents/configuration values enter this report.
        return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()

    def apply(self, actor, identity, *, review_token, instructions, location=None):
        """Review token pins inventory. Ambiguous/stale copies need an explicit location.

        Instructions are manager-reviewed ordinary commands, not automatically
        rewritten shell text. No runtime repair or implicit package changes.
        """
        self.bindings._actor(actor)
        record = self.resources.store.get(identity)
        if record is None:
            raise KeyError('legacy_registration_missing')
        self.resources.require_manager(record, actor)
        journal = self._journal(identity)
        if journal and journal[0] != 'rolled_back':
            if journal[0] == 'complete':
                return {'state': 'complete', 'binding_name': journal[1]['binding_id'], 'files_moved': []}
            if journal[0] == 'rolling_back':
                raise ValueError('legacy_rollback_in_progress')
            return self._finish(actor, identity, journal[1])
        report = next(item for item in self.preview(actor) if item['artifact_id'] == identity)
        if report['review_token'] != review_token:
            raise ValueError('legacy_preview_changed')
        if report['reconciliation'] != 'registered' and location is None:
            raise ValueError('legacy_explicit_reconciliation_required')
        selected = record
        if location is not None:
            if not {'storage_kind', 'entrypoint_path', 'execution'} <= set(location) or set(location) - {'storage_kind', 'entrypoint_path', 'execution', 'skill_id', 'legacy_root_path'}:
                raise ValueError('legacy_location_invalid')
            selected = replace(record, **location)
        self.resources.validate(selected, allow_missing_configuration=False)
        detail = self.resources.describe(selected, actor, management=True)
        skill = self.resources.skills._editable(selected.skill_id)
        path = contained_path(Path(skill.directory), 'SKILL.md')
        before = path.read_text()
        instructions = str(instructions).strip()
        if not instructions or 'artifact_id' in instructions or 'ALPHONSE_ARTIFACT_' in instructions or identity in instructions:
            raise ValueError('legacy_standard_instructions_required')
        name = 'legacy-' + hashlib.sha256(identity.encode()).hexdigest()[:20]
        # Keep private paths/instructions outside the household-shared package.
        sections = re.split(r'(?m)(?=^## Artifact: )', before)
        after = ''.join(section for section in sections if f'Artifact ID: `{identity}`' not in section.partition('Operating instructions:')[0])
        note = f'\n\n## Retained resource {name}\n\nUse the actor-scoped `{name}` binding with native Bash and this skill. Read the authorized local resource guidance at activation.\n'
        after = after.rstrip() + note
        payload = dict(source=record.to_dict(), selected=selected.to_dict(), detail=detail,
                       skill_id=skill.skill_id, path=str(path), before=before, after=after,
                       instructions=instructions, binding_id=name, review_token=review_token)
        payload = json.loads(json.dumps(payload))
        with self.resources.store._connect() as conn:
            conn.execute("DELETE FROM v2_legacy_migrations WHERE artifact_id=? AND state='rolled_back'", (identity,))
            conn.execute('INSERT INTO v2_legacy_migrations VALUES (?, ?, ?)',
                         (identity, 'prepared', json.dumps(payload)))
        return self._finish(actor, identity, payload)

    def _finish(self, actor, identity, payload):
        from alphonse.agent_v2.artifacts import ArtifactRecord
        selected = ArtifactRecord(**payload['selected'])
        self.resources.require_manager(selected, actor)
        # Revalidate containment/readiness on resume; never trust the journal's paths.
        self.resources.validate(selected, allow_missing_configuration=False)
        detail = self.resources.describe(selected, actor, management=True)
        if json.loads(json.dumps(self.resources.store.get(identity).to_dict())) != payload['source']:
            raise ValueError('legacy_source_changed')
        if detail != payload['detail']:
            raise ValueError('legacy_resource_changed')
        path = contained_path(Path(self.resources.skills._editable(selected.skill_id).directory), 'SKILL.md')
        if str(path) != payload['path'] or path.read_text() not in (payload['before'], payload['after']):
            raise ValueError('legacy_skill_changed_reconcile_required')
        self.bindings.import_legacy(payload)
        if path.read_text() != payload['after']:
            self._write_instructions(path, payload['after'])
        self._state(identity, 'complete')
        return {'state': 'complete', 'binding_name': payload['binding_id'], 'files_moved': []}

    def rollback(self, actor, identity):
        self.bindings._actor(actor)
        journal = self._journal(identity)
        if not journal:
            raise KeyError('legacy_migration_missing')
        state, payload = journal
        self.resources.require_manager(self.resources.store.get(identity), actor)
        if state == 'rolled_back':
            return {'state': state}
        path = contained_path(Path(self.resources.skills._editable(payload['skill_id']).directory), 'SKILL.md')
        if str(path) != payload['path'] or path.read_text() not in (payload['after'], payload['before']):
            raise ValueError('legacy_rollback_conflict: reverse later migrations/edits first')
        self.bindings.check_legacy(payload)
        self._state(identity, 'rolling_back')
        # Revocation precedes restoration; interruption is safe to retry.
        self.bindings.remove_legacy(payload)
        self._write_instructions(path, payload['before'])
        self._state(identity, 'rolled_back')
        return {'state': 'rolled_back', 'files_moved': []}

    def guidance(self, skill_id, actor):
        self.bindings._actor(actor)
        result = []
        for record in self.resources.store.list():
            journal = self._journal(record.artifact_id)
            target_skill = journal[1]['skill_id'] if journal and journal[0] == 'complete' else record.skill_id
            if target_skill != skill_id or not self.resources.can_use(record, actor):
                continue
            if not journal or journal[0] != 'complete':
                result.append('A retained legacy resource requires explicit migration review before use.')
                continue
            payload = journal[1]
            try:
                self.bindings.prepare(actor_id=actor, skill_id=skill_id, names=[payload['binding_id']])
            except (ValueError, PermissionError):
                result.append('Retained resource binding unavailable or revoked; reconcile access/setup before use.')
                continue
            result.append(f"Binding {payload['binding_id']}: {payload['instructions']}\n"
                          f"Verified local paths/runtime: {json.dumps({k: v for k, v in payload['detail'].items() if k != 'artifact_id'})}\n"
                          'Use native.bash with skill_id and binding_names; use ordinary commands, '
                          'explicit cwd via cd, runtime and prerequisite checks. Verify effects by readback.')
        return '\n\n'.join(result)
