"""Coordinate isolated coding and verification providers with skill publication.

The service has no default provider: callers must supply independently isolated
coding and verification providers. Do not bind host Codex or trusted-host Bash as
those providers without a separately enforced filesystem/credential boundary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from alphonse.agent_v2.coding_jobs import CodingJobStore
from alphonse.agent_v2.skills import SkillStore, _has_private_state, _validate_skill_name


class CodingSkillService:
    def __init__(self, *, jobs: CodingJobStore, skills: SkillStore, workspace_root: Path,
                 provider, verifier, is_admin):
        self.jobs, self.skills = jobs, skills
        self.root = Path(workspace_root)
        self.provider, self.verifier, self.is_admin = provider, verifier, is_admin

    def reconcile(self, *, actor: str, task_id: str, job_id: str) -> dict:
        """Stop a known interrupted job; never infer publication or restart it."""
        if not actor or not self.is_admin(actor):
            raise PermissionError('coding_skill_manager_required')
        job = self.jobs.get(job_id=job_id, actor=actor, task_id=task_id)
        if job is None:
            raise PermissionError('coding_skill_job_not_found')
        if job['state'] == 'interrupted' and job['result_code'] == 'containers_removed':
            return job
        if job['state'] != 'running':
            raise ValueError('coding_skill_job_not_running')
        stop_build = getattr(self.provider, 'stop', None)
        stop_verify = getattr(self.verifier, 'stop', None)
        if not callable(stop_build) or not callable(stop_verify):
            raise RuntimeError('coding_skill_reconciliation_unavailable')
        job_key = job_id.removeprefix('coding-job:')
        if not stop_build(job_key) or not stop_verify(job_key):
            raise RuntimeError('coding_container_reconciliation_required')
        if not self.is_admin(actor):
            raise PermissionError('coding_skill_access_revoked')
        return self.jobs.transition(job_id=job_id, actor=actor, task_id=task_id, expected='running',
                                    state='interrupted', result_code='containers_removed')

    def build(self, *, actor: str, task_id: str, operation_id: str, name: str, brief: str, checks: list[str],
              cancelled=lambda: False) -> dict:
        if not actor or not self.is_admin(actor):
            raise PermissionError('coding_skill_manager_required')
        name = _validate_skill_name(name)
        if not isinstance(brief, str) or not brief.strip() or len(brief.encode()) > 8192:
            raise ValueError('coding_skill_brief_invalid')
        if not isinstance(checks, list) or not 1 <= len(checks) <= 20 or any(
                not isinstance(check, str) or not check.strip() or len(check) > 4096 for check in checks):
            raise ValueError('coding_skill_checks_required')
        if self.provider is None or self.verifier is None:
            raise RuntimeError('coding_skill_isolated_provider_unavailable')
        request = json.dumps({'name': name, 'brief': brief, 'checks': checks}, ensure_ascii=True, sort_keys=True)
        request_hash = hashlib.sha256(request.encode()).hexdigest()
        job = self.jobs.reserve(actor=actor, task_id=task_id, operation_id=operation_id, request_hash=request_hash)
        identity = dict(job_id=job['job_id'], actor=actor, task_id=task_id)
        if job['state'] == 'completed':
            record = self.skills.get(name)
            if record is None or self.skills._digest(Path(record.directory)) != job['result_code']:
                raise RuntimeError('coding_skill_publication_reconciliation_required')
            return {'job_id': job['job_id'], 'skill_id': record.skill_id, 'revision': job['result_code'], 'replayed': True}
        if job['state'] != 'queued':
            raise RuntimeError(f"coding_skill_job_reconciliation_required: job_id={job['job_id']} task_id={task_id}")
        if cancelled():
            self.jobs.transition(**identity, expected='queued', state='cancelled')
            raise RuntimeError('coding_skill_cancelled')
        self.jobs.transition(**identity, expected='queued', state='running')
        published = False
        try:
            if self.root.is_symlink() or not self.root.is_dir():
                raise ValueError('coding_skill_workspace_root_invalid')
            workspace = self.root / job['job_id'].split(':', 1)[1]
            workspace.mkdir(mode=0o700)  # Existing paths require reconciliation, not reuse.
            if self.skills.get(name) is not None:
                raise ValueError('coding_skill_already_exists')
            result = self.provider.run(workspace=workspace, name=name, brief=brief, cancelled=cancelled)
            if result != 'completed':
                raise RuntimeError('coding_skill_provider_failed')
            if cancelled():
                raise RuntimeError('coding_skill_cancelled')
            package = workspace / name
            if package.is_symlink() or not package.is_dir():
                raise ValueError('coding_skill_package_missing')
            record = self.skills._read_skill(package)
            if record.name != name or _has_private_state(package):
                raise ValueError('coding_skill_package_invalid')
            revision = self.skills._digest(package)
            if self.verifier.verify(package=package, checks=checks, cancelled=cancelled) is not True:
                raise ValueError('coding_skill_verification_failed')
            if cancelled():
                raise RuntimeError('coding_skill_cancelled')
            if not self.is_admin(actor):
                raise PermissionError('coding_skill_access_revoked')
            if self.skills._digest(package) != revision:
                raise ValueError('coding_skill_verification_changed_package')
            record = self.skills.install_directory(package, owner=actor)
            published = True
            installed_revision = self.skills._digest(Path(record.directory))
            if installed_revision != revision or cancelled() or not self.is_admin(actor):
                raise RuntimeError('coding_skill_publication_reconciliation_required')
            self.jobs.transition(**identity, expected='running', state='completed', result_code=revision)
            return {'job_id': job['job_id'], 'skill_id': record.skill_id, 'revision': revision, 'replayed': False}
        except Exception as exc:
            if published:
                raise RuntimeError('coding_skill_publication_reconciliation_required') from exc
            if str(exc) == 'coding_container_reconciliation_required':
                # The container may still be alive. Keep the running receipt so
                # actor-scoped reconciliation can remove it; never claim failure
                # implies that execution stopped.
                raise RuntimeError(f"coding_container_reconciliation_required: job_id={job['job_id']} task_id={task_id}") from exc
            state = 'cancelled' if cancelled() or str(exc) == 'coding_skill_cancelled' else 'failed'
            self.jobs.transition(**identity, expected='running', state=state, result_code='build_not_published')
            raise
