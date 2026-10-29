"""Recoverable skill-resource transactions. Scripts run only in the native layer."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from alphonse.agent_v2.skills import MAX_SKILL_FILE_BYTES, SKILL_FILE, _skill_path, _validate_package_files


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


@contextmanager
def maintenance_lock(store, name):
    """Serialize maintenance across tasks/processes; never follow state links."""
    path = store._state_root(name) / 'maintenance.lock'
    if path.is_symlink():
        raise ValueError('skill_state_symlink_not_allowed')
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@contextmanager
def draft_package(store, record, files):
    """Overlay reusable resources; never remove or edit household state."""
    from alphonse.agent_v2.skill_resources import reusable_path
    if not isinstance(files, dict) or not files or len(files) > 100:
        raise ValueError('skill_maintenance_files_required: supply 1-100 resource edits')
    if SKILL_FILE not in files:
        raise ValueError('skill_maintenance_instructions_required: include updated SKILL.md operating instructions')
    source = Path(record.directory)
    _validate_package_files(source)
    staging = Path(tempfile.mkdtemp(prefix='.skill-maintain-', dir=store.skills_dir))
    root = staging / record.name
    try:
        shutil.copytree(source, root)
        for relative, content in files.items():
            parts = Path(relative).parts
            if (relative != SKILL_FILE and (len(parts) < 2 or parts[0] not in {'scripts', 'references', 'assets'})) or not reusable_path(relative):
                raise ValueError('skill_maintenance_resource_path_required')
            path = _skill_path(root, relative)
            if not isinstance(content, str) or '\x00' in content or len(content.encode('utf-8')) > MAX_SKILL_FILE_BYTES:
                raise ValueError('skill_maintenance_text_resource_required')
            if path.is_file():
                try:
                    previous = path.read_text(encoding='utf-8')
                except UnicodeError:
                    raise ValueError('skill_file_not_text_editable') from None
                if '\x00' in previous:
                    raise ValueError('skill_file_not_text_editable')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        store._read_skill(root)
        _validate_package_files(root)
        yield root
    except BaseException:
        # Keep useful draft work for explicit reconciliation, outside the active package.
        if root.is_dir():
            recovery = store._state_root(record.name) / ('draft-' + uuid.uuid4().hex)
            root.rename(recovery)
        raise
    finally:
        # If preserving the draft or restoring the backup itself failed, keep
        # staging as evidence instead of deleting the only remaining resource.
        if not (staging / 'previous').exists() and not root.exists():
            shutil.rmtree(staging, ignore_errors=True)


def publish_draft(store, record, draft, expected_revision, *, owner):
    """Compare-and-publish with complete previous content retained and rollback.

    Two directory renames have a short unavailable interval. Readers fail closed;
    any ordinary exception restores the previous directory. A process crash keeps
    the backup/draft and an incomplete operation receipt for explicit recovery.
    """
    current = store.get(record.skill_id)
    if current is None or current.directory != record.directory or store._digest(Path(current.directory)) != expected_revision:
        raise ValueError('skill_maintenance_conflict: active package changed; reconcile before retrying')
    store._read_skill(draft)
    _validate_package_files(draft)
    target = store.skills_dir / record.name
    if target.is_symlink():
        raise ValueError('skill_operational_symlink_not_allowed')
    local = Path(record.directory) == target
    prior = store._provenance(record.name) if local else {}
    backup = store._retain(record, provenance=prior)
    provenance = prior or {'source': record.directory, 'base_revision': expected_revision,
                           'override': not local, 'owner': owner, 'audience': 'household'}
    provenance = {**provenance, 'installed_revision': store._digest(draft)}
    moved = draft.parent / 'previous'
    if local:
        target.rename(moved)
    installed = False
    try:
        draft.rename(target)
        installed = True
        save_json(store._state_root(record.name) / 'provenance.json', provenance)
    except BaseException:
        if installed:
            target.rename(draft)
        if local:
            moved.rename(target)
        raise
    if moved.exists():
        shutil.rmtree(moved, ignore_errors=True)
    return {'skill_id': record.skill_id, 'revision': provenance['installed_revision'],
            'recovery_directory': str(backup), 'published': True, 'exit_code': 0,
            'recovery': 'Restore reviewed resource files from the recovery directory in a new maintenance transaction; keep current household data and bindings.'}


def operation_path(store, name, actor, task_id, operation_id):
    identity = json.dumps([actor, task_id, operation_id], ensure_ascii=True)
    return store._state_root(name) / ('operation-' + hashlib.sha256(identity.encode()).hexdigest() + '.json')
