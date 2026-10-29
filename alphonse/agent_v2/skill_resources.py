"""Activation and contained, on-demand package reads (never script execution).

The current skill library is household-shared. A store's can_use policy is checked
on every load; legacy private Artifact sections/resources retain their own ACLs.
This resolver does not sandbox generic Bash or authorize live data bindings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NoReturn

from yaml import YAMLError

from alphonse.agent_v2.skills import MAX_SKILL_FILE_BYTES, _skill_path
from alphonse.agent_v2.system_one import SystemOneUnavailableError

MAX_PACKAGE_REVISION_FILES = 4096


class SkillActivationError(SystemOneUnavailableError):
    """A selected package cannot be used completely and consistently."""


def fail(task, reason: str, skill_id: str = "") -> NoReturn:
    selection = task.metadata.setdefault("v3_skill_selection", {})
    selection.update(status=reason, loaded_skill_ids=[], recovery=(
        "Restore access/the pinned package, or explicitly restart skill selection and replan. "
        "For instruction budget overflow, increase the configured budget or revise the package; "
        "essential instructions must never be clipped or silently omitted."
    ))
    raise SkillActivationError(f"v3_skill_activation:{reason}:{skill_id}")


def authorized(store, skill_id: str, actor: str) -> bool:
    # Missing policy fails closed. Selection itself is never the access check.
    return bool(actor and store is not None and callable(getattr(store, "can_use", None))
                and store.can_use(skill_id, actor))


def require_skill(task, store, skill_id: str):
    if not authorized(store, skill_id, str(task.user or "")):
        fail(task, "access_denied", skill_id)
    try:
        skill = store.get(skill_id)
    except (OSError, UnicodeError, ValueError, YAMLError):
        fail(task, "package_unavailable", skill_id)
    if skill is None:
        fail(task, "package_unavailable", skill_id)
    return skill


def private_legacy_package(skill, actor: str, resources) -> bool:
    # A legacy resource may store arbitrary private files anywhere inside the
    # package, not just at its declared entrypoint. Deny supporting-file reads
    # conservatively until that data is migrated into explicit bindings.
    return bool(resources is not None and any(
        record.skill_id == skill.skill_id and record.storage_kind == "skill"
        and not resources.can_use(record, actor) for record in resources.access_records()
    ))


def reusable_path(relative: str) -> bool:
    path = Path(relative)
    return not (any(part.startswith('.') or part in {'__pycache__', 'node_modules'} for part in path.parts)
                or path.suffix.lower() in {'.db', '.sqlite', '.sqlite3', '.db3', '.pem', '.key', '.pyc'}
                or path.name.lower() in {'credentials.json', 'secrets.json'})


def package_revision(skill, *, private: bool = False) -> str:
    """Fingerprint instructions and file identity without eagerly reading resources.

    Resource size/inode/mtime/ctime detect external edits conservatively (even a
    touch invalidates a task). Only an explicitly requested read hashes its bytes.
    This local activation fingerprint is not a portable lifecycle content digest.
    """
    root = Path(skill.directory).resolve()
    inventory = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if not reusable_path(relative) or (private and relative != 'SKILL.md'):
            continue
        resolved = _skill_path(root, relative)
        if not resolved.is_file():
            continue
        stat = resolved.stat()
        inventory.append((relative, stat.st_size, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns))
        if len(inventory) > MAX_PACKAGE_REVISION_FILES:
            raise ValueError('skill_package_revision_budget_exceeded')
    content = json.dumps({'instructions': skill.instructions, 'name': skill.name,
                          'description': skill.description, 'metadata': skill.package_metadata(),
                          'files': inventory}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def pin_skill(task, skill, resources=None) -> dict[str, Any]:
    private = private_legacy_package(skill, str(task.user or ''), resources)
    try:
        revision = package_revision(skill, private=private)
    except ValueError as exc:
        reason = 'revision_budget_exceeded' if str(exc) == 'skill_package_revision_budget_exceeded' else 'package_unavailable'
        fail(task, reason, skill.skill_id)
    except OSError:
        fail(task, 'package_unavailable', skill.skill_id)
    entry = {'root': skill.directory, 'revision': revision}
    pins = task.metadata.setdefault('v3_skill_selection', {}).setdefault('packages', {})
    previous = pins.get(skill.skill_id)
    if previous is not None and previous != entry:
        fail(task, 'package_changed_replan_required', skill.skill_id)
    pins[skill.skill_id] = entry
    return entry


def read_resource(arguments, *, context, store):
    if context is None:
        raise PermissionError('skill_resource_context_required')
    task = context.task
    if task.status == 'cancelled' or (context.cancellation_checker and context.cancellation_checker()):
        raise PermissionError('skill_resource_cancelled')
    skill_id = str(arguments.get('skill_id') or '')
    selection = task.metadata.get('v3_skill_selection') or {}
    if skill_id not in selection.get('selected_skill_ids', []):
        raise PermissionError('skill_not_selected')
    skill = require_skill(task, store, skill_id)
    resources = context.artifact_resources
    if private_legacy_package(skill, str(task.user or ''), resources):
        raise PermissionError('skill_private_legacy_resources_denied')
    pin = pin_skill(task, skill, resources)
    relative = str(arguments.get('path') or '.')
    if not reusable_path(relative):
        raise PermissionError('skill_resource_requires_data_binding')
    root = Path(skill.directory).resolve()
    path = _skill_path(root, relative)
    if path == root / 'SKILL.md':
        # Resolve before comparing: ./SKILL.md must not bypass legacy filtering.
        raise PermissionError('skill_instructions_use_activation')
    if resources is not None:
        for record in resources.access_records():
            if record.skill_id != skill.skill_id or record.storage_kind != 'skill':
                continue
            for value in record.execution.get('configuration_paths', []):
                if path == _skill_path(root, value):
                    raise PermissionError('skill_resource_requires_data_binding')
    if arguments.get('operation', 'read') == 'list':
        if not path.is_dir():
            raise ValueError('skill_resource_directory_not_found')
        entries = []
        for child in sorted(path.iterdir()):
            name = child.relative_to(root).as_posix()
            if not reusable_path(name) or child.is_symlink():
                continue
            entries.append({'path': name, 'kind': 'directory' if child.is_dir() else 'file',
                            'size_bytes': child.stat().st_size if child.is_file() else None})
            if len(entries) > 200:
                raise ValueError('skill_resource_listing_too_large: inspect a narrower directory')
        pin_skill(task, skill, resources)
        if task.status == 'cancelled' or (context.cancellation_checker and context.cancellation_checker()):
            raise PermissionError('skill_resource_cancelled')
        receipt = {'skill_id': skill.skill_id, 'path': relative, 'revision': pin['revision'], 'operation': 'list'}
        task.metadata.setdefault('v3_skill_resource_reads', []).append(receipt)
        return {**receipt, 'entries': entries}
    if not path.is_file():
        raise ValueError('skill_resource_not_found')
    if path.stat().st_size > MAX_SKILL_FILE_BYTES:
        raise ValueError('skill_resource_too_large: split the reference into smaller files')
    data = path.read_bytes()
    if len(data) > MAX_SKILL_FILE_BYTES:
        raise ValueError('skill_resource_too_large')
    try:
        content = data.decode('utf-8')
    except UnicodeError as exc:
        raise ValueError('skill_resource_binary: use an authorized native media capability') from exc
    if '\x00' in content:
        raise ValueError('skill_resource_binary')
    # Detect package edits during the read as well as between phase calls.
    pin_skill(task, skill, resources)
    if task.status == 'cancelled' or (context.cancellation_checker and context.cancellation_checker()):
        raise PermissionError('skill_resource_cancelled')
    receipt = {'skill_id': skill.skill_id, 'path': path.relative_to(root).as_posix(),
               'revision': pin['revision'], 'sha256': hashlib.sha256(data).hexdigest()}
    task.metadata.setdefault('v3_skill_resource_reads', []).append(receipt)
    return {**receipt, 'content': content}
