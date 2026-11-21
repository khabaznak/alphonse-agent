"""Opt-in direct-owner host Bash dispatch, not enabled by default.

Bash runs as the daemon's OS user with its host files, network and environment.
A same-user process can reach authority state; this adapter is not a sandbox or
an established multi-user authorization boundary. An explicit trusted local
development opt-in accepts this limitation; it is not a household release mode.

Only process facts are returned. They do not establish an external effect or
task success. The staged caller must reauthorize output and effect claims.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
from .owner_atomic_checkpoint import OwnerAtomicCheckpoint


def authorized_host_skill_context(context, task, arguments):
    """Recheck package access and revision; never infer authority from selection.

    Binding-backed execution remains unavailable here until the exact original
    binding identities and revisions can be persisted and revalidated. Generic
    host commands are unaffected and remain trusted host execution.
    """
    if not isinstance(arguments, dict) or 'binding_names' in arguments:
        return False
    if 'skill_id' not in arguments:
        return True
    try:
        from pathlib import Path
        from alphonse.agent_v2.skill_resources import authorized, package_revision, private_legacy_package
        skill_id = arguments['skill_id']
        selection = task.metadata.get('v3_skill_selection')
        store = context.skill_store
        if (not isinstance(skill_id, str) or not skill_id or not isinstance(selection, dict)
                or skill_id not in selection.get('selected_skill_ids', [])
                or not authorized(store, skill_id, str(task.user or ''))):
            return False
        pin = selection.get('packages', {}).get(skill_id)
        skill = store.get(skill_id)
        if not isinstance(pin, dict) or skill is None or set(pin) != {'root', 'revision'}:
            return False
        root = Path(skill.directory).resolve(strict=True)
        if (not root.is_dir() or not (root / 'SKILL.md').is_file()
                or str(root) != str(Path(pin['root']).resolve(strict=True))):
            return False
        private = private_legacy_package(skill, str(task.user or ''), context.artifact_resources)
        return package_revision(skill, private=private) == pin['revision']
    except Exception:
        return False


def valid_host_output(observation):
    """Validate bounded process facts without treating them as an effect."""
    return bool(isinstance(observation, dict)
        and set(observation) == {'exit_code', 'timed_out', 'cancelled', 'effect_status', 'stdout', 'stderr'}
        and type(observation['exit_code']) is int
        and type(observation['timed_out']) is bool and type(observation['cancelled']) is bool
        and observation['effect_status'] == 'unverified'
        and all(isinstance(observation[key], str) and len(observation[key].encode('utf-8')) <= 65536
                for key in ('stdout', 'stderr')))


def valid_host_binding(checkpoint, context, binding, history):
    """Reauthorize and verify persisted historical process output."""
    from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
    from .owner_atomic_checkpoint import _decision_digest, observation_digest
    from .owner_authority_bridge import context_authority_active
    try:
        args, output = binding['arguments'], binding['host_bash_observation']
        digest = observation_digest(output)
        return (context_authority_active(context) and getattr(context, '_owner_host_bash_enabled', False) is True
            and context.cancellation_checker() is False
            and checkpoint.queue.verify_trusted_local_owner_claim(checkpoint.receipt, checkpoint.task)
            and binding['actor'] == checkpoint.task.user and binding['tool_id'] == 'native.bash'
            and isinstance(args, dict) and authorized_host_skill_context(context, checkpoint.task, args)
            and valid_host_output(output)
            and binding['observation_sha256'] == digest and isinstance(history, list)
            and any(isinstance(item, dict) and item.get('claim_digest') == _claim_digest(checkpoint.receipt)
                and item.get('arguments_sha256') == _decision_digest(args)
                and item.get('observation_sha256') == digest and item.get('effect_status') == 'unverified'
                for item in history))
    except Exception:
        return False


def dispatch_owner_host_bash(*, atomic, context, arguments, reservation_id, authorize):
    """Run the exact reserved command once and preserve an unresolved receipt.

    The trusted host supplies ``authorize(actor, task_id, arguments)``; a
    model-provided flag is never authority. This point-in-time policy cannot
    confine a same-UID child or protect the daemon's authority state from it.
    """
    if (type(atomic) is not OwnerAtomicCheckpoint or not isinstance(context, CoreLoopContext)
            or context.messages is not atomic.queue or context.claim_receipt is not atomic.receipt
            or not isinstance(arguments, dict) or not isinstance(reservation_id, str)
            or not reservation_id or not callable(authorize)):
        raise ValueError('owner_host_bash_unavailable')
    task = atomic.task

    def stop():
        try:
            return (context.is_cancelled() is not False
                    or atomic.queue.verify_trusted_local_owner_claim(atomic.receipt, task) is not True
                    or authorize(task.user, task.task_id, deepcopy(arguments)) is not True
                    or atomic.check_claim(reservation_id, 'tool', tool_id='native.bash', arguments=arguments) is not True)
        except Exception:
            return True

    if stop():
        raise PermissionError('owner_host_bash_unavailable')
    # Persist a one-way dispatch marker first. A crash between this marker and
    # process start is deliberately uncertain and cannot cause a replay.
    atomic.mark_host_bash_dispatch(expected_version=atomic.version, reservation_id=reservation_id,
                                   arguments=arguments)
    if stop():
        raise PermissionError('owner_host_bash_unavailable')
    execution = replace(context.tool_execution_context(task), cancellation_checker=stop)
    prepared = []
    result = execute_bash(deepcopy(arguments), context=execution, _binding_observer=prepared.append)
    if len(prepared) != 1:
        raise ValueError('owner_host_bash_unavailable')
    # Store only the native executor's redacted output, using the exact bindings
    # used for this call. This is process evidence, not proof of an effect.
    atomic.record_bash_return(expected_version=atomic.version, reservation_id=reservation_id,
        arguments=arguments, exit_code=result['exit_code'], timed_out=result['timed_out'],
        cancelled=result['cancelled'], stdout=result['stdout'], stderr=result['stderr'],
        output_bindings=prepared[0])
    if stop():
        raise PermissionError('owner_host_bash_unavailable')
    return {'exit_code': result['exit_code'], 'stdout': result['stdout'], 'stderr': result['stderr'],
            'timed_out': result['timed_out'], 'cancelled': result['cancelled'],
            'effect_status': 'unverified'}
