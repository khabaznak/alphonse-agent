"""Native entrypoint for an explicitly configured isolated coding service."""
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'operation': {'type': 'string', 'enum': ['build', 'reconcile'], 'default': 'build'},
        'name': {'type': 'string', 'minLength': 1, 'maxLength': 64},
        'brief': {'type': 'string', 'minLength': 1, 'maxLength': 8192,
                  'description': 'Authorized specification and acceptance behavior; omit secrets and private household data.'},
        'operation_id': {'type': 'string', 'minLength': 1, 'maxLength': 128,
                         'description': 'Stable operation identity for retry/reconciliation.'},
        'checks': {'type': 'array', 'minItems': 1, 'maxItems': 20,
                   'items': {'type': 'string', 'minLength': 1, 'maxLength': 4096},
                   'description': 'Independent sandboxed shell checks of the requested behavior using synthetic inputs.'},
        'job_id': {'type': 'string', 'description': 'Exact interrupted job ID for reconciliation.'},
        'target_task_id': {'type': 'string', 'description': 'Original task ID for an explicitly requested recovery.'},
    },
    'allOf': [{
        'if': {'properties': {'operation': {'const': 'reconcile'}}, 'required': ['operation']},
        'then': {'required': ['job_id', 'target_task_id']},
        'else': {'required': ['name', 'brief', 'operation_id', 'checks']},
    }],
}


def build_coding_skill_tool_definition(service):
    descriptor = ToolDescriptor(
        'native.coding_skill_build', 'coding_skill_build', ToolKind.NATIVE,
        'Delegate creation of a standard skill package to the configured isolated coding service. '
        'Provide a concise task specification, acceptance checks and a stable operation ID; omit private data and credentials. '
        'This publishes a verified skill but does not complete the household task or authorize external effects. '
        'After installation, select the new skill and use its documented native capabilities to complete and verify the original request. '
        'An interrupted job requires explicit reconcile with its original task and job IDs; this stops both containers '
        'and never reruns work or claims a skill was installed. Inspect the job and publication before retrying.',
        SCHEMA, ('skill_management',), ('native', 'skills', 'coding'), read_only=False,
        metadata={'v3_capabilities': ['skill_management'], 'side_effect_class': 'local_skill_installation'},
    )
    def execute(arguments, context=None):
        if context is None or not context.task.task_id:
            raise PermissionError('coding_skill_context_required')
        task = context.task
        if arguments.get('operation', 'build') == 'reconcile':
            return service.reconcile(actor=str(task.user or ''), task_id=arguments['target_task_id'], job_id=arguments['job_id'])
        cancelled = lambda: task.status == 'cancelled' or bool(
            context.cancellation_checker and context.cancellation_checker())
        result = service.build(actor=str(task.user or ''), task_id=str(task.task_id),
                               operation_id=arguments['operation_id'], name=arguments['name'],
                               brief=arguments['brief'], checks=arguments['checks'], cancelled=cancelled)
        # Installation is an intermediate effect. Pin the published package for
        # this task so the next tactical action can use its instructions, without
        # treating the household request as complete or granting new authority.
        try:
            from pathlib import Path
            from alphonse.agent_v2.skill_resources import require_skill, pin_skill
            skill_id = result['skill_id']
            skill = require_skill(task, service.skills, skill_id)
            if cancelled() or service.skills._digest(Path(skill.directory)) != result['revision']:
                raise ValueError('coding_skill_activation_conflict')
            pin = pin_skill(task, skill, context.artifact_resources)
            selection = task.metadata.setdefault('v3_skill_selection', {})
            selected = selection.setdefault('selected_skill_ids', [])
            if skill_id not in selected:
                selected.append(skill_id)
            task.metadata.setdefault('v3_skill_publications', []).append({
                'skill_id': skill_id, 'from': None, 'to': pin, 'revision': result['revision'],
            })
            return {**result, 'activated_for_task': True}
        except Exception as exc:
            return {**result, 'activated_for_task': False,
                    'exception': {'code': 'coding_skill_activation_reconciliation_required',
                                  'message': 'Skill was published but activation could not be verified; reconcile it before use.'}}
    return ToolDefinition(descriptor, execute, SCHEMA, accepts_context=True)
