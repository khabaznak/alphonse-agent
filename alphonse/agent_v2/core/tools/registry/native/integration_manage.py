"""Conversation access to the same account and check management boundary."""
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'action': {'type': 'string', 'enum': ['catalog', 'project', 'findings', 'review', 'apply_review', 'set_status',
        'save_grant', 'revoke_grant', 'bind_grant', 'select_binding', 'pause', 'disconnect', 'retry_deletion']},
    'values': {'type': 'object', 'description': 'Exact management inputs. project: project_id. review: definition, optional policy_id/expected_revision. apply_review: review_id, activate. set_status: policy_id, expected_revision, status. save_grant: grant_id, connection_id, expected_revision, beneficiary_user_id, operations, resources, recipient_user_ids, expires_at (ISO with offset), optional project_id. revoke_grant: grant_id, expected_revision. bind_grant: grant_id, skill_id. select_binding: binding_id, skill_id, name (skill role). pause/disconnect/retry_deletion: connection_id, expected_revision. Use exact catalog IDs; never invent recipients, resources or authority.'}},
    'required': ['action', 'values']}


def build_integration_manage_tool_definition(runtime):
    descriptor = ToolDescriptor('native.integration_manage', 'integration_manage', ToolKind.NATIVE,
        'Manage optional account grants and project checks for the authenticated requester. '
        'catalog returns owned account health, explicit grants, members and usable skill source choices. '
        'project returns own policies, run status and audience-authorized findings. '
        'Resolve missing goal, account/source, recipients, timezone, days/time, retrieval window, duration, '
        'lookback and budgets with ask_question before review. Connection alone never starts checks. '
        'review requires a complete definition: project_id, source_bindings (skill_id, role, binding_id, '
        'connection_id, grant_id, operation, resources), query {text}, trigger {kind:schedule, weekdays:[0..6], '
        'local_time:HH:MM}, timezone, retrieval_window {weekdays,start,end}, duration {ongoing,starts_at,ends_at}, '
        'budgets {max_pages,max_items,max_bytes,max_model_calls,max_runtime_seconds}, initial_lookback_days, '
        'recipient_user_ids, skill_hints and notification_mode. Show the concrete review summary. '
        'apply_review activates only within the requester\'s explicit instruction; use the returned review_id '
        'in the same task. Existing explicit recurring scope needs no approval per run. Revisions fence changes. '
        'Never substitute a scheduled prompt, widen a grant, guess missing scope, or treat monitoring as delivery. '
        'Provider consent is completed through Desktop Integrations. No tokens or browser consent URLs in chat.',
        SCHEMA, ('scheduling',), ('native', 'integrations'), read_only=False,
        metadata={'v3_capabilities': ['scheduling']})

    def execute(arguments, context=None):
        if context is None or not context.task.user:
            raise PermissionError('management_authenticated_task_required')
        if context.task.status == 'cancelled' or (context.cancellation_checker and context.cancellation_checker()):
            raise PermissionError('management_cancelled')
        if runtime is None or runtime.integration_management is None:
            raise ValueError('management_service_unavailable')
        return runtime.integration_management.request(AccountSession(context.task.user),
            arguments['action'], arguments['values'], channel='task:' + context.task.task_id)

    return ToolDefinition(descriptor, execute, SCHEMA, accepts_context=True)
