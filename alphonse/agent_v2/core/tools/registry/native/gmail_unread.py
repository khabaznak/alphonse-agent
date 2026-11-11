"""Native on-demand Gmail count tool; actor identity comes only from the task."""
from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.services.project_check_worker import CheckStopped

SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    'label_id': {'type': 'string', 'description': 'Exact authorized Gmail label ID; omit for INBOX.'},
    'unit': {'type': 'string', 'enum': ['messages', 'threads'], 'description': 'Count unread messages or unread threads.'},
    'connection_id': {'type': 'string', 'description': 'Optional account filter. Multiple authorized accounts still require a verified user choice.'},
    'choice_question_id': {'type': 'string', 'description': 'ID of a previously answered server-issued Gmail account choice, if requested.'},
}, 'required': []}


def build_gmail_unread_tool_definition(service=None, *, runtime=None, delivery_is_private=None):
    descriptor = ToolDescriptor('native.gmail_unread_count', 'gmail_unread_count', ToolKind.NATIVE,
        'Read the current provider-reported unread message or thread count for one explicitly authorized Gmail label. '
        'Defaults to INBOX messages; multiple accounts currently require a verified selection flow. '
        'No project, monitoring, message contents, new grant, or mailbox mutation. '
        'Report the returned label, unit, observation time and any blocker honestly.',
        SCHEMA, ('mail_read',), ('native', 'gmail'), read_only=True,
        metadata={'v3_capabilities': ['mail_read']})

    def execute(arguments, context=None):
        if context is None or not context.task.user:
            raise PermissionError('gmail_authenticated_task_required')
        # An authenticated sender is not proof that the response destination is
        # private. A future integration must supply a trusted audience verifier;
        # model arguments and provider-origin metadata cannot grant access.
        def private_delivery():
            try:
                return callable(delivery_is_private) and delivery_is_private(context) is True
            except Exception:
                return False

        if not private_delivery():
            return {'status': 'blocked', 'reason': 'gmail_private_delivery_unverified'}

        def guard():
            if context.task.status == 'cancelled' or (context.cancellation_checker and context.cancellation_checker()):
                raise CheckStopped('cancelled', 'task_cancelled')
            if not private_delivery():
                raise CheckStopped('blocked', 'gmail_private_delivery_unverified')
        guard()
        active_service = service
        if active_service is None and runtime is not None:
            management = getattr(runtime, 'integration_management', None)
            accounts = getattr(management, 'accounts', None)
            oauth = getattr(management, 'oauth_factory', None) or getattr(management, 'oauth', None)
            if accounts is not None and oauth is not None:
                from alphonse.agent_v2.integrations.gmail_unread import GmailUnreadService
                active_service = GmailUnreadService(accounts, oauth)
        if active_service is None:
            return {'status': 'blocked', 'reason': 'gmail_count_service_unavailable'}
        return active_service.count(AccountSession(context.task.user), label_id=arguments.get('label_id', 'INBOX'),
                                    unit=arguments.get('unit', 'messages'), connection_id=arguments.get('connection_id'),
                                    choice_question_id=arguments.get('choice_question_id'),
                                    choice_store=getattr(context, 'question_store', None), task=context.task,
                                    guard=guard)
    return ToolDefinition(descriptor, execute, SCHEMA, accepts_context=True)
