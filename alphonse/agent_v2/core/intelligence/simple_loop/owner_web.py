"""Validate bounded, claim-scoped public-web observations for the owner loop.

Web content is a point-in-time untrusted source, not an instruction, proof of
truth, or authorization for subsequent effects. Recovery retains the observed
response and never repeats an external request.
"""
import json

from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry.native.web import WEB_FETCH_TOOL_ID, WEB_SEARCH_TOOL_ID

WEB_TOOL_IDS = frozenset((WEB_FETCH_TOOL_ID, WEB_SEARCH_TOOL_ID))
MAX_WEB_OBSERVATION_BYTES = 48000

# Only failures explicitly returned by the native read tools are known
# outcomes. Uncaught exceptions and malformed results remain unconfirmed.
_WEB_FAILURE_CODES = {
    WEB_SEARCH_TOOL_ID: frozenset({'searxng_timeout', 'searxng_http_error', 'searxng_invalid_json'}),
    WEB_FETCH_TOOL_ID: frozenset({'web_fetch_timeout', 'web_fetch_http_error', 'web_fetch_unresolvable_host',
        'web_fetch_output_too_large', 'web_fetch_output_invalid'}),
}


def web_failure_observation(tool_id, result):
    """Project an explicit tool failure without retaining external error text."""
    if (tool_id not in WEB_TOOL_IDS or not isinstance(result, dict) or result.get('status') != 'failed'
            or not isinstance(result.get('error'), dict) or not isinstance(result.get('result'), dict)):
        return None
    raw = result['result']
    if set(raw) != {'output', 'exception'} or raw['output'] is not None:
        return None
    exception = raw['exception']
    if (not isinstance(exception, dict)
            or set(exception) not in ({'code', 'message', 'retryable'},
                                      {'code', 'message', 'retryable', 'status_code'})
            or not isinstance(exception.get('message'), str)
            or type(exception.get('retryable')) is not bool):
        return None
    code = exception.get('code')
    if not isinstance(code, str) or code not in _WEB_FAILURE_CODES[tool_id] or result['error'].get('code') != code:
        return None
    # A remote authentication/access rejection is not a service outage and
    # must not be reframed as a reason to bypass that restriction.
    if 'status_code' in exception:
        status = exception['status_code']
        if type(status) is not int or not 400 <= status <= 599 or status in (401, 403):
            return None
    return {'status': 'failed', 'tool_id': tool_id, 'error_code': code}


def is_web_failure_observation(tool_id, observation):
    return (tool_id in WEB_TOOL_IDS and isinstance(observation, dict)
            and set(observation) == {'status', 'tool_id', 'error_code'}
            and observation['status'] == 'failed' and observation['tool_id'] == tool_id
            and isinstance(observation['error_code'], str)
            and observation['error_code'] in _WEB_FAILURE_CODES[tool_id])


def authorized_web_action(actor, tool_id, arguments, *, context, task):
    try:
        from .owner_authority_bridge import context_authority_active
        descriptor = context.tools.get(tool_id)
        if (tool_id not in WEB_TOOL_IDS or task.user != actor or context.is_cancelled() is not False
                or context_authority_active(context) is not True
                or descriptor is None or descriptor.read_only is not True or not isinstance(arguments, dict)):
            return False
        ToolInvocationService._validate(descriptor.argument_schema, arguments)
        return True
    except Exception:
        return False


def valid_web_observation(actor, tool_id, arguments, observation, *, context, task):
    try:
        if not authorized_web_action(actor, tool_id, arguments, context=context, task=task):
            return False
        if not isinstance(observation, dict) or 'exception' in observation:
            return False
        encoded = json.dumps(observation, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')
        if len(encoded) > MAX_WEB_OBSERVATION_BYTES:
            return False
        if is_web_failure_observation(tool_id, observation):
            return True
        if tool_id == WEB_SEARCH_TOOL_ID:
            return (set(observation) == {'query', 'provider', 'result_count', 'results', 'answers', 'suggestions'}
                    and observation['query'] == arguments.get('query', '').strip()
                    and observation['provider'] == 'searxng' and type(observation['result_count']) is int
                    and isinstance(observation['results'], list)
                    and observation['result_count'] == len(observation['results']) <= 20
                    and isinstance(observation['answers'], list) and isinstance(observation['suggestions'], list)
                    and all(isinstance(item, dict) for item in observation['results']))
        return (set(observation) == {'url', 'final_url', 'status_code', 'content_type', 'title', 'text', 'truncated'}
                and observation['url'] == arguments.get('url', '').strip()
                and all(isinstance(observation[key], str) for key in ('url', 'final_url', 'content_type', 'title', 'text'))
                and type(observation['status_code']) is int and 200 <= observation['status_code'] < 400
                and type(observation['truncated']) is bool)
    except (ValueError, TypeError, KeyError, AttributeError):
        return False
