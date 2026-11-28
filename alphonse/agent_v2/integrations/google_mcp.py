"""Bounded, stateless transport for the official Google Workspace MCP servers."""
from datetime import datetime, timezone
import json
import re


_ENDPOINTS = {
    'gmail': 'https://gmailmcp.googleapis.com/mcp/v1',
    'calendar': 'https://calendarmcp.googleapis.com/mcp/v1',
}
_VERSION = '2026-07-28'
_TOOL_NAME = re.compile(r'[a-z][a-z0-9_]{0,63}')
_STATUS = {
    'UNAUTHENTICATED': 401,
    'PERMISSION_DENIED': 403,
    'NOT_FOUND': 404,
    'ABORTED': 409,
    'RESOURCE_EXHAUSTED': 429,
    'UNAVAILABLE': 503,
    'DEADLINE_EXCEEDED': 504,
}


class GoogleMcpError(Exception):
    """A safe failure. Provider text, content, and continuation state stay private."""

    def __init__(self, status: int, reason: str = 'request_failed'):
        self.status = status
        self.reason = reason
        super().__init__(reason)


def _error(wire, http_status):
    if not isinstance(wire, dict):
        wire = {}
    error = wire.get('error')
    result = wire.get('result')
    result = result if isinstance(result, dict) else {}
    if result.get('resultType') == 'input_required':
        return GoogleMcpError(409, 'approval_required')
    if isinstance(error, dict) and error.get('code') == -32021:
        return GoogleMcpError(409, 'approval_required')

    meta = result.get('_meta')
    structured = meta.get('google.com/error') if isinstance(meta, dict) else None
    if not isinstance(structured, dict):
        structured = error if isinstance(error, dict) else {}
    status = structured.get('code')
    if type(status) is not int or not 400 <= status <= 599:
        status = _STATUS.get(str(structured.get('status')), http_status if http_status >= 400 else 502)
    details = structured.get('details')
    if isinstance(details, list) and any(isinstance(item, dict) and
            item.get('reason') in ('SERVICE_DISABLED', 'accessNotConfigured') for item in details):
        return GoogleMcpError(403, 'service_disabled')

    # The live Google endpoint sometimes supplies only its fixed diagnostic.
    content = result.get('content')
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
        diagnostic = content[0].get('text')
        if isinstance(diagnostic, str):
            diagnostic = diagnostic.casefold()
            if status == 403 and 'has not been used in project' in diagnostic and 'or it is disabled' in diagnostic:
                return GoogleMcpError(403, 'service_disabled')
            if diagnostic.rstrip('.') in {'requested entity was not found', 'not found'}:
                return GoogleMcpError(404, 'not_found')
    return GoogleMcpError(status, 'unavailable' if status == 429 or status >= 500 else 'request_failed')


def _remaining(deadline):
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise GoogleMcpError(504, 'deadline')
    return remaining


def _read(response, deadline, max_bytes):
    received = 0
    buffer = bytearray()
    data = []
    event_stream = response.headers.get('Content-Type', '').split(';', 1)[0].strip() == 'text/event-stream'
    for chunk in response.iter_content(chunk_size=8192):
        _remaining(deadline)
        received += len(chunk)
        if received > max_bytes:
            raise GoogleMcpError(413, 'bytes_budget')
        buffer.extend(chunk)
        if event_stream:
            while (newline := buffer.find(b'\n')) >= 0:
                line = bytes(buffer[:newline]).removesuffix(b'\r')
                del buffer[:newline + 1]
                if line.startswith(b'data:'):
                    data.append(line[5:].removeprefix(b' '))
                elif not line and data:
                    envelope = json.loads(b'\n'.join(data))
                    data.clear()
                    if isinstance(envelope, dict) and 'id' in envelope:
                        return envelope, received
    _remaining(deadline)
    if event_stream:
        raise GoogleMcpError(502, 'response_invalid')
    return json.loads(buffer) if buffer else {}, received


class GoogleMcpClient:
    def call(self, product, tool, arguments, *, token, deadline, max_bytes, guard=None):
        """Call a reviewed adapter operation. Provider approval stops the call."""
        import requests

        if (product not in _ENDPOINTS or not isinstance(tool, str) or not _TOOL_NAME.fullmatch(tool)
                or not isinstance(arguments, dict) or not isinstance(token, str) or not token
                or type(max_bytes) is not int or max_bytes <= 0):
            raise GoogleMcpError(400, 'request_invalid')
        envelope = {
            'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
            'params': {'name': tool, 'arguments': arguments, '_meta': {
                'io.modelcontextprotocol/protocolVersion': _VERSION,
                'io.modelcontextprotocol/clientInfo': {'name': 'Alphonse', 'version': '2'},
                'io.modelcontextprotocol/clientCapabilities': {},
            }},
        }
        headers = {
            'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream', 'Accept-Encoding': 'identity',
            'MCP-Protocol-Version': _VERSION, 'Mcp-Method': 'tools/call', 'Mcp-Name': tool,
        }
        if guard is not None:
            guard()
        status = 0
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.post(_ENDPOINTS[product], json=envelope, headers=headers,
                                  timeout=min(20, _remaining(deadline)), stream=True,
                                  allow_redirects=False) as response:
                    status = response.status_code
                    if 300 <= status < 400:
                        raise GoogleMcpError(502, 'redirect_denied')
                    wire, size = _read(response, deadline, max_bytes)
        except requests.RequestException:
            raise GoogleMcpError(503, 'unavailable') from None
        except (ValueError, UnicodeError):
            if status >= 400:
                raise _error({}, status) from None
            raise GoogleMcpError(502, 'response_invalid') from None
        if status != 200:
            raise _error(wire, status)
        if (not isinstance(wire, dict) or wire.get('jsonrpc') != '2.0'
                or type(wire.get('id')) is not int or wire['id'] != 1):
            raise GoogleMcpError(502, 'response_invalid')
        result = wire.get('result')
        if ('error' in wire or not isinstance(result, dict) or result.get('isError')
                or result.get('resultType') == 'input_required'):
            raise _error(wire, status)
        payload = result.get('structuredContent')
        if result.get('resultType') not in (None, 'complete') or not isinstance(payload, dict):
            raise GoogleMcpError(502, 'response_invalid')
        if guard is not None:
            guard()
        return payload, size
