"""Optional Gmail reader. Fixed GET endpoints; no watch, send or mutation surface.

History is a change signal, not a query engine. Changed mailboxes get a bounded
query refresh using the original baseline lower bound. This deliberately trades
extra reads for truthful query/label coverage (including archived mail).
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.message import EmailMessage
import hashlib
import json
import re
from urllib.parse import quote

from alphonse.agent_v2.integrations.acquisition import CursorExpired, SourcePage, SourceRecord
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped

SCOPE = SCOPES['google']['mail_read']
ALL_MAIL = 'gmail:all'  # Explicit mailbox-wide grant, including Spam and Trash.
_ID = re.compile(r'^[A-Za-z0-9_-]{1,4096}$')


class ScopeMoved(PermissionError):
    pass


class GmailHTTPError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__('gmail_http_error')


class GmailTransport:
    """Lazy requests transport, streamed decoded byte cap and no redirects/retries."""
    def get(self, path, *, params, token, deadline, max_bytes):
        import requests
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise CheckStopped('partial', 'gmail_deadline')
        try:
            with requests.get('https://gmail.googleapis.com/gmail/v1/users/me/' + path,
                    params=params, headers={'Authorization': 'Bearer ' + token},
                    timeout=min(20, remaining), stream=True, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise GmailHTTPError(response.status_code)
                body = bytearray()
                for chunk in response.iter_content(chunk_size=8192):
                    if datetime.now(timezone.utc) >= deadline:
                        raise CheckStopped('partial', 'gmail_deadline')
                    if len(body) + len(chunk) > max_bytes:
                        raise CheckStopped('partial', 'gmail_bytes_budget')
                    body.extend(chunk)
                return json.loads(body), len(body)
        except requests.RequestException:
            raise CheckStopped('partial', 'gmail_transport_unavailable') from None
        except (ValueError, UnicodeError):
            raise CheckStopped('partial', 'gmail_response_invalid') from None


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('gmail_id_invalid')
    return quote(value, safe='')


def _decode(value):
    try:
        return base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)
    except (ValueError, TypeError):
        raise ValueError('gmail_content_invalid') from None


class GmailAdapter:
    version = 'gmail-readonly-1'

    def __init__(self, oauth, *, transport=None, clock=None):
        # oauth may be a lazy session -> OAuthService factory; constructing this
        # adapter never loads credentials, optional packages or provider clients.
        self.oauth = oauth
        self.transport = transport or GmailTransport()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _authorize(self, context):
        context.guard()
        if self.clock() >= context.deadline:
            raise CheckStopped('partial', 'gmail_deadline')
        h = context.grant
        if h.provider_key != 'google' or h.operation != 'mail_read' or not h.resources:
            raise PermissionError('gmail_scope_denied')
        for r in h.resources:
            if r != ALL_MAIL:
                _id(r)
        oauth = self.oauth(context.session) if callable(self.oauth) else self.oauth
        authorization = dict(connection_id=h.connection_id, grant_id=h.grant_id,
            operation='mail_read', resources=list(h.resources), recipient_user_ids=list(h.recipient_user_ids),
            required_scopes=[SCOPE], project_id=context.project_id, expected_grant_revision=h.grant_revision)
        oauth.accounts.authorize(context.session, **authorization)
        return oauth, authorization

    def _get(self, context, path, params, budget):
        oauth, authorization = self._authorize(context)
        if budget[0] <= 0:
            raise CheckStopped('partial', 'gmail_bytes_budget')
        token = oauth.token(context.session, authorization=authorization)
        try:
            value, size = self.transport.get(path, params=params, token=token,
                deadline=context.deadline, max_bytes=budget[0])
        except GmailHTTPError as exc:
            if exc.status == 401:
                raise CheckStopped('blocked', 'gmail_reauth_required') from None
            if exc.status == 403:
                raise CheckStopped('blocked', 'gmail_read_denied') from None
            if exc.status == 429 or exc.status >= 500:
                raise CheckStopped('partial', 'gmail_throttled_or_unavailable') from None
            raise
        if type(size) is not int or size < 0 or size > budget[0] or not isinstance(value, dict):
            raise CheckStopped('partial', 'gmail_response_budget')
        budget[0] -= size
        context.on_bytes(size)
        # Revocation/cancellation during I/O fences this result and the next call.
        self._authorize(context)
        oauth.accounts.authorize(context.session, **authorization)
        return value

    def health(self, context):
        value = self._get(context, 'profile', {}, [context.limits.bytes])
        return {'provider': 'google', 'status': 'healthy', 'history_available': bool(value.get('historyId'))}

    def capabilities(self, context):
        self._authorize(context)
        return ('search', 'read', 'attachment', 'changes')

    def _key(self, context, query):
        if set(query) != {'text'} or not isinstance(query['text'], str) or len(query['text']) > 4096:
            raise ValueError('gmail_query_invalid')
        h = context.grant
        return hashlib.sha256(json.dumps([h.connection_id, h.grant_id, h.grant_revision,
            h.resources, query], sort_keys=True).encode()).hexdigest()

    def _cursor(self, context, query, cursor):
        try:
            state = json.loads(cursor)
            if state['key'] != self._key(context, query):
                raise ValueError()
            return state
        except (ValueError, TypeError, KeyError):
            raise CursorExpired() from None

    def _stamp(self, state):
        return json.dumps(state, sort_keys=True)

    def _record(self, context, resource, source_id, budget):
        value = self._get(context, 'messages/' + _id(source_id), {'format': 'full'}, budget)
        if value.get('id') != source_id:
            raise ValueError('gmail_identity_mismatch')
        if resource != ALL_MAIL and resource not in value.get('labelIds', []):
            raise ScopeMoved('gmail_message_outside_scope')
        timestamp = datetime.fromtimestamp(int(value['internalDate']) / 1000, timezone.utc).isoformat()
        # Convert provider MIME to inert RFC822, leaving attachment data unopened.
        count = [0]
        def mime(part):
            count[0] += 1
            if count[0] > 256:
                raise CheckStopped('partial', 'gmail_mime_budget')
            msg = EmailMessage()
            kind = part.get('mimeType', 'application/octet-stream')
            if kind.startswith('multipart/'):
                msg.set_type(kind)
                for child in part.get('parts', []):
                    msg.attach(mime(child))
            elif kind in {'text/plain', 'text/html'} and not part.get('filename') and not part.get('body', {}).get('attachmentId'):
                msg.set_type(kind)
                for header in part.get('headers', []):
                    if header['name'].lower() == 'content-type':
                        original = EmailMessage()
                        original['Content-Type'] = header['value']
                        charset = original.get_content_charset()
                        if charset:
                            msg.set_param('charset', charset)
                msg['Content-Transfer-Encoding'] = 'base64'
                msg.set_payload(base64.b64encode(_decode(part.get('body', {}).get('data', ''))).decode())
            else:
                msg.set_type(kind)
                msg['Content-Disposition'] = 'attachment'
                attachment_id = part.get('body', {}).get('attachmentId')
                if attachment_id:
                    msg['X-Alphonse-Attachment-Id'] = _id(attachment_id)
                msg.set_payload('')
            return msg
        msg = mime(value.get('payload', {}))
        for header in value.get('payload', {}).get('headers', []):
            if header['name'].lower() in {'subject', 'from', 'to', 'date', 'message-id'}:
                # Header injection is rejected by the standard email library.
                msg[header['name']] = header['value']
        return SourceRecord(resource, source_id, str(value['historyId']), timestamp, msg.as_bytes())

    def search(self, context, *, query, since, cursor=None):
        self._authorize(context)
        if context.limits.pages <= 0 or context.limits.items <= 0:
            raise CheckStopped('partial', 'gmail_page_budget')
        budget = [context.limits.bytes]
        state = self._cursor(context, query, cursor) if cursor else dict(key=self._key(context, query),
            mode='scan', since=since.isoformat(), resource_index=0, page=None,
            history=str(self._get(context, 'profile', {}, budget)['historyId']))
        if state['mode'] != 'scan':
            raise CursorExpired()
        resource = context.grant.resources[state['resource_index']]
        # Parenthesize user query: OR must never escape the bounded time clause.
        lower = int(datetime.fromisoformat(state['since']).timestamp()) - 1
        params = dict(q=f"({query['text']}) after:{lower}" if query['text'].strip() else f'after:{lower}',
            maxResults=min(100, context.limits.items), includeSpamTrash='true')
        if resource != ALL_MAIL:
            params['labelIds'] = resource
        if state['page']:
            params['pageToken'] = state['page']
        try:
            value = self._get(context, 'messages', params, budget)
        except GmailHTTPError as exc:
            if exc.status in {400, 404}:
                raise CursorExpired() from None
            raise
        records, coverage = [], True
        for message in value.get('messages', []):
            if len(records) >= context.limits.items:
                raise CheckStopped('partial', 'gmail_item_budget')
            try:
                records.append(self._record(context, resource, message['id'], budget))
            except GmailHTTPError as exc:
                if exc.status != 404:
                    raise
                coverage = False  # Deleted between list and get; never quiet no-match.
            except ScopeMoved:
                coverage = False  # Label changed during this scan.
        state['page'] = value.get('nextPageToken')
        if not state['page']:
            state['resource_index'] += 1
        complete = not state['page'] and state['resource_index'] == len(context.grant.resources)
        if complete:
            state.update(mode='history', page=None, resource_index=0)
        return SourcePage(tuple(records), self._stamp(state), complete, coverage,
                          context.limits.bytes - budget[0])

    def changes(self, context, *, query, cursor):
        self._authorize(context)
        if context.limits.pages <= 0 or context.limits.items <= 0:
            raise CheckStopped('partial', 'gmail_page_budget')
        state = self._cursor(context, query, cursor)
        if state['mode'] == 'scan':
            return self.search(context, query=query, since=datetime.fromisoformat(state['since']), cursor=cursor)
        budget = [context.limits.bytes]
        resource = context.grant.resources[state['resource_index']]
        params = dict(startHistoryId=state['history'], maxResults=min(100, context.limits.items))
        if resource != ALL_MAIL:
            params['labelId'] = resource
        if state.get('page'):
            params['pageToken'] = state['page']
        try:
            value = self._get(context, 'history', params, budget)
        except GmailHTTPError as exc:
            if exc.status == 404:
                raise CursorExpired() from None
            raise
        records = {}
        for entry in value.get('history', []):
            for removed in entry.get('labelsRemoved', []):
                for removed_resource in set(removed.get('labelIds', [])) & {resource}:
                    message = removed['message']
                    records[(removed_resource, message['id'])] = SourceRecord(removed_resource, message['id'],
                        str(entry['id']), self.clock().isoformat(), b'', deleted=True)
            for deleted in entry.get('messagesDeleted', []):
                message = deleted['message']
                # The history request itself is label-scoped. Tombstones carry no content.
                r = SourceRecord(resource, message['id'], str(entry['id']),
                    self.clock().isoformat(), b'', deleted=True)
                records[(resource, message['id'])] = r
        if len(records) > context.limits.items:
            raise CheckStopped('partial', 'gmail_item_budget')
        state['dirty'] = state.get('dirty', False) or bool(value.get('history'))
        state['page'] = value.get('nextPageToken')
        complete = False
        state['next_history'] = str(min(int(state.get('next_history', value['historyId'])), int(value['historyId'])))
        if not state['page']:
            state['resource_index'] += 1
            if state['resource_index'] == len(context.grant.resources):
                state['history'] = state.pop('next_history')
                state['resource_index'] = 0
                if state.pop('dirty', False):
                    state['mode'] = 'scan'
                else:
                    complete = True
        return SourcePage(tuple(records.values()), self._stamp(state), complete, True,
                          context.limits.bytes - budget[0])

    def read(self, context, *, resource, source_id):
        if resource not in context.grant.resources:
            raise PermissionError('gmail_resource_denied')
        return self._record(context, resource, source_id, [context.limits.bytes])

    def attachment(self, context, *, resource, source_id, attachment_id, max_bytes):
        if resource not in context.grant.resources:
            raise PermissionError('gmail_resource_denied')
        if not 0 < max_bytes <= min(context.limits.bytes, 5 * 1024 * 1024):
            raise ValueError('gmail_attachment_budget')
        budget = [context.limits.bytes]
        value = self._get(context, 'messages/' + _id(source_id), {'format': 'full'}, budget)
        if resource != ALL_MAIL and resource not in value.get('labelIds', []):
            raise ScopeMoved('gmail_message_outside_scope')
        if value.get('id') != source_id:
            raise ValueError('gmail_identity_mismatch')
        pending, found, count = [value.get('payload', {})], None, 0
        while pending:
            part = pending.pop()
            count += 1
            if count > 256:
                raise ValueError('gmail_mime_budget')
            if part.get('body', {}).get('attachmentId') == attachment_id:
                found = part
            pending.extend(part.get('parts', []))
        if found is None:
            raise CheckStopped('partial', 'gmail_attachment_missing')
        if found.get('mimeType') not in {'text/plain', 'text/csv', 'application/pdf'}:
            raise PermissionError('gmail_attachment_type_denied')
        if found['body'].get('size', max_bytes + 1) > max_bytes:
            raise CheckStopped('partial', 'gmail_attachment_budget')
        try:
            value = self._get(context, 'messages/' + _id(source_id) + '/attachments/' + _id(attachment_id), {}, budget)
        except GmailHTTPError as exc:
            if exc.status == 404:
                raise CheckStopped('partial', 'gmail_attachment_missing') from None
            raise
        raw = _decode(value.get('data', ''))
        if len(raw) > max_bytes or value.get('size') != len(raw):
            raise CheckStopped('partial', 'gmail_attachment_budget')
        return raw
