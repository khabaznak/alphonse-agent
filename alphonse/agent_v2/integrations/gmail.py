"""Managed Gmail reads through the official Google Workspace MCP server."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr
import hashlib
import json
import re
from urllib.parse import quote

from alphonse.agent_v2.accounts import GMAIL_CONTENT_OPERATION, exact_gmail_label_id, valid_gmail_label_name
from alphonse.agent_v2.integrations.acquisition import CursorExpired, SourcePage, SourceRecord
from alphonse.agent_v2.integrations.google_mcp import GoogleMcpClient, GoogleMcpError
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped

SCOPE = SCOPES['google']['mail_read']
ALL_MAIL = 'gmail:all'  # Explicit mailbox-wide grant, including Spam and Trash.
GMAIL_LABEL_SEARCH_MAX = 20
_ID = re.compile(r'^[A-Za-z0-9_-]{1,4096}$')
_MIME_TYPE = re.compile(r'[a-zA-Z0-9.+_-]+/[a-zA-Z0-9.+_-]+')


class ScopeMoved(PermissionError):
    pass


class GmailHTTPError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__('gmail_http_error')


class GmailTransport:
    """Fixed Gmail capabilities with shared transport budgets and no RAW reads."""

    def __init__(self, client=None):
        self.client = client if client is not None else GoogleMcpClient()

    def get(self, path, *, params, token, deadline, max_bytes, guard=None):
        if path == 'labels' or path.startswith('labels/'):
            tool, arguments = 'list_labels', {}
        elif path == 'search_threads':
            tool, arguments = 'search_threads', params
        elif path.startswith('messages/'):
            message_id = path.removeprefix('messages/')
            _id(message_id)
            tool, arguments = 'get_message', {'messageId': message_id,
                'messageFormat': {'metadata': 'METADATA_ONLY', 'full': 'FULL_CONTENT'}[params['format']]}
        else:
            raise CheckStopped('partial', 'gmail_mcp_operation_unavailable')
        try:
            value, size = self.client.call('gmail', tool, arguments, token=token,
                deadline=deadline, max_bytes=max_bytes, guard=guard)
        except GoogleMcpError as exc:
            if exc.reason == 'approval_required':
                raise CheckStopped('blocked', 'gmail_provider_approval_required') from None
            if exc.reason in ('deadline', 'bytes_budget', 'response_invalid'):
                raise CheckStopped('partial', 'gmail_' + exc.reason) from None
            if exc.status in (400, 401, 403, 404, 410, 429) or exc.status >= 500:
                raise GmailHTTPError(exc.status) from None
            raise CheckStopped('partial', 'gmail_transport_unavailable') from None
        if tool != 'list_labels':
            return value, size
        labels = _mcp_labels(value)
        if path == 'labels':
            return {'labels': list(labels.values())}, size
        return labels.get(path.removeprefix('labels/'), {}), size


def _mcp_labels(value):
    items = value.get('labels', []) if isinstance(value, dict) else None
    if not isinstance(items, list) or len(items) > 10100:
        raise CheckStopped('partial', 'gmail_response_invalid')
    labels = {}
    for item in items:
        if not isinstance(item, dict):
            raise CheckStopped('partial', 'gmail_response_invalid')
        label_id = item.get('labelId')
        if not isinstance(label_id, str) or not _ID.fullmatch(label_id) or label_id in labels:
            raise CheckStopped('partial', 'gmail_response_invalid')
        labels[label_id] = {'id': label_id, **{key: item[key] for key in
            ('name', 'messagesUnread', 'threadsUnread') if key in item}}
    return labels


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('gmail_id_invalid')
    return quote(value, safe='')


def _decode(value):
    try:
        return base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)
    except (ValueError, TypeError):
        raise ValueError('gmail_content_invalid') from None


def _mcp_query_group(query):
    """Keep caller query syntax inside the independent provider scope clauses."""
    groups = []
    quoted = escaped = False
    endings = {')': '(', '}': '{'}
    for char in query:
        if not char.isprintable():
            raise ValueError('gmail_query_invalid')
        if escaped:
            escaped = False
        elif char == '\\':
            if not quoted:
                raise ValueError('gmail_query_invalid')
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted:
            if char in '({':
                groups.append(char)
            elif char in endings and (not groups or groups.pop() != endings[char]):
                raise ValueError('gmail_query_invalid')
    if groups or quoted or escaped:
        raise ValueError('gmail_query_invalid')
    return f'({query})' if query else ''


class GmailAdapter:
    version = 'gmail-mcp-readonly-1'

    def __init__(self, oauth, *, transport=None, clock=None):
        # oauth may be a lazy session -> OAuthService factory; constructing this
        # adapter never loads credentials, optional packages or provider clients.
        self.oauth = oauth
        self.transport = transport if transport is not None else GmailTransport()
        self._mcp = isinstance(self.transport, GmailTransport)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _authorize(self, context, operation='mail_read'):
        context.guard()
        if self.clock() >= context.deadline:
            raise CheckStopped('partial', 'gmail_deadline')
        h = context.grant
        if h.provider_key != 'google' or h.operation != operation or not h.resources:
            raise PermissionError('gmail_scope_denied')
        for r in h.resources:
            if r != ALL_MAIL:
                _id(r)
        oauth = self.oauth(context.session) if callable(self.oauth) else self.oauth
        authorization = dict(connection_id=h.connection_id, grant_id=h.grant_id,
            operation=operation, resources=list(h.resources), recipient_user_ids=list(h.recipient_user_ids),
            required_scopes=[SCOPE], project_id=context.project_id, expected_grant_revision=h.grant_revision)
        if operation == GMAIL_CONTENT_OPERATION:
            authorization['owner_gmail_content'] = True
        oauth.accounts.authorize(context.session, **authorization)
        return oauth, authorization

    def _get(self, context, path, params, budget, *, operation='mail_read'):
        oauth, authorization = self._authorize(context, operation)
        if budget[0] <= 0:
            raise CheckStopped('partial', 'gmail_bytes_budget')
        token = oauth.token(context.session, authorization=authorization)
        context.guard()
        try:
            options = {'guard': context.guard} if self._mcp else {}
            value, size = self.transport.get(path, params=params, token=token,
                deadline=context.deadline, max_bytes=budget[0], **options)
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
        self._authorize(context, operation)
        oauth.accounts.authorize(context.session, **authorization)
        return value

    def health(self, context):
        if self._mcp:
            self._get(context, 'labels', {}, [context.limits.bytes])
            return {'provider': 'google', 'status': 'healthy', 'history_available': False}
        value = self._get(context, 'profile', {}, [context.limits.bytes])
        return {'provider': 'google', 'status': 'healthy', 'history_available': bool(value.get('historyId'))}

    def capabilities(self, context):
        self._authorize(context)
        if self._mcp:
            return ('search', 'read')
        return ('search', 'read', 'attachment', 'changes')

    def _key(self, context, query):
        if set(query) != {'text'} or not isinstance(query['text'], str) or len(query['text']) > 4096:
            raise ValueError('gmail_query_invalid')
        h = context.grant
        return hashlib.sha256(json.dumps([self.version, context.session.actor_user_id, context.project_id,
            h.connection_id, h.binding_id, h.connection_revision, h.grant_id, h.grant_revision,
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

    @staticmethod
    def _mcp_message_scope(value, resource, source_id, operation, window):
        if value.get('id') != source_id:
            raise ValueError('gmail_identity_mismatch')
        labels = value.get('labelIds')
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError('gmail_message_labels_missing')
        if resource != ALL_MAIL and resource not in labels:
            raise ScopeMoved('gmail_message_outside_scope')
        if (operation == GMAIL_CONTENT_OPERATION and resource not in ('SPAM', 'TRASH')
                and ('SPAM' in labels or 'TRASH' in labels)):
            raise ScopeMoved('gmail_message_outside_scope')
        stamp = value.get('internalDate')
        if not isinstance(stamp, str) or not stamp.isascii() or not stamp.isdecimal() or len(stamp) > 16:
            raise ValueError('gmail_message_time_invalid')
        received = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=int(stamp))
        if window is not None and not window[0] <= received <= window[1]:
            raise ScopeMoved('gmail_message_outside_window')
        return received

    def _mcp_record(self, context, resource, source_id, budget, *, operation, window):
        path = 'messages/' + _id(source_id)
        metadata = self._get(context, path, {'format': 'metadata'}, budget, operation=operation)
        self._mcp_message_scope(metadata, resource, source_id, operation, window)
        value = self._get(context, path, {'format': 'full'}, budget, operation=operation)
        received = self._mcp_message_scope(value, resource, source_id, operation, window)
        version = value.get('historyId')
        if not isinstance(version, str) or not version.isascii() or not version.isdecimal() or len(version) > 64:
            raise ValueError('gmail_message_version_invalid')

        plain, html = value.get('plaintextBody', ''), value.get('htmlBody', '')
        if not isinstance(plain, str) or not isinstance(html, str):
            raise ValueError('gmail_message_content_invalid')
        message = EmailMessage()
        if plain or not html:
            message.set_content(plain)
            if html:
                message.add_alternative(html, subtype='html')
        else:
            message.set_content(html, subtype='html')
        subject, sender = value.get('subject'), value.get('sender')
        if isinstance(subject, str):
            message['Subject'] = subject
        if isinstance(sender, str) and sender:
            names = value.get('addressDisplayNames', [])
            if not isinstance(names, list):
                raise ValueError('gmail_sender_metadata_invalid')
            name = next((item['displayName'] for item in names if isinstance(item, dict)
                         and item.get('emailAddress') == sender and isinstance(item.get('displayName'), str)), '')
            message['From'] = formataddr((name, sender))

        attachments = value.get('attachments', [])
        if not isinstance(attachments, list) or len(attachments) > 250:
            raise CheckStopped('partial', 'gmail_mime_budget')
        for attachment in attachments:
            if not isinstance(attachment, dict):
                raise ValueError('gmail_attachment_metadata_invalid')
            content_type = attachment.get('mimeType')
            if not isinstance(content_type, str) or not _MIME_TYPE.fullmatch(content_type):
                content_type = 'application/octet-stream'
            content_type = content_type.lower()
            if content_type.startswith(('message/', 'multipart/')):
                content_type = 'application/octet-stream'
            maintype, subtype = content_type.split('/')
            message.add_attachment(b'', maintype=maintype, subtype=subtype)
        return SourceRecord(resource, source_id, version, received.isoformat(), message.as_bytes())

    def _mcp_page(self, context, resource, query, window, budget, *, operation, state):
        if resource in ('DRAFT', 'SPAM'):
            raise CheckStopped('partial', 'gmail_label_search_unsupported')
        grouped = _mcp_query_group(query)
        scope = ''
        if resource == 'INBOX':
            scope = 'in:inbox'
        elif resource == 'TRASH':
            scope = 'in:trash'
        elif resource != ALL_MAIL:
            label = self._get(context, 'labels/' + _id(resource), {}, budget, operation=operation)
            name = label.get('name')
            if label.get('id') != resource or not valid_gmail_label_name(name) or any(c in name for c in '\\"'):
                raise CheckStopped('partial', 'gmail_label_search_unavailable')
            scope = f'label:"{name}"'
        since, until = window
        # The MCP tool documents days in Gmail's search timezone. Widen these
        # metadata-only coarse bounds and enforce exact timestamps per message.
        first = (since.astimezone(timezone.utc) - timedelta(days=1)).date()
        last = (until.astimezone(timezone.utc) + timedelta(days=2)).date()
        bounds = f'after:{first:%Y/%m/%d} before:{last:%Y/%m/%d}'
        search = ' '.join(value for value in (grouped, bounds, scope) if value)
        arguments = {'query': search, 'pageSize': state['size'], 'view': 'THREAD_VIEW_METADATA_ONLY',
                     'includeTrash': operation != GMAIL_CONTENT_OPERATION or resource == 'TRASH'}
        marker = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        if state.get('page'):
            if state.get('query_marker') != marker:
                raise CursorExpired()
            arguments['pageToken'] = state['page']
        try:
            value = self._get(context, 'search_threads', arguments, budget, operation=operation)
        except GmailHTTPError as exc:
            if state.get('page') and exc.status in (400, 404, 410):
                raise CursorExpired() from None
            raise

        threads, continuation = value.get('threads', []), value.get('nextPageToken')
        if (not isinstance(threads, list) or len(threads) > state['size']
                or continuation is not None and (not isinstance(continuation, str)
                    or not continuation or len(continuation) > 4096 or continuation == state.get('page'))):
            raise CheckStopped('partial', 'gmail_response_invalid')
        covered = not value.get('omittedContent') and resource != ALL_MAIL
        estimate = value.get('resultCountEstimate')
        if (not state.get('page') and not continuation and isinstance(estimate, str)
                and estimate.isascii() and estimate.isdecimal() and len(estimate) <= 20
                and int(estimate) > len(threads)):
            covered = False
        records, seen = [], set()
        for thread in threads:
            if not isinstance(thread, dict) or not isinstance(thread.get('messages'), list):
                covered = False
                continue
            members = thread['messages']
            # Gmail matches the thread and may include nonmatching members.
            # Every body remains subject to exact label and time checks.
            count = thread.get('messageCount')
            if (type(count) is not int or count != len(members) or not 1 <= len(members) <= 5
                    or query and count != 1):
                covered = False
            previous = len(seen)
            for candidate in members[:5]:
                if not isinstance(candidate, dict) or not isinstance(candidate.get('labelIds'), list):
                    covered = False
                    continue
                labels = candidate['labelIds']
                if resource != ALL_MAIL and resource not in labels:
                    continue
                message_id = candidate.get('id')
                if not isinstance(message_id, str) or not _ID.fullmatch(message_id) or message_id in seen:
                    covered = False
                    continue
                if len(seen) >= context.limits.items:
                    return records, continuation, False, marker
                seen.add(message_id)
                try:
                    records.append(self._mcp_record(context, resource, message_id, budget,
                        operation=operation, window=window))
                except GmailHTTPError as exc:
                    if exc.status != 404:
                        raise
                    covered = False
                except CheckStopped as exc:
                    if exc.outcome != 'partial':
                        raise
                    return records, continuation, False, marker
                except (ScopeMoved, OverflowError, TypeError, ValueError):
                    covered = False
            if len(seen) == previous:
                covered = False
        return records, continuation, covered, marker

    def _mcp_snapshot(self, context, *, query, since, cursor):
        self._authorize(context)
        if context.limits.pages <= 0 or context.limits.items <= 0:
            raise CheckStopped('partial', 'gmail_page_budget')
        budget = [context.limits.bytes]
        if cursor:
            state = self._cursor(context, query, cursor)
            try:
                since, until = (datetime.fromisoformat(state[key]) for key in ('since', 'until'))
                if (not since.tzinfo or not until.tzinfo or since >= until
                        or type(state['resource_index']) is not int
                        or not 0 <= state['resource_index'] < len(context.grant.resources)
                        or type(state['size']) is not int or not 1 <= state['size'] <= 50):
                    raise ValueError()
            except (TypeError, KeyError, ValueError):
                raise CursorExpired() from None
        else:
            until = self.clock()
            if since.tzinfo is None or since >= until:
                raise ValueError('gmail_snapshot_window_invalid')
            state = dict(key=self._key(context, query), since=since.isoformat(), until=until.isoformat(),
                        size=min(50, context.limits.items), resource_index=0)
        resource = context.grant.resources[state['resource_index']]
        records, page, covered, marker = self._mcp_page(context, resource, query['text'].strip(),
            (since, until), budget, operation='mail_read', state=state)
        state.update(page=page, query_marker=marker)
        if not page:
            state['resource_index'] += 1
        complete = state['resource_index'] == len(context.grant.resources)
        return SourcePage(tuple(records), None if complete else self._stamp(state), complete, covered,
                          context.limits.bytes - budget[0])

    def _record(self, context, resource, source_id, budget, *, operation='mail_read', window=None):
        if self._mcp:
            return self._mcp_record(context, resource, source_id, budget, operation=operation, window=window)
        value = self._get(context, 'messages/' + _id(source_id), {'format': 'full'}, budget, operation=operation)
        if value.get('id') != source_id:
            raise ValueError('gmail_identity_mismatch')
        labels = value.get('labelIds')
        if resource != ALL_MAIL and (not isinstance(labels, list) or resource not in labels):
            raise ScopeMoved('gmail_message_outside_scope')
        if (operation == GMAIL_CONTENT_OPERATION and resource not in ('SPAM', 'TRASH')
                and ('SPAM' in labels or 'TRASH' in labels)):
            raise ScopeMoved('gmail_message_outside_scope')
        received = datetime.fromtimestamp(int(value['internalDate']) / 1000, timezone.utc)
        if window is not None and not window[0] <= received <= window[1]:
            raise ScopeMoved('gmail_message_outside_window')
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
        return SourceRecord(resource, source_id, str(value['historyId']), received.isoformat(), msg.as_bytes())

    def search_label(self, context, *, query, since, until):
        """Read one page for the owner's exact authorized label."""
        operation = GMAIL_CONTENT_OPERATION
        resources = context.grant.resources
        if (len(resources) != 1 or not exact_gmail_label_id(resources[0]) or context.project_id is not None
                or context.limits.pages != 1 or not 0 < context.limits.items <= GMAIL_LABEL_SEARCH_MAX
                or not isinstance(query, str) or len(query) > 256 or since >= until):
            raise PermissionError('gmail_label_search_scope_denied')
        resource, = resources
        budget = [context.limits.bytes]
        if self._mcp:
            records, continuation, covered, _ = self._mcp_page(context, resource, query,
                (since, until), budget, operation=operation, state={'size': context.limits.items})
            return SourcePage(tuple(records), complete=not continuation, coverage_complete=covered,
                              bytes_read=context.limits.bytes - budget[0])
        # Gmail query syntax is only a search filter. Recheck both the exact
        # provider label and returned message time before accepting any body.
        bounds = f'after:{int(since.timestamp()) - 1} before:{int(until.timestamp()) + 1}'
        params = {'q': f'({query}) {bounds}' if query else bounds, 'labelIds': resource,
                  'maxResults': context.limits.items,
                  'includeSpamTrash': 'true' if resource in ('SPAM', 'TRASH') else 'false'}
        value = self._get(context, 'messages', params, budget, operation=operation)
        items, continuation = value.get('messages', []), value.get('nextPageToken')
        if (not isinstance(items, list) or
                ('nextPageToken' in value and (not isinstance(continuation, str) or not continuation))):
            raise CheckStopped('partial', 'gmail_response_invalid')
        records, seen = [], set()
        covered = len(items) <= context.limits.items
        for item in items[:context.limits.items]:
            try:
                message_id = item['id']
                _id(message_id)
                if message_id in seen:
                    covered = False
                    continue
                seen.add(message_id)
                records.append(self._record(context, resource, message_id, budget,
                                            operation=operation, window=(since, until)))
            except GmailHTTPError as exc:
                if exc.status != 404:
                    raise
                covered = False
            except CheckStopped as exc:
                if exc.outcome != 'partial':
                    raise
                covered = False
                break
            except (ScopeMoved, AttributeError, KeyError, OverflowError, TypeError, ValueError):
                covered = False
        return SourcePage(tuple(records), complete=not continuation, coverage_complete=covered,
                          bytes_read=context.limits.bytes - budget[0])

    def search(self, context, *, query, since, cursor=None):
        if self._mcp:
            return self._mcp_snapshot(context, query=query, since=since, cursor=cursor)
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
        if self._mcp:
            # A previous transport's history cursor cannot be converted into
            # an MCP cursor. The acquisition service starts a bounded baseline.
            raise CursorExpired()
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
        if self._mcp:
            self._authorize(context)
            raise CheckStopped('partial', 'gmail_attachment_unavailable')
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
