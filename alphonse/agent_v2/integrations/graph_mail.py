"""Optional delegated Graph mail reads. No provider SDK, account I/O or jobs at import.

One transport request per operation. Composite cursors are private acquisition
state, containing verbatim provider links for each explicitly selected folder.
They must never enter prompts, UI, logs or events. Message bodies are untrusted.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
from urllib.parse import quote, unquote, urlencode, urlsplit

from alphonse.agent_v2.integrations.acquisition import CursorExpired, ReadRetryAfter, SourcePage, SourceRecord
from alphonse.agent_v2.integrations.oauth import SCOPES
from alphonse.agent_v2.services.project_check_worker import CheckStopped

ROOT = 'https://graph.microsoft.com/v1.0/me/mailFolders/'
FIELDS = 'id,parentFolderId,changeKey,lastModifiedDateTime,receivedDateTime,subject,from,body,hasAttachments,webLink'


class GraphReadError(ValueError):
    """Static safe error; never include provider response text or opaque URLs."""


class GraphHTTP:
    """Bounded streaming GET only, no redirects, decompression or implicit retries."""
    def get(self, url, *, token, deadline, max_bytes, headers):
        import requests
        # urllib3 DEBUG includes the full request URL (opaque delta tokens).
        # Match the managed OAuth policy: disable provider HTTP diagnostics
        # before any credential-bearing operation; application logging stays on.
        for name in list(logging.Logger.manager.loggerDict):
            if name.startswith(('requests', 'urllib3')):
                logging.getLogger(name).disabled = True
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            raise CheckStopped('partial', 'source_deadline')
        try:
            with requests.Session() as session:
                # Ignore netrc and ambient credential/proxy configuration.
                session.trust_env = False
                with session.get(url, headers={**headers, 'Authorization': 'Bearer ' + token,
                        'Accept-Encoding': 'identity'}, stream=True, allow_redirects=False,
                        timeout=min(20, remaining)) as response:
                    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                        raise GraphReadError('graph_encoding_invalid')
                    chunks, size = [], 0
                    for chunk in response.iter_content(chunk_size=min(8192, max_bytes + 1)):
                        size += len(chunk)
                        if size > max_bytes:
                            raise CheckStopped('partial', 'source_content_budget')
                        if datetime.now(timezone.utc) >= deadline:
                            raise CheckStopped('partial', 'source_deadline')
                        chunks.append(chunk)
                    return response.status_code, dict(response.headers), b''.join(chunks)
        except requests.RequestException:
            raise GraphReadError('graph_transport_unavailable') from None


def _identifier(value):
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or any(c in value for c in ('/', '\\', '%', '?', '#'))
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise GraphReadError('graph_identifier_invalid')
    return quote(value, safe='')


def _folder_path(resource):
    return ROOT + _identifier(resource) + '/messages'


def _scope(context, query):
    grant = context.grant
    return hashlib.sha256(json.dumps([context.session.actor_user_id, context.project_id,
        grant.connection_id, grant.connection_revision, grant.grant_id, grant.grant_revision,
        list(grant.resources), list(grant.recipient_user_ids), query], sort_keys=True).encode()).hexdigest()


def _link(link, resource):
    if not isinstance(link, str) or len(link) > 16000:
        raise GraphReadError('graph_link_invalid')
    uri = urlsplit(link)
    expected = urlsplit(_folder_path(resource) + '/delta')
    if (uri.scheme != 'https' or uri.netloc != 'graph.microsoft.com' or uri.fragment
            or unquote(uri.path) != unquote(expected.path) or not uri.query
            or any(c in link for c in ('\r', '\n', '\\'))):
        raise GraphReadError('graph_link_invalid')
    return link  # Preserve opaque query exactly; never decode/rebuild state tokens.


class GraphMailAdapter:
    version = 'graph-mail-folder-delta-1'

    def __init__(self, oauth, *, transport=None, clock=lambda: datetime.now(timezone.utc)):
        # Callable form lets the daemon lazily configure OAuth only on managed use.
        self.oauth, self.transport, self.clock = oauth, transport or GraphHTTP(), clock

    def _authorization(self, context):
        context.guard()
        g = context.grant
        if g.provider_key != 'microsoft' or g.operation != 'mail_read' or not g.resources:
            raise PermissionError('graph_mail_scope_denied')
        service = self.oauth(context.session) if callable(self.oauth) else self.oauth
        args = dict(connection_id=g.connection_id, grant_id=g.grant_id, operation='mail_read',
            resources=list(g.resources), recipient_user_ids=list(g.recipient_user_ids),
            required_scopes=[SCOPES['microsoft']['mail_read']], project_id=context.project_id,
            expected_grant_revision=g.grant_revision)
        record, _ = service.accounts.authorize(context.session, **args)
        if record.provider_key != 'microsoft' or record.connection['revision'] != g.connection_revision:
            raise PermissionError('graph_mail_account_changed')
        return service, args

    def _get(self, context, url):
        if self.clock() >= context.deadline:
            raise CheckStopped('partial', 'source_deadline')
        if context.limits.pages <= 0 or context.limits.items <= 0 or context.limits.bytes <= 0:
            raise CheckStopped('partial', 'source_page_budget')
        service, args = self._authorization(context)
        token = service.token(context.session, authorization=args)
        status, headers, raw = self.transport.get(url, token=token, deadline=context.deadline,
            max_bytes=context.limits.bytes, headers={'Accept': 'application/json',
                'Prefer': f'IdType="ImmutableId", odata.maxpagesize={min(100, context.limits.items)}'})
        self._authorization(context)  # Fence revocation/reconnect while I/O was running.
        if self.clock() >= context.deadline:
            raise CheckStopped('partial', 'source_deadline')
        if not isinstance(raw, bytes) or len(raw) > context.limits.bytes:
            raise CheckStopped('partial', 'source_content_budget')
        context.on_bytes(len(raw))
        if status == 429 or status == 503:
            retry = next((v for k, v in headers.items() if k.lower() == 'retry-after'), '5')
            try:
                seconds = int(retry)
            except (ValueError, TypeError):
                try:
                    seconds = int((parsedate_to_datetime(retry) - self.clock()).total_seconds()) + 1
                except (ValueError, TypeError, OverflowError):
                    seconds = 5
            raise ReadRetryAfter(max(1, seconds))
        if status == 410:
            raise CursorExpired()
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError):
            raise GraphReadError('graph_response_invalid') from None
        if not isinstance(data, dict):
            raise GraphReadError('graph_response_invalid')
        code = str(data.get('error', {}).get('code', '')).lower() if isinstance(data.get('error', {}), dict) else ''
        if status >= 400 and code in {'syncstatenotfound', 'resyncrequired', 'errorinvalidsyncstatedata'}:
            raise CursorExpired()
        if status == 401:
            record, _ = service.accounts.authorize(context.session, **args)
            service._require_reconnect(record)
            raise CheckStopped('blocked', 'graph_reauth_required')
        if status in {403, 404}:
            return None, len(raw)
        if status != 200:
            raise GraphReadError('graph_read_failed')
        return data, len(raw)

    def health(self, context):
        service, args = self._authorization(context)
        record, _ = service.accounts.authorize(context.session, **args)
        return dict(provider='microsoft', status=record.connection['status'],
                    coverage='selected_folders_only')

    def capabilities(self, context):
        self._authorization(context)
        return ('mail_read',)

    def _state(self, context, query, since, cursor):
        folders = list(context.grant.resources)
        if len(folders) != len(set(folders)) or len(folders) > 100:
            raise GraphReadError('graph_folder_selection_invalid')
        scope = _scope(context, query)
        if cursor is None:
            if since is None or since.tzinfo is None:
                raise GraphReadError('graph_baseline_required')
            return dict(scope=scope, since=since.isoformat(), index=0,
                        links={f: None for f in folders}, counts={f: 0 for f in folders}, gaps=[], truncated=False)
        try:
            state = json.loads(cursor)
            if (state['scope'] != scope or list(state['links']) != folders
                    or type(state['truncated']) is not bool or set(state['counts']) != set(folders) or not isinstance(state['gaps'], list)
                    or type(state['index']) is not int or not 0 <= state['index'] <= len(folders)
                    or datetime.fromisoformat(state['since']).tzinfo is None):
                raise ValueError()
            for folder, link in state['links'].items():
                if link is not None:
                    _link(link, folder)
                if type(state['counts'][folder]) is not int or state['counts'][folder] < 0:
                    raise ValueError()
            if state['index'] == len(folders):
                state.update(index=0, gaps=[], counts={f: 0 for f in folders})
            return state
        except (ValueError, TypeError, KeyError, AttributeError):
            raise GraphReadError('graph_cursor_invalid') from None

    def search(self, context, *, query, since, cursor=None):
        return self._page(context, query, since, cursor)

    def changes(self, context, *, query, cursor):
        if not cursor:
            raise GraphReadError('graph_cursor_required')
        return self._page(context, query, None, cursor)

    def _page(self, context, query, since, cursor):
        self._authorization(context)
        state = self._state(context, query, since, cursor)
        folders = list(context.grant.resources)
        folder = folders[state['index']]
        link = state['links'][folder]
        if link is None:
            # Query text describes the project goal; semantic relevance belongs
            # to existing triage. Do not translate prose into brittle OData syntax.
            url = _folder_path(folder) + '/delta?' + urlencode({
                '$select': FIELDS, '$filter': 'receivedDateTime ge ' + state['since']})
        else:
            url = _link(link, folder)
        data, size = self._get(context, url)
        records = ()
        if data is None:
            # Denied/deleted folders remain selected, unadvanced and retryable
            # next round. Other authorized folders can still produce evidence.
            state['gaps'].append(folder)
            state['index'] += 1
        else:
            values = data.get('value')
            if not isinstance(values, list):
                raise GraphReadError('graph_page_invalid')
            if len(values) > context.limits.items:
                raise CheckStopped('partial', 'source_page_budget')
            next_link, delta = data.get('@odata.nextLink'), data.get('@odata.deltaLink')
            if bool(next_link) == bool(delta):
                raise GraphReadError('graph_page_link_invalid')
            state['links'][folder] = _link(next_link or delta, folder)
            if state['links'][folder] == link and next_link:
                raise GraphReadError('graph_cursor_stalled')
            records = tuple(self._record(folder, item) for item in values)
            state['counts'][folder] += len(values)
            # Graph caps filtered delta at 5000 messages; never claim exhaustive
            # coverage when that provider limit might have truncated the baseline.
            if state['counts'][folder] >= 5000:
                state['truncated'] = True
            if delta:
                state['index'] += 1
        encoded = json.dumps(state, separators=(',', ':'))
        if len(encoded) > 65536:
            raise CheckStopped('partial', 'source_cursor_budget')
        coverage = {f: ('unavailable' if f in state['gaps'] else
            'incomplete' if state['truncated'] else
            'complete' if i < state['index'] else
            'in_progress' if i == state['index'] and state['links'][f] else 'pending')
            for i, f in enumerate(folders)}
        return SourcePage(records, encoded, state['index'] == len(folders),
            not state['gaps'] and not state['truncated'], size, coverage)

    def _record(self, folder, item):
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']:
            raise GraphReadError('graph_message_invalid')
        if '@removed' in item:
            return SourceRecord(folder, item['id'], 'removed', self.clock().isoformat(), b'', deleted=True)
        if item.get('parentFolderId') != folder:
            raise PermissionError('graph_folder_identity_mismatch')
        if not item.get('changeKey') or not item.get('lastModifiedDateTime'):
            raise GraphReadError('graph_message_version_missing')
        body = item.get('body', {})
        if not isinstance(body, dict) or body.get('contentType', '').lower() not in {'html', 'text'}:
            raise GraphReadError('graph_message_body_invalid')
        message = EmailMessage()
        try:
            message['Subject'] = item.get('subject', '')
            sender = item.get('from', {}).get('emailAddress', {}).get('address', '')
            if sender:
                message['From'] = sender
            message.set_content(body.get('content', ''), subtype='html' if body['contentType'].lower() == 'html' else 'plain')
        except (ValueError, TypeError, AttributeError):
            raise GraphReadError('graph_message_invalid') from None
        return SourceRecord(folder, item['id'], item['changeKey'], item['lastModifiedDateTime'],
                            message.as_bytes(), item.get('webLink', ''))

    def read(self, context, *, resource, source_id):
        if resource not in context.grant.resources or not source_id:
            raise PermissionError('graph_mail_resource_denied')
        data, size = self._get(context, _folder_path(resource) + '/' + _identifier(source_id) + '?' + urlencode({'$select': FIELDS}))
        if data is None:
            raise CheckStopped('partial', 'graph_message_unavailable')
        record = self._record(resource, data)
        if record.source_id != source_id:
            raise PermissionError('graph_message_identity_mismatch')
        return record

    def attachment(self, context, *, resource, source_id, attachment_id, max_bytes):
        if resource not in context.grant.resources or not source_id or not attachment_id:
            raise PermissionError('graph_mail_resource_denied')
        if type(max_bytes) is not int or not 0 < max_bytes <= context.limits.bytes:
            raise ValueError('source_attachment_budget')
        url = _folder_path(resource) + '/' + _identifier(source_id) + '/attachments/' + _identifier(attachment_id)
        data, size = self._get(context, url)
        if data is None:
            raise CheckStopped('partial', 'graph_attachment_unavailable')
        # Reference/item attachments never trigger linked-site requests or recursion.
        if (data.get('@odata.type') != '#microsoft.graph.fileAttachment' or data.get('id') != attachment_id
                or data.get('contentType') not in {'text/plain', 'text/csv', 'application/pdf'}):
            raise CheckStopped('partial', 'graph_attachment_unsupported')
        try:
            result = base64.b64decode(data['contentBytes'], validate=True)
        except (ValueError, TypeError, KeyError):
            raise GraphReadError('graph_attachment_invalid') from None
        if len(result) > max_bytes:
            raise CheckStopped('partial', 'source_attachment_budget')
        return result
