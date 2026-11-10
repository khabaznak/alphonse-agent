"""Actor-scoped, one-off Gmail label counts; no message bodies or monitoring.

The service is deliberately independent of project-check acquisition. A caller
must construct the AccountSession from the authenticated task, never tool input.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from alphonse.agent_v2.accounts import AccountHandle, AccountSession
from alphonse.agent_v2.integrations.acquisition import ReadContext, ReadLimits
from alphonse.agent_v2.integrations.gmail import ALL_MAIL, SCOPE, GmailAdapter, GmailHTTPError, _id
from alphonse.agent_v2.integrations.oauth import OAuthError
from alphonse.agent_v2.services.project_check_worker import CheckStopped


class GmailUnreadService:
    """Read one provider-reported label count using a pre-existing one-off grant."""

    def __init__(self, accounts, oauth, *, transport=None, clock=None):
        self.accounts = accounts
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.adapter = GmailAdapter(oauth, transport=transport, clock=self.clock)

    def count(self, session, *, label_id="INBOX", unit="messages", connection_id=None,
              guard=lambda: None, max_bytes=16384, timeout_seconds=20,
              choice_question_id=None, choice_store=None, task=None):
        if not isinstance(session, AccountSession) or not session.actor_user_id:
            raise PermissionError("gmail_authenticated_session_required")
        if unit not in {"messages", "threads"}:
            raise ValueError("gmail_count_unit_invalid")
        escaped_label = _id(label_id)
        if connection_id is not None and (not isinstance(connection_id, str) or not connection_id.strip()):
            raise ValueError("gmail_connection_invalid")
        if type(max_bytes) is not int or not 0 < max_bytes <= 65536:
            raise ValueError("gmail_count_budget_invalid")
        if type(timeout_seconds) not in {int, float} or not 0 < timeout_seconds <= 30:
            raise ValueError("gmail_count_timeout_invalid")
        try:
            guard()
            self.accounts._active(session.actor_user_id)
            candidates = self._candidates(session, label_id, None)
            guard()
            if not candidates:
                return {"status": "blocked", "reason": "gmail_authorized_count_unavailable"}
            by_account = {}
            for record, grant, resource in candidates:
                by_account.setdefault(record.integration_id, []).append((record, grant, resource))
            if choice_store is not None and task is not None:
                from alphonse.agent_v2.integrations.gmail_choice import choice_was_issued
                choice_required = choice_was_issued(choice_store, task=task, label_id=label_id, unit=unit)
                # Once a question was issued, omitted proof or a changed
                # candidate set cannot silently choose a remaining account.
                if choice_required and (not choice_question_id or len(by_account) < 2):
                    return {"status": "blocked", "reason": "gmail_account_choice_unavailable"}
            elif choice_question_id is not None:
                return {"status": "blocked", "reason": "gmail_account_choice_unavailable"}
            if len(by_account) > 1:
                # Tool arguments are model-composed: a supplied ID is not proof
                # of a user choice. Recheck grants before showing account names.
                current = self._candidates(session, label_id, None)
                guard()
                current_accounts = {item[0].integration_id: item[0] for item in current}
                if len(current_accounts) < 2:
                    return {"status": "blocked", "reason": "gmail_account_choices_changed"}
                options = [
                    {"connection_id": key, "display_name": current_accounts[key].display_name}
                    for key in sorted(current_accounts)
                ]
                chosen = None
                if choice_store is not None and task is not None and getattr(task, 'user', None) == session.actor_user_id:
                    from alphonse.agent_v2.integrations.gmail_choice import resolve_choice
                    chosen = resolve_choice(choice_store, task=task, question_id=choice_question_id,
                                            accounts=options, label_id=label_id, unit=unit)
                if chosen is None:
                    return {"status": "account_choice_required", "accounts": options}
                # Candidates were reauthorized above and the selected grant is
                # checked again around provider I/O by the Gmail adapter.
                by_account = {}
                for record, grant, resource in current:
                    if record.integration_id == chosen:
                        by_account.setdefault(chosen, []).append((record, grant, resource))
                if not by_account or (connection_id is not None and connection_id != chosen):
                    return {"status": "blocked", "reason": "gmail_authorized_count_unavailable"}
            if connection_id is not None and connection_id not in by_account:
                return {"status": "blocked", "reason": "gmail_authorized_count_unavailable"}
            # Prefer the narrow grant to the mailbox-wide grant. Stable tie
            # breaking does not alter scope or bypass authorize().
            values = next(iter(by_account.values()))
            record, grant, resource = sorted(values, key=lambda v: (v[2] == ALL_MAIL, v[1]['grant_id']))[0]
            handle = AccountHandle(record.integration_id, "", grant['grant_id'],
                record.connection['revision'], grant['revision'], 'google', 'mail_read', (resource,), ())
            context = ReadContext(session, handle, self.clock() + timedelta(seconds=timeout_seconds),
                ReadLimits(1, 1, max_bytes), project_id=None, guard=guard)
            value = self.adapter._get(context, 'labels/' + escaped_label, {}, [max_bytes])
            # Do not trust a wrong label or assume a missing field means zero.
            field = 'messagesUnread' if unit == 'messages' else 'threadsUnread'
            count = value.get(field)
            if value.get('id') != label_id or type(count) is not int or count < 0:
                return {"status": "unknown", "reason": "gmail_count_response_incomplete"}
            guard()
            return {"status": "complete", "provider": "gmail", "connection_id": record.integration_id,
                    "label_id": label_id, "unit": unit, "count": count,
                    "coverage": "provider_label_count", "observed_at": self.clock().isoformat()}
        except CheckStopped as exc:
            return {"status": exc.outcome, "reason": exc.code}
        except (PermissionError, OAuthError):
            return {"status": "blocked", "reason": "gmail_authorization_unavailable"}
        except GmailHTTPError:
            return {"status": "unknown", "reason": "gmail_provider_error"}
        except (OSError, TimeoutError):
            return {"status": "unknown", "reason": "gmail_provider_unavailable"}

    def _candidates(self, session, label_id, connection_id):
        """List only presently authorized grants, never provider content or tokens."""
        with self.accounts.integrations._connect() as conn:
            rows = conn.execute('SELECT definition FROM v2_account_grants').fetchall()
        candidates = []
        for row in rows:
            try:
                grant = json.loads(row['definition'])
                if (grant.get('beneficiary_user_id') != session.actor_user_id
                        or grant.get('project_id') is not None or grant.get('revoked_at')
                        or 'mail_read' not in grant.get('operations', [])):
                    continue
                if connection_id is not None and grant.get('connection_id') != connection_id:
                    continue
                record = self.accounts.integrations.get(grant['connection_id'])
                if record is None or record.provider_key != 'google':
                    continue
                resources = grant.get('resources', [])
                resource = label_id if label_id in resources else ALL_MAIL if ALL_MAIL in resources else None
                if resource is None:
                    continue
                self.accounts.authorize(session, connection_id=record.integration_id,
                    grant_id=grant['grant_id'], operation='mail_read', resources=[resource],
                    recipient_user_ids=[], required_scopes=[SCOPE], project_id=None,
                    expected_grant_revision=grant['revision'])
                candidates.append((record, grant, resource))
            except (PermissionError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return candidates
