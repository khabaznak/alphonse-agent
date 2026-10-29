# Trusted account setup and exact effect authorization

This package deliberately adopts a reusable Google Calendar implementation into
product source. Household configuration, OAuth client files/tokens, grants and
calendar records remain outside the package/repository. There is no artifact or
per-script registry. Installation uses the normal standard skill installation flow.

## Connect an account

An authenticated administrator uses the existing `SQLiteIntegrationStore` and
`SkillBindings` facilities from trusted host management code. No generic provider
OAuth UI is added by this package. Enable Calendar API in a Google Cloud project,
create a desktop OAuth client, configure its consent screen/test users, and install
`google-auth-oauthlib` in the trusted setup environment. Follow Google's
[desktop OAuth flow](https://developers.google.com/identity/protocols/oauth2/native-app)
using a loopback redirect, state/PKCE and offline access. Review the selected Google
identity and requested scopes in the browser. Do not paste tokens into a conversation.

For example, trusted setup code (not a model-issued Bash command) can use
`InstalledAppFlow.from_client_secrets_file(private_client_path, scopes=[...])`
and `.run_local_server(port=0, access_type="offline", prompt="consent")`.
Use `calendar.readonly` for event listing and availability; `calendar` supports
these plus creation/update. An events-only scope cannot query free/busy; if using
`calendar.events`, also consent to `calendar.freebusy`. Store the actually granted
scopes from the OAuth response, not a requested-scope assumption. Missing refresh
token or partial scope grants require reconnection. Keep client files in protected
host storage, outside installed/exportable skills.

Persist credentials directly to the existing integration store; for example:

```python
# Authenticated setup context only; never log creds or call to_dict(mask_secrets=False).
integrations.upsert(
    integration_id=chosen_account_id, provider_key="google-calendar",
    display_name=chosen_display_name, enabled=True,
    config={"granted_scopes": " ".join(actual_granted_scopes)},
    secrets={"client_id": creds.client_id, "client_secret": creds.client_secret,
             "refresh_token": creds.refresh_token},
)
bindings.configure(
    actor_id=authenticated_admin_id, owner_id=authenticated_member_id,
    binding_id=chosen_binding_id, skill_id="skill:family-calendar",
    kind="account", target=chosen_account_id,
    required_secrets=("client_id", "client_secret", "refresh_token"),
    executables=("python3",),
    configuration={"calendars": {explicit_calendar_id:
                                ["list", "availability", "get", "create", "update"]}},
)
bindings.select(actor_id=authenticated_member_id, skill_id="skill:family-calendar",
                name="calendar", binding_id=chosen_binding_id)
```

Resolve the exact calendar ID through Google's Calendar settings or an authorized
CalendarList request in trusted setup. Review the calendar's owner, role and intended
family audience. No `primary` alias or email/name guessing is needed. Read-only
bindings should omit create/update operations. Sharing an account binding shares
its configured calendar operations; use separate bindings for narrower grants.
The Google account itself must have the required reader/writer calendar ACL.

Each command refreshes an ephemeral access token from that exact bound account.
Tokens are never written to package files or task evidence. Google rejects revoked
refresh tokens (static reconnect error); missing scopes or calendar ACLs fail closed.
The client prefers scope evidence in the fresh token response, using stored actual
grants only when Google omits scope. New consent requires updating host account
state. Disable the integration or binding to revoke managed subsequent dispatch;
revoke OAuth consent with Google to invalidate provider credentials. An in-flight
request cannot be recalled. Existing integration SQLite storage is not an encrypted
vault; the host file protections and trusted-code boundary still apply.

## Requested effects

Account/calendar access is not permission for every write. After authenticating the
requester and approving the exact intended change through the host's normal effect
policy, trusted management code computes `digest(request)` from `scripts/google_calendar.py`
(canonical UTF-8 JSON, sorted keys, compact separators, SHA-256) and provisions a
short-lived **configuration** binding with:

```python
bindings.configure(
    actor_id=authenticated_admin_id, owner_id=authenticated_requester_id,
    binding_id=unique_grant_id, skill_id="skill:family-calendar", kind="configuration",
    configuration={"request_sha256": approved_request_hash,
                   "expires_at": explicit_rfc3339_expiry},
)
bindings.select(actor_id=authenticated_requester_id, skill_id="skill:family-calendar",
                name="authorization", binding_id=unique_grant_id)
```

Do not expose grant issuance as a model tool or let the model approve its own hash.
This initial route intentionally needs trusted management provisioning for writes;
a future host UI may automate this only from authenticated approved effects. It is
an operation grant, not a script registration. It includes exact calendar, payload,
operation ID, ETag/recurrence scope and `send_updates="none"`. The script compares
that hash and expiry before any network call. A different actor cannot resolve the
binding unless explicitly shared; do not share operation grants. Expired grants
require renewed authorization even for write reconciliation (GET remains available).
Disable the grant after verified completion. It permits replay of the same approved
request until expiry to reconcile uncertain outcomes, rather than promising a
single-use distributed transaction. Concurrent updates are protected by If-Match;
concurrent creates converge on the same event ID. No extra local journal is needed.

Native dispatch retains phase effects, cancellation and checkpoint checks. The
package's checks are deterministic for the supplied managed execution path, not an
OS boundary against trusted Bash reading the integration database or replacing code.
No Jev answer supplies access, authorizes an effect or substitutes for readback.
