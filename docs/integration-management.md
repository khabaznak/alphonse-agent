# Optional account and project check management

Desktop Settings → Integrations contains optional accounts alongside the existing
Telegram and Discord setup. Project details contains Project checks. Scheduled
tasks contains the project check agenda and an audience-scoped finding inbox.
Typed project checks are managed through their policy controls rather than the
legacy prompt schedule editor. No interactive terminal client is required.

## Local identity and consent

The optional management IPC method authenticates the local OS peer against the
owner-only Unix socket. It binds that principal to the daemon's canonical runtime
user. It never accepts a payload user/actor as authentication. A UI profile picker
cannot impersonate another household member. Other members use their authenticated
conversation identity or their own local daemon principal. Administrators do not
receive another owner's accounts, grants, check definitions or findings merely
because of their role. The visible household directory includes names and IDs only.
This is a trusted local-host boundary, not an application sandbox against programs
already running as the same OS user.

Configure the existing `ALPHONSE_GOOGLE_OAUTH_CLIENT_ID` / `_REDIRECT_URI` or
`ALPHONSE_MICROSOFT_OAUTH_CLIENT_ID` / `_REDIRECT_URI` environment variables before
starting the daemon. Desktop requires an explicitly registered
`http://127.0.0.1:<port>/<callback-path>` redirect. HTTPS redirects require a separate
trusted transport and are rejected by this Desktop flow. Consent opens only the
Google or Microsoft HTTPS authorization hosts. The daemon binds the loopback port
before consent and handles the exact callback path. Existing single-use state,
actor, PKCE, revision, expiry, subject and scope checks remain authoritative.
Callback codes, token bytes and consent URLs do not enter conversation history.
Restart loses pending consent; reconnect explicitly to retry. Provider qualification
and actual account setup remain release/deployment work, not proof from these tests.

Choose capabilities before connecting or reconnecting. Connect alone starts no
monitoring. Health distinguishes connected, paused, reconnect-required, disconnected,
credential storage failure, provider revocation pending and credential deletion
pending. Pause/disconnect stop managed access; reconnection never silently restores
a previously suspended project policy. Local disconnect remains effective while
provider cleanup is pending. Retry deletion explicitly after resolving storage or
network problems.

## Grants and sources

The owner chooses a beneficiary, exact capabilities and source IDs, recipients,
expiry and optional project restriction, then reviews the concrete grant. No raw
OAuth scopes are required. Gmail sources are label IDs such as `INBOX`; Outlook
sources are exact folder IDs. Wildcards are rejected. Create a skill binding for the
grant; its beneficiary separately chooses the skill role described in that skill's
instructions. The check editor exposes only usable, explicitly selected bindings.
Grant revision, expiry, ownership, source limits, project restriction, binding ACL,
skill visibility and recipient limits are independently enforced at use time.
Revise or revoke grants from Integrations; revisions invalidate stale authorization.

## Checks and evidence

Choose the goal, accounts/sources, recipients, weekdays, check time, timezone,
retrieval start/end, initial lookback, duration, per-run budgets, notification mode
and missed-run behavior. Duration must have an end or be explicitly ongoing. The
retrieval end is exclusive, including one-minute windows. Schedule/window times use
the selected IANA timezone; duration fields explicitly use the viewer's local time.
No recurrence syntax is needed. Review includes the current project goals and all
execution limits. Save a disabled draft or explicitly authorize activation.

Reviews last five minutes, are actor/channel-bound and single-use. Activation
rechecks the exact reviewed project snapshot and revision, selected bindings,
connection and grant revisions. The policy service checks reviewed authority again
before persistence. Changed authority, an expired review or a stale revision requires
a fresh review. Cancellation, delivery and each operation's normal authorization
and window fences still apply. Resume only preserves a previously authorized paused
scope; needs-review policies require revision/re-authorization. No approval per run
is added within an established scope.

Project and agenda views show last/next run, stale/offline receipts, missed-window
errors, coverage, partial/blocked outcomes and pending analysis separately. Raw
provider cursors and model checkpoints are withheld. Classified irrelevant evidence
does not remain pending; relevant/unclassified evidence remains pending until its
CAPD checkpoint. Findings show uncertainty, due date, timezone and actual per-recipient
delivery state. Pending/queued is never labeled delivered. The finding inbox resolves
recipient access even when that recipient has no access to the source project;
revocation removes the finding from subsequent reads.

Conversation uses `native.integration_manage`, the same service with the task's
trusted identity and cancellation check. Use catalog/project/findings to inspect,
resolve missing fields through questions, review a complete definition, present its
summary, and apply the review within the explicit user instruction. The model and
Jev cannot grant account authority. Browser consent stays in Desktop. No account
credentials, provider writes or real household records are used for validation.

## Reproducible UI fixture

Run Desktop's Vite development server and open
`/test-fixtures/integration-management.html`. This fixture replaces the Tauri transport
with synthetic accounts and records calls in `window.__fixtureCalls`. It performs no
daemon or provider calls. Validate recipient filtering, form/review heading focus,
Tab → activation, cancellation focus restoration, account pause/disconnect and
390-pixel layout. The fixture is excluded from the production entrypoint/build.
