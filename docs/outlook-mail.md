# Optional Outlook/Hotmail read-only acquisition (#118)

`GraphMailAdapter` implements the shared `ReadAdapter` contract for Microsoft
personal Outlook.com/Hotmail and eligible work/school mailboxes. It uses delegated
`Mail.Read` from the existing managed authorization-code + PKCE lifecycle; it has
no application credentials, write, send, modify, subscription or provider-job API.
Microsoft registration must explicitly support the intended account types and the
tenant must permit consent. These are deployment prerequisites, not claims that a
synthetic login proves production eligibility.

Follow [managed OAuth setup](oauth-accounts.md), install the `accounts-oauth`
optional dependencies, and configure the public Microsoft client ID and exact
callback. The daemon registers the adapter without importing optional OAuth
libraries or opening a connection. OAuth configuration is resolved lazily on
managed use. An unconfigured provider fails closed; connecting an account creates
no check, monitoring subscription or schedule. A separately authorized project
policy, explicit account/skill role, read grant, recipients, resource selection,
retrieval window and budgets are required by the existing pipeline. Trusted native
callers can use the same adapter for authorized one-off reads with a `ReadContext`
and no project; a project-specific grant cannot be reused that way.

Select exact folder IDs in the grant and policy. Include folders to which mailbox
rules move school mail. Folder IDs, including nested folders, must be explicitly
provided; the adapter neither enumerates nor silently includes additional folders.
Use actual Graph folder IDs rather than well-known aliases such as `inbox`, since
message `parentFolderId` is verified against the selected ID. Folder IDs are stable
container IDs; all requests include `Prefer: IdType="ImmutableId"`. Message IDs
remain stable for moves within the same mailbox, while provenance remains scoped
to the folder where each version was observed. Archive-mailbox transfers and
export/reimport can change message identity. No cross-mailbox guessing or ID
conversion requiring broader access occurs.

Each bounded step makes one GET request. A filtered folder delta baseline uses
`receivedDateTime ge <initial-lookback-start>` and explicitly selects message body,
version and provenance. Project query prose is passed through to the existing
System One triage; it is never interpreted as OData or a provider search expression.
Domain skills need no provider query syntax. Bodies are packaged as MIME for the
shared inert normalization path. Remote images, scripts and arbitrary linked sites
are never fetched. Explicit attachment reads accept only bounded file attachments
of the shared allowed types; reference and item attachments stay unsupported.
Full response bytes, including JSON/base64 overhead, are charged for expansions.

Private composite cursors hold the exact next/delta link for every selected
folder, current round position, bounded baseline date, coverage gaps and provider
filter-limit state. Scope binds actor, project, exact account/revision, grant/
revision, resources, recipients and query. The shared stream key also pins policy,
binding and adapter version. Cursor state and accepted evidence commit together
in the existing acquisition transaction; analysis checkpoints remain independent.
No other cursor store, scheduler, event queue or model-controlled progress exists.
Links are opaque, are reused verbatim and are validated against the fixed global
Graph HTTPS origin and exact selected folder delta endpoint before credentials
are attached. Redirects, ambient netrc/proxy credentials, compressed responses and
implicit transport retries are disabled. This implementation targets global Graph;
national cloud endpoints require separately reviewed configuration.

Coverage receipts list only the selected resource IDs and their individual safe
statuses. An Inbox-only receipt never asserts whole-mailbox or hierarchy coverage.
403/404 folders remain unavailable without losing evidence from other folders;
the next round retries their unadvanced links. 401 blocks pending reconnect.
410/expired sync states trigger the shared bounded resync and a persistent gap for
that run, with overlap/version deduplication. Graph's filtered delta limit of 5,000
messages is treated conservatively as incomplete when reached; later delta rounds
cannot erase that baseline uncertainty. A new explicitly bounded scope/baseline is
needed to clear it. Page/item/byte/time limits can also terminate a run as partial.

429 and 503 responses become the shared safe `ReadRetryAfter` signal. The existing
worker persists the provider delay (integer or HTTP date) without sleeping or
advancing the cursor, and rechecks the window/grant before retry. Delays beyond
the authorized window produce partial coverage. No-match is quiet only after all
selected folders complete with no gaps; partial, revoked and failed reads cannot
become no-match. Provider descriptions, response bodies, tokens and opaque links
never enter errors, coverage events or model prompts.

## Disposable verification

`tests/test_graph_mail.py` exercises delegated personal/work-school scenarios,
folder pagination, rule-moved mail, stable IDs, scope changes, malicious links/body,
partial access, expired sync states, throttling, bounded GET transport and allowed/
unsupported attachments. `tests/test_graph_mail_acquisition.py` runs the real
adapter through the shared durable pipeline: per-folder persistence/restart,
immutable source provenance, incremental replay, crash after acceptance, model
failure after retrieval, revocation, byte accounting, durable backoff and quiet
no-match. The existing acquisition, account-grant, OAuth and scheduling conformance
suites continue to apply. All fixtures use disposable databases and credentials;
no household account, OS keychain or provider write is used. Production readiness
and a real-account pilot still require #123 and explicit consent.

Provider references checked 2026-11-02:

- [Folder message delta and delegated permissions](https://learn.microsoft.com/en-us/graph/api/message-delta?view=graph-rest-1.0)
- [Folder delta links, query restrictions and filter limit](https://learn.microsoft.com/en-us/graph/delta-query-messages)
- [Immutable Outlook IDs and move boundaries](https://learn.microsoft.com/en-us/graph/outlook-immutable-id)
- [429 and Retry-After](https://learn.microsoft.com/en-us/graph/throttling)
