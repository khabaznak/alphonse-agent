# Optional Gmail read access

Gmail is an opt-in native `ReadAdapter` (`integrations/gmail.py`), available to the
existing source acquisition service under provider key `google`. Importing it
loads no provider SDK, credentials or network connections. Connecting an account
alone installs no checks and starts no monitoring. There are no send, modify,
watch, delete, or mailbox configuration operations.

## Clean install and authorization

Follow [managed OAuth setup](oauth-accounts.md) and install the optional
`accounts-oauth` extra plus the appropriate secure credential backend. Configure
`ALPHONSE_GOOGLE_OAUTH_CLIENT_ID` and `ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI` for the
registered public OAuth client. Use the existing authenticated account lifecycle;
request only capability `mail_read`, mapping to
`https://www.googleapis.com/auth/gmail.readonly`. No service account, domain-wide
delegation, client secret in the desktop app, or Gmail write scope is required.
Missing configuration/credentials remain an optional blocked capability and do
not prevent clean startup or affect existing message integrations.

The owner must explicitly grant `mail_read` to an authenticated member, exact
resources and recipients, optionally for a specific project. Resources are exact
Gmail label IDs, or the explicit sentinel `gmail:all` for the entire mailbox,
including Spam and Trash. A label grant includes archived/rule-moved messages
that still carry that label. Removing the authorized label ends access; an Inbox
only grant cannot promise coverage of archived mail. Multiple resources are a
union scanned independently, never an implicit whole-mailbox grant. Stable source
IDs are Gmail message IDs; versions are message history IDs. The connection ID
supplies account identity, so identical IDs in different accounts cannot collide.

The existing project policy authorizes goals, query (`{"text": "Gmail query"}`),
initial lookback, retrieval window, budgets and audience. Authorized one-off
trusted callers can construct a `ReadContext` with `project_id=None` for a
non-project grant, a finite UTC deadline, read limits and a cancellation `guard`.
Use the acquisition service's native read gate for message/attachment expansion
within a project run. Raw provider content must remain untrusted evidence.
These APIs are trusted Python composition points, not unauthenticated legacy IPC
commands. Client management/workflow composition remain #121/#122.

## Coverage and recovery

A first scan captures mailbox history before listing and reads at most 100
messages per logical page (also capped by remaining items). The time-bounded
query is parenthesized before appending the baseline lower bound. Listing is not
limited to Inbox. Each page has at most one profile read, one list and N message
reads; each request rechecks cancellation, current grants, deadline and remaining
transport bytes. Fixed GET endpoints use streamed responses, bounded timeouts,
no redirects and no implicit retries. The acquisition worker owns page retries
and cumulative page/item/byte/runtime budgets. Transport bytes include JSON and
base64 overhead; direct expansions report these bytes to the common gate.

Opaque cursor state includes account/grant/resources/query identity, original
lower bound, resource/page position and history anchor. Acquisition additionally
keys streams by policy/binding/grant revision, query digest and adapter version.
A query/scope change creates a fresh bounded baseline. The service commits accepted
source evidence, page receipt and cursor together before processing advances;
failed processing cannot erase fetched evidence. Overlap records deduplicate by
account/resource/message/version, including replay after failed acceptance.

Gmail history cannot apply arbitrary Gmail queries. An unchanged history advances
without fetching messages; any changes trigger a bounded scoped query refresh
from the original lower bound, after consuming all history pages. It deliberately
re-reads overlap rather than claiming a query-filtered delta. The pre-scan anchor
ensures changes during the scan are considered on the next occurrence. Large
mailboxes or busy history may exhaust budgets: the receipt is partial, never
no-match. This implementation does not promise coverage before the declared
lookback or complete mailbox indexing. History deletions and label removal emit
content-free tombstones so prior source references become unavailable.

An expired history ID (HTTP 404) raises `CursorExpired`. The shared acquisition
service records a gap and attempts one bounded resync using the policy lookback;
it keeps the coverage outcome partial even if that resync finishes. Expired page
tokens also restart only through bounded recovery. Concurrent deletion/label move
between list and message read marks incomplete coverage. HTTP 401/403 blocks;
429/5xx and transport errors report partial without internal retry. Reconnect or
operator policy action is separate from provider evidence and never inferred from
an empty error response. Provider diagnostics, tokens and raw content are absent
from error messages and health output.

Gmail full-message payloads become inert MIME text, preserving quoted text and
charset. HTML normalization never follows links/images or executes scripts.
Attachments remain metadata requiring a separate authorized expansion; the
normalized manifest exposes an attachment ID for that operation. The adapter
verifies the attachment belongs to the currently authorized message, restricts
content types to plain text, CSV and PDF, and enforces a maximum of 5 MiB plus
remaining transport/decoded-byte limits. Missing, oversized or unsupported
attachments are failures, not absent evidence. Returned bytes are never opened
or executed. PDF text extraction is a separate authorized bounded workflow;
attachment-only evidence does not manufacture dates, weights or commitments.

## Validation and pilot gate

Run synthetic tests without credentials or household account access:

```sh
python -m pytest -q tests/test_gmail_adapter.py tests/test_source_acquisition.py \
  tests/test_oauth_lifecycle.py tests/test_account_grants.py \
  tests/test_project_checks.py tests/test_project_check_scheduling.py \
  tests/test_project_triage.py tests/test_findings.py
```

Fixtures use disposable databases and in-memory Gmail responses. They cover
buried/archived notices, MIME and attachment-only evidence, unknown date/weight,
malicious bodies, pagination, history expiry/deletions, throttling, scoped query
changes, revocation/cancellation and failure before/after durable acceptance.
No real-account pilot is authorized by #117. The milestone readiness gate (#123),
backup/recovery and retention qualification, verified Google client configuration,
and explicit owner consent for exact sources, retrieval windows and recipients
must be satisfied before any pilot. Keep setup and readiness gaps visible;
provider connection alone never grants proactive authority.

Provider references: [message listing](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/list),
[history](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.history/list),
[synchronization](https://developers.google.com/workspace/gmail/api/guides/sync),
[attachment reads](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages.attachments/get).
