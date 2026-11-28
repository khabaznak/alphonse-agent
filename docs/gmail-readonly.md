# Optional Gmail read access

Gmail is an opt-in native `ReadAdapter` (`integrations/gmail.py`), available to the
existing source acquisition service under provider key `google`. The adapter
uses Google’s managed Gmail MCP endpoint. Importing it opens no connections. Connecting an account
alone installs no checks and starts no monitoring. There are no send, modify,
watch, delete, or mailbox configuration operations.

## Desktop search and summaries

In **Settings → Integrations → Gmail**, connect the account and grant content
access to Inbox or to a label you select from Gmail. Choose an optional end date
or leave access active until you revoke it. Each grant covers one exact Gmail
label. Inbox content access also covers the unread-count check in the owner's
Desktop conversation. The count-only grant is available on its own. Label
selection and Gmail content searches can renew Google credentials with the
existing connection. Reconnect if Google withdraws or expires its authorization.

When asked in the owner’s Desktop Conversation, Alphonse searches one granted
label per attempt and can summarize excerpts from up to 20 matching messages.
Use the full Gmail name for a nested label, such as `Family/School`. A message may be
archived or outside Inbox and still match the selected label. A request that
omits a label defaults to Inbox. The search planner sees up to 32 unambiguous
label names with live content grants. When a search completes with zero
matches, the planner may make up to two different authorized Gmail searches.
It sees the empty searches' query and scope during refinement. The search
normally covers 90 days and may extend to 365 days. The reply identifies its scope, sources and any incomplete
coverage. When asked, the summary draws out deadlines and action items with
numbered references to the reviewed messages. [Gmail search semantics](https://developers.google.com/workspace/gmail/api/guides/filtering)

The configured inference model receives the bounded matching excerpts in a
summary step with no Alphonse tools. Email text remains untrusted evidence.
The private result is rendered as a read-only A2UI card. Action reviews put
dated rows before undated tasks and keep search coverage visible. Numbered
citations open and focus the exact email excerpt within the card. Headings,
lists and emphasis are formatted in Alphonse's synthesis. Provider fields
and excerpts are literal text. Older Desktop versions can display the
plain-text reply. The task checkpoint
holds bounded excerpts and the draft for verified pickup. Ordinary conversation
history records a receipt without message content. Google access, the local
grant, and the owner claim are rechecked before pickup.

Authenticated owner Desktop history can reconstruct an acknowledged card from
its retained snapshot. The daemon rechecks the same account and label grant.
When the snapshot or grant is unavailable, Desktop shows the receipt.

## Clean install and authorization

Follow [managed OAuth setup](oauth-accounts.md) and install the optional
`accounts-oauth` extra plus the appropriate secure credential backend. Configure
`ALPHONSE_GOOGLE_OAUTH_CLIENT_ID` and `ALPHONSE_GOOGLE_OAUTH_REDIRECT_URI` for the
registered public OAuth client and enable [Gmail MCP](https://developers.google.com/workspace/guides/configure-mcp-servers)
in its Google Cloud project. Use the existing authenticated account lifecycle;
request only capability `mail_read`, mapping to
`https://www.googleapis.com/auth/gmail.readonly`. No service account, domain-wide
delegation, or Gmail write scope is required. Some Google Desktop clients require
their client credential at token exchange; it is not bundled with Alphonse and
can be imported into the configured credential vault from the downloaded client
configuration. Set `ALPHONSE_GOOGLE_OAUTH_CLIENT_SECRET_REF` to the resulting
opaque reference; do not put the credential itself in an environment variable.
Missing configuration/credentials remain an optional blocked capability and do
not prevent clean startup or affect existing message integrations.

The owner must explicitly grant `mail_read` to an authenticated member, exact
resources and recipients, optionally for a specific project. Resources are exact
Gmail label IDs, or the explicit sentinel `gmail:all` for the entire mailbox,
which reports limited coverage because the provider cannot search Spam. A label
grant includes archived or rule-moved messages
that still carry that label. Removing the authorized label ends access; an Inbox
only grant cannot promise coverage of archived mail. Multiple resources are a
union scanned independently, never an implicit whole-mailbox grant. Stable source
IDs are Gmail message IDs; versions are message history IDs. The connection ID
supplies account identity, so identical IDs in different accounts cannot collide.

The existing project policy authorizes goals, query (`{"text": "Gmail query"}`),
initial lookback, retrieval window, budgets and audience. Authorized one-off
trusted callers can construct a `ReadContext` with `project_id=None` for a
non-project grant, a finite UTC deadline, read limits and a cancellation `guard`.
Use the acquisition service's native read gate for message expansion
within a project run. Raw provider content must remain untrusted evidence.
These APIs are trusted Python composition points, not unauthenticated legacy IPC
commands. Client management/workflow composition remain #121/#122.

## Coverage and recovery

The adapter searches threads in bounded date snapshots with
`THREAD_VIEW_METADATA_ONLY`. Before fetching a message body, it checks the
message ID, exact granted label and received timestamp with `METADATA_ONLY`.
It checks them again against the full response before accepting content.
The adapter reports partial coverage for capped or mixed results, provider
omissions, or an exhausted budget. Spam and Drafts searches are unavailable.
Approved archived messages remain discoverable under their retained label.

Each cursor binds account, grant, resources, query, time window and page
position. Acquisition also keys streams by policy, binding and adapter
version. The worker commits accepted source evidence, receipt and cursor
together. The Gmail MCP server exposes no mailbox History feed. The worker
rescans a bounded window on the next occurrence. A message missing from the
snapshot cannot prove deletion or label removal. Direct message access
rechecks the live grant and label.

Each request rechecks authorization, deadline and remaining transport bytes.
Expired page tokens use the worker’s bounded recovery. Provider approval
requests block a check. Transport failures, concurrent moves and incomplete
responses surface without claiming exhaustive absence. The transport uses
fixed endpoints, bounded responses and no automatic retries.

Message text remains inert. Normalization never follows links or images.
Google returns attachment metadata without a standalone download tool.
Attachment bytes are unavailable through managed Gmail. The adapter does
not fetch whole-message RAW as a substitute.

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
malicious bodies, pagination, incomplete scope, throttling, scoped query
changes, revocation, cancellation and failure before or after durable acceptance.
No real-account pilot is authorized by #117. The milestone readiness gate (#123),
backup/recovery and retention qualification, verified Google client configuration,
and explicit owner consent for exact sources, retrieval windows and recipients
must be satisfied before any pilot. Keep setup and readiness gaps visible;
provider connection alone never grants proactive authority.

See Google’s [thread search](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/search_threads),
[message read](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/get_message)
and [label list](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/list_labels)
schemas.
