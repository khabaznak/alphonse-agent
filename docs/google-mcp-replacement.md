# Google Workspace MCP replacement plan

The goal is to replace Alphonse’s direct Google Workspace product API calls with
Google’s official MCP servers. The replacement covers Gmail, Calendar and the
existing Classroom integration. It includes Calendar writes and adds separately
authorized Gmail drafts and sending after the underlying controls are proven.
The owner is the sole current user, so plan one coordinated reconnect and
cutover for Google. Keep the household account and grant model in place. The
plan also evaluates an MCP replacement for personal Outlook.com and Hotmail.

The [operation and scope map](google-mcp-operation-map.md) records the current
calls, documented replacements and gaps to resolve before switching providers.

Google publishes generally available Gmail and Calendar MCP servers. Classroom
requires Developer Preview enrollment. Google says enforcement of its new agent
access requirements begins February 1, 2027. That is the planning deadline for
retiring direct agent access to Google Workspace product APIs. Google OAuth
continues to handle consent, identity, refresh and revocation. [Google setup](https://developers.google.com/workspace/guides/configure-mcp-servers) · [Rollout timeline](https://developers.google.com/workspace/guides/quota-usage-tiers) · [Workspace AI Tools Addendum](https://workspace.google.com/terms/ai-tools/)

## What moves

- **Gmail.** Route Settings label discovery, unread message and thread counts,
  owner search and private evidence review, scheduled checks, and source
  expansion through the same managed adapter. Qualify attachments where the
  existing flow authorizes them. Review draft creation and recipient-visible
  sending as separate capabilities, each with its own consent and runtime
  approval. The existing mail grants stay read-only. [Current integration](gmail-readonly.md) · [Google Gmail MCP reference](https://developers.google.com/workspace/gmail/api/reference/mcp)
- **Calendar.** Move calendar discovery, Desktop Agenda, conversation reads,
  scheduled acquisition, event expansion, and the bundled Family Calendar
  skill’s list, availability, create and update operations. Move the skill’s
  provider calls into the daemon so the skill uses authorized native operations
  and the daemon retains custody of the credentials. Invitations, RSVP and
  deletion are separate capabilities that require a concrete workflow and
  approval design before exposure. [Current integration](family-calendar.md) · [Google Calendar MCP reference](https://developers.google.com/workspace/calendar/api/v3/reference/mcp)
- **Classroom.** Qualify published coursework and announcements for the same
  selected active course and connected student. Check whether the preview can
  establish the existing student enrollment proof without expanding to broad
  roster, teacher or submission access. Show the feature as unavailable if
  preview access, school authorization or equivalent enrollment proof is
  missing at cutover. [Current integration](classroom-readonly.md) · [Google Classroom MCP reference](https://developers.google.com/workspace/classroom/reference/mcp)

Route every entry point through the replacement, including provider health
checks, the Gmail label lookup that currently lives in `OAuthService`, the
simpler loop’s owner services and daemon `SourceAcquisition`. The finishing
audit also checks installed copies of the bundled Family Calendar skill, which
can retain the old script even after the bundled source changes. Preserve
household edits and reconcile legacy bindings explicitly through the standard
skill lifecycle. [Integration management](integration-management.md) · [Standard skills contract](standard-skills-contract.md)

## Outlook and Hotmail

Keep personal Outlook.com and Hotmail on Microsoft Graph. Work IQ MCP is the
Microsoft-hosted counterpart for organizational Microsoft 365 mail and calendar.
It uses Microsoft Entra and tenant policy for work or school accounts. Microsoft
explicitly excludes personal accounts from its Work IQ integration guidance.
No Microsoft-hosted MCP equivalent for personal mail was identified in the
published documentation. [Work IQ MCP overview](https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/mcp/overview) · [Work IQ account prerequisites](https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/chatgpt-plugin)

Preserve both Graph paths. The personal Desktop Inbox count uses delegated
`Mail.ReadBasic` for the selected Inbox metadata. The project acquisition
adapter uses separately authorized `Mail.Read` for exact folders, bounded
delta reads and source expansion. The Google rollout needs regression coverage
for both paths through the shared account and scheduling services. [Current Outlook integration](outlook-mail.md) · [Graph folder read](https://learn.microsoft.com/en-us/graph/api/mailfolder-get?view=graph-rest-1.0) · [Graph message delta](https://learn.microsoft.com/en-us/graph/api/message-delta?view=graph-rest-1.0)

Reopen consumer cutover when a supported MCP endpoint accepts personal account
consent and can preserve the metadata-only Inbox permission, exact folders,
stable message identity and bounded change tracking. If an organizational
account enters scope separately, qualify Work IQ through the shared MCP client.
Check tenant consent and policy, costs, and per-operation resource controls.
Bind Work IQ’s generic resource paths and actions server-side to reviewed
Alphonse operations. [Work IQ tools and policy](https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/mcp/tool-reference) · [Work IQ tenant setup](https://learn.microsoft.com/en-us/microsoft-365/copilot/extensibility/work-iq/enable-work-iq)

## Reusable MCP support

The current tree has no Alphonse-owned MCP client for these account integrations.
Add one shared remote client in the daemon and qualify it against Google’s
Streamable HTTP endpoints. Use an SDK if it supports the protocol versions and
form elicitation Google actually serves. Scope client state and credentials to
the exact configured server and account. Centralize protocol compatibility,
tool discovery and schema checks, response limits, deadlines, cancellation,
typed errors and approval requests. Keep Google’s OAuth and data mapping in its
provider adapters. A future connector should need its own configuration,
credential handling, reviewed operations and adapter tests, and should reuse
the transport and lifecycle. [MCP protocol versioning](https://modelcontextprotocol.io/specification/versioning) · [Google form elicitation](https://developers.google.com/workspace/guides/mcp-elicitation)

The simpler loop proposes Alphonse operations through the current tool registry.
The daemon resolves the authenticated actor, account, grant revision, selected
resource, permitted effect and audience before it calls the corresponding MCP
tool. Server discovery informs compatibility checks. Only reviewed operations
enter the registry. Recheck access, cancellation and deadlines before each
page, retry, approval response, persistence or delivery. Treat server outputs
and provider content as untrusted input. A server request for approval never
substitutes for Alphonse’s own authorization. [Existing MCP equivalence contract](optional-integration-extensions.md#mcp-equivalence-contract-unimplemented-reopening-gates)

Use the primary `/mcp/v1` endpoint and route form elicitation through an
authenticated Desktop review bound to the exact pending call. Keep approvals
single-use within Alphonse and recheck scope before responding to Google. A
scheduled job that requires interactive approval reports that state and stops
within its retrieval window. Google documents that the self-mode endpoint can
return a restricted subset for search. Any explicitly selected background mode
must preserve coverage semantics and surface omitted results. [Google approval and endpoint behavior](https://developers.google.com/workspace/guides/mcp-elicitation)

## Qualification that determines cutover scope

- **Gmail scope and coverage.** `search_threads` can return up to five related
  messages per thread. Verify exact label membership per message, minimize the
  data fetched and release content only for authorized messages. Qualify the
  narrowest search response and keep any out-of-scope metadata in transient
  daemon processing. A capped or filtered thread cannot establish an exhaustive
  match set. The `list_labels` schema includes unread message and thread counts.
  Verify those fields on real responses for the selected label. The public tool
  list does not show a History API equivalent or standalone attachment fetch.
  For scheduled checks, a bounded rescan is acceptable if it preserves the
  authorized window, deduplicates evidence and reports gaps honestly. Show
  unavailable or limited coverage where exact counts or attachments cannot be
  supported. [Gmail tools](https://developers.google.com/workspace/gmail/api/reference/mcp) · [Search response schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/search_threads)
- **Calendar reads and conflicts.** The documented event list supports
  pagination but does not expose the REST sync token. Test bounded window
  snapshots for recurring instances, cancellations, private or free/busy-only
  records, and time zones. Infer removals only when the coverage supports that
  conclusion. Verify that availability can be scoped to the selected calendar
  and that the result distinguishes busy intervals from suggested times.
  A live MCP update accepted a nonmatching `If-Match` header and overwrote a
  newer test description. The interim owner-approved update flow rechecks
  the event before dispatch and warns about last-second manual edits. Create
  and readback passed for one guest-free test event. Qualify
  recovery from an actual lost response and duplicate prevention before
  production create. [Event list schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/list_events) · [Live write evidence](agent-loop/evidence/google-mcp-discovery-2026-11-27.md#calendar-writes) · [Current write validation](family-calendar.md#opt-in-real-account-validation)
- **Gmail and Calendar write outcomes.** Bind the review to the exact account,
  resource, effect and fields. For mail, show recipients, subject and body. For
  Calendar, show the affected event or series, dates and notification behavior.
  Retain the approved intent before calling Google, verify the result against
  provider evidence, and reconcile an ambiguous timeout before any retry. If a
  send cannot be safely verified, report its outcome as unknown and avoid
  resending. A successful approval is evidence of consent.
- **Insights, evidence and retention.** Preserve the workflows producing useful
  insights by retrieving task-scoped Gmail and Calendar evidence when needed.
  For example, a weekly commitments review can combine a granted mail label and
  calendar window, cite each source and identify incomplete coverage.
  Inventory retained excerpts, digests, findings, queued deliveries, indexes,
  logs and backups. Map each data class to the Addendum’s limits on active
  authorized use, retention of Google Technical Information, and retention
  after disconnect or revocation.
  Build deletion and invalidation into the credential lifecycle and protect
  independently authored household records. Use disposable synthetic data for
  evaluation datasets. [Google Workspace AI Tools Addendum](https://workspace.google.com/terms/ai-tools/)

## Calendar update protection

Google Calendar’s REST API returns a resource ETag and evaluates `If-Match`
when committing an update. If someone changes the event after Alphonse reads
it, the original ETag fails with HTTP 412. The retired bundled Calendar script
used this guarantee. Both live MCP catalogs expose `updated` on `get_event`
and no precondition on `update_event`. Google’s Workspace CLI generates its
Calendar commands from the same MCP catalog and exposes no precondition flag.
The live write probe confirms that the MCP endpoint accepts a request carrying
a nonmatching `If-Match` header and still commits the change. [Google Calendar versioning](https://developers.google.com/workspace/calendar/api/guides/version-resources) · [Workspace CLI command discovery](https://developers.google.com/workspace/guides/workspace-cli#complete-command-reference) · [Qualification evidence](agent-loop/evidence/google-mcp-discovery-2026-11-27.md#calendar-writes)

The provider capability we need is an opaque event version returned by
`get_event` and an expected-version input on `update_event`. Google must
compare the expected version and apply the patch atomically, returning a
distinct conflict outcome that leaves the current event intact. This can be
an ETag or another provider-documented version token with the same guarantee.
The MCP `updated` timestamp can help show a user what changed. A fresh read
still leaves time for another writer before the update, and Calendar’s REST
timestamp omits reminder-only changes. A local lock coordinates Alphonse
workers only. [Calendar event fields](https://developers.google.com/workspace/calendar/api/v3/reference/events) · [Calendar MCP update schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/update_event)

Once the provider supports conditional updates, bind the exact event version
to the existing Alphonse effect review alongside the account, calendar ID,
event or instance ID, proposed fields and notification behavior. Send the
reviewed version with the update. On conflict, read the current event, show
the difference and require a fresh review before another write. After an
ambiguous response, read back the exact event and reconcile the outcome before
retrying. A version fetched for reconciliation does not refresh the user’s
earlier authorization.

The owner approved an interim Desktop flow. A separate OAuth connection for
Calendar editing requests `calendar.events`. Alphonse additionally requires
an owner-only expiring grant for the exact calendar. Conversation prepares a
short-lived review of the exact event and fields. The owner sees the affected
event, recurrence scope and before/after values. Desktop alone can approve or
dismiss the review. The approval is single-use. Before dispatch, Alphonse
checks the live grant again and fetches the exact event. A changed provider
snapshot invalidates the proposal and requires a new review. A local lock
serializes Alphonse edits to the same event. After dispatch, Alphonse verifies
the reviewed fields by reading back the event. An ambiguous response remains
unknown when readback cannot resolve it. Alphonse never retries the update
automatically. Pending reviews live in the daemon and expire on restart.

The review asks the owner to wait for Alphonse to finish before editing the
event in Google Calendar. A manual change can still land between the final
read and write. The owner can approve adding guests by exact email address
and request notifications to the entire guest list. The review shows both
existing and newly added guests. Readback verifies membership. Simultaneous
edits can still change who Google notifies. Event creation, guest removal,
conference changes, and unattended edits remain outside this flow.

Qualification for a future provider precondition requires an unchanged event
to accept the reviewed patch and another client’s edit after the final read to
cause a conflict without changing the newer event. For the interim flow,
verify that a change visible during the final read blocks dispatch and that a
lost response triggers exact readback without a blind retry. Include explicit
recurrence scope and same-event attempts on a disposable calendar. The local
check cannot certify what happened between the final read and write.

## Delivery and cutover

1. **Freeze the capability inventory.** Record each affected entry point, Google
   schema and required scope, expected source IDs, coverage, approval behavior
   and failure states. Confirm the owner’s agent-enabled OAuth client setup and
   Classroom eligibility. Record any unsupported capability and its visible
   product behavior before choosing cutover scope.
2. **Build the client and wire all adapters.** Exercise representative wire
   fixtures for discovery and schema changes, pagination, refresh, prompt
   injection in provider content, partial results and late replies. Test account
   and grant separation with multiple disposable identities even though the
   rollout has one owner. Test one second provider-shaped fixture to check that
   the client is reusable without introducing another live integration.
3. **Qualify the actual product paths.** Extend the existing Dojo exercises for
   Gmail search and counts, Calendar and scheduled work. Use disposable records
   and the daemon/Desktop paths. Cover wrong account or label, revoked consent
   during a read, duplicate schedule delivery, a Calendar write racing a human
   edit, and a send whose response is lost. Run fake-server conformance first,
   then separately qualify Google’s servers with a disposable mailbox, calendar
   and approved Classroom fixture where available. Record observable provider
   effects, partial coverage, model/provider mode, costs and untested paths.
   [Dojo](agent-loop/dojo.md) · [Gmail search](agent-loop/dojo-gmail-content.md) · [Gmail counts](agent-loop/dojo-gmail-inbox.md) · [Owner Calendar](agent-loop/dojo-calendar-owner.md) · [Scheduled capabilities](agent-loop/dojo-scheduled-capabilities-2026-11-22.md)
   Recheck the [personal Hotmail Inbox path](agent-loop/dojo-outlook-inbox.md)
   and Graph project acquisition against disposable fixtures before switching
   the shared services over.
4. **Reconnect and switch the owner.** Pause affected schedules and fence pending
   work. Reconnect through Settings and verify the same Google subject and
   selected label, calendar and course IDs before reusing grants. Require a fresh
   review wherever authority or coverage changes. Keep saved policies and permitted
   evidence provenance. Reset transport-specific cursors and establish one bounded
   baseline per approved scope. Resume scheduling under its existing authorization
   rules and reconcile pending writes before allowing new attempts.
5. **Retire direct Workspace calls.** Remove the product REST transports and
   bundled script execution path when the capability disposition is explicit
   and verified. Verify no production entry point routes around the daemon client.
   Retire superseded credentials and apply the retention decisions. Recovery
   pauses affected features or restores a known-good MCP build with compatible
   state. After cutover there is no automatic REST fallback.
