# Gmail and Calendar MCP qualification on November 27, 2026

Google’s actual endpoints were queried from the development machine. This run
qualified public tool discovery and the unauthenticated boundary. Account-level
checks use the owner-approved existing Google account and limit test effects
to disposable resources. The discovery requests sent no OAuth token or
household data.

## Provider evidence

`server/discover` and `tools/list` returned HTTP 200 at both Google endpoints
for both products. An unauthenticated `tools/call` for `list_labels` or
`list_calendars` returned HTTP 401 at all four endpoints. Repeating discovery
with Google’s documented `2026-07-28` protocol headers and form-elicitation
metadata returned the same tool catalogs. Hashes below use the UTF-8 JSON tool
array with sorted object keys and compact separators.

| Google endpoint | Tools | Catalog SHA-256 |
| --- | ---: | --- |
| `gmailmcp.googleapis.com/mcp/v1` | 30 | `2271a1c1bcb8e26cac182bc2f9f8765cf753a8d6af9c387dbcfd052ee2c9ab3a` |
| `gmailmcp.googleapis.com/mcp/v1self` | 25 | `a29a6fd79ba00e81aef9585fecdba96fab73503d731096ad6d41ad56f5d07144` |
| `calendarmcp.googleapis.com/mcp/v1` | 9 | `cf54b0a6cc9b3161f694dd5d46b7aff20b9d8400a8bb557b0500d13ab8c37dfc` |
| `calendarmcp.googleapis.com/mcp/v1self` | 8 | `6dfea81424c02e8350c3c2cba8658cdcf903e14a1cc2d85f5584d0911082d7d4` |

The live schemas exposed these details on both endpoint modes:

- Gmail `get_message` declares `internalDate` in epoch milliseconds and
  `historyId` for the last change to that message. These are candidates for
  preserving the current evidence timestamp and version. Verify their presence
  in an authenticated content response. No discovered tool exposes a mailbox
  history feed or a single-attachment download. Attachment metadata declares
  ID, MIME type, filename and part ID. It has no size or byte field. `RAW` is
  still a whole-message read.
- Gmail `list_labels` declares both unread counts without making them required.
  `search_threads` declares `includeTrash`, `nextPageToken`,
  `omittedContent` and `resultCountEstimate`. Test actual count presence and
  preserve the distinction between a count and a search estimate. Check empty
  pages with continuations and filtered content before drawing absence conclusions.
- Calendar `list_events` declares `nextPageToken` and no incremental sync
  token or deletion switch. Its event schema declares `recurringEventId` and
  `originalStartTime`, with no ETag or `iCalUID`. `suggest_time` declares time
  slots and no busy-interval list keyed by a selected calendar. The currently
  published guide describes omitted attendees, which were absent from this
  discovery response’s `suggest_time` output schema.

## Existing Alphonse connection

The owner approved using Alphonse’s existing Google account for the next
phase. The saved Calendar connection had a current token with
`calendar.events.readonly`. An authenticated `get_event` call for a nonexistent
test resource returned HTTP 403 with Google’s API-disabled diagnostic. A
Gmail control request for a nonexistent message with the Calendar credential
returned HTTP 403 for insufficient scope. The saved Gmail access token had
expired. The control tests reached Google and returned no account content.

The Cloud Console showed Gmail MCP and Calendar MCP disabled for Alphonse’s
project and **Use this client for an AI-powered agent** already selected on
the existing Desktop OAuth client. The Console listed the Workspace AI Tools
Addendum under the terms for these APIs. The saved Calendar grant does not
include `calendar.events`. Provider writes require consent for that scope and
an exact test calendar. [Google setup](https://developers.google.com/workspace/guides/configure-mcp-servers)

After the owner enabled Gmail MCP and Calendar MCP, the same Alphonse client
and its managed grants reached both endpoint modes. Gmail returned the expected
not-found result for the deliberately nonexistent message after managed token
refresh. Calendar returned a tool-level error for the deliberately nonexistent
event, replacing the prior API-disabled response.

## Authenticated Gmail fixture

The owner’s existing Gmail account received a synthetic self-addressed message.
Gmail UI showed it under a dedicated test label, archived, and unread. On both
`/mcp/v1` and `/mcp/v1self`, the live MCP server:

- Returned one unread message and one unread thread for the exact test label.
- Returned that archived message from a subject-and-label search with a
  metadata-only thread view. The response had no continuation or reported
  omissions for this one-message case.
- Returned the expected marker, message ID and label from `get_message`, with
  populated `internalDate` and `historyId`.

Calls declared no form elicitation capability and completed without Google
approval. These observations establish the behavior of this fixture. They leave
mixed-label threads, multiple pages, filtering, attachments and unattended
coverage on real scheduled runs open.

## Scheduled checks

The existing Alphonse regression suite passed 206 tests across Gmail reads and
counts, owner scheduled Gmail, project scheduling, source acquisition, Calendar
reads and the Family Calendar skill. These tests use fixtures and exercise the
current path. They supply the baseline for the MCP migration.

The authenticated Gmail fixture above tested provider calls made without form
elicitation. The Calendar run exercised Alphonse’s `ScheduledTaskWorker`,
`ProjectCheckDispatcher` and `ProjectCheckWorker` with a controlled clock, an
isolated project and schedule database, and a temporary read-only MCP adapter.
The exact private test calendar grant was configured in the fixture. The probe
used the owner’s existing `calendar.events.readonly` token for live provider
calls. It sent only the explicitly selected test calendar and date range.
The independently seeded event was visible in Google Calendar before the run.

| Scheduled scenario | Live provider calls | Durable outcome |
| --- | ---: | --- |
| In-window `/mcp/v1`, no form elicitation | 1 | Complete, one expected event item checkpointed |
| In-window `/mcp/v1self`, no form elicitation | 1 | Complete, one expected event item checkpointed |
| Worker starts at the exclusive end of the retrieval window | 0 | Partial, `window_expired`, zero items and pages |

All three rerun scenarios passed. The initial harness invocation failed to
load the repository’s disposable credential fixture and made no scheduled
provider calls. Separate direct preflight reads returned exactly the expected
event on both endpoints, with neither a continuation nor a sync token. A
second worker claim found no remaining work in each scenario. No household
schedule, finding or notification was created. The shipped daemon still needs
its managed MCP adapter. Multi-page, moved and recurring events, provider
approval challenges, and token refresh during an occurrence remain open.
[Scheduled Dojo](../dojo-scheduled-capabilities-2026-11-22.md) · [Google approval protocol](https://developers.google.com/workspace/guides/mcp-elicitation)

## Calendar writes

The live `create_event` and `update_event` schemas accept an exact `calendarId`
and `notificationLevel=NONE`. Create has no advertised caller-supplied event ID
or idempotency key. Update has no advertised version precondition or recurrence
update field. `get_event` has no advertised ETag. These are confirmed gaps in
tool discovery.

The temporary OAuth receiver expired before exchanging the first authorization
code. A silent reauthorization returned `interaction_required`. The interactive
retry showed a Google consent screen granting event editing across all calendars
in the personal account. Approval review rejected submitting that broader grant
for the test-calendar-only exercise. The consent flow was closed and its local
receiver stopped. The synthetic event was seeded in Google Calendar for the
read-only scheduled experiment. This attempt ended before any MCP write.
Google lists `calendar.events` and `calendar` for both write tools. The narrower
Calendar REST scopes for owned or app-created calendars are absent from the
published MCP write tool scopes. [Create scopes](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/create_event#authorization-scopes) · [Update scopes](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/update_event#authorization-scopes) · [Calendar API scopes](https://developers.google.com/workspace/calendar/api/auth)

After the owner explicitly approved account-wide OAuth consent in Google, the
temporary runner verified that Google returned the existing Alphonse account’s
subject. The access token stayed in memory for the run. All writes specified
the same private calendar ID and `notificationLevel=NONE`. The runner used
novel synthetic content and left one write fixture for inspection.

| Live scenario | Observed result |
| --- | --- |
| Create and readback | One `create_event` dispatch created a single matching event for November 28, 2026, 2:30–2:45 p.m. in Mexico City. A fresh `list_events` located it without using the ID from the create response. `get_event` verified the exact timing and fields, zero attendees and no Meet link. |
| Update and readback | `update_event` changed the description. A fresh `get_event` verified the same event ID and the new description. |
| Stale precondition | `update_event` with a deliberately nonmatching `If-Match` header succeeded and overwrote the newer test description. No conditional-write rejection occurred. |
| Independent readback | With the existing read-only credential, both `/mcp/v1` and `/mcp/v1self` returned the same single fixture, the stale probe’s description, zero attendees and no Meet link. |

The runner made eight MCP calls for the write sequence and four for the separate
readback. The create response arrived normally. Ignoring its ID exercised
reconciliation for this single fixture without resubmitting the operation. A
transport failure and behavior on an actual retry remain untested. The stale
probe demonstrates that sending an `If-Match` header to the tested MCP endpoint
does not preserve the previous atomic update guard. A provider-enforced version
precondition or equivalent is needed to restore that guarantee. [Family Calendar validation contract](../../family-calendar.md#opt-in-real-account-validation)

### Conditional update follow-up

Fresh unauthenticated discovery on both Calendar endpoint modes returned
`updated` as the only version-related `get_event` output and no version or
precondition input on `update_event`. Google Workspace CLI `1.0.6.1026726559`
generated its Calendar commands from the live MCP server. Its `get_event`
schema exposed `updated`, and its `update_event` help exposed neither a version
flag nor a request-header flag. This follow-up used public discovery only.
[Workspace CLI reference](https://developers.google.com/workspace/guides/workspace-cli#complete-command-reference) · [Calendar update protection plan](../../google-mcp-replacement.md#calendar-update-protection)

## Migrated client verification

The shared production MCP client reached both `/mcp/v1` endpoints with the
existing Alphonse account and read-only grants. Gmail returned one unread
message and one unread thread for the disposable label. A label-and-date
search found the archived fixture. Its metadata read populated `internalDate`,
`historyId` and the expected label, and omitted the body. Calendar returned two
events for the explicit offset-bearing test window on the private test calendar.
An empty window returned the owner access role and calendar timezone, and omitted
the `events` field. The adapter accepts that shape as an empty result only for
full-detail access with no continuation or reported omission.

### Scheduled checks with the migrated adapters

The actual scheduler, dispatcher, project worker and source acquisition service
ran in temporary databases with a controlled occurrence clock, the new adapters,
and live Google MCP reads. Scope was the disposable Gmail label and private test
calendar. The earlier Calendar checks used the temporary adapter described above.

| Migrated check | Live calls | Outcome |
| --- | ---: | --- |
| Gmail within window | 4 | Complete, one evidence item and one page |
| Calendar within window | 1 | Complete, two evidence items and one page |
| Gmail at exclusive window end | 0 | Partial, `window_expired`, zero items and pages |
| Calendar at exclusive window end | 0 | Partial, `window_expired`, zero items and pages |

All four cases passed in 2.93 seconds. A repeated worker claim found no remaining
work. The live calls made reads only and no household schedule was created.

### Owner conversation and approval Dojo

The Dojo exercised the owner worker and authenticated Unix IPC with disposable
databases, the real `gpt-6-luna` model, and simulated Calendar MCP data. The
authorized agenda case made one exact `list_events` call, displayed the expected
timed and all-day events privately, and omitted the cancelled fixture. The
no-grant case made zero calls and returned setup guidance. They took 12.578 and
17.491 seconds, respectively. The authorized case reported 32,713 input and 197
output tokens. The no-grant case reported 51,145 input and 264 output tokens.

The model interpreted a Spanish request to change the location of a named event.
The private review showed only the requested location change. The simulated
provider recorded `list_events` and `get_event` before approval, followed by
`get_event`, one `update_event` with no notifications, and another `get_event`
after the separate IPC approval. The simulated event reflected the new location.
The approval token and unrelated event description were absent from model input.
The first run failed at the response budget after staging. After correcting that
budget, the rerun completed in 16.238 seconds and reported 30,509 input and 237
output tokens across two calls. Status copy appeared in English. Disposable
regressions separately exercised conflict, authority changes, provider approval,
and an ambiguous response. The Dojo used IPC and a simulated provider. Review
rendering was covered by a separate Desktop component test.

## November 28 Desktop rollout

The installed macOS app and its bundled daemon were updated using the existing
Alphonse installation. The Calendar editing connection matched the existing
Google account. The live test grant allowed only the private disposable calendar
and had a 45-minute expiry. Proposing the edit left the provider event unchanged.

During rollout the daemon lost its SQLite and socket connections. A disposable
stress run with the default 256-file limit reproduced those failures. The local
macOS session now runs the bundled daemon with a higher limit scoped to that
process. The previous app bundle is preserved for rollback.

The owner found that the Calendar review was below the text area and its buttons
were clipped. The review now appears above the composer. Long changes scroll
inside the card, with the warning and actions outside that scrolling region.
The exact calendar ID is available in a collapsed disclosure. The installed
native app showed the review and its actions before the text area and Send.

The Critic independently exercised the actual frontend with simulated IPC at
1120×760 and 760×600 with At hand open. Approve, Dismiss and Send stayed reachable
with a long edit and no overlap. Keyboard approval produced one simulated dispatch
and focused the result status. The independent review did not exercise live
Google behavior. The Desktop test suite passed 366 tests after this fix.

## Inline comparison and guest invitation review

The approved inline design was implemented as a separate Conversation surface
above the composer. A disposable Calendar response included a description change,
one existing guest, and one new guest. The card showed the two exact addresses
and the notification effect before approval. An approval without the explicit
notification level, including a request from an older Desktop, made zero provider
calls. A valid approval dispatched once. A separate read verified the guest list.
The simulated results distinguish confirmed membership from email delivery.

The independent UI Tester and The Critic rendered the actual frontend with
synthetic IPC in Chromium. The Critic issued **PASS** after checking 1120×760
with At hand open and closed, and 760×600 with At hand open. Both recipients and
all approval controls remained reachable. When field comparisons overflowed,
**Scroll to description** focused and scrolled the comparison while the composer
stayed in place. The fixture preserved the active draft and selection, dispatched
one approval on keyboard activation, and reported missing guest verification as
unknown. These checks used disposable data and simulated Calendar effects. Native
WebKit and live email delivery were outside their scope.

### Installed invitation build

The signed local Desktop bundle and its daemon were switched to the invitation
build. The prior app was kept for rollback. The same Google identity was verified
across the saved Gmail and Calendar connections. A self-addressed disposable
message established the single test recipient. A separate Calendar read
confirmed the selected private test event started without guests.

The installed daemon staged one owner review on the exact private test calendar
under a 45-minute grant. The native window showed the comparison, description
shortcut, single-recipient notification notice, and **Approve and notify** above
the composer. The proposal used the prior fields from the live event and made
no update. The owner controls the approval action in Desktop.

The first review expired. Subsequent Google Calendar readback showed the original
fields and an empty guest list. The exact proposal was staged again, and Desktop
reported verified completion after the owner approved it. An independent Google
Calendar read verified the approved location and description and the single test
guest. The guest was the connected Google account itself. The Gmail check for
the fixture title across all folders returned no matching message, so this run
verified the Calendar write and guest addition. Gmail delivery remains unverified.

### Conversation receipts and Agenda refresh

A disposable owner Dojo case exercised the conversation proposal, separate
Desktop IPC approval, Calendar write, and provider readback. The verified result
was also visible from the approval day's history endpoint. A separate SQLite
reopen check recovered the receipt. The fixture rejected retrieval as another
member and retained no approval token or calendar ID in the receipt. Unknown
results left history empty.

The running Agenda widget reported an unavailable result for both Today and
Month. Its existing read grant selected a different calendar from the private
write test. The configured read connection also held a persisted OAuth refresh
lease and had an expired access token. A read-only qualification using an
in-memory refresh with the production request shape returned the expected
synthetic event and projected it into Agenda. The existing reconnect procedure
clears the interrupted refresh lease with a connection revision check.

The desktop and daemon fixtures exercised an invitation accepted in Google and
subsequently found by Agenda on focus and polling. A successful Desktop edit
invalidated an open widget immediately. These scenarios used disposable
provider data. The live grant selection and OAuth connection were untouched.

The Critic passed the updated production frontend at 1120×760 and 760×600 with
synthetic IPC. The card remained in approval-time order through date navigation
and reload, wrapped long addresses, preserved the draft and selection, and
presented no edit actions. The visual check used Chromium.

### Live Agenda and receipt qualification

The signed app and its daemon were switched to the receipt build. The daemon
answered the owner-only receipt history request. The existing Calendar read
account was reconnected at the same scope, clearing the stale refresh lease.
The existing owner read grant selected only the private test calendar.

The native At hand Agenda showed the original Nov 27 synthetic event in Week
view. An independent Calendar read confirmed it was a non-recurring event
with no guests. The new build staged a title-only approval above the composer.
The original title was still present in Agenda before approval.

The first review's approval window expired. Independent Calendar readback
and the native Agenda still showed the original title, and the owner
history contained no matching receipt. The exact title-only review was
staged again for owner approval.

After the short approval windows elapsed, the owner requested a longer window.
The authenticated local proposal accepted a 30-minute window for this review.
The ordinary proposal still uses five minutes, and the exact Calendar write
grant bounds the review's expiration. The native app showed the extended
pending card and the original event title in Agenda Week before approval.

The owner approved the title-only change in Desktop at 11:14 a.m. Mexico City
time. Native Agenda Week displayed the new title without a manual refresh,
and a separate provider read confirmed it on the exact test event. The daemon's
Agenda result contained one event with the new title and zero with the original
title. Native Conversation showed a read-only **Approved and saved** card for
the Nov 28 approval, with the original Nov 27 event time and no edit controls.
The receipt matched its SQLite row and was absent from the Nov 27 approval
history. The test event still had no guests.

### Final Desktop merge gate

The Critic independently returned **PASS** on the final Desktop manifest
`d59651385fc6ae61b49ad10cf5a73443648321d36e9df88b0efd8ddb4a9f5b37`
against base `f9001327`. The actual React app rendered in Chromium at
1120×760 and 760×600, in light and dark themes and with At hand open and
closed. The review checked visible current and proposed values, exact guest
addresses, notification scope, and access to approval and dismissal.
Long-title, multi-review, and expanded-details fixtures checked body
scrolling, field navigation, focus, and composer position.

The gate also verified the owner receipt's chronology, focus, and persistence,
Agenda recovery and refresh, Gmail label access, and the exact Calendar edit
grant setup. The visual run used synthetic IPC and made no provider writes.
The separate live Desktop run above exercised the installed app and Google.

The Desktop production build, 378 Desktop tests, and 347 focused Python tests
passed. Under the same sandbox, every failure or error in the full migration
suite also occurred on clean main.
