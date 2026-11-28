# Google Workspace operations and MCP equivalents

Repository inventory and published Google schemas checked November 27, 2026.
This is the operation inventory for the [full replacement plan](google-mcp-replacement.md).
Each row lists the current provider call, a documented MCP tool where available,
and an accepted scope to qualify. The scope names omit the common prefix
`https://www.googleapis.com/auth/`. Listed scopes are alternatives unless a row
explicitly combines tools. Google OAuth consent and Alphonse’s resource grants
apply independently. The [live Gmail and Calendar discovery receipt](agent-loop/evidence/google-mcp-discovery-2026-11-27.md)
records the schemas actually returned by Google. Account-level tool behavior
and consent still need qualification on the intended account.

## Gmail

The current managed capability uses `gmail.readonly` for discovery, count, content
and project acquisition. Keep the count-only and content grants distinct in
Alphonse even when they share a Google OAuth connection. Source paths include
[Settings discovery](../alphonse/agent_v2/integrations/oauth.py), [unread counts](../alphonse/agent_v2/integrations/gmail_unread.py), [the owner’s Inbox status](../alphonse/agent_v2/core/intelligence/simple_loop/owner_gmail_search.py), and [the mail adapter](../alphonse/agent_v2/integrations/gmail.py).

| Current operation | Official MCP tool and scope | Cutover check |
| --- | --- | --- |
| `GET labels` for owner Settings label IDs and names | `list_labels` with existing `gmail.readonly`. Also documents narrower `gmail.labels.readonly`. | Normalize MCP `labelId` and `name` to the existing Settings contract. Keep it an owner-only discovery step. [Schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/list_labels) |
| `GET labels/{labelId}` for exact unread messages or threads, including the owner’s Inbox status lookup | `list_labels` with `gmail.readonly`. Qualify `gmail.labels.readonly` if a separate count-only consent is useful. | The response schema includes `messagesUnread` and `threadsUnread`. Verify that the fields are populated for Inbox and custom labels, including zero. The tool returns all labels. Keep other labels and counts in transient daemon processing. Missing or invalid counts are unavailable. [Schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/list_labels) |
| `GET profile` for adapter health | `list_labels` with `gmail.readonly` can probe mailbox reachability. | Report tool reachability separately from history support. Use the bound OAuth subject for account identity. [Gmail tool catalog](https://developers.google.com/workspace/gmail/api/reference/mcp) |
| `GET profile` for initial mailbox `historyId`, and `GET history` for paged deltas, label removals, deletions and rescan triggers | No documented MCP tool or scope provides the Gmail history feed. A bounded rescan would use `search_threads` and `get_message`, both accepting `gmail.readonly`. | Replace REST cursors with a fresh scoped baseline. Establish how to invalidate stale evidence for known IDs. Search omission, exhausted budgets and interrupted pagination cannot prove deletion or label removal. Surface the missing change-feed guarantee if no equivalent is available. [Gmail tool catalog](https://developers.google.com/workspace/gmail/api/reference/mcp) · [Search schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/search_threads) |
| `GET messages` for exact-label owner search and bounded project discovery | `search_threads` with `gmail.readonly` or `gmail.messages.readonly`. `get_thread` with the same scope is a candidate for metadata-only expansion of capped threads. | Search takes label display names in its query, while grants store immutable label IDs. Resolve the current name from the trusted ID and verify the label on each message before fetching content. Validate renames, nested labels, date/sender filters, archived mail, same-thread nonmatches, multiple pages and Spam/Trash. The search returns at most five related messages per thread and may filter some messages without reporting them. Define truthful partial and no-match results. [Search schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/search_threads) · [Thread schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/get_thread) |
| `GET messages/{messageId}` with `format=full` for normalization, evidence expansion and attachment preflight | `get_message` with `gmail.readonly` or `gmail.messages.readonly`. Use explicit `METADATA_ONLY` for scope checks and bounded `FULL_CONTENT` for authorized content. | Match ID, label and time before release. The live schema declares `internalDate` and `historyId`, beyond the published reference. Verify their presence in authenticated responses. Validate HTML/text normalization and attachment provenance since the REST MIME part tree is absent. [Live schema evidence](agent-loop/evidence/google-mcp-discovery-2026-11-27.md#provider-evidence) · [Published schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/get_message) |
| `GET messages/{messageId}/attachments/{attachmentId}` for one approved text, CSV or PDF attachment | No dedicated attachment tool. `get_message` exposes attachment metadata and a `RAW` MIME format under `gmail.readonly` or `gmail.messages.readonly`. | Test whether `RAW` includes the approved attachment bytes and whether the exact attachment ID/type/size can be matched and bounded before release. Whole-message RAW can fetch unrelated attachments and can exceed the current budget. Mark attachment fetching unavailable unless the replacement preserves the existing scope and limits. [Schema](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/get_message) |

## Calendar

Managed Calendar discovery currently uses `calendar.calendarlist.readonly`.
Managed event reads use `calendar.events.readonly`. The bundled Family Calendar
script additionally accepts `calendar.events` or `calendar` for writes and
`calendar.freebusy`, `calendar.events.freebusy`, `calendar.readonly` or `calendar`
for availability. See the [managed Calendar adapter](../alphonse/agent_v2/integrations/google_calendar.py)
and the [bundled skill script](../alphonse/agent_v2/builtin_skills/family-calendar/scripts/google_calendar.py).

The live scheduled experiment succeeded with `calendar.events.readonly` on
both endpoint modes for one private fixture event. For writes, Google’s
`calendar.events` consent grants provider access across the account’s calendars.
Alphonse’s exact-calendar restriction applies locally. The published MCP write
tools omit narrower scopes for owned and app-created calendars. [Qualification receipt](agent-loop/evidence/google-mcp-discovery-2026-11-27.md#scheduled-checks) · [Google OAuth scopes](https://developers.google.com/workspace/calendar/api/auth)

| Current operation | Official MCP tool and scope | Cutover check |
| --- | --- | --- |
| `GET users/me/calendarList` for explicit Calendar setup and discovery | `list_calendars` with `calendar.calendarlist.readonly`. | Map and paginate exact IDs, names, time zones and access roles. Discovery alone does not select or grant event access. [Schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/list_calendars) |
| `GET calendars/{calendarId}/events` for the adapter health probe, Desktop Agenda, owner conversation, bundled list and the all-day expansion time-zone lookup | `list_events` with `calendar.events.readonly` or existing write consent `calendar.events` where appropriate. | Pass explicit `calendarId` on every request. Test exact date and time bounds, all-day dates, time zones, recurrence expansion, ordering, paging and redaction. MCP defaults to a subset of event types. Qualify the required explicit types and verify which fields can reach Desktop or the model. [Schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/list_events) |
| `GET calendars/{calendarId}/events` for scheduled baseline and incremental sync | `list_events` with `calendar.events.readonly` for a bounded snapshot. No documented `syncToken` or `nextSyncToken`. | Design a transport-versioned baseline and bounded reconciliation. Qualify cancellation, renamed events, moved instances, window exits and private/free-busy coverage before replacing REST tombstone behavior. The schema exposes recurrence and original start time, but does not document REST `iCalUID` or ETag. Define evidence identity/version migration and expose incomplete coverage. [Schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/list_events) |
| `GET calendars/{calendarId}/events/{eventId}` for source expansion, skill get and write verification | `get_event` with `calendar.events.readonly` for reads or `calendar.events` for write readback. | Pass explicit calendar and event IDs. Preserve event-window checks and all-day time-zone handling. Verify returned IDs, recurrence, timing and effect fields. This schema does not document an ETag for safe update approval. [Schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/get_event) |
| `POST freeBusy` for busy intervals on one exact selected calendar | `suggest_time` accepts `calendar.freebusy` or `calendar.events.freebusy`. Its output contains suggestions and omitted attendees. | No documented equivalent for returning busy intervals keyed by the selected calendar ID. Qualify any alternative with the same resource boundary and exact interval semantics. A suggestion or an omitted attendee cannot prove availability. [Schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/suggest_time) |
| `POST calendars/{calendarId}/events` for authorized create | `create_event` and `get_event` with `calendar.events`. | A live private-calendar create and readback verified timing, zero attendees and no Meet link with `notificationLevel=NONE`. A fresh listing found one event after ignoring the create response ID. REST uses a caller-generated ID derived from the operation ID. MCP does not document that input or an idempotency key. Field and recurrence parity, an actual lost response and duplicate prevention on retry remain open. [Live write evidence](agent-loop/evidence/google-mcp-discovery-2026-11-27.md#calendar-writes) · [Create schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/create_event) |
| `PATCH calendars/{calendarId}/events/{eventId}` for authorized conditional update | `update_event` and `get_event` with `calendar.events`. | A live description edit succeeded. A subsequent request with a deliberately nonmatching `If-Match` header overwrote that edit. The schema exposes neither a version precondition nor a recurrence update field. The interim owner Desktop flow requires exact review, distinguishes instance from series, blocks attendee edits, checks the event immediately before writing and reads it back. An external change after the final check can still be overwritten. [Live write evidence](agent-loop/evidence/google-mcp-discovery-2026-11-27.md#calendar-writes) · [Published schema](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/update_event) |

## Classroom Developer Preview

The current coursework and announcement capabilities use
`classroom.coursework.me.readonly` and `classroom.announcements.readonly`
respectively. Both currently also use `classroom.courses.readonly` and
`classroom.rosters.readonly` for the active-course and self-enrollment checks.
Each content page and direct read requires both checks. See the
[Classroom adapter](../alphonse/agent_v2/integrations/classroom.py) and
[Google’s preview tool catalog](https://developers.google.com/workspace/classroom/reference/mcp).

| Current operation | Official MCP tool and scope | Cutover check |
| --- | --- | --- |
| `GET courses/{courseId}` for exact active course and adapter health | `list_courses` with `classroom.courses.readonly`. | The result documents course ID and state. Qualify finding the exact `ACTIVE` course with bounded pagination. Discard unrelated course data. There is no documented exact-course get tool. [Schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/list_courses) |
| `GET courses/{courseId}/students/me` for the caller’s enrollment | `list_courses` with `classroom.courses.readonly` documents the caller’s course `role`, including `STUDENT`. | Verify that the role is for the connected account in the exact active course and is authoritative under school policy. If qualified, this can replace both preflights without separate roster consent. `get_course_roster` is paginated across other members and does not provide an exact `students/me` call. [Course schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/list_courses) · [Roster schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/get_course_roster) |
| `GET courses/{courseId}/courseWork` for published coursework initial list and scheduled refresh | `list_course_work_summaries` plus `get_course_work` where detail is required, both with `classroom.coursework.me.readonly`. | Disable submission overview. Check stable item IDs, result budgets, published state and description coverage. The summary is not a substitute for the current payload or version and updated-time fields. [List schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/list_course_work_summaries) · [Detail schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/get_course_work) |
| `GET courses/{courseId}/courseWork/{itemId}` for direct coursework evidence | `get_course_work` with `classroom.coursework.me.readonly`. | Verify course binding, exact item ID and published state. Qualify description, maximum points and date-only vs explicit due time. Preserve unknowns if the localized date response does not provide original time or version evidence. Retain no submissions, grades or unrelated materials. [Schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/get_course_work) |
| `GET courses/{courseId}/announcements` for published announcement list and scheduled refresh | `list_announcement_summaries` plus `get_announcements` for full text, both with `classroom.announcements.readonly`. | Summaries truncate text. Bound expansion to selected published IDs and preserve item/course provenance. Qualify whether evidence can distinguish edited announcements from new ones within the existing lookback. [List schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/list_announcement_summaries) · [Detail schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/get_announcements) |
| `GET courses/{courseId}/announcements/{itemId}` for direct announcement evidence | `get_announcements` with a single `announcementIds` entry and `classroom.announcements.readonly`. | Verify the exact ID and published state under the requested course. Use full text without fetching linked materials. Qualify timestamp semantics for both announcement and coursework refresh. Publication date alone cannot stand in for creation/update timestamps. [Schema](https://developers.google.com/workspace/classroom/reference/mcp/tools_list/get_announcements) |

## OAuth and connection lifecycle

| Current operation | MCP cutover disposition |
| --- | --- |
| Open Google’s authorization endpoint with authorization code, PKCE, `openid`, consent and offline access | Keep the Google OAuth flow. Verify the intended Desktop OAuth client is marked for agent use and asks for only the scopes of enabled operations. |
| Exchange the code and refresh tokens at Google’s token endpoint, including the legacy Calendar script refresh | Managed token exchange and refresh stays in the daemon. The retired skill binding can no longer receive Google credentials. |
| Read Google OpenID Connect `userinfo` and verify `sub` | Keep this identity check. Verify subject continuity and credential revision on reconnect. |
| Revoke at Google’s OAuth revocation endpoint | Keep revocation. Fence grants, scheduled work, stored evidence access and any pending MCP approval before release or delivery. |

These are OAuth endpoints used by [the current authorization service](../alphonse/agent_v2/integrations/oauth.py). Google’s [MCP client setup](https://developers.google.com/workspace/guides/configure-mcp-servers) documents the agent-enabled OAuth client.

## New Gmail writes in the replacement plan

Alphonse currently exposes Gmail reads only. Qualify proposed drafts with
`create_draft` under `gmail.drafts.create` and `get_draft` under
`gmail.drafts.readonly` or existing `gmail.readonly`, subject to actual consent
availability. Qualify sending with `send_message` under `gmail.send` on `/mcp/v1`.
Review the exact draft content before sending and use an independent authorized
read to verify provider evidence where supported. An ambiguous send needs
reconciliation before any retry. These capabilities require separate Alphonse
grants and approval, even if the OAuth token holds both scopes. [Create draft](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/create_draft) · [Read draft](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/get_draft) · [Send message](https://developers.google.com/workspace/gmail/api/reference/mcp/tools_list/send_message)

## Cutover decisions to close

1. **Qualify Gmail reads under real search constraints.** Prove safe label/name
   resolution and per-message evidence using mixed-label threads, multiple pages,
   sparse or filtered results and archived mail. Test both exact counts and missing
   count fields. Agree on visible coverage for Spam, filtered content, history and
   attachments that the server cannot supply.
2. **Prove Calendar effects and coverage.** Exercise a concurrent human edit and
   a create that committed before its response was lost. Verify exact busy-interval
   semantics or record the feature gap. Prove that bounded snapshots can invalidate
   stale evidence without creating false removals or losing recurring instances.
3. **Prove Classroom eligibility and provenance.** Test an authorized enrolled
   student, a non-student and an unavailable course with disposable records.
   Measure expansion costs and decide how to represent unavailable edit timestamps
   and exact due-time fields. Keep the preview capability unavailable if the
   school, account or evidence requirements cannot be met.
4. **Test the approval and scheduling path per operation.** Capture live tool
   schemas and granted scopes during qualification. Exercise denial, session expiry,
   token refresh and Google’s form approval on the primary endpoint. Scheduled
   checks that require interaction must surface that state within their existing
   window. Self mode can give reduced results. A filtered or blocked response
   must produce a coverage or authorization state before cutover. [Google approval behavior](https://developers.google.com/workspace/guides/mcp-elicitation)
