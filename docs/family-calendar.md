# Family calendar package adoption and validation (#41)

The shipped `family-calendar` Agent Skills package directs Calendar work through
Alphonse’s managed Google MCP integration. Owner Agenda and project checks read
authorized calendar windows. Desktop Conversation can prepare an edit for explicit
owner approval. Connecting an account starts no monitoring.

Stored legacy script bindings and household files are preserved. The managed
runtime no longer gives Google Calendar credentials to those scripts. Use the
managed account and exact calendar grants for future requests.

## Owner Desktop agenda

The owner can ask for event titles and times in a date range of up to 32 days in
Desktop Conversation. Enable the [Google Calendar MCP](https://developers.google.com/workspace/guides/configure-mcp-servers)
in the Google Cloud project that owns Alphonse’s OAuth client. In **Settings → Integrations → Google Calendar**, connect
the Google account, then choose **Manage access**. Copy the exact calendar ID from
that calendar’s **Integrate calendar** settings in Google Calendar and choose an
expiry for the owner-only grant. Google’s `primary` alias is unavailable here. If
more than one calendar is authorized, choose which one to keep enabled in access
settings. An expired access token requires reconnecting the Calendar account.

Alphonse reads up to 40 event occurrences from one exact calendar and reports
when there are more results. Ask a separate Calendar question for a shorter
range when needed. The Desktop Agenda handles reads.

Event titles and times appear in a day-grouped card in the live Desktop
conversation. Timed events use the owner's local timezone. Multi-day all-day
events include their last day, and an event with only a start time says its end
is unspecified where Google supplies that field. The conversation
history records that the result was displayed and the model receives only the
read status. The existing local task checkpoint holds the bounded observation
to support verified pickup without another provider read. That database remains
within the same local OS principal’s trust boundary. A grant change, lost claim,
or expired observation invalidates pickup. The live event details are cleared
when the conversation view reloads. Telegram and scheduled requests cannot use
this agenda path.

## Live events in the Agenda widget

The owner’s Agenda widget and Detailed Agenda show event titles and times from
the same explicitly granted calendar alongside scheduled work. Today, Week,
and Month read the displayed range. All-day dates follow the calendar’s dates,
and timed events use the owner’s timezone. If a response exceeds the one-page
limit of 200 events, Agenda marks coverage as limited. An unavailable read
leaves scheduled work visible.

The open view refreshes on navigation, Calendar configuration changes, window
focus and a 30-second poll. Only verified local owner requests with the exact
active grant can receive events. The direct Agenda request uses managed OAuth
renewal if its token expires. A revoked authorization requires reconnecting.
Agenda sends event data to the active Desktop view and does not send it to the
model or store it in conversation history.

## Owner-approved edits

In **Settings → Integrations → Google Calendar editing**, connect the same Google
account and grant editing to the exact calendar with an expiry. Google’s consent
permits event editing across calendars. Alphonse applies its own exact calendar
grant for each proposal and checks the grant again when the owner approves.

Ask in private Desktop Conversation to change an exact event. Review the event,
occurrence or series, and **Current** and **Proposed** values in the separate
review card above the composer. Read long descriptions at full width in the
review. Choose **Approve and update** for an ordinary edit.

When the request adds guests, provide their exact email addresses. The card shows
existing and new guests and who Google will notify. **Approve and notify** applies
the reviewed changes and requests email notifications for all guests, including
existing guests. Alphonse checks the saved guest list by reading the event back.
That read verifies guest membership. Email delivery requires separate evidence.

Wait until Alphonse reports the result before editing the same event in Google
Calendar. If the provider snapshot changed before dispatch, request a new proposal.
A simultaneous edit can also change which guests Google notifies.
If the outcome is unknown, inspect the event in Google Calendar before retrying.
[Calendar update protection](google-mcp-replacement.md#calendar-update-protection)

## Managed project read context (#119)

Use trusted `OAuthService` setup with capabilities chosen independently:
`calendar_discover` requests only `calendar.calendarlist.readonly` (plus identity),
while `calendar_read` requests only `calendar.events.readonly` (plus identity).
Neither implies mail access or write consent. Discovery uses an explicit
`calendar_discover` grant with resource `calendar:list`. Its bounded, paginated
`CalendarAdapter.discover(ReadContext, cursor=...)` returns candidate metadata;
it never selects a binding or authorizes event reads. It is a trusted setup API,
not a proactive policy operation. Discovery pages obey the caller's deadline,
guard and item/byte allowance; setup callers must charge pages across continuations.

For project reads, explicitly bind the installed `skill:family-calendar` calendar
role through `SkillBindings` to the exact managed Google connection and project
grant. The operation is `calendar_read`, resources are exact calendar IDs, and
the audience must fit both grant and project policy. Do not use `primary`, guess
an email/calendar ID, substitute accounts or broaden a denied calendar selection.
The optional policy query field `calendar_window` is mandatory for Calendar
sources, with explicit RFC3339 offsets and a positive duration of at most 366 days:

```json
{"text": "school assignments", "calendar_window": {
  "time_min": "2026-11-01T00:00:00-06:00",
  "time_max": "2026-12-01T00:00:00-06:00"
}}
```

This is the **event-time** window, distinct from the policy's allowed retrieval/
analysis hours. Editing it requires normal policy review/authorization and a new
cursor scope. An old recurring series may have an occurrence inside this window;
its creation/update date is not a lower bound. Mixed Calendar/mail policies keep
mail's original bounded text query. Calendar reads collect the authorized event
window without provider text filtering so renamed events cannot leave stale
matches. Jev performs bounded relevance judgments downstream; System Two composes
interpretations and canonical identities.

Calendar MCP lists bounded windows with `list_events`. Each selected calendar has
its own page continuation. The acquisition transaction persists accepted evidence
before advancing a page. Google’s MCP schema has no incremental sync token, ETag,
or recurrence-expansion switch. Alphonse uses fresh snapshots, retains provider
occurrence identity when returned, and treats explicit cancellations as tombstones.
An event missing from a snapshot cannot prove deletion or movement. All-day dates
stay local with exclusive ends. Direct event reads verify the authorized window.

Query, actor, project, exact account, grant revision, resource and adapter version
determine cursor scope. Provider failures, redaction and exhausted budgets surface
as blocked or partial coverage. Each provider call checks authority before and
after I/O. Read results do not execute HTML or fetch links and attachments.

Calendar/mail reconciliation uses the existing `FindingService`: explicit
source-backed canonical person/commitment IDs can collect separate source
references on one finding. Similar titles or shared event UIDs alone cannot
establish assignment identity. Unknown identity remains unresolved. Calendar
presence is only `calendar_presence`, never proof of submission, payment or
completion. Revoked/absent access is a coverage/authority gap, never evidence of
assignment absence. Each source grant and recipient remains independently
required for finding reads and delivery; loss of Calendar access cannot be filled
by mail access.

See the provider’s [event list](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/list_events)
and [calendar discovery](https://developers.google.com/workspace/calendar/api/v3/reference/mcp/tools_list/list_calendars)
schemas.

## Synthetic validation

Run `python3 -m pytest -q tests/test_calendar_acquisition.py tests/test_calendar_mcp_edits.py tests/test_source_acquisition.py
 tests/test_oauth_lifecycle.py tests/test_project_checks.py tests/test_findings.py tests/test_family_calendar.py tests/test_agent_v2_skill_bindings.py
 tests/test_skill_script_execution.py tests/test_agent_v2_skills.py` (on one line).
The calendar tests inject synthetic OAuth and HTTP responses; no real network or
calendar records are used. Binding and native execution tests cover authenticated
actors, sharing/removal, revoked accounts, redaction, cancellation and checkpoints.

## Opt-in real-account validation

Do not run during tests, installation or daemon startup. A human must explicitly
opt in, connect their chosen test Google account through trusted setup, select an
empty dedicated test calendar and review its ACLs. Never use a production family
calendar for smoke validation. Keep all resulting records and tokens outside Git.

1. Compare a finite Agenda window against Google Calendar and exercise one exact
   scheduled read within its retrieval window.
2. Seed a test event in Google Calendar with no attendees. Propose a change in
   Desktop. Verify the exact review and confirm dismissal leaves the event intact.
3. After explicit approval for the live write, approve a fresh review and verify
   the reviewed fields in Google Calendar. Check that the same review cannot run
   again. Prepare another review, manually change the event before approving, and
   verify that Alphonse stops when its preflight sees the newer change.
4. Run permission loss, ambiguous responses and conflicting requests with
   disposable provider fixtures. Revoke test grants and remove test events
   manually in Google Calendar after the authorized live run.

The test plan covers fixture-backed behavior and opt-in live checks separately.
[Google MCP qualification evidence](agent-loop/evidence/google-mcp-discovery-2026-11-27.md)
