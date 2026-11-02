# Family calendar package adoption and validation (#41)

`alphonse/agent_v2/builtin_skills/family-calendar/` is explicitly adopted as a shipped
reusable Agent Skills package. This is product-maintained service support, not a
copy of an individual household artifact. Its SKILL.md, ordinary Python script and
references use existing skill discovery/installation, account bindings and native
Bash. There are no calendar tools, script manifests or catalog entries. Package
exports contain no account state. See its references for connection, request and
authorization procedures. No automatic invitations are supported.

Project-scoped read context and incremental sync extend this package through
`integrations/google_calendar.py` and the existing `SourceAcquisition` service
(#119). They reuse account bindings, secure managed OAuth and the existing project
scheduler/evidence/finding stores. The interactive script retains its exact write
authorization, readback and retry guards. No new Calendar tool, registry, scheduler
or automatic invitation is introduced. Connecting an account starts no monitoring.
Legacy `google-calendar` script bindings are preserved; they are not silently
converted to managed `google` grants. An owner explicitly connects/selects the
managed account and calendar IDs, following [secure account setup](secure-credentials.md).

## Managed project read context and sync (#119)

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

Baseline lists use `singleEvents=true`, `showDeleted=true`, `timeMin` and
`timeMax`. Provider event IDs, ETags, iCalUID, recurringEventId, originalStartTime,
exception start/end, recurrence fields and status are preserved as inert JSON
text in normalized source evidence. Instances retain separate provider identities;
a shared series UID is not a unique occurrence ID. All-day dates remain local
DATEs with exclusive end dates. Calendar timezone and timed offsets/IANA zones
are retained, including daylight-saving transitions. Cancelled sparse instances
are source tombstones, never a completed assignment. New reads do not execute
HTML, follow links or fetch attachments.

Each selected calendar has its own opaque sync token and page continuation. The
acquisition transaction persists accepted pages/evidence before advancing tokens.
Incremental pages reuse the original sync token through the last page. Google's
API forbids `timeMin`, `timeMax` and `q` with `syncToken`; delta transport therefore
reads changes in the **selected calendars**, then filters event data against the
explicit window before retention. Consent to this selected-calendar change read
is part of enabling sync. Out-of-window changes retain only an ID/version
tombstone to redact stale cached evidence. Delta pages can be large for recurring
series: page/item/byte/runtime budgets apply and exhaustion is partial coverage,
never an unbounded scan or successful absence. Direct event expansion also checks
the authorized event window, using the actual calendar zone for all-day dates.

HTTP 410 triggers the shared service's single bounded baseline resync and records
a coverage gap; it never widens the event window or certifies old history. Query,
actor, project, exact account, grant revision, resource and adapter-version changes
invalidate cursor scope. HTTP 401/403/404 is blocked access, and redacted/busy-only
records, missing final sync tokens, throttling and budget exhaustion never mean
no-match. Every GET checks current authority before and after I/O. No retries,
redirects, credential forwarding, provider writes or live-account probes occur
at startup or in synthetic tests.

Calendar/mail reconciliation uses the existing `FindingService`: explicit
source-backed canonical person/commitment IDs can collect separate source
references on one finding. Similar titles or shared event UIDs alone cannot
establish assignment identity. Unknown identity remains unresolved. Calendar
presence is only `calendar_presence`, never proof of submission, payment or
completion. Revoked/absent access is a coverage/authority gap, never evidence of
assignment absence. Each source grant and recipient remains independently
required for finding reads and delivery; loss of Calendar access cannot be filled
by mail access.

Provider contracts: [event list and sync constraints](https://developers.google.com/workspace/calendar/api/v3/reference/events/list),
[incremental sync](https://developers.google.com/workspace/calendar/api/guides/sync),
and [calendar discovery](https://developers.google.com/workspace/calendar/api/v3/reference/calendarList/list).
This is synthetic qualification of the shared service extension; real-account
and provider behavior still require separately authorized opt-in validation.

## Synthetic validation

Run `python3 -m pytest -q tests/test_calendar_acquisition.py tests/test_source_acquisition.py
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

1. Confirm Python/timezone readiness. List a finite date window and compare to the
   Google UI. Query free/busy and verify busy intervals; a provider error is unknown.
2. Confirm an unselected actor/calendar is rejected before any network call. Disable
   the binding and confirm managed denial; re-enable only with explicit owner action.
3. Only after separate approval for a real write, provision a short-lived exact hash
   grant. Create a uniquely identified test event with no attendees and
   `send_updates=none`. Read back its ID, title, offset, timezone and dates. Repeat the
   exact request with the same operation ID and confirm one event, including after
   simulating interrupted readback. This test must not send invitations.
4. Read the ETag, approve the precise update and grant its hash. Change the title,
   read back, and repeat to verify no second PATCH. Make a manual conflicting edit
   and confirm the old ETag fails without overwriting it.
5. With explicit series approval, test a COUNT=2 weekly series across a daylight
   saving transition in a relevant IANA timezone. Confirm local wall time in Google,
   list expanded instances, then authorize only one instance update and verify that
   the other occurrence is unchanged. Confirm ambiguous dates/nonexistent local
   times require clarification or fail before mutation.
6. Revoke OAuth consent and confirm reconnect guidance without token/error-body
   disclosure; test missing scopes using a separate read-only consent. Remove test
   events manually in Google (deletion is outside this package), disable grants and
   test bindings, and retain only a redacted pass/fail report outside the package.

Real-account validation is intentionally not claimed by the synthetic suite. Google
provider delivery/concurrency behavior remains an opt-in integration check. Requests
with attendees, invitation delivery, conferencing, deletion and complex recurrence
are outside the initial implementation and must never be approximated silently.
