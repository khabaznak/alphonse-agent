# Agenda (#71)

The Agenda workspace and At hand Agenda cards provide Month, Week and Today views,
previous/next navigation, a civil-date picker and a return-to-today control. Weeks
start on Monday; Month loads six weeks. The configured member timezone determines
today, day boundaries, grouping and displayed times. Schedule timezone remains
visible when it differs. At small desktop widths, days become a readable two-column
list; compact drawer cards use one column.

## Sources and semantics

The owner explicitly confirmed that #71 should finish with existing sources and
an explicit missing-calendar state. The repository has no structured appointment
or external household-calendar feed. The UI says those feeds are not connected;
no prose memory, task creation date or mock appointment becomes a calendar event.
A future calendar source must apply its own authorization before projection.

`agenda` IPC is a read-only projection of authorized schedule definitions and their
recorded executions. It requires an active, explicit actor, selects only that
member's schedules and filters associated projects through current non-admin
visibility. Being an administrator does not broaden the Agenda audience. It reads
no prompts, private reasoning or execution error bodies. Existing management
endpoints retain their owner/admin authorization and mutations.

A projected occurrence is an expected time under the saved recurrence, not proof
that a task ran. Schedule status (including paused/cancelled) is independent of
execution status. Recorded occurrences retain their scheduled time even when the
worker started later. Legacy records without a usable occurrence timestamp are
labelled **Actual execution** at their recorded start time. Past recurrence slots
without a record say **No execution recorded**. Recurrences lacking explicit DTSTART
use the definition's creation time as a stable projection anchor; this cannot
recover an absent historical recurrence anchor or prove past execution.

## Bounds and retained workflows

Requests must cover 1–42 civil days, using exclusive end dates converted to UTC
at each local midnight (including 23/25-hour DST days). The definition scan is
capped at 1,000; range-filtered execution reads at 501 per schedule; recurrence
expansion at 500 per definition; and the result at 500 items. Expensive old
secondly/minutely/hourly definitions that would require more than 100,000 seek
steps show a limit notice and their next occurrence/recorded history where available.
All limits are reported rather than implying completeness. This adds a bounded
projection, not a new scheduling engine or mutation of saved recurrence data.

Agenda refreshes every 30 seconds while active; closed drawers and inactive
workspaces stop their refresh timers. Refreshes clear old data, show loading or
errors, and ignore superseded/unmounted responses. Schedule mutations refresh the
detailed Agenda immediately. A source revocation is reflected on the next read;
this is not a push revocation protocol.

Selecting an occurrence opens the existing schedule editor and recent execution
history (eight records), using stable schedule IDs. Schedule management is loaded
only when explicitly opened, includes status/authorized owner filters and reports
its 1,000-definition cap. Name/prompt editing, suspend, resume, cancel and permanent
delete remain available. Creation remains in Conversation. The existing API has
no direct time/recurrence editor: cancel and request a replacement in Conversation.
The Agenda and widget both link to Projects & Tasks; widgets link to the detailed
Agenda. Removing or hiding a widget does not change schedules or execution records.

## Validation

- Blockers #8, #69 and #70 were confirmed closed and present in the integration base.
- 82 Python tests pass across Agenda, IPC, Household overview and native scheduling.
  Agenda fixtures cover spring/fall DST midnights, recurrence across DST, overlaps,
  past slots, delayed execution, legacy execution timestamps, cancelled state,
  private/revoked project access, invalid actor/ranges, immutability and density limits.
- All 101 Desktop tests and the TypeScript/Vite production build pass. Date fixtures
  cover Monday/year boundaries, leap days, short months and timezone-local today.
  The existing Vite bundle-size advisory remains non-fatal.
- Disposable Playwright fixtures exercised real Agenda components for Month/Week/Today,
  year navigation, stable schedule selection, Projects & Tasks routing, loading,
  empty/error/retry, stale-data removal and bounded requests. A real App fixture
  verified occurrence-to-editor/history navigation, name editing, pause/resume/cancel
  and the Conversation creation route. Light/dark and 760 × 600/1440 × 1000 layouts
  were inspected. Browser data and scripts are disposable and are not product sources.
- Real household use and daily-use acceptance remain #78; no live schedule or
  household data was created, edited or deleted during these checks.
