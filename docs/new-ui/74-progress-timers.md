# Progress, visual time and activity widgets — #74

Both blockers (#8 and #69) were verified closed before implementation. This adds
fixed native renderers to their existing authorized skill/context bridge and
At hand placement lifecycle. It reuses #73's contained, read-only snapshot reader.

## Skill resource contract

An installed skill can declare an activity binding in `widgets.json`:

```json
{
  "version": 1,
  "widgets": [{
    "key": "exercise-session",
    "title": "Exercise progress",
    "binding": {
      "type": "activity", "family": "tracker",
      "artifact_id": "artifact.example", "path": "activity.json"
    }
  }]
}
```

Families are `progress`, `tracker`, and `timer`. The existing snapshot envelope is
`{version: 1, updated_at, expires_at, items}`. Dates are explicit ISO timestamps
with timezones. Snapshot freshness is at most 24 hours; the owning skill refreshes
the envelope even when presenting older session records. Desktop displays dates
in the member's timezone, so yesterday's session stays an actual dated record,
not a moving relative-day label.

Every item requires a stable `id`, `title`, `scope`, `recorded_at`, and `status`.
Scopes are `session` (recorded session), `project` (project progress), `habit`
(habit/activity record) and `sample` (explicitly illustrative). No percentage is
invented from prose, project status or animation.

Progress/tracker items additionally require `value`, `target`, `unit`, and
`display` (`bar` or `dots`). Status is `active`, `completed`, or `cancelled`.
Quantities are finite, nonnegative numbers; targets are positive. Dots require
integer counts and targets of at most 100. Counts and units remain visible and
are exposed through accessible progressbar labels, values and value text. Values
above target retain their actual text while visual fill is capped. Reaching the
target never changes source status. A tracker can contain multiple dated activity
records. For example, a session can record 3 of 5 sets alongside a separate
project's 2 of 10 sessions and an explicitly labeled illustrative percentage.

Timer items instead require `elapsed_ms`, `duration_ms`, and `running_since`.
Status is `running`, `paused`, `completed`, or `cancelled`. Planned duration is
positive; elapsed time is nonnegative. Running timers require a timezone-qualified
`running_since` at or before the snapshot revision. Other states require null.

- Running elapsed = persisted elapsed + max(0, current time − running_since).
  Remaining time is clamped to zero. Every frame derives from timestamps;
  poll frequency, rerenders, reload and reconnect never add ticks or reset time.
- Pause: the owning workflow persists accumulated elapsed and clears
  `running_since`. Paused time remains fixed. Resume preserves that elapsed and
  stores the new running timestamp. These writes belong to the owning skill.
- Completion/cancellation: the source freezes elapsed and clears the timestamp.
  Only a completed source displays “Source reports completed.” Local expiry
  displays “Time elapsed · completion unconfirmed”; it performs no mutation,
  notification, tool call, task completion or verification bypass.
- Pause/resume buttons create conversation drafts using the existing #73 route.
  They do not send a message or optimistically modify state. The authorized
  owning workflow rereads its source, verifies effects and publishes a new
  snapshot. Other completion/recovery requests use ordinary conversation.

## Trust, privacy and lifecycle

The catalog must authorize the polling actor and match the artifact's owning
skill. Existing root containment, enabled checks, sharing and administrator
policy apply on every poll. Files remain in registered storage, with a 64 KB file
limit and at most 50 items. Unknown fields, malformed values, duplicate IDs,
future/naive timestamps, stale envelopes, traversal and symlink escape fail
closed. Opening a widget never runs its program or writes its data. Registration,
artifact tool exposure and Jev/System Two responsibilities are unchanged.

Existing context selection, stable member/skill/key identities, expiry, pins,
locks, dismissal, metadata-only layout persistence and out-of-order poll guards
remain authoritative. Disconnect hides payloads; reconnect rereads the source.
Revoked or expired sources show unavailable even on a locked placement. Source
payloads are never put into persistent drawer storage. There is no fill
transition or entry animation, so polling does not replay motion and reduced
motion requires no animated fallback. The visual time ring is decorative;
remaining/elapsed units and status are textual, with a non-live `timer` role to
avoid screen-reader announcements every second.

## Exercise handoff and limits

Issue #4 still owns the Personal Trainer capability, exercise prescriptions,
recording, verification and its skill-owned program/data. This change provides
its UI integration contract only. That skill can publish actual dated session
counts and interval timestamps through the binding above, selected through the
existing context bridge. No household skill, generated exercise implementation,
or real session data is copied into this repository. Missing trainer data is
unavailable; sample values are never substituted for it. All examples and tests
use disposable fixtures. Real household and packaged-native acceptance remains
in #78.

## Validation — October 29, 2026 (member timezone)

- 115 Desktop tests passed, including timestamp-derived frames, duplicate polls,
  serialized reload/reconnect, pause/resume/cancel/completion, zero-time
  non-completion, dated session/project/sample labels, quantities/units,
  malformed payload rejection, locks, expiry and offline/member data suppression.
- 45 targeted Python tests passed: activity, coordination, widget bridge and
  artifact regressions. Real IPC fixture polling checks source updates, retained
  identities, missing/disabled resources, actor denial, malformed clocks/counts,
  stale data and containment. Existing coordination behavior remains covered.
- TypeScript/Vite production build passed (existing bundle-size advisory).
- Playwright disposable component fixture passed tracker updates, offline hiding,
  reconnect deduplication, fixed paused time, request draft creation, lock and
  identity persistence across reload, reduced-motion computed styles and drawer
  containment at 760×600. Visually inspected recorded session dots and timer ring
  at 1120×760. Local evidence: `/private/tmp/issue74-widgets.png`,
  `/private/tmp/issue74-timer.png`, `/private/tmp/issue74-minimum.png`.
  This fixture renders the real drawer/components, without a household daemon;
  it does not claim a real trainer integration or packaged Tauri acceptance.

Integrated the newly merged care-widget PR #99 before delivery. Shared bridge
conflicts were resolved by retaining both native binding validators/renderers and
care access behavior. The combined result passes all 119 Desktop tests, 57
Python tests (including care regressions), and the TypeScript/Vite production
build. No care capability implementation is included in the #74 diff.
