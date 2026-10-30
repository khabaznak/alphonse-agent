# Daily coordination widgets — #73

Both execution blockers, #8 and #69, were closed and merged into
`codex/v3-hierarchical-capd` before implementation. This extends their At hand
catalog, authorization bridge, member layout, contextual resurfacing and locks.

## Data and actions

- **To-dos** shows the polling member's latest eight active delegated tasks via
  `work_items`. These are execution records, so checking a box must not turn
  unfinished agent work into success. Open a task in Projects & Tasks to see its
  criteria/history, or draft a conversation request about its exact task ID.
  Editable personal to-do lists use the owning-skill snapshot below.
- **Reminders** shows up to 100 stored schedules filtered to active/paused direct
  delivery. Other scheduled agent work is not mislabeled as a reminder. Pause,
  resume and explicit cancel use the existing authorized IPC operations. Controls
  stay disabled while a submission is pending. The returned schedule ID, owner
  and expected state must match before showing confirmation; errors and
  unconfirmed outcomes stay visible. Details/edit opens the exact schedule in
  Agenda, preserving the existing schedule editor and execution history.
- **Shopping** and **Appointments**, plus editable to-dos and skill reminders,
  render a bounded current snapshot from a registered, authorized skill resource.
  There is no built-in shopping or calendar provider invented by this issue.
  Without an installed owning skill and a current source, these show unavailable.
  An empty valid snapshot instead shows an empty list. Appointments carry an
  offset-aware time rendered in the member's configured timezone.
- Skill item completion, list edits, and appointment review/rescheduling open an
  ordinary **conversation draft** containing the source, stable item ID, and
  requested change. The member reviews and sends it using the existing composer.
  Opening a draft does not enqueue work, save a list or claim completion. The
  owning skill must re-read, authorize, execute and verify using its existing
  instructions/native capabilities. The regular task activity/question surfaces
  report workflow failures. Items change only after the skill publishes its
  confirmed source state. No generic artifact executor or new runtime tool exists.

All four families are available in Add widget. Selected skills can also surface
specific lists through #69's existing Jev-selected skill context. Catalog views
reuse the same accepted poll snapshot as contextual cards and item details.
Unselected sources are not read just because a catalog family is visible: ask
Alphonse to use the owning skill first, or retain/pin its surfaced card. Native
work/reminder reads run only while a relevant drawer is open and connected, and
refresh every five seconds. Member changes invalidate old requests. Poll
responses retain the existing generation guard; reads started before a reminder
mutation cannot overwrite its result. Repeated rendering performs no mutations.

A skill item opens as a selected item within Agenda or Projects & Tasks using
artifact ID and item ID. These IDs are transient lookup hints. Details re-resolve
against the current authorized poll snapshot rather than holding old payloads;
revocation, missing skills, disconnection and expiry hide the content. Native task
links select the task in the current member's bounded active-work page. A task
that has left that page must be located through the workspace's history and paging
controls; it is never manufactured from an old widget record.

## Skill prerequisite and snapshot contract

The installed skill declares a fixed renderer in its existing `widgets.json`:

```json
{
  "version": 1,
  "widgets": [{
    "key": "groceries",
    "title": "Shopping list",
    "binding": {
      "type": "coordination",
      "family": "shopping",
      "artifact_id": "artifact.example-list",
      "path": "widget-list.json"
    }
  }]
}
```

`family` is exactly `todos`, `shopping`, `reminders` or `appointments`. The artifact
must already be enabled, registered to this installed skill and accessible under
its catalog ownership, explicit sharing or administrator rules. Selection never
grants access. Source/owner labels come from catalog metadata, never from claims
inside the snapshot. The relative path must resolve inside its registered
storage root, including symlink/traversal checks. Shared, skill-package and legacy
project roots retain their existing resolution. Registration and files are not
changed by this reader. No program is executed on polling, pinning or navigation.

The owning skill/program publishes this JSON **in its registered storage**, using
an atomic file replacement after a verified read or change:

```json
{
  "version": 1,
  "updated_at": "2026-11-02T08:00:00-06:00",
  "expires_at": "2026-11-02T09:00:00-06:00",
  "items": [{"id": "item-123", "title": "Milk", "status": "pending"}]
}
```

This is a schema example, not an installed program or household record. Provider
credentials, account selection, delivery destinations and write verification
remain prerequisites owned by the skill. Snapshots report source state; their
existence does not independently prove an external provider write. The skill
must describe its update/verification workflow in SKILL.md and keep snapshot
identities stable across edits. Publish failures leave the last valid state only
until its expiry; the conversation must report the failure, never claim a save.

The reader rejects unknown snapshot/item fields, duplicate IDs, malformed JSON,
invalid timestamps, naive times, future updates, expired data, validity windows
longer than 24 hours, files over 64 KB, more than 50 items, IDs over 128 characters,
titles over 300 characters, and statuses outside `pending`, `completed`,
`cancelled`. Optional `when` must be offset-aware. Missing, disabled, inaccessible,
wrong-skill or unsafe-path resources produce unavailable without old payloads.
Snapshot lifetime is independent of a placement's contextual expiry: a lock
preserves position, never stale data or access. Text remains escaped React text.

No generated household programs, provider configuration, snapshot files or user
data are product source. Disposable fixtures cover only this framework contract.
Layout storage continues to contain IDs, modes and ordering only.

## Validation

Validation counts and browser evidence are recorded on the implementation PR.
Python fixtures exercise actual Desktop polling with synthetic skill resources
for all four families, source replacement, stable IDs, no writes on opening,
expiry, missing sources, catalog sharing/revocation, mismatched owning skills,
disabled resources, schema rejection and traversal/symlink containment. Existing
IPC schedule lifecycle and owner-denial checks are retained.

Desktop tests cover fixed renderers, escaped text, provenance, timezone display,
draft semantics, empty/loading/unavailable/save-failure states, replay/locks,
member scope, disconnection and malformed snapshots. The disposable browser
fixture mocks Tauri/IPC rather than accessing household records. It checks
pending mutation suppression, verified native effects, failures, source updates
across repeated views, selected-item routing, revocation, locks, metadata-only
persistence and 1120×760 / 760×600 layouts. This does not qualify real providers,
a packaged native release, or household daily use; that acceptance remains #78.

Initial validation before integration: 102 Desktop tests, 65 targeted Python tests
and the production build passed. The complete isolated browser fixture passed;
its initially incomplete synthetic Tauri metadata and schedule fields were fixed
and the full run repeated. Local screenshots: `output/playwright/issue-73-widgets.png`
and `output/playwright/issue-73-minimum.png`.
