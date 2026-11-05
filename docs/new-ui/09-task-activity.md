# Task activity and work logs — #9 / #54

The current V3 integration baseline already had human-readable tool labels,
credential redaction, delete-surface archiving, and failure cleanup. It still
retired every other task when a task emitted progress, exposed legacy
`internal_state` as intention, rendered raw checkpoint diagnostics, and emitted
pending closure before the last progress update. A final update in that same
poll could recreate the just-closed card. #67/PR #87 supplied the flowing
conversation layout; it did not complete this lifecycle work.

## Presentation and contract

Each `task-progress:<task_id>` owns its activity independently. Activity shows
its request, bounded subgoal objective (or the legacy user request), current tool
label and actual tool status. Past actions are collapsed; terminal work becomes
a collapsed work log with the task outcome, timestamp, inspectable failed action
and guidance for retrying. Responses enter the ordinary conversation immediately
and do not replace the log. The old minimum-duration/morph timers are removed.

The progress snapshot and A2UI root data model carry separate `tool_id`,
`tool_name` and `tool_status` fields. Each step has its own `tool_id`, display
label, status, and sanitized input/result. Display labels are not machine IDs.
Selecting a new legacy call cannot borrow a previous call's successful result.
Failed effects remain failed even if later task-level work completes. Execution,
authorization, ownership, acceptance verification and Jev decisions are unchanged.
This change owns the overlapping projection regressions tracked by #54.

Legacy `internal_state`, reasoning-keyed values, and raw checkpoint diagnostics
are excluded from the presentation. Existing call-time context manifests and
owner-authorized source expansion remain available. This is a presentation
change, not deletion of task history or memory.

The daemon publishes lifecycle snapshots for completed/failed/cancelled work and
retry waits, including when no reply exists. Desktop-origin checks remain in
place. Terminal snapshots are journaled, so a reconnect/replay receives their
correct status. Delete envelopes follow all final updates in a poll. A later
explicit running/retry update can resume the same task identity. Delayed durable
queue retries are counted separately from ready work, preventing a temporary
idle queue from retiring their activity.

Disconnect presents interrupted activity without asserting completion; a fresh
poll restores authoritative state. A daemon identity change clears obsolete
activity before accepting the new daemon's events. Legacy idle cleanup labels
unconfirmed outcomes as interrupted, never successful. Work logs remain an
in-memory Desktop projection, not a new persistent audit store; they can be
reconstructed only within the existing bounded activity journal. Existing
conversation history remains persistent.

## Validation

Disposable Python/SQLite fixtures cover the original terminal failure without a
reply, final-update/closure ordering, journal replay, independent task activity,
success/failure/cancellation/abandonment, delayed durable retry and channel
exclusion. The two #54 tests assert display labels and separate machine
identity/status. Privacy and secret-redaction assertions remain active.

Desktop regression tests exercise `applyA2uiEvent` → `projectTaskProgress` →
`TaskProgressBubble`, plus timeline placement beside the correct request and
reply, concurrent work, interrupted state, and retry/resumption. Production build
and the existing Desktop suite are included in final validation.

A disposable full-shell Playwright fixture consumes real `poll_desktop` A2UI
envelopes from an in-memory daemon, with synthetic requests and tool failures.
It checks two active tasks, failure beside independent ongoing work, expanding
failed action details, timestamps, stable retry identity, terminal cleanup,
repeated polling, disconnect/reconnect and daemon restart. Light and dark skins
were inspected at desktop and 760 × 650 dimensions, with no workspace horizontal
overflow. No fixture, household data or generated artifact program is shipped.
The browser run uses a mocked native transport; it is not a live provider or
native WebKit endurance test. Daily-use acceptance remains #78.

Final checks on October 29, 2026: 151 targeted Python tests, all 95 Desktop tests,
and the TypeScript/Vite production build pass. Fifteen full-shell browser
assertions pass. Vite retains the existing bundle-size advisory.

Integration includes PR #92 (skill appearances) and PR #93 (At hand). The sole
App import conflict retains both At hand and the extracted activity renderer.
The complete checks and full-shell browser transitions pass on this combined tree.
