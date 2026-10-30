# Projects & Tasks — #70

## API gap and scope

Before this change Desktop exposes manageable projects, schedules and schedule
execution history, plus day-scoped conversation history. It has no listing API
for ordinary delegated tasks. Persisted `v2_task_checkpoints` already hold their
stable IDs, owners, project IDs, status and acceptance contracts. A narrow
read-only `work_items` IPC projection will expose these records; it will not
create a new task domain or change task execution, delegation or authorization.
The database status column is authoritative even when serialized state is stale.
Only explicit public fields are returned; plans, reasoning, raw metadata and
checkpoint diagnostics are excluded.

Task visibility is owner-scoped. Explicit valid actors may read their own work;
only administrators may select another owner. Project membership does not grant
access to another person's task checkpoint. Existing project and schedule
management APIs remain authoritative. Their legacy Desktop administrator
normalization is outside this issue; #51 owns project membership/error-contract
regressions. No broader access is inferred from a `shared` label.

Active work is the default. History is an explicit, bounded paginated view.
System home projects are presented as everyday work, preserving their real IDs.

## Delivered behavior

The work overview follows the accepted project-summary / nested-task / everyday
hierarchy without duplicating the shell's navigation. Project details still
contain creation/import, editing, sharing, lifecycle/archive, files and removal.
Active projects and work are the default. Needs input / blocked includes waiting
tasks, failed tasks and required blocked criteria; completed/history includes
persisted done, failed and cancelled tasks and completed/archived projects.
Tasks are paginated (50 per page, server cap 99). Project completion counts
cover all persisted task statuses for the selected owner and project, independent
of pagination. Failed/cancelled work is counted separately and never as completed.
No progress is invented when acceptance criteria are missing.

Schedule rows stay attached to their project or everyday bucket, label schedule
state separately from the latest execution state, and open the existing Agenda
item with its execution history and mutation controls. The current schedule API
returns a bounded 100-record window; the overview labels the cap when reached.
Agenda calendar views and expanded retrieval remain #71.

Task details show the stable ID, actual persisted state, acceptance progress and
creation/update timestamps. These timestamps are not a fabricated event log.
The checkpoint store retains snapshots, not a full revision history. A related
conversation link opens the task's creation day. Other owners' conversations
remain in their own channels; delegated questions display the actual pending
respondent and only offer the current respondent's answer route. No per-task
completion checkbox, retry executor or cancellation API is invented. Ordinary
work is delegated through conversation. The existing shell emergency stop and
schedule controls remain available.

Work refreshes on workspace return and explicit Refresh; no extra poll loop or
agent session is introduced. Responses from stale requests are discarded.

## Validation — October 29, 2026

- Seven disposable daemon/SQLite tests cover stable IDs, authoritative terminal
  status despite stale serialized state, allowlisted output without private
  plans/metadata, owner/member/unrelated/admin reads, private and explicitly
  shared projects, cross-owner denial, pagination and limits, home/missing
  project identity, optional/superseded blockers, delegated answer authority and
  project totals across pages.
- The containing IPC/project regressions pass, including existing project and
  schedule lifecycle/ownership controls. The #51 store regression cases pass
  on this revision; #51 remains the owner of its broader contract reconciliation.
- Disposable browser fixture checks cover nested/standalone hierarchy, active
  default, attention/history filters, task focus, conversation day and question
  routes, selected schedule and failed execution history, retained project
  editor/sharing/archive controls, pagination, owner switches, loading, retry,
  partial schedule errors and useful empty states. At 760 × 600 there is no
  document horizontal overflow; light and dark appearances are visually checked.
- Fixture checks do not contact the household daemon or mutate household data.
  Real household end-to-end and daily-use acceptance remain #78.

Final validation: 66 Desktop tests and the TypeScript/Vite production build pass;
91 targeted Python tests pass, including the newly merged Settings regressions. Vite retains the existing bundle-size advisory.
