# Project configuration

Each registered project folder has a machine-readable `project_config.json`.
It records project-specific context and is the only project context file loaded
for V3 curation. Existing `project_context.md` files are preserved as-is for
their owner's manual review and are never used as prompt context.

The initial schema is:

```json
{
  "schema_version": 1,
  "scope": "",
  "goals": [],
  "owner_user_id": "user-id",
  "duration": {
    "ongoing": true,
    "start_date": null,
    "end_date": null
  },
  "directory_structure": []
}
```

`goals` is a list of project-specific desired outcomes. `duration` may describe
an ongoing effort or use ISO 8601 dates. `directory_structure` is a declared
layout; it does not authorize filesystem changes by itself. Project files remain
inside the registered project folder.

Project lifecycle states are `active`, `paused`, `completed`, and `archived`.
Sharing is private or shared with explicitly selected family-member user IDs.
The owner is always retained in the project registry and cannot be changed by
editing this config.

The first daemon start after this change converts legacy family-wide shared
projects into grants for all currently active family members. This preserves
their existing access while recording it explicitly; owners can then narrow
the member list in project management.
New shared projects do not receive automatic member grants.

Project management lists up to 1,000 visible non-hidden files below the project
folder. File additions copy user-selected files into the project root. Removal
is limited to regular files inside the project folder; config and legacy
context files are protected. The declared directory structure does not itself
create or delete folders.

## Existing project migration

When the project store initializes, it writes a default `project_config.json`
only where one does not already exist. It leaves project folders and existing
`project_context.md` files untouched. Owners can review and extract useful
material from legacy files before deleting them themselves.

Jev's System One context curation receives only authorized project-config
candidates. It can select multiple projects; every candidate whose relevance
probability meets the configured threshold is retained, including equal-scoring
candidates. Project files and config are data, not instructions or authorization.

## Optional account checks

The [optional-account implementation contract](optional-account-integrations.md)
defines separate persisted, revisioned policies for project-directed proactive
checks. This is future work, not an extension already supported by this config
schema. Editing goals may invalidate a policy for review; editing this file or
sharing a project can never activate monitoring, select an account, expand an
account grant or add notification recipients. Ordinary authorized one-off account
queries remain project-optional. Connecting an account starts no proactive work.

## Authorized proactive checks (#112)

`ProjectCheckService`, available as `runtime.project_checks`, stores version-1
policies in `v2_project_check_policies` in the shared project database. Creating a
connection, sharing a project or adding policy-looking keys to this config starts
no monitoring. Ordinary authorized one-off account queries still need no project.

Trusted conversation/management code calls `save(AccountSession(...),
definition=..., activate=True, authorization_reference=...)` only after the user's
instruction authorizes the concrete summary. The session comes from authenticated
host context, never policy/model arguments. An explicit conversation instruction
can authorize the recurring policy; routine runs need no repeated approval.
`activate` defaults false. Drafts are disabled and cannot be resumed into authority.
The policy owner is its executing principal, who must have access to an active
project; this does not grant access to any account.

A definition supplies:

- `project_id`, `source_bindings`, and `query: {"text": "bounded provider query"}`.
  Each source specifies `skill_id`, `role`, exact `binding_id`, `connection_id`,
  `grant_id`, read `operation` and nonempty explicit `resources`. Existing binding
  choice, ACL, installed skill, connection, provider scopes and grant checks all
  apply. The service pins grant, connection, binding and role-choice revisions.
- `trigger: {"kind": "schedule", "weekdays": [0], "local_time": "08:00"}`,
  `timezone: "America/Mexico_City"`, and
  `retrieval_window: {"weekdays": [0], "start": "08:00", "end": "09:00"}`.
  Weekdays use Monday=0 through Sunday=6. Version 1 accepts same-day windows;
  the trigger must fit the window. Trigger fields cannot contain scripts/prompts.
- `duration: {"ongoing": false, "starts_at": "<UTC timestamp>",
  "ends_at": "<later UTC timestamp>"}` or an explicit `ongoing: true` with
  `ends_at: null`. Project-config date bounds independently restrict execution.
  Renewing duration uses a newly authorized policy revision.
- Positive bounded `budgets`: `max_pages`, `max_items`, `max_bytes`,
  `max_runtime_seconds`, plus nonnegative `max_model_calls`.
- Optional `initial_lookback_days` (default 30, maximum 365), `skill_hints`,
  `recipient_user_ids` (default empty), `notification_mode` (`material_changes`
  by default, or `requested_digest`), and `missed_run`.
  Missed-run default is `{"kind": "next_allowed_window"}`; explicit bounded late
  catch-up uses `{"kind": "bounded_late", "max_lateness_seconds": 3600}`.
  Neither mode authorizes replay of every missed occurrence.

The service derives the validated goal snapshot/digest from project scope, goals,
owner, description and duration. It records query digest, authorizing principal,
instruction reference, authorization time, stable policy ID and monotonic revision.
These server-owned fields cannot be supplied in a definition. Source account grants
independently bound each recipient; membership never expands the audience or gives
raw mailbox access. An empty audience permits no delivery.

`save` and `set_status` use expected-revision compare-and-swap. Goal/scope/query,
source, recipient and duration revisions need fresh concrete authorization.
`set_status` supports pause, disable, end and resume of an authorized paused policy.
Disabled, ended and `needs_review` records need `save` with fresh authorization.
Resume produces a new revision and cannot make an old queued occurrence current.

Consumers call `guard(policy_id, expected_revision=..., cancelled=...)` at dispatch,
each provider/page/attachment call, CAPD entry and delivery, including after an
in-flight read. The guard revalidates project access/lifecycle, goal digest, duration,
skills, source revisions and grant/audience limits. Project/member changes carry
persistent revision fences, including pause/resume cycles between checks. Direct
file edits are checked by digest before use. A dependency mismatch durably suspends
an active policy as `needs_review`; restoring the file, member, or connection cannot
revive the old revision. A cancelled run stops at the guard and must not deliver;
it does not revoke the policy for future authorized runs. Already returned bytes
cannot be recalled.

This is the policy authority layer. It creates no schedule, queue job, provider read
or notification. #113 installs typed jobs in the existing scheduler and enforces
occurrence identity, windows/deadlines and missed-run behavior in addition to this
fence. #114 bounds acquisition and checkpoints; #121 supplies management UI/IPC.
No parallel scheduler, event store or delivery channel is introduced. Existing
sensor automations and reminders retain their behavior.
