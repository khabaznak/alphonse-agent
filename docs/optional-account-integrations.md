# Optional accounts and project-directed checks

Status: implementation contract for [#105](https://github.com/khabaznak/alphonse-agent/issues/105), 2026-10-31. Reconciles the October 30 design snapshot in [delivery index #104](https://github.com/khabaznak/alphonse-agent/issues/104).
Baseline: `e224e368b7bb80f556b51b7b4411bb60e4c4a4a4` (current main inspected for this contract).
This document is the delivery contract. Requirements below are future work unless
identified as shipped. #109 secure credential storage and #110 member-owned account
grants/binding resolution, #111 managed Google/Microsoft OAuth lifecycle APIs, and
#112 validated project-policy persistence/lifecycle fences now ship; see
[OAuth implementation](oauth-accounts.md), [project policies](project-config.md#authorized-proactive-checks-112)
and [binding implementation](skill-bindings.md#member-owned-connections-and-bounded-grants-110).
The per-issue dependencies in #104 govern delivery.

## Outcome and non-negotiable boundaries

A member can connect an optional account without starting any proactive work.
Ordinary authorized one-off reads remain project-optional. A project owns proactive
goals, duration, schedules, retrieval windows and recipients; account access is
independently authorized. Conversation can create a project and configure its checks,
without making people manually select a project for everyday questions.

The first acceptance slice is a finite school-trimester project checking a parent's
chosen Gmail account on Monday mornings for consequential commitments. Household
administration can later review overdue notices, which are evidence of a sender's
claim, never proof that a bill remains unpaid. Outlook/Hotmail, Calendar and eligible
Classroom extend the same contracts. No real-account connection, live retrieval or
provider write is authorized by this document or its implementation tests.

Reuse the existing event store, scheduler, inbound queue, V3 CAPD and outbox. Do not
add a second event bus, polling LLM loop, scheduler or notification channel. Native
optional adapters implement managed service boundaries. Standard skills supply
expertise and ordinary resources/scripts; no per-script catalog, Artifact registry,
script tools or tools per account are introduced. Project sharing, editable config,
package installation, Jev relevance and account connection confer no new authority.

New project-check acquisition is read-only. Existing explicitly authorized Calendar
creation/update remains supported by its shipped package; it is not activated by a
read policy. Mail sending/marking/archiving, calendar writes from checks, payments,
invitations and school submissions are outside this delivery.

## Reconciliation with shipped foundations

- [#32](https://github.com/khabaznak/alphonse-agent/issues/32): the accepted
  [standard skills contract](standard-skills-contract.md) makes packages the
  reusable unit. Preserve package/resource containment, provenance, update/export/
  uninstall behavior and the [native script execution](skill-script-execution.md)
  boundary. Generic Bash is trusted host execution, not an OS sandbox.
- [#36](https://github.com/khabaznak/alphonse-agent/issues/36):
  [SkillBindings](skill-bindings.md) already persists `v2_skill_bindings` and
  `v2_skill_binding_choices` in `SQLiteIntegrationStore`. Choices map authenticated
  `(actor, skill, role)` to an exact binding whose account target is an existing
  integration ID. `configure`, `select`, `prepare` and `use` enforce actor/owner,
  explicit audience, enabled account, installed skill and readiness. Extend this
  resolver; do not build a second selection or sharing system. Today provisioning
  is administrator-only, integration records lack member ownership, and credentials
  reside in plaintext `secret_json` protected by host permissions. Masking is not
  encryption. Managed process delivery/redaction exists; secure OAuth storage does not.
- [#39](https://github.com/khabaznak/alphonse-agent/issues/39): the
  [starter library](household-starter-skills.md) already supplies family coordination,
  household administration and skill authoring. Extend these workflows or add a
  focused portable school workflow; do not replace them with provider-specific
  personas. No personal facts, credentials or live data belong in shipped packages.
- [#41](https://github.com/khabaznak/alphonse-agent/issues/41):
  [family-calendar](family-calendar.md) is a shipped reusable Google Calendar package,
  with ordinary `scripts/google_calendar.py`, `calendar` binding and separate exact
  `authorization` binding for writes. It lists events/availability and verifies
  approved creates/updates by readback, stable operation IDs and ETags. There are
  no Calendar tools/catalog entries. Extend this package and its service access
  for bounded project reads/sync; do not rebuild Calendar or remove its write guards.

These completed issues retain their completed state and original milestone.
[Skills management](skill-management.md) already reports binding readiness in Desktop;
full optional-account/grant/check management remains #121. Legacy Artifact migration
is documented in [resource migration](skill-artifact-resources.md); this work does
not revive that catalog or migrate household programs into product source.

Current runtime anchors and gaps, verified against the baseline:

- [Local workers](local_event_workers.md), [`automations.py`](../alphonse/agent_v2/automations.py):
  worker/type registration, schema validation, exact top-level filters, bounded
  history and worker/event-ID deduplication exist. Worker ID is a registry lookup,
  not process authentication. Matching/history lack private-account scope; pruning
  deletes the row currently used for deduplication. Claims have no recovery lease.
- [`daemon.py`](../alphonse/agent_v2/daemon.py) `publish_event` claims and queues
  fixed automation prompts with `metadata.event`, then marks executions enqueued.
  This is not restart-safe end-to-end completion. The owner-only Unix socket is a
  same-OS-user boundary, not a restricted independently authenticated worker API.
- [`scheduled_tasks.py`](../alphonse/agent_v2/core/scheduled_tasks.py) and
  [`scheduled_worker.py`](../alphonse/agent_v2/services/scheduled_worker.py) already
  have durable occurrences, leases/retries, stable queue IDs and `pdca`/`direct`
  delivery. Today creation requires a prompt and accepts only those two modes.
  Typed acquisition is an additive job interface, not an existing third mode.
- [`context.py`](../alphonse/agent_v2/core/intelligence/v3/context.py) does not
  explicitly render `metadata.event` as triggering evidence. There is no
  `v3/projections.py` on this baseline, despite the old draft's reference. #108
  must audit every actual context/input-manifest path when it lands; queue metadata
  alone does not establish that Jev or System Two saw the evidence.
- [Project configuration](project-config.md) already provides goals, duration,
  lifecycle and explicit sharing. It is context, not operational authorization.

## Version 1 record and service contracts

The following field names and invariants are the shared implementation contract,
not claims of existing tables or IPC methods. All records have `schema_version: 1`,
opaque stable string IDs, UTC RFC3339 creation/update timestamps and a monotonically
increasing integer `revision` starting at 1 for mutable records. Provider timestamps
are separate from observation timestamps; local schedules carry an IANA timezone.
Never recycle IDs or retarget an existing ID to another account/owner. Store new
operational records through validated services in the existing shared database;
credentials/content reside behind references, never inside policy/event payloads.

### Connection — #109, #110, #111

`connection_id` is the existing `integration_id`, not a parallel account identity.
Add `owner_user_id`, `provider_key`, `provider_subject`, `display_name`,
`granted_scopes`, normalized `capabilities`, `credential_ref`, `credential_generation`,
`status`, `health_code`, `last_success_at` and `revocation_pending`. Provider subject
is the provider's immutable identity (including tenant where needed); an address is
only display metadata. Many connections per provider/member are valid. Status is
`connected`, `paused`, `reauth_required` or `disconnected`. Only `connected` with
healthy required capabilities permits managed retrieval. Tokens never appear in
record projections. A newly connected account creates no policies or schedules.

### Grant and existing binding — #110

`grant_id`, `connection_id`, `grantor_user_id`, `beneficiary_user_id`, optional
`project_id`, allowed `operations`, provider `resources`, permitted
`recipient_user_ids`, `expires_at`, `revoked_at`, and revision define operation,
resource, purpose and disclosure limits. Empty recipients authorize no delivery;
no wildcard recipients or implicit project-member expansion. Only an authenticated
owner/authorized grant manager may issue or expand grants. Administrator management
rights remain subject to the explicit binding choice and service policy; they are
not an account-selection fallback. Provider consent and Alphonse grants are separate.

Keep existing binding IDs, exact account targets and member choices. Extend account
binding requirements with `grant_id`/capability requirements; service resolution
returns a sanitized connection handle, selected resource constraints and effective
grant revision. Multiple bindings may narrow the same connection for different
skills/members. Binding ACLs and the grant both apply (intersection, never union).
A legacy shared binding is not silently converted to broad new provider grants.
No usable grant means setup/reconciliation, not implicit authorization.

Every managed use resolves the existing explicit `(actor, skill, role)` choice,
checks active actor/owner, installed available skill, binding access/enabled state,
exact connection status, grant beneficiary/project/expiry/revocation, resource/
operation constraints and actual provider scopes. `get_by_provider`, an address in
a message and Jev selection may not pick an account. Missing or ambiguous choices
fail visibly. A selection change cannot retarget an already authorized policy:
pin its binding ID, connection and grant; require policy review before switching.

### Project check policy — #112

`policy_id`, `project_id`, `executing_user_id`, `goal_snapshot`, `goal_digest`,
`source_bindings` (skill ID, role, binding ID, connection ID, grant ID and approved
grant revision), `query`/`query_digest`, `skill_hints`, `trigger`, `retrieval_window`,
`duration`, `initial_lookback_days`, `missed_run`, `budgets`,
`recipient_user_ids`, `notification_mode`, `authorized_by`, `authorized_at` and
revision are persisted authorization, separate from project files. Status is
`disabled` (default), `active`, `paused`, `needs_review` or `ended`.

Version 1 defaults: scheduled local execution, initial lookback 30 days, missed
run `next_allowed_window`, notification `material_changes`, no recipients until
explicitly selected. Activation requires a concrete reviewed timezone/window,
finite duration or explicit ongoing choice, query/resources, grants, budgets and
audience. Conversation may authorize this concrete summary; execution within it
needs no repeated approval. No arbitrary prompt/script/credential in the trigger.
Changes use expected-revision compare-and-swap; widening needs fresh authorization.
Relevant goal, query, resource, account, recipient or duration edits invalidate old
work and require review. Project file edits can invalidate scope but cannot activate
or broaden it. Schedule/policy revisions must be installed atomically, never leave
an old schedule pointing at a newly broadened policy.

### Check run — #113, #114

`check_run_id`, `policy_id`, `policy_revision`, `occurrence_key`, `executing_user_id`,
`effective_grants` (IDs/revisions), `window_start`, `window_end`, `query_digest`,
`coverage` (requested/covered ranges, resources, gaps), `acquisition_checkpoint`,
`analysis_checkpoint`, `started_at`, `finished_at`, `outcome`, `error_code`,
`publication_id` and revision establish an auditable receipt. Unique
`(policy_id, policy_revision, occurrence_key)` identifies one logical run across
retries. Lease/attempt IDs are not new run IDs. Cursor state is opaque and bound to
connection, resource, query digest, scope and adapter version.

Operational state is `pending`, `leased`, `retry_wait` or `terminal`; terminal
outcome is `complete`, `no_match`, `partial`, `blocked`, `failed` or `cancelled`.
`no_match` requires complete authorized acquisition and completed required triage,
not empty output after an error. `complete` means required processing finished,
not successful alert delivery. Window expiry with unfinished work is `partial`;
revocation is `blocked`, explicit cancellation is `cancelled`. Terminal runs are
immutable; subsequent permitted work gets a new occurrence with links to prior
coverage. Acquisition progress and analysis progress never share a single cursor.

### Source evidence — #114

`evidence_id`, `connection_id`, `owner_user_id`, `resource_id`, `resource_version`,
`provider_timestamp`, `observed_at`, `check_run_id`, `grant_id`, `grant_revision`,
`content_ref`, `content_digest`, `safe_source_link`, `audience_user_ids`,
`retention_deadline`, `state` and revision preserve provenance. The identity key is
connection + provider resource + version; the same source can support independently
authorized projects. Use an adapter-defined digest when a provider has no version.
A source link is a convenience, never authority to browse or forward credentials.
State is `available`, `expired`, `revoked` or `deleted`; terminal content removal
retains only a content-free receipt. Scope-check references before loading bytes.

### Finding and notification — #116

`finding_id`, `project_id`, `goal_digest`, `evidence_ids`, `facts`, `uncertainty`,
optional `due_at`/`due_date` and timezone, `audience_user_ids`, `state`,
`supersedes_finding_id`, `related_finding_ids`, `material_revision` and notification
history record an interpretation, not a provider event or a task-success claim.
State is `open`, `resolved`, `corrected`, `withdrawn` or `unresolved`. Missing dates,
grade weights and ambiguous children remain unknown. Do not merge two assignments
or invoices merely because their wording matches. Prefer current authoritative
notices to old quoted text; corrections link to prior findings.

Audience is bounded by the policy and each supporting grant's allowed disclosure.
Broader derived disclosure requires an explicit grant, not an assumption that a
summary is public. Project membership alone never grants raw-source access or
expands prior findings. Outbox idempotency is `(finding_id, material_revision,
recipient_user_id, channel)`; store recipient-level pending/sent/failed/ambiguous
outcomes separately. Do not announce delivery from queue acceptance.

## Deterministic timing, lifecycle and execution

Before dispatch, each provider call/page/attachment/retry/continuation, CAPD entry,
evidence expansion and delivery, recheck trusted actor, grants, project lifecycle,
policy/revision, cancellation and relevant timing/audience. A grant revision change
fences queued work: resolve again and review scope, never reuse a cached credential
payload as authority. Project pause/completion/archive/end invalidates queued
checks. Connection pause/revocation blocks dependents only; resume never revives
stale revisions or replays every missed run. Cancellation cannot recall an in-flight
read or already delivered message; stop subsequent calls and report the boundary.

“Only Mondays” restricts retrieval **and background analysis**, not merely alerts.
A concrete example is Monday 08:00–09:00 America/Mexico_City, half-open window
`[start, end)`. Stop/checkpoint at 09:00; no Tuesday continuation. Delivery is also
window-bound by default; any separate delivery window needs explicit authorization.
A one-off user read uses its own task authorization without altering the policy.
Auth maintenance may refresh credentials outside a retrieval window but may not
read source content. Human consent is required when reauthorization is needed.

Use existing timezone-aware recurrence. Nonexistent local times skip with a visible
gap; a repeated local time fires once using the earlier offset and stable occurrence
key. #113 tests this contract against the scheduler and adds deterministic handling
where necessary. Default catch-up is the next allowed window, with a visible missed
coverage interval. Late catch-up requires an explicitly bounded authorized window.
Disallow concurrent runs per policy; coalesce overlap into one recorded gap/next
window, not an unbounded backlog. Separate projects can independently use the same
account under their own grants and coverage receipts.

### Typed scheduled-job interface — #113

Extend the existing store/worker with `job_kind` and `job_version`, separate from
legacy `delivery_mode`. Migrate existing schedules to `job_kind="prompt"`,
`job_version=1`, preserving `pdca` and `direct`, prompts, IDs and recurrence exactly.
A new `project_check`, version 1, has only this typed payload:

```json
{
  "job_kind": "project_check",
  "job_version": 1,
  "payload": {"policy_id": "check_school_trimester", "policy_revision": 1}
}
```

The trusted scheduler creates the immutable occurrence context: `occurrence_key`,
`scheduled_for`, `window_start`, `window_end`, lease token/generation and deadline.
Actor/project/grants/query come from the persisted policy, never payload claims.
Validate payload strictly, reject unknown fields/versions and require no prompt
for typed acquisition. Preserve prompt validation for old jobs. Do not encode a
provider poll as a prompt or fall through an unknown kind into CAPD.

The handler contract is `dispatch(job, occurrence_context) -> {check_run_id,
state, error_code?}`. Within a short transaction, validate revision and upsert the
unique logical run, then wake a bounded acquisition worker outside scheduler ticks.
No network/model calls in the scheduler transaction. Repeated dispatch reconnects
to the same run. Fence all writes by lease generation; heartbeat active leases and
reject stale worker commits. Reuse scheduled claim/retry/backoff conventions (five
attempts maximum by default); retry only within the window. Record acceptance,
processing, publication and outbox completion separately. Unknown versions are
blocked with a static `unsupported_job_version`, visible for operator recovery.

## Acquisition, events and V3

Shared account service operations are `capabilities`, `health`, bounded
`search/list`, `read`, permitted `read_attachment` and `iterate_changes`. Context
contains authenticated actor, exact connection/binding/grant, task ID, optional
policy/revision/run/occurrence, deadline, cancellation and remaining budgets.
No project is required for an authorized ad hoc query. Return evidence references,
coverage, continuation, counts and static typed errors, never credentials.

Default run limits are 100 pages, 1,000 candidate resources, 10 MiB normalized
content, 5 MiB per attachment, 10 minutes wall time and the occurrence end,
whichever comes first. Default analysis limits are 1,000 bounded Jev candidate
sets and 20 System Two calls. Persist consumed budgets across retries. Exceeding
any limit yields partial coverage/pending counts, not no-match. #115 calibrates
thresholds and #124 qualifies these conservative defaults on synthetic school/bill
cases before a pilot; raising limits is an explicit policy revision.

Persist accepted page evidence and replay identities before advancing acquisition.
Only advance analysis after required processing is durable. Query/folder/scope or
adapter-version changes require an explicit bounded baseline/backfill, not cursor
reuse. Cursor expiry triggers bounded resync/overlap deduplication and a visible
gap if completeness is unknown. Provider denial, inaccessible folders, throttling,
attachment failures and model failure never become clean no-match.

Normalize MIME/HTML deterministically, preserve needed quoted evidence, do not
execute scripts or remote images. Fetch only permitted needed attachment types
and sizes. Message links authorize no arbitrary browsing or credential forwarding.
Provider text is external untrusted data, separate from runtime constraints and
stored prompts; it cannot change permissions, recipients or tool activation.

#106 implements the fail-closed event gate (see [managed event boundary](local_event_workers.md#managed-private-event-boundary-106)). Live authority services/adapters remain #110/#112/#113 and provider work. The server-owned versioned event envelope contains: `publication_id`, registered
worker binding, authenticated owner, connection/grant, project/policy revision,
check-run ID, type/version and occurrence/observation times. Bind from trusted run
state, not worker payload assertions. Scope-filter matching/history/status before
exact payload filters. Use bounded references and coverage, not whole messages or
model-authored instructions. Daemon-controlled adapters are the initial workers;
independent restricted worker authentication is a #126 decision, not a guarantee
of the existing owner-only socket. Existing sensor schemas/fixed prompts survive.

#107 adds atomic execution leases, retry deadlines, recovery ticks and idempotent
queue insertion keyed by execution ID. Crash after queue insertion reconnects to
the original message. Recover pending/expired work without needing another publish;
no transaction spans network/model work. Publish at most one candidate/attention
result per run with stable publication identity; no-match persists a quiet receipt.
Use independent receipts for publication, run, source version, finding and delivery
deduplication. Two authorized projects may legitimately consume one source.

#108 renders bounded, scope-checked triggering evidence into every actual V3
curation/planning/tactical/review path and records exact IDs/revisions/digests in
input manifests. Explicit coverage/freshness/uncertainty and untrusted-source
labels must survive clipping; truncated evidence is partial, not absent.

[Jev/System One](v3-hierarchical-capd/06-system-one-check-act.md) answers only bounded
Noul/Choice/Score relevance, commitment-presence, change and urgency questions on
authorized content. Independent questions run in parallel. Record question/model
versions and probabilities without private content in telemetry. System Two owns
interpretation, extraction, reconciliation, strategy, calls and user language.
Jev never extracts open-ended prose, grants access, composes calls or turns failed
effects into success. Ambiguity retains pending candidates for bounded CAPD;
unavailability yields degraded/retry state, never irrelevance or an automatic
high-volume LLM replacement for bounded classification.

[Progressive reveal](v3-hierarchical-capd/03-progressive-tool-revealing.md) continues
to expose native capabilities, not scripts/accounts. Add stable provider-native
read tools only when the managed service boundary needs them, with reviewed Jev
questions and deterministic exposure/execution gates. Calendar remains its package
workflow by default. Shared account management can be native; each connection is a
resource argument, never another tool definition.

## Credentials and supported deployment decision — #109, #111

#109 implementation: [secure credential storage and operations](secure-credentials.md).
Credential references, vault backends and generation fencing are available; OAuth
consent, provider refresh serialization and account/grant services remain #110/#111.

Desktop v1 targets macOS with the daemon's OS user accessing macOS Keychain.
Headless v1 targets Linux with an explicitly configured encrypted file backend:
AEAD-encrypted credential records, owner-only directory/file permissions and an
independent 256-bit key supplied by a protected systemd credential or separately
mounted owner-only key file outside the credential database/backups. Use a maintained
cryptographic library, fresh nonces and authenticated connection/generation metadata;
no homemade cipher or key stored beside ciphertext. #109 owns implementation and
backend conformance tests. Other OS/backend combinations remain unsupported for new
OAuth credentials until qualified; unrelated daemon functions still start normally.

Backend interface: `put`, `get`, generation-checked `replace`, `delete`, `health`.
Return opaque references to metadata clients; only trusted credential consumers
receive bytes. Missing, locked or corrupt backend blocks account use with reconnect/
unlock guidance; never fall back to plaintext `secret_json`. Secret backend/key
configuration is operator setup, not a skill dependency or an LLM tool argument.

Use provider-supported authorization-code + PKCE/state, exact registered callbacks
and system-browser consent. Desktop uses a loopback callback, never a confidential
client secret as proof of public-client identity. Headless uses an operator-assisted
browser/loopback on the daemon host (SSH forwarding where needed); no deprecated
out-of-band code paste. Unattended first consent is unsupported. Exact provider
registration/client-type qualification is #111/#123, a pilot blocker, not assumed
supported because a library supplies a flow.

Serialize refresh per connection and use credential-generation compare-and-swap
for rotation. Persist rotated refresh tokens securely before making them active;
never let a stale refresh overwrite a newer token. Ambiguous rotation/storage
failure blocks for reconciliation. Credentials must stay out of IPC, project/package
files, event history, checkpoints, exports, inference, arguments and routine logs.
Extend `SkillBindings.prepare/use` to dereference credentials inside trusted
execution. Preserve narrowly requested `ALPHONSE_SKILL_BINDINGS` process delivery
for compatible scripts; strip inherited payloads and redact before output clipping.
Managed services should retain credentials in-process and return handles/results.
Trusted Bash can still read OS-accessible files/process state: this is not
host-level household isolation or a sandbox against malicious package code.

Disconnect first fences local use/revokes dependent grants, then deletes active
credential/cache references and attempts provider revocation. An offline disconnect
is locally effective with `revocation_pending=true`; use a separately protected,
short-lived backend revocation record for retries (maximum seven days), never a
usable connection token. After expiry erase it and show manual provider-revocation
guidance. Do not claim provider revocation succeeded offline.

## Retention and replay defaults — #107, #114, #116, #123

Defaults are maximum lifetimes, never a promise to retain content after revocation.
Services enforce deadlines on reads and cleanup; event history count limits remain
additional bounds. All backup/checkpoint/export paths must respect the same scope.

- Raw/normalized source cache, attachments and excerpts: seven days from acquisition.
  Pending active runs may pin necessary evidence up to 30 days total; at that hard
  limit expire bytes and terminate blocked/partial with truthful coverage. A pin
  never defeats revocation/deletion. Reacquisition requires current authority/window.
- Optional-account event payloads, terminal run diagnostics and content-free input
  manifests: 30 days. Active references pin only the minimum metadata required for
  recovery; purge terminal diagnostics after 30 days. Never put message bodies,
  subjects, addresses or attachment names into telemetry/diagnostic receipts.
- Content-free deduplication receipts/tombstones: 90 days after acceptance/terminal
  completion, whichever is later. Maximum replay age is 90 days from original
  trusted acceptance time, not a retriable caller timestamp. Old publications are
  rejected as expired even after receipts are pruned. Over-horizon recovery is an
  explicit new authorized run, not an exactly-once promise. Keep minimal live source
  identity references while a retained finding needs them, independently of replay.
- Findings: project lifetime plus 90 days by default; ongoing projects review retained
  findings annually. Explicit durable retention is a separate user choice, with
  audience/source restrictions retained. Do not copy mailboxes into Memory Ledger.
- Private evidence in CAPD checkpoints/prompt caches: same seven-day deadline (or
  bounded active-run pin), references preferred. Expiration/revocation invalidates
  dependent views before resume. No private account content in routine exports;
  explicit authorized data exports carry scope/expiry metadata and explain that an
  already downloaded or delivered copy cannot be recalled.

Source deletion or revoked access immediately denies expansion and invalidates
cached views, pending tasks and outbox deliveries. Cleanup removes/redacts dependent
excerpts, prompt caches and checkpoints; findings become withdrawn/unavailable or
retain only separately authorized facts with independent surviving evidence. Backups
are encrypted, expire within 30 days, and are not readable serving stores; a restore
must apply deletion/revocation tombstones before enabling any reads. #123 rehearses
this across recovery/export paths before any private-data pilot. Provider rules may
require shorter retention; #123 records deployment-specific restrictions and blockers.
Longer content retention requires explicit user authorization and provider allowance.

## Provider extensions and rollout ownership

- #117 Gmail: optional read-only retrieval/history, bounded baseline and expired-history
  resync; do not request send/modify access. Synthetic buried assignment, correction,
  quoted old notice, missing due date, attachment-only notice and malicious message
  must produce source-backed uncertainty and one material-change alert.
- #118 Outlook/Hotmail: delegated Graph mail reads for qualified personal/work/school
  registrations, folder-aware delta and throttling. Inbox-only coverage cannot claim
  to include mail moved by rules; partial folder access stays visible.
- #119 Calendar: extend the shipped family-calendar package and managed binding path
  for selected-calendar project read context, incremental sync, exceptions,
  cancellations, recurring-instance identity and all-day/timezone semantics. Preserve
  its existing explicit write authorization/readback and no-invitation behavior.
  Do not invoke its current token-refresh/pagination script unchanged as if it already
  enforced per-page background windows; managed acquisition must add those checks.
- #120 Classroom: validate course/account role and administrator restrictions before
  read access. Guardian status is not student API permission; guardian-summary email
  can use mail adapters. Missing due dates, submission status or grade weight remain
  unknown; points alone do not establish a percentage of the course grade.
- #122 extends family-coordination/household-administration or supplies focused school
  guidance atop the same provider-independent evidence/finding contracts. Scripts
  remain ordinary package files, live data and credentials remain user-owned.

Provider scopes, registrations, account eligibility, public distribution verification
and processor retention/training terms must be verified against current provider
requirements by #111/#123 before live use. Remote Jev and System Two receive private
data and count as off-host processors. #123 publishes separate dated personal-test
and public-release readiness evidence; missing approvals remain blockers. This
contract does not assert provider approval or qualified production OAuth support.

Optional provider modules/dependencies load only when enabled. Missing SDK/backend
cannot break unrelated daemon startup, existing sensors, reminders or messaging.
The [#126 extension decision](optional-integration-extensions.md) defers continuous
polling, provider push, independent restricted workers and MCP transport for this
milestone. Local scheduled operation is the first path to qualify. That decision
records reopening triggers, conformance gates and rollout ownership; it implements
none of the deferred interfaces. Hosted availability and a general third-party
plugin marketplace require separate explicit user approval outside this milestone.
No connection implicitly enables monitoring; any future push is only a validated
hint to retrieve within existing grants/windows.

## Supported surfaces and terminal retirement

#121 owns Desktop connections (owner, consent, explicit grants, health, pause,
reconnect/disconnect), project policy review/activation, last/next run and coverage,
findings and per-recipient delivery. Reuse existing settings, Skills readiness,
project and work/automation views plus authenticated daemon/IPC services. Conversation
uses the same validated services with a concrete policy summary. Display natural
schedules, not RRULE/scope strings. Healthy no-change checks stay quiet; surface
meaningful findings/corrections, persistent failure and required user action.
Offline/sleeping hosts display stale coverage and the selected catch-up behavior.

Coordinate with [#79](https://github.com/khabaznak/alphonse-agent/issues/79) terminal
inventory, [#80](https://github.com/khabaznak/alphonse-agent/issues/80) setup/recovery
gaps, [#81](https://github.com/khabaznak/alphonse-agent/issues/81) shared runtime/
identity decoupling, [#82](https://github.com/khabaznak/alphonse-agent/issues/82)
retirement and [#83](https://github.com/khabaznak/alphonse-agent/issues/83) supported
operations documentation. No new TUI screens/parity requirement. Preserve necessary
daemon/status/diagnostic/admin commands, Telegram and shared identities/APIs until
their consumers are accounted for. #121/#127 supply account/check setup and recovery
evidence to those owners; this document neither retires a client nor blocks useful
account work on completion of the entire terminal milestone.

## Migration, compatibility and verification ownership

Use additive versioned schema changes with features disabled by default. Schema
version describes serialization; record revision fences authority; credential
generation fences rotation; job/event version selects a handler. These are distinct.
Unknown major versions fail closed without dropping work; additive optional fields
may be read compatibly only when their defaults cannot broaden scope. Management
mutations require expected revision; actor comes from trusted session context,
never a model-supplied owner field. Return static typed errors and sanitized IDs.

#109 owns credential-reference migration and rollback. #110 owns member ownership,
grants and binding-choice compatibility. Inventory exact existing integration IDs
and bindings; an administrator/owner explicitly reconciles missing/ambiguous ownership
or multiple account copies. Preserve IDs, choices, audiences and files. No inference
from project sharing, worker payloads or guessed email addresses. For an opted-in
legacy Calendar account: securely write/verify credentials, atomically switch its
reference, then remove plaintext values; recover interrupted migration without
falling back or losing the account. Existing #41 behavior is legacy protected-host
support until explicitly migrated, not proof of secure new OAuth storage. Unrelated
Telegram/Discord secrets are not prerequisites and must not be migrated here.

#106/#107 own sensor/event scope and receipt migrations; preserve existing IDs,
schemas and fixed-prompt subscriptions. Legacy trusted sensors stay in their current
local scope; no private-account event is allowed through an unscoped registration.
#112/#113 own policy/schedule migrations; old prompts remain old jobs, never new
account checks. #114/#116 own evidence/finding schema and deletion handling. #127
rehearses backups, restart, feature-disable and downgrade: stop/fence new job kinds,
finish/cancel active runs, preserve records and use a compatible binary/database.
Never downgrade OAuth references back to plaintext or replay unsupported jobs as
prompts. Document irreversible schema changes before rollout; no household file
moves/deletions or generated-program adoption are part of metadata upgrades.

Validation gates belong to the indexed implementation issues, not to claims of
runtime completion in #105:

- #106–#108: spoofed owner, cross-user history, concurrent claims, crash at every
  handoff, replay after pruning, pending-history protection, restart recovery,
  cancellation and exact V3 evidence manifests; existing sensors still work.
- #109–#112: two owners/two same-provider accounts, explicit selection, revoked or
  expired grants, narrowed scope, inactive members, missing/locked backend, OAuth
  state/PKCE/rotation races, goal/config edits and no secret-bearing projections.
- #113–#116: Monday-only zero off-window fetches/analysis, DST, pagination/retry
  crossing boundaries, overlap, cursor expiry/query changes, acquisition/model
  failure independence, budgets, deletion/retention and recipient-level deduplication.
- #117–#125: shared synthetic adapter conformance; recurring events, guardian denial,
  cross-source identity/uncertainty, no-match/partial distinction, Jev outage and
  prompt injection. #124 qualifies the Gmail school slice; #125 qualifies providers
  and household isolation. Real-account tests require separate explicit opt-in.
- #123/#126/#127: processor/provider readiness, optional transport decision, supported
  client setup/recovery, disconnected startup, migration/rollback and staged release.
  Index #104 remains open until release; contract publication completes only #105.

Baseline validation on 2026-10-31 at the revision above used disposable fixtures:

```sh
python -m pytest -q tests/test_agent_v2_event_automations.py tests/test_project_provenance_waiting.py tests/test_family_calendar.py tests/test_agent_v2_skill_bindings.py tests/test_skill_script_execution.py tests/test_agent_v2_skills.py
```

Result: **154 passed, 35 skipped, 2 failed**. The existing failures are
`test_new_conversation_creates_protected_home_project` and
`test_independent_message_receives_exact_turns_ahead_notice` in
`test_project_provenance_waiting.py`, both expecting an automatically created Home
project that is absent. Event-routing tests pass. This reproduces the draft's
baseline rather than attributing those failures to this documentation change.
Skipped legacy cases and these synthetic tests establish no real-provider effects,
secure-backend implementation or end-to-end proactive-check readiness. Record the
final documentation commit and relevant checks on the implementation PR.
