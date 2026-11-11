# Single-loop reliability implementation plan

Status: proposed; foundation work in progress. The candidate remains disabled until qualified. The current and proposed diagrams in the local output directory are diagnostic artifacts, not runtime specifications.

## Objective

Reduce repeated planning, per-action semantic reviews and unbounded prompt growth while preserving actor-scoped access, authorized effects, cancellation, durable evidence, uncertain-effect recovery and honest completion. The approach is a hypothesis to evaluate against the existing V3 engine, not a proven improvement. Issue [#184](https://github.com/khabaznak/alphonse-agent/issues/184) contains preceding work and should be reused.

## Work and dependencies

| Issue | Scope | Ready condition |
| --- | --- | --- |
| [#210](https://github.com/khabaznak/alphonse-agent/issues/210) | Bounded, authorized, source-linked evidence projection | Independent |
| [#211](https://github.com/khabaznak/alphonse-agent/issues/211) | Scoped one-off Gmail unread count | Independent; provider readiness remains separately tracked |
| [#212](https://github.com/khabaznak/alphonse-agent/issues/212) | Paired synthetic evaluation and independent oracles | Independent; candidate arm follows #214 |
| [#213](https://github.com/khabaznak/alphonse-agent/issues/213) | Durable task-wide budget primitives | Independent |
| [#216](https://github.com/khabaznak/alphonse-agent/issues/216) | Fenced superseded-run and queue handoff | Independent lifecycle foundation |
| [#214](https://github.com/khabaznak/alphonse-agent/issues/214) | Opt-in single-loop processor and safety integration | #210, #213 and #216; use #212 for evaluation |
| [#215](https://github.com/khabaznak/alphonse-agent/issues/215) | Comparative qualification and staged rollout | #212 and #214; #211 for the real capability scenario when available |

GitHub native blocking links express the hard dependencies for #214 and #215. An issue completing a standalone primitive does not imply that the candidate engine uses it; integration is owned by #214. Existing hierarchical task checkpoints must continue on their stamped engine. Qualification includes rollback and uncertain delivery/effect reconciliation.

## Coordination rules

- Start independent work in parallel only when file ownership is clear; reconcile shared interfaces before integration. Preserve existing uncommitted work in the current checkout.
- Use disposable synthetic cases and independent effect oracles. Do not put household messages, mailbox data, credentials or proprietary household programs into fixtures, telemetry or issues.
- Record verified outcomes separately from unknowns, blocked runs and measured versus estimated usage. Define thresholds before comparison, rather than fitting them to the Gmail incident.
- Account for actions, model calls, retries, tokens and time across mixed outcomes and restarts. Budget exhaustion must lead to a truthful bounded response, not success.
- Jev makes bounded probabilistic judgments; System Two plans, composes calls and speaks; deterministic runtime code enforces authority, effects, scope, cancellation and terminal invariants.
- A real provider test, live Jev experiment or paid comparison requires its own explicit scope, data disclosure and budget. No provider experiment is authorized by this plan.
- Keep the existing V3 engine available as the rollback path; do not change defaults until qualification and the separate rollout decision.

## Initial coordinated work

The working tree now contains standalone prototypes for projection, budget accounting, an unregistered read-only loop, a Gmail label-count service, and a paired synthetic evaluation. These are uncommitted and do not close the associated issues or change the production engine. One frozen scripted read-only fixture completes in both actual V3 and the prototype; provider token usage remains unknown, and the comparison is not a live-model qualification.

The Gmail tool is intentionally unregistered: private delivery to the authenticated actor and authenticated account-choice binding remain unresolved. The candidate loop cannot perform writes and uses synthetic host hooks; real authority, atomic checkpoint/effect fencing, provider limits, questions, scheduling, engine selection and lifecycle integration remain. The evaluation needs representative and held-out adapters, independent safety/effect oracles and meaningful comparison. Continue according to issue dependencies and keep all issues open until their actual acceptance criteria are met.

## Coordinated progress — 2026-11-10

Standalone projection and budget primitives have expanded authorization, omission, restart and clock-rollback tests. The unregistered candidate has an experimental SQLite task-row reservation/settlement adapter for a synthetic model step. Generic legacy checkpoint replacement and terminal-success paths now fence experimental rows; cancellation remains available. This does not yet authorize real provider or tool dispatch, and tool settlement remains blocked without durable actor/action-bound evidence.

The paired evaluator includes scripted reads, denied access, cancellation and a limited held-out variant, plus separate prototype crash and V3 seeded-resumption scenarios. Missing comparison arms are reported unavailable. A V3 cancellation diagnostic was corrected to report cancellation rather than a generic runtime error. Tool and task UI events were scoped and reviewed against synthetic two-user and malformed-payload cases. These improvements do not secure shared-channel Gmail delivery: the one-off Gmail capability remains unregistered until private destination, data propagation, account choice and effect/restart fencing are proven end to end.

The full Python suite passed with local socket and subprocess permissions: 2,809 passed, 37 skipped. Focused UI/question/checkpoint tests also passed after a subsequent malformed-event guard. This is test evidence, not live-provider qualification; all six issues remain open and candidate rollout is blocked.

A production tactical prompt-size guard was investigated and removed: a checkpoint CAS loss cannot safely use the existing failure or scheduled-handoff lifecycle without risking a stale memory finish or notification. #216 tracks the missing queue-claim and superseded-run contract. Until then the current tactical prompt remains unbounded.

Further work added a default-off queue-claim generation and renewal slice, experimental durable holds and owner-only inspection, and adversarial tests. These paths are not enabled: mixed-version workers can bypass generation checks; hold recovery and crash-safe user deletion are missing. Current processor/UI paths cannot safely publish a superseded outcome until publication is fenced. The outbox now validates that an injected SQLite transaction targets its own main database and avoids temporary-table shadowing, but its current schema has no non-dispatchable stage; claim-fenced publication remains unimplemented. See [#216](https://github.com/khabaznak/alphonse-agent/issues/216) and [26-superseded-runtime-integration.md](26-superseded-runtime-integration.md).

After the outbox and queue work, the complete Python suite passed with local socket/process permissions: **2,907 passed, 37 skipped**. The current V3 tactical prompt remains unbounded: the attempted size guard was withdrawn because a superseded checkpoint could still cause stale memory or notification processing. All issues remain open and these working-tree changes are uncommitted.

## Further coordination — 2026-11-10

Core and daemon startup/run-once now reject an unsupported experimental claim-fenced queue before claiming work or starting workers. User deletion now rejects an admin target before deleting associated data, though the broader multi-store deletion remains non-atomic and is not crash-safe. Outbox migration rejects unknown states and unrecognized future status schemas before modification, uses one SQLite write transaction, and refuses legacy schema extensions it cannot preserve. The oldest recorded four-status schema is covered; noncanonical schemas fail closed for explicit recovery.

A disposable regression demonstrates that socket-scoped daemon locks do not fence two processes using the same database through different socket paths or direct SQLite access. TUI construction can also precede daemon locking. A database-wide protocol, old-worker/effect drain, durable deletion gate and journal, claim propagation, fenced publication and authorized hold/effect recovery remain prerequisites; the experimental queue and hold stay disabled. The full Python suite passed with local socket/process permissions: **2,941 passed, 37 skipped**; a later focused diagnostics run passed **35**. #216 and its dependents remain open; no live-provider qualification is implied.

## Coordinated progress after 31bb1f52

The queue refuses concurrent incompatible claim modes among clients using the current binary, and rejects malformed generation values; this does not fence older binaries or raw SQL. A proposed core/daemon conflict diagnostic was withdrawn because a conflict after preflight could falsely mark an event processing. `claim_mode_blocked()` is a diagnostic snapshot, not an integrated ownership or liveness guarantee.

Outbox idempotency now rejects a reused key when routing, audience, payload or stable operation metadata differs, and does not suppress unrelated constraint violations. A conflicting regenerated early acknowledgment is reported as a delivery failure rather than sent. Managed user/project cleanup retains identifying database rows until cleanup succeeds and uses descriptor-relative no-follow deletion on supported platforms; unavailable roots and unsupported platforms fail closed. Deletion remains non-atomic across stores, lacks a hold gate/journal, and filesystem operations cannot be rolled back. These are incremental safeguards, not completion of #216 or activation of any experimental path.

The full Python suite passed with local socket/process permissions: **2,976 passed, 37 skipped**. These changes are uncommitted and do not establish live-provider reliability.

## Coordination after commit 31bb1f52

Outbox reads and delivery updates use the main schema; supplied transactions are checked against the store, and incompatible idempotency-key reuse fails without claiming a different message was sent. Managed project/profile cleanup keeps the identifying rows until cleanup succeeds, using descriptor-relative no-follow deletion; missing roots and unsupported platforms fail closed. The queue rejects incompatible simultaneous claim modes among current-binary clients. These are incremental safeguards, not a database-wide writer barrier or complete deletion recovery.

An experimental deletion journal and its daemon route were removed during review: ordinary and older writers do not honor its gate, current caller authentication and pre-filesystem authority need stronger fencing, and safe retirement of identifying gate state is undefined. The proposed core/daemon claim-conflict diagnostic was also withdrawn because it could mark an unclaimed event processing. Experimental engine, queue, holds, staging and Gmail remain disabled. #216 and dependent issues stay open. The complete Python suite passed with local socket/process permissions: **2,980 passed, 37 skipped**. Changes after 31bb1f52 are uncommitted.
