# Test-suite review against the dependable-loop mission

Snapshot: 13 November 2026, local working tree based on `3f043d15`, with
parallel gym and measurement edits in progress. This is an inventory of the
**Python pytest suite**; it does not inventory Desktop, TypeScript, manual,
integration-deployment, or future tests. The suite changes while the milestone
is active, so the snapshot is not a permanent count.

## What was examined and what remains

`pytest --collect-only -q` collected **3,941 cases** from **2,469 distinct
functions** in **245 files**. Every collected node maps to a source function and
line in [the inventory](test-inventory-2026-11.csv); parametrized cases are
counted against their function. The CSV records a single, filename-based
triage category for each function, not a verdict on its value. Its
`automated_triage_not_individual_review` label is intentional: the code and
assertions of every test have **not** been individually reviewed. Therefore no
total for tests that are safe to delete can be derived from this inventory.

Automated first-pass areas (functions / collected cases) are:

| Area | Functions | Cases | Interpretation |
| --- | ---: | ---: | --- |
| Shared host, product and integrations | 820 | 1,174 | Household-facing capabilities and infrastructure across both loops. |
| V3 bridge and evidence | 423 | 743 | Current production bridge and historical comparison; review before retirement. |
| Gym and measurement | 370 | 679 | Fixture/oracle, comparison and measurement coverage; not real model performance. |
| Skills, accounts, privacy and migration | 246 | 466 | Authorization, credentials, skill and data lifecycle boundaries. |
| Candidate, claims and fencing | 314 | 457 | Candidate experiments, failure diagnostics and safety primitives, not production qualification. |
| Other or needs manual classification | 243 | 369 | Filename did not fit a priority rule; some may belong in any category above. |
| Retired V2 internal implementation group | 53 | 53 | Eight narrowly selected files, not every `agent_v2` file. |

These categories are approximate and mutually exclusive by matching priority;
they can misclassify individual tests and must not be used as deletion lists.
For example, the package name `agent_v2` also contains the **current** daemon,
tools and household services. Some V3-named files are evaluation and some gym
tests exercise real shared boundaries.

## Recommendations from direct inspection

### First removal candidates, with explicit lost coverage

1. **Removed:** `tests/test_agent_v2_basic_intelligence.py` had two cases for the old
   `BasicIntelligenceProcessor` echo and its `/project new` placeholder. The
   repository search found no use of the class outside its own two tests and
   its definition/export; the sole installed command enters the daemon CLI.
   Neither case exercises the normal conversation path, actual command
   execution, dependable-loop behavior or a safety invariant. They are strong
   low-value candidates for removal. **Tradeoff:** the class is explicitly
   exported from the distributable Python package, so removing the tests gives
   up coverage of an importable placeholder; unknown external users cannot be
   ruled out. The owner accepted that tradeoff on 14 November; the exported
   implementation remains in place, but these placeholder behaviors no longer
   have direct test coverage. This is not proof of zero risk.
2. Removed `tests/test_agent_v2_pdca_graph.py::test_pdca_do_node_stub_returns_task_state_unchanged`:
   it checked object identity on an empty, retired V2 node; it did not test an
   actual household outcome. `test_agent_v2_do_node.py` separately tests the
   no-unexecuted-call behavior with a real registry and checks no invocation.
   Removing the identity-only assertion loses an internal identity convention,
   not that no-effect behavior.
3. Removed `tests/test_agent_v2_pdca_graph.py::test_pdca_act_node_returns_task_state`:
   this checked only whether the retired node returned the identical Python
   object, using an injected decision fixture. It asserted no decision,
   household outcome, authorization, or other behavior. Other tests exercise
   the old node's routing and truthful failure; they do not preserve the
   identity assertion, which is intentionally abandoned here.

Two identity-only tests and the two placeholder tests were removed in this review. These candidates do not
justify deleting the old Python implementation or its exported symbols.
The remaining eight-file V2 group plus deterministic-replay and replay-harness
checks passed together: **56 passed in 1.11 seconds**. The CSV is the earlier
collection snapshot, so it still lists the two removed functions.
For each deletion, verify collection and focused tests and record the deliberate
loss; do not treat a reduction in count as an outcome in itself.

### Retirement and consolidation candidates requiring more work

- The 53 cases in the eight-file narrow V2-internal snapshot were individually
  inspected, including the four now removed. The remaining 49 cover behavior or structure
  tied to that implementation. Examples include evidence-backed completion,
  failure and cancellation behavior, graph routing and prompt contracts. They
  are lower priority for the selected target, but not automatically valueless.
  Some are candidates for later consolidation; do not infer a safe deletion
  count. V2 is retired
  from normal production dispatch, but `evaluation/deterministic_adapters.py`
  still invokes PDCA for historical replay and `test_agent_v3_deterministic_replay.py`
  and `test_agent_v3_replay_harness.py` rely on that adapter. Preserve those
  baselines or explicitly retire them, transfer mission-relevant invariants to
  the new loop, and only then remove tests tied solely to old internals. Do not
  confuse the retained replay baseline with a current production engine.
- The former 48-case invalid-judgment matrix in
  `test_agent_v3_delegated_outcome.py` exercised `outcome_assessment`, a helper
  with no repository consumer outside that test, and then translated its
  result into valid synthetic Act judgments. It did not inject malformed
  judgments into the active V3 provider. See the detailed review below; the
  separate real-provider and outer-controller malformed-answer checks remain.
- The 100 held-out fixture variations in `test_heldout_memory.py` and the 41
  paired fixture variations in `test_paired_memory.py` validate data and oracle
  boundaries. Keep full corpus checks, and consider a smaller fast selection
  **in addition** to a routinely run comprehensive job. These are not 141
  separate product features, nor proof that candidate conversations work.
- Some claim and delivery diagnostics deliberately reproduce current unsafe
  legacy behavior. Keep them visible as open defect evidence; once the behavior
  is corrected, invert or replace the assertions with regressions for the safe
  outcome. Synthetic grant and effect tests cannot substitute for real provider
  and authentication coverage.

An automated exact-AST-body scan found one identical test-body pair, Gmail and
Classroom limit tests. Inspection showed their `setup` fixtures instantiate
different adapters, so identical source is **not** evidence that either case is
redundant. This scan finds only exact body matches and does not detect semantic
duplication. More importantly, two superficially similar tests may protect
different actor, lease, cancellation, revocation, or delivery boundaries.

### Claim and fence review: seven files

Individually inspected all **71 test functions** in `test_claim_fence_foundation.py`,
`test_checkpoint_claim_fence.py`, `test_claim_fenced_outbox_diagnostics.py`,
`test_experimental_core_fence.py`, `test_experimental_queue_claims.py`,
`test_claim_bound_atomic_step.py` and `test_fenced_queue_daemon_preflight.py`.
The counts below refer to this working-tree review, not the earlier CSV snapshot.
The seven files currently collect **82 cases**, which passed in **0.77 seconds**
in one local run (`pytest -q --durations=10` on these files). This is not a
full-suite runtime or an estimate of production reliability.

- The claim foundation and checkpoint wrapper protect different boundaries:
  the former tests transaction-level lease, identity, rollback and clock
  handling, while the latter also binds the owner, task and checkpoint version.
  Staging and synthetic dispatch call the checkpoint wrapper; several other
  experimental checkpoint and recovery paths call the underlying fence. Both
  remain useful until equivalent normal-path coverage exists.
- Two outbox diagnostics intentionally demonstrate current shortcomings:
  a live claim alone can publish an obsolete task version, and an expired
  outbox claimant can settle after reclaim. They document unresolved behavior,
  not safe behavior; invert or replace them after the corresponding fixes.
  Cross-database, SQLite temporary-table and alias checks exercise distinct
  outbox connection boundaries.
- Core and daemon guard tests overlap in subject but protect different entry
  points and effects. The daemon test for an unadvertised receipt explicitly
  shows that prework still occurs before core refuses the claim. Keep that gap
  visible until acquisition and all affected sinks are fenced.
- The model-step tests observe reservations, source reads and chooser calls
  around claim loss, expiry and cancellation. They prove the disposable path
  fails closed in those cases, not that the daemon propagates claims or that
  a real provider effect is reconciled. Two expiry tests advance a fake clock
  after a fixed number of clock calls; that binds them to implementation call
  order and can break after harmless refactoring. Replace that timing technique
  with explicit barriers at observable dispatch boundaries when the integrated
  host supplies them; do not simply delete the expiry scenarios.
- Several queue renewal tests currently exercise a deliberately unavailable
  method, so their green result must not be interpreted as working renewal.
  They check separate inputs and no-mutation/hold conditions that will matter
  when a renewal protocol is introduced; review and replace them alongside
  that implementation. The core/daemon refusal tests likewise need to evolve
  when a reviewed supported path exists, rather than mandate refusal forever.

One parametrized checkpoint test ran the same expiry-and-rollback scenario
twice, once for each unrelated checkpoint mutation. It now checks both status
and project mutations and runs the expiry scenario once, saving one collected
case while retaining the three exercised behaviors. A repeated adjacent
identical acknowledgement assertion was also removed. Neither change proves
semantic equivalence of other similar-looking tests; no safety case was
removed wholesale. The earlier CSV still reflects its collection snapshot.

### Memory corpus and oracle review: two files

Individually inspected all **37 test functions** in `test_heldout_memory.py`
and `test_paired_memory.py`; their **183 collected cases** passed in **7.08
seconds** locally. This includes 100 held-out source-readback fixtures and 41
paired fixture-isolation checks. The 100 cases check their own source contents
and verify that a success sentence cannot stand in for effects. The 41 cases
verify the available tool and storage boundary for each fixture. They are not
141 separate product features, but deleting them by case count alone could
hide input-specific fixture corruption or privacy failures.

The remaining checks protect distinct behaviors: frozen corpus/order and
incomplete runs; source-delivery ranges and private-source filtering;
recipient, recurrence and readback requirements for reminders; partial searches
and pagination; failure and interruption evidence; and crashes across a
subprocess restart. The seemingly similar access-transition checks assert
different boundaries: paired cases exercise changed source access before the
next model input, while the held-out-file check keeps non-held-out fixture
readback available after that transition. No duplicate was proved safe to
remove in this group.

The older paired-memory experiment calls `HierarchicalCAPDProcessor` for both
arms, toggling a context-projection flag. Its arm called `candidate` is therefore
the V3 projection experiment, not the selected simpler loop. These tests protect
the current bridge and historical measurement harness but do not qualify the
new loop. A scripted provider or synthetic receipt does not establish model
quality or delivery. A Jev test in this group observes its fixture clock in a
mocked call manifest; it is instrumentation coverage rather than a live model
or calibration evaluation.

In this one run, the slowest case was the subprocess crash/restart drill at
2.11 seconds; the next was the paired-abort drill at 0.28 seconds. When the
team designs fast/comprehensive test jobs, those measured costs and their
different safety roles are more useful than the 100/41 parameter counts. Do
not remove crash/restart coverage merely because it is slower. These timings
are a single local observation and the CSV remains the earlier snapshot.

### Offline measurement and journals: three files

Individually inspected all **19 functions** in `test_benchmark_summary.py`,
`test_offline_three_arm.py`, and `test_synthetic_suite_journal.py`, representing
**29 collected cases** at review time. These tests address different boundaries:
benchmark-summary counts preserve denominators and unknowns; the three-arm
fixture runner isolates each arm and independently checks fixture output; the
synthetic journal records interrupted attempts and does not overwrite a run.
No whole test was identified as a safe duplicate in this group.

At review time repository search found `benchmark_summary.summarize` used only
by its own tests. A subsequent change added `synthetic_journal_summary`, which
connects a validated synthetic-suite journal to that accounting helper. The
three-arm runner still uses a separate schema, and the new bridge remains local
and uncommitted. Passing these accounting tests therefore does not demonstrate
a measured external baseline.
The summary accepts caller-reported grades and observations; the journal trusts
fixture adapter results. The offline executor declaration is not a network
sandbox. Keep those limitations visible when results are reported.

### V3 completion, clarification and retry: five files

Individually inspected all **54 original test functions** in
`test_agent_v3_delegated_outcome.py`, `test_agent_v3_completion_recovery.py`,
`test_agent_v3_failure_reporting.py`, `test_agent_v3_clarification_resolution.py`
and `test_agent_v3_retry_continuity.py`; one function was replaced with three
more explicit functions during this review. Also examined the relevant malformed
Act-answer tests in `test_agent_v2_system_one.py` and
`test_agent_v3_outer_controller.py`; those two whole files were not individually
reviewed. A focused run across all seven files passed **241 cases in 1.05 s**.

The active V3 bridge checks exercise several distinct user-facing boundaries:
failed effects cannot become completion, incomplete or conflicting user answers
remain unresolved, corrections invalidate prior assessment, source access is
rechecked across checkpoints, failure explanations omit secrets and remain
associated with the correct task, and a question answer can arrive before or
after delivery without a second resume. These use synthetic providers; they do
not establish model calibration or real delivery. Preserve them while V3 is the
production bridge and until equivalent supported-path coverage replaces them.

The old 48-case cross product in `test_agent_v3_delegated_outcome.py` looked
like invalid Jev-response coverage but exercised only the test-owned
`outcome_assessment` helper. No production or evaluation module imports that
helper. `run_case` converted its result to valid `.01`/`.99` values for the
real Act provider; it never sent the invalid original value to that provider.
The current provider and outer controller use `ACT_QUESTIONS`, and their own
malformed-answer tests remain unchanged. This corrects the earlier claim that
the 48 cases directly protected the active V3 invalid-answer boundary.

The helper validates every field using one uniform predicate. Its Cartesian
matrix was reduced from **48 to 14 cases**: eight check each field with a
missing value, five check the other invalid value classes, and one retains the
fixture-to-Act route. This deliberately drops the other per-field/per-invalid-
value combinations, so a future field-specific validator would need its own
cases. A local probe made ten small in-memory mutations (skip each of eight
fields, accept booleans, remove bounds); both the original 48 pairs and the
reduced field/value set detected all ten. This is limited mutation evidence,
not proof of semantic equivalence or measured full-suite speedup.

The TypeSafe live documentation was unavailable through the web tool during
this review, and no installed TypeSafe SDK was found; the installed skill and
repository System One code were used as fallback. No live Jev request was made.

### Skill activation, bindings, execution and appearances: four files

On 14 November, individually inspected all **58 original test functions**
in `test_skill_activation.py` (18), `test_agent_v2_skill_bindings.py` (19),
`test_skill_script_execution.py` (13), and `test_skill_appearances.py` (8).
These four files collected **90 cases** before this change. The historical CSV
remains a snapshot; this review does not turn its automated labels into
individual verdicts. Disposable fixtures are used throughout.

| File | Individual coverage and disposition |
| --- | --- |
| Activation | Retain: full-instruction budget, on-demand resources and receipts, seven distinct rejected paths, symlink escape and alias, discovery/read access recheck, changed/missing package, legacy privacy, bounded selection and prompt privacy, project-free direct reply, planning/invocation, missing-package failure, no eager resource bytes, actual executor dispatch, explicit resource failures, budget-before-inference, changed revision during inference, Bash access/revision recheck, and live cancellation. The selection/planning tests exercise the current V3 bridge; they are not proof that the new loop can do the same. |
| Bindings | Retain: explicit choice/sharing/revocation, three actor states, inactive owner, admin-only attachment, disabled/deleted/missing-secret account, lifecycle and data persistence, missing/retargeted data, four invalid data locations, missing dependency, nested and Bash redaction, Bash authorization, key selection/rotation, configuration with SQLite, environment isolation, cross-skill identity, cancellation redaction, and safe preflight. Consolidate the separate context-service identity assertion into the existing Bash authorization test, which now executes a real bound command using the core-generated execution context and verifies redaction. |
| Script execution | Retain: spaces and working directory independence, normal executable, six preflight failures, interpreter link and symlink escape, dependency diagnostics, selection/removal recheck, cancellation and partial output, write/resume/readback, four prior mutation states, failed command despite a positive judgment, project-free authenticated execution, timeout evidence, and child-process cancellation. These protect distinct production or bridge behaviors; the recovery fixture is not evidence of a qualified end-to-end candidate loop. |
| Appearances | Retain: install/update/removal fallback, 11 incompatible specifications, unsafe/corrupt/animated images, install/update link laundering, both shipped presets, explicit leading-skill selection, IPC authorization and active-task projection, and selection checkpoint before reply. These serve an existing Desktop feature; their priority is lower than conversation reliability, but similar assertions do not establish duplication. |

The consolidation removes **one collected case and one function**, leaving
**89 cases** in these files. It loses the standalone assertion that the exact
same service object is copied into the execution context; the remaining test
instead proves the service actually supports a bound command and redacts the
result through that context. It does not cover every production invocation
path. The other cases were kept because their inputs, boundary, failure mode,
or consumer differ; a high case count alone is not evidence of waste.
Validation: `.venv/bin/python -m pytest -q tests/test_skill_activation.py
tests/test_agent_v2_skill_bindings.py tests/test_skill_script_execution.py
tests/test_skill_appearances.py` — **89 passed in 2.80 seconds**. No model
calls or live household resources were used. This is not a full-suite run.

### Scheduled work and uncertain delivery: three files

On 14 November, individually inspected **49 test functions / 66 collected
cases** in `test_agent_v2_native_scheduled_task_tool.py` (19 functions),
`test_project_check_scheduling.py` (24), and
`test_scheduled_delivery_uncertainty.py` (6). These files were not part of
the preceding individual passes. All use synthetic data and fake or stubbed
providers; this does not establish real provider delivery or new-loop
qualification.

| File | Individual coverage and disposition |
| --- | --- |
| Native scheduled task | Retain the one-off and recurring stores, lifecycle and history, project migration, editable/deleted state, native registry and actor/project context, configured timezone, relative duration, missing prompt, registered Telegram address, do-node result, idempotent direct reminder, one-off and recurring runners, inactive/not-due runner, planning prompt, stubbed plan-to-do path, TUI result, and exhausted recurrence. The two PDCA-specific tests for planning and stubbed selection are lower-priority historical replay coverage, not evidence the model chooses correctly or the candidate loop supports reminders. Their integration boundary is not duplicated by the separate direct do-node and tool tests; revisit when the replay baseline is retired. |
| Project checks | Retain durable typed dispatch, four off-window offsets, six operation boundaries, retry/window cutoff, retry and pagination budgets after restart, claim recovery and concurrency, four authority changes, missed/coalesced runs, bounded lateness, overlap and policy isolation, end date and absent adapter, three DST examples, five invalid schemas, owner-scoped DST gaps, owner-scoped offline receipts, duplicate dispatch, unsupported stored version, in-flight lease expiry, concurrent in-memory claims, legacy prompt migration, acquisition budget, runtime deadline, and independently advancing acquisition/analysis. Similar tests protect different entry points or outcomes. |
| Uncertain delivery | Retain failed send after restart, expired lease, V3 receipt timeout, non-scheduled retry migration, scheduled prior-attempt migration, and scheduled-occurrence migration with matching receipt. These cases distinguish an unknown external effect from an ordinary retry and protect different stored records. They do not show that a provider actually sent or reconciled a message. |

No tests were removed in this pass: direct inspection did not establish a
duplicate whose loss was justified. One test's name overstates its coverage:
`test_byte_item_and_runtime_budgets_are_not_silently_complete` supplies 101
items and observes runtime accounting, but does not exceed the byte budget;
it is not evidence of byte-limit enforcement. The separate deadline test
checks runtime enforcement. This is a coverage gap to address in behavior
work, not a reason to claim a passing byte-limit regression.

Validation: `.venv/bin/python -m pytest -q
tests/test_agent_v2_native_scheduled_task_tool.py
tests/test_project_check_scheduling.py tests/test_scheduled_delivery_uncertainty.py`
— **66 passed in 2.37 seconds**. This was a focused run, not the full suite.

### Account choice, grants and OAuth: four files

On 14 November, individually inspected **57 test functions / 82 collected
cases** in `test_gmail_account_choice.py` (4 functions),
`test_account_grants.py` (26), `test_oauth_lifecycle.py` (23), and
`test_oauth_loopback.py` (4). They use disposable identities and credentials;
the loopback tests use a real localhost listener, while provider calls are
synthetic. None verifies a live household account or candidate-loop
end-to-end authorization.

| File | Individual coverage and disposition |
| --- | --- |
| Gmail choice | Retain authenticated answer with context changes, rejection of model-authored challenges, task/actor isolation, and supersession of old answers. These protect distinct receipt and authorization boundaries. |
| Account grants | Retain exact selection, secret-free handle, operation/resource/project/audience intersection, explicit project grant, restart rechecks, inactive actor/owner, connection and consent changes, disconnect, missing records, expiry, owner control, binding/grant intersection, revision/identity guards, package lifecycle, skill availability, legacy adoption, sharing removal, secure disconnect, redaction with in-flight revocation, locked vault, session identity, concurrent revisions, reserved ownership, tampered binding configuration, stale ACL, and explicit owner reconciliation. The actor, scope, state and storage mutations are independent failure modes, even where all produce denial. |
| OAuth lifecycle | Retain minimal scopes/health, malformed callback variants, state owner/expiry/replay, missing scope/refresh token, reconnect identity/revocation, account isolation, authority change during refresh, pause during callback, serialized rotation, revoked consent/redaction, offline revoke, crash lease, PKCE/library transport and logging, optional dependency failure, trusted daemon session, refresh uncertainty/persistence failure, locked local deletion, token redaction, pending cleanup, provider revoke transport, project-policy lifecycle, and calendar/classroom scopes. Mocked library requests validate request formation, not live provider behavior. |
| OAuth loopback | Retain callback/replay/log privacy, wrong path/state/consent, two provider paths on one listener, and non-loopback redirect rejection. The shared-listener case uses the same actor for both paths despite `keep_actors` in its name; it does **not** establish isolation between different actors. |

No tests were removed: direct inspection did not establish redundant behavior.
The initial sandboxed run reported **79 passed and 3 failed** because socket
binding to `127.0.0.1` was prohibited. A separate permitted localhost run of
`tests/test_oauth_loopback.py` passed **4 tests in 1.65 seconds**. The
first run used `.venv/bin/python -m pytest -q` on all four files; the second
used the same command on the loopback file outside that sandbox restriction.
Together these runs exercise all **82 cases**, but neither is a full-suite run.

### Gmail acquisition and unread count: two files

On 14 November, individually inspected **36 functions / 58 cases** in
`test_gmail_adapter.py` (19 functions) and `test_gmail_unread.py` (17).
All decisions below are **retain**. Names omit the `test_` prefix. Fixtures
use synthetic data and transports; no Gmail account or model is contacted.

| Function in `test_gmail_adapter.py` | Reason retained |
| --- | --- |
| `buried_archived_notice_pagination_and_stable_identity` | Archived content, page cursor, bounded scope, identity and untrusted text. |
| `query_or_cannot_escape_time_bound_and_scope_cursor_changes` | Query grouping and rejecting cursor scope changes. |
| `history_paginated_tombstones_then_bounded_query_refresh` | History pagination, deletion, refresh and anchor. |
| `expired_history_and_deleted_during_scan_are_honest` | Expired history and incomplete coverage on deletion. |
| `provider_errors_never_no_match_or_implicit_retry` | Four error classes retain blocked/partial semantics. |
| `revocation_fences_io_and_no_followup` | Revocation before and during provider IO. |
| `cancellation_between_list_and_message_get` | No subsequent message fetch after cancellation. |
| `malicious_html_is_inert_and_quotes_preserved` | Untrusted HTML sanitization and quoted content. |
| `attachment_only_evidence_requires_permitted_expansion` | Attachment evidence and authorized expansion. |
| `missing_or_disallowed_attachments_are_not_empty_success` | Four distinct missing, type, size and provider failures. |
| `direct_read_scope_and_no_write_surface` | Requested and returned label checks; no write API. |
| `limits` | Four independent pre-IO resource limits. |
| `gmail_failure_before_accept_replays_fetch_without_skipping` | Durable worker failure before evidence acceptance. |
| `gmail_crash_after_accept_keeps_evidence_and_cursor` | Recovery after acceptance with page cursor preserved. |
| `mime_charset_and_attachment_manifest_id` | Character encoding and attachment identity. |
| `capabilities_reauthorize_and_health_is_redacted` | Capability reauthorization and private health fields. |
| `fixed_get_transport_stream_budget_no_redirect_or_retry` | Four fixed transport outcomes and no redirect/retry. |
| `multiple_labels_history_keeps_original_anchor_and_separate_scope` | Independent label scopes and history anchor. |
| `explicit_all_mail_does_not_add_implicit_inbox_label` | Explicit all-mail scan and history without inbox restriction. |

| Function in `test_gmail_unread.py` | Reason retained |
| --- | --- |
| `scoped_count_is_one_label_request_without_message_content` | Message/thread counts and no content disclosure. |
| `no_grant_or_project_only_grant_cannot_read_one_off` | A project grant is not ad-hoc read permission. |
| `label_scope_narrow_and_all_mail_grant` | Label-specific, all-mail and path validation. |
| `multiple_accounts_are_choice_not_implicit_selection` | Multiple accounts require a choice. |
| `actor_cannot_use_someone_elses_grant_or_invent_account` | Other actor cannot access the grant/account. |
| `revocation_during_io_and_cancellation_fence_response` | Revocation and cancellation after IO withhold result. |
| `provider_failures_do_not_fabricate_zero` | Eight error/malformed-response cases never fabricate count. |
| `native_tool_uses_authenticated_task_not_arguments` | Authenticated task supplies identity. |
| `grant_beneficiary_can_read_but_other_accounts_are_not_exposed` | Explicit beneficiary can read; owner without grant cannot. |
| `inflight_byte_budget_never_returns_a_count` | Oversize in-flight response cannot become a count. |
| `tool_cancellation_and_missing_service_are_safe` | Missing service and cancellation fail safely. |
| `fixture_registered_gmail_tool_curation_reveal_and_tactical_execution` | Scripted V3 registry/curation/execution integration; explicitly not live model accuracy. |
| `choice_candidates_are_reauthorized_before_disclosure` | Revocation between discovery and disclosure. |
| `group_channel_and_unproven_audience_cannot_read_mail` | Discord and Telegram destination privacy is not assumed. |
| `only_server_issued_authenticated_choice_selects_between_accounts` | Authorized question and answer actually select a mailbox. |
| `account_choice_is_invalidated_by_grant_revocation` | Revocation invalidates prior choice and prevents fallback. |
| `private_audience_revoked_during_io_withholds_count` | Destination privacy change during IO withholds result. |

No removals or consolidations were supported by direct inspection: even where
the two files discuss the same grant or provider, one tests source acquisition
and the other tests the count service or native-tool boundary. The V3 fixture
manually registers a tool whose production registration is disabled; it does
not prove normal conversations can use it. The cancellation test's guard
depends on a count of internal checks and may be brittle; replace that timing
mechanism with an observable boundary before relying on it as durable coverage.
Validation: `.venv/bin/python -m pytest -q tests/test_gmail_adapter.py
tests/test_gmail_unread.py` — **58 passed in 1.15 seconds**, not a full-suite run.

## Mission coverage and next review passes

Keep behavior that protects real household outcomes while the relevant feature
exists: actor-scoped privacy and authorization; skills and credential binding;
bounded invocation and cancellation; factual evidence and truthful completion;
data/history retention and migration; stale-claim fencing; restart and unknown
external effects; and actual delivery/user-facing integration. Preserve V3
bridge checks until a qualified replacement is in production. A claimed or
synthetic success is not a delivery receipt, and scripted gym results are not
evidence of autonomous reliability.

The current gap is not more isolated matrix cases: the first fenced
conversation has not crossed the real daemon, candidate and delivery
boundaries. The acceptance target is in `first-fenced-conversation-slice.md`.
For review, work through the CSV one area at a time, record the concrete user
behavior/failure each function detects and its remaining consumer, then mark
retain, consolidate, retire with dependency, or remove. A confident removal
needs observed overlapping coverage or an explicit decision to abandon the
behavior, not merely a similar test name. Run an appropriate targeted check
after each removal batch; retain the full suite until an independently checked
fast selection and comprehensive job exist.

For runtime context, the preceding audit measured one full run at 181.60 s
(3,890 passed, 37 skipped) before subsequent changes. It is not a baseline for
this exact inventory snapshot. The largest collected files or matrices need
not be the slowest or least valuable; see `test-suite-value-audit-2026-11.md`.
