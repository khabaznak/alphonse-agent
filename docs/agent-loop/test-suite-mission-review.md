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
   implementation was subsequently removed as part of V2 retirement; its
   public import is no longer available. This is not proof of zero risk.
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

### Calendar and Classroom acquisition: two files

On 14 November, individually inspected **32 functions / 59 cases** in
`test_calendar_acquisition.py` (18 functions) and
`test_classroom_adapter.py` (14). All decisions below are **retain**. Names
omit the `test_` prefix. Synthetic provider responses and disposable accounts
exercise adapter and worker behavior, not real provider semantics or a
qualified new-loop conversation.

| Function in `test_calendar_acquisition.py` | Reason retained |
| --- | --- |
| `recurrence_exception_identity_dst_and_all_day_dates` | Recurring identity, exceptions, DST, all-day values and scoped read. |
| `paginated_baseline_and_delta_anchor_cancellation_and_move` | Cursor progression, cancelled and moved event behavior. |
| `expiry_and_changed_scope_require_bounded_resync` | Expiry and project/resource/window scope changes. |
| `revoked_calendar_and_provider_failure_never_absence` | Five provider errors stay blocked/partial, not absent. |
| `grant_revocation_fences_results` | Revocation before and during IO. |
| `partial_redacted_events_and_missing_token_are_not_empty_success` | Redacted data and absent sync token do not imply complete coverage. |
| `discovery_consent_is_independent_of_mail_and_event_reads` | Discovery consent and read permissions remain separate. |
| `direct_read_exact_resource_identity_and_window` | Exact resource, event identity, window and read-only surface. |
| `budgets_before_io` | Four distinct resource limits prevent IO. |
| `multiple_selected_calendars_keep_separate_tokens` | Independent per-calendar sync cursors. |
| `fixed_transport_bounds_and_no_credential_forwarding` | Fixed URL, streaming, bounded data, timeout and no redirect following. |
| `calendar_worker_persists_page_before_cursor_and_recovers` | Durable acceptance and cursor after a simulated crash. |
| `calendar_worker_failure_coverage_is_never_no_match` | Expired, redacted and revoked worker outcomes. |
| `calendar_and_mail_references_reconcile_without_completing_assignment` | Cross-source evidence, incomplete assignment and revoked access. |
| `all_day_exclusive_end_uses_calendar_zone_at_window_boundary` | Calendar timezone at an exclusive all-day boundary. |
| `cancellation_and_window_mismatch_prevent_io` | Cancellation, mismatched window and non-exact ID deny IO. |
| `busy_only_visibility_keeps_coverage_partial` | Two visibility restrictions remain partial. |
| `incremental_page_size_remains_stable_as_budget_shrinks` | Pagination parameters stay stable across a shrinking budget. |

| Function in `test_classroom_adapter.py` | Reason retained |
| --- | --- |
| `student_role_checked_before_content_and_unknowns_preserved` | Role before content and unknown assignment fields. |
| `guardian_only_or_school_restricted_role_never_reads_content` | Three missing/restricted role responses stop before content. |
| `school_restriction_revoked_consent_and_unavailable_are_explicit` | Five provider failures preserve blocked/partial outcomes. |
| `revocation_and_cancellation_fence_io` | Revocation and cancellation prevent unauthorized continuation. |
| `repeated_corrected_and_missing_assignment_fields` | Stable and corrected versions, with missing fields unknown. |
| `announcements_require_explicit_operation_and_scope` | Announcement-specific operation and consent. |
| `pages_scope_cursor_expiry_and_refresh_preserve_original_lower_bound` | Paging, changed scope and lower-bound preservation. |
| `multiple_courses_have_independent_truthful_coverage` | Per-course pending and completed coverage. |
| `limits_before_io` | Four budget and deadline cases stop IO. |
| `read_scope_identity_and_no_write_or_impersonation_surface` | Scoped read, no attachment, no write/impersonation surface. |
| `transport_fixed_get_bounded_stream_and_no_redirect` | Fixed endpoint, streamed budget, timeout and redirect rejection. |
| `durable_crash_resume_and_corrected_assignment_versions` | Resume after acceptance, revised evidence and publication. |
| `durable_replay_deduplicates_same_assignment_version` | Repeated identical evidence is deduplicated. |
| `permission_denied_worker_never_reports_no_match` | Provider denial remains blocked, not a negative finding. |

No test was removed: similar budget and transport cases run different provider
implementations, and the durable recovery tests use distinct adapters and
evidence. The calendar transport test demonstrates that redirects are disabled
and only one fixed-endpoint request is made; it does not inspect an actual
redirect destination or prove every credential-forwarding path despite its
name. The Classroom role test models missing/restricted student responses,
not an actual guardian account or live school policy. In the parametrized
Classroom revocation test, the identical cancellation check runs twice; that
small duplicate setup is a candidate for tidying, but removing a case would
also remove a distinct pre/during-revocation scenario.

Validation: `.venv/bin/python -m pytest -q tests/test_calendar_acquisition.py
tests/test_classroom_adapter.py` — **59 passed in 1.30 seconds**, not a
full-suite run.

### Source acquisition and findings: two files

On 14 November, individually inspected **36 functions / 57 cases** in
`test_source_acquisition.py` (19 functions) and `test_findings.py` (17).
All decisions below are **retain**. Names omit the `test_` prefix. These use
disposable accounts, synthetic source pages and delivery stores; they are not
evidence of a real provider effect or a qualified simple-loop conversation.

| Function in `test_source_acquisition.py` | Reason retained |
| --- | --- |
| `bounded_context_durable_page_and_independent_model_failure` | Scoped context, persisted evidence and independent failed analysis. |
| `quiet_no_match_and_partial_is_never_no_match` | Completed empty acquisition has no-match outcome and no event. |
| `incomplete_and_expired_cursor_bounded_resync` | Two partial/expired cursor paths preserve coverage gaps. |
| `restart_after_atomic_accept_never_skips_or_replays_page` | Recovery after page and cursor acceptance. |
| `transaction_rolls_back_evidence_and_cursor_together` | Storage failure rolls back page and stream changes. |
| `publication_crash_replay_one_scoped_event` | Crash after journal write, replay deduplication and scoped event. |
| `adapter_scope_and_budget_violations_are_not_success` | Four adapter scope, budget and stalled-cursor violations. |
| `revocation_prevents_next_page_and_private_expansion` | Revocation stops another fetch and evidence disclosure. |
| `attachment_limits_window_and_grant_enforced` | Authorized attachment, byte accounting, wrong resource and window. |
| `inert_html_and_quoted_evidence` | Sanitization, quoted evidence, link safety and size. |
| `processing_outside_window_cannot_manufacture_completion` | Processing checkpoints cannot advance outside the window. |
| `query_and_folder_revision_start_new_baseline` | Changed query causes a new bounded scan. |
| `crash_after_final_page_recovers_without_another_read` | Final-page crash recovery does not fetch again. |
| `incremental_cursor_then_expiry_and_adapter_version_baseline` | Incremental cursor, expiry and subsequent version change. |
| `content_removal_leaves_only_provenance` | Expiry, revocation and source deletion remove content. |
| `folder_change_uses_new_bounded_baseline` | Changed authorized folder gets a new cursor and scope. |
| `adapter_version_change_between_pages_never_reuses_old_cursor` | Mid-scan adapter change discards old page cursor. |
| `failed_acquisition_publishes_failure_without_private_error` | Failure event excludes provider diagnostic. |
| `subscription_before_first_result_routes_existing_event_execution` | Existing subscription gets the first scoped event. |

| Function in `test_findings.py` | Reason retained |
| --- | --- |
| `persistence_scope_provenance_and_reference_only_checkpoint` | Durable provenance, reference-only checkpoint and actor/audience access. |
| `reported_sources_cannot_manufacture_verified_outcomes` | Four outcome predicates cannot be inferred as verified. |
| `sender_statements_remain_reports` | Three sender assertions stay reports, not verified facts. |
| `missing_due_date_timezone_and_grade_units` | Unknown dates/timezone and explicit grade units. |
| `correction_duplicate_and_old_quote_create_one_alert_per_version` | Corrections, duplicate evidence, old quotes and alert versioning. |
| `unrelated_and_ambiguous_identity_never_merge` | Five identity changes cannot merge unrelated commitments. |
| `replay_after_checkpoint_crash_recovers_without_new_version` | Checkpoint crash replay avoids duplicate finding/delivery. |
| `ambiguous_send_is_durable_and_never_automatically_retried` | Exception and expired lease preserve unknown delivery. |
| `revocation_deletion_propagates_all_versions_and_local_delivery` | Five authority/retention changes gate and purge local versions. |
| `historical_version_revalidates_its_own_retention` | Historical record honors its own retention deadline. |
| `restored_evidence_cannot_override_current_revocation` | Stale restored evidence cannot override current authority. |
| `unwired_outbox_consumer_fails_closed` | Unconfigured consumer cannot claim protected outbox item. |
| `partial_coverage_remains_explicit` | Partial source coverage reaches the notification. |
| `recipient_delivery_and_ambiguity_are_independent` | Separate recipients retain separate delivery states. |
| `retained_findings_preserve_history_after_cache_expiry_or_pause` | Expired cache and paused policy preserve appropriate history. |
| `independent_accounts_reconcile_one_finding_with_separate_provenance` | Two source accounts reconcile while retaining provenance. |
| `forged_evidence_and_cross_actor_cannot_record` | Forged evidence and different actor cannot create a finding. |

No removals were justified: apparently similar cases exercise different
transaction, identity, revocation, or lifecycle boundaries. Two names
overstate individual assertions: `quiet_no_match_and_partial_is_never_no_match`
checks the no-match branch; the next parametrized test checks partial coverage.
`query_and_folder_revision_start_new_baseline` changes the query only; the
separate folder-change test covers folders. The attachment/window test uses
`pytest.raises(Exception)` for the expired window, so an unrelated exception
could satisfy that assertion; it is weaker evidence of the intended reason.
Its other checks and the separate processing-window test remain useful.

Validation: `.venv/bin/python -m pytest -q tests/test_source_acquisition.py
tests/test_findings.py` — **57 passed in 7.02 seconds**, not a full-suite run.

### Conversation history and session routing: three files

On 14 November, individually inspected **30 test functions / 31 collected
cases** in `test_agent_v2_conversations.py` (16 functions),
`test_agent_v2_memory_sessions.py` (7), and
`test_agent_v2_project_sessions.py` (7). The project-session cross-channel
question case is parameterized over two source channels. These are synthetic
store/router and V3-bridge checks, not evidence that the simple-loop candidate
can resume questions or deliver a household response.

| Conversation function suffix | Disposition and behavior |
| --- | --- |
| `channel_records_project_user_turn_once` | Retain: duplicate inbound message ID records one turn. |
| `desktop_history_combines_all_projects_for_one_local_day_in_order` | Retain: Desktop combines project, projectless, and cross-channel records in observed order. |
| `desktop_history_refresh_sees_telegram_turns_without_daemon_restart` | Retain: a post-first-read Telegram turn becomes visible without restart. |
| `desktop_history_uses_users_timezone_at_day_boundary` | Retain: UTC timestamps fall into the correct local dates. |
| `user_seen_cursor_and_history_survive_store_restart` | Retain: persisted read marker and history survive reopening. |
| `seen_cursor_is_scoped_to_user_and_local_day` | Retain for separate local-day cursors; the fixture has only one user, so the name does **not** demonstrate cross-user isolation. |
| `blank_project_migration_keeps_unassociated_history_unassociated` | Retain: startup migration preserves projectless and associated scope. |
| `startup_imports_legacy_user_and_project_ledgers_once_with_original_scope` | Retain: startup migration retains project scopes and avoids repeat imports. |
| `timestamps_are_canonical_utc_and_sort_true_chronological_order` | Retain: offset normalization and chronological order. |
| `same_timestamp_order_is_stable` | Retain: stable tie ordering and sequence on repeated reads. |
| `project_seen_cursor_is_authoritative_and_project_scoped` | Retain: seeing one project does not mark a second project seen. |
| `legacy_sequence_migration_initializes_existing_history_as_seen` | Retain: migrated sequence/order and subsequent unseen event. |
| `store_startup_migrates_legacy_offset_timestamps_idempotently` | Retain: old timestamps normalize across two restarts. |
| `legacy_ledger_recovers_only_visible_turns` | Retain: import excludes internal plan/tool entries. |
| `desktop_history_imports_legacy_ledger_only_once` | Retain: import idempotence and unknown-date handling; remove its final `repeated == history` assertion because the earlier `history == repeated == []` already asserts it. No collected case was removed. |
| `conversation_store_searches_memory_across_projects` | Retain for matching the requested text; only one project has a match, and the other owner's record does not match the query. It does **not** prove cross-project search or cross-owner isolation. |

| Memory-session function suffix | Disposition and behavior |
| --- | --- |
| `bounded_context_never_loads_full_session_ledger` | Retain: bounded projection and truncation metadata while stored ledger is longer. The test inspects returned context and file length, not actual disk-read volume. |
| `session_close_keeps_historical_view_without_promoting_project_memory` | Retain: stored summary remains session-scoped. |
| `router_stamps_selected_session_before_queueing_and_supports_commands` | Retain: explicit-project session selection and cross-channel binding, with ordinary conversation projectless. |
| `legacy_ledgers_are_archived_once_without_promoting_prose` | Retain: one-time archive without creating a project summary. |
| `memory_session_store_keeps_bindings_per_project_and_rejects_closed_sessions` | Retain: distinct project bindings and closed-session behavior. |
| `operational_task_prompt_omits_memory_context` | Retain: explicit prompt option excludes memory bytes. |
| `memory_setting_persists_configurable_context_budget` | Retain: persisted budget survives store reopen. |

| Project-session function suffix | Disposition and behavior |
| --- | --- |
| `project_session_never_routes_ordinary_messages_or_falls_back_to_home` | Retain: selected session does not force ordinary messages into a project across channels/threads. |
| `explicit_project_context_is_authorized_and_projectless_task_is_valid` | Retain: explicit accessible project, projectless request and denied actor. |
| `project_commands_are_deterministic_and_do_not_queue_capd_work` | Retain: commands update selected project and do not queue model work. |
| `project_context_mutation_requires_owner_and_unknown_slash_bypasses_capd` | Retain: member cannot change owner context and unknown command does not invoke model work. |
| `project_commands_do_not_grant_admins_access_to_other_users_private_projects` | Retain: listing/selection enforce owner and shared membership even for an admin. |
| `cross_channel_reply_resumes_pending_question_with_original_task_context` | Retain both channel variants: an answer is correlated and retains original task/session context; it does not test simple-loop or actual message delivery. |
| `explicit_desktop_project_request_is_new_work_and_checks_access` | Retain: explicit Desktop request bypasses pending-question routing and checks project authority. |

Validation: all **31 cases passed** after the redundant assertion removal.
Tests were run with a temporary pytest isolation plugin redirecting the
default legacy users root to a disposable directory and with a temporary empty
skills path, avoiding household storage; fixtures that explicitly provide a
test users root keep it. This focused run is not a full-suite validation. The
plugin changes the default-root environment and therefore does not itself
verify the real default users-root configuration.

### Questions and visible follow-through: three files

On 14 November, individually reviewed **25 test functions / 31 collected
cases** in `test_agent_v2_ask_question_ui.py` (5 functions),
`test_agent_v2_native_ask_question_tool.py` (8), and
`test_desktop_questions.py` (12). Questions are a current shared user-facing
boundary even though some tests drive retired PDCA. Disposable fixtures and
scripted processors do not establish model judgment or simpler-loop delivery.

| Question UI function | Review decision and behavior |
| --- | --- |
| `tui_renders_yes_no_question_interrupt` | Retain: renders the supported yes/no response format. |
| `tui_answer_resumes_pending_question` | Retain: queues the matching question ID and normalized answer. |
| `tui_invalid_choice_answer_does_not_resume` | Retain: rejects an invalid choice without queueing. |
| `ag_ui_adapter_emits_interrupt_run_finished_with_snapshots` | Retain: interrupt identity and boolean response schema reach the AG-UI consumer. |
| `ag_ui_resume_resolves_question_through_store` | Retain: resolved interrupt maps to the corresponding resumed task. |

| Native question function | Review decision and behavior |
| --- | --- |
| `native_registry_registers_ask_question_tool` | Retain: active native descriptor exposes the expected question schema and interrupt capability. |
| `question_project_migration_backfills_checkpoint_idempotently` | Retain: legacy migration preserves project and survives restart. |
| `do_node_executes_ask_question_and_parks_task` | Retain for historical PDCA replay: tool invocation creates a pending question; this does not test the selected simple loop. |
| `question_answer_resumes_serialized_task_state` | Retain: answer routing updates the matching stored task and execution result. |
| `question_store_normalizes_multi_choice_and_datetime_answers` | Retain: choice labels and timezone conversion are distinct forms. |
| `multiple_pending_questions_are_ambiguous_without_direct_reference` | Retain: no arbitrary continuation when several questions match. |
| `pdca_processor_returns_parked_status_for_waiting_question` | Retain for historical PDCA replay: processor-to-UI parked event; not current simple-loop coverage. |
| `core_parked_result_leaves_loop_available` | Retain: parked processing returns the shared core to available; its processor is retired PDCA, so it cannot establish current V3 or simple-loop execution. |

| Desktop question function | Review decision and behavior |
| --- | --- |
| `desktop_rejects_wrong_respondent_without_impersonating_owner` | Retain: wrong respondent cannot answer or queue work. |
| `desktop_cancel_requires_respondent` | Retain: separate cancel authorization. |
| `question_answer_queues_exact_checkpoint_once` | Retain five cases: form-specific payloads, exact resume identity, duplicate answer, and Desktop surface removal. |
| `poll_reconciles_terminal_questions_and_reconnect` | Retain three cases: expire, cancel and external answer each clear stale surfaces. |
| `desktop_missing_actor_and_stale_context_do_not_authorize` | Retain: missing actor and unavailable context do not grant authority. The tested member poll is rejected, but this is not a complete multi-user privacy study. |
| `question_context_checks_current_artifact_access_and_availability` | Retain: delegated resource availability, revoked sharing and respondent ownership. |
| `missing_checkpoint_does_not_report_answer_success` | Retain: missing stored task cannot become a successful answer. |
| `ipc_answer_runs_resumed_continuation_once` | Retain: daemon consumes one IPC-resumed task and does not rerun it on duplicate answer, with a scripted fixture processor and disposable file. |
| `native_question_context_schema_persists_explicit_context` | Retain: native context persists and invalid HTML-shaped data is rejected. |
| `question_cannot_resurrect_terminal_checkpoint` | Retain: stale question cannot restart a cancelled checkpoint. |
| `question_delivery_and_history_share_identity_and_occurrence_time` | Retain: simulated outbox, Desktop acknowledgement, history identity and restart consistency. This is not external recipient delivery. |
| `delayed_question_delivery_stays_on_question_day` | Retain: question chronology despite next-day outbox record. |

No function or case was removed: overlaps across interfaces are not equivalent
entry points or outcomes, and three older tests remain explicitly scoped to
PDCA. Focused validation with a temporary pytest plugin redirecting fallback
users and skills paths to per-test disposable directories: **31 passed in
0.70 seconds**. This does not test configured household roots or a real model,
channel, or new-loop resume.

### Conversation continuity and bounded memory formatting: two files

On 14 November, individually reviewed all **17 functions / 21 collected
cases** in `test_memory_continuity.py` (11 functions) and
`test_memory_formatting.py` (6 functions, including five budget cases).
These use invented data and in-memory or temporary SQLite stores; they do not
execute a live model or the new conversation host.

| Function in `test_memory_continuity.py` | Decision and behavior |
| --- | --- |
| `complete_exchange_omission_and_exact_manifest` | Retain: old conversation exchange omitted together, both omissions identified, stored content preserved. |
| `recent_exchanges_preserve_retry_pronouns_and_channel_changes` | Retain: recent context accompanies short follow-ups across project and channel changes. |
| `completed_history_does_not_accumulate_and_repeated_yes_remains_distinct` | Retain: bounded recent history, repeated identical user messages and source/omission counts. |
| `multiline_steering_is_one_checkpoint_record` | Retain: multiline steering survives checkpoint as one indivisible record and is entirely omitted if over budget. |
| `legacy_markdown_is_omitted_as_one_unit` | Retain: oversized legacy transcript does not leave fragments beside new steering. |
| `tiny_budgets_never_slice_any_record` | Retain: exhaustive small transcript budgets do not return a partial message. |
| `question_answer_and_constraints_survive_restart_and_compaction` | Retain: SQLite restart, routed answer and user constraints remain available in continuity context. |
| `multiple_tasks_require_explicit_answer_target_and_owner` | Retain: ambiguity, wrong-owner refusal, correct question routing and unrelated pending task. |
| `pending_question_and_failed_evidence_do_not_become_success` | Retain: pending and failed states appear honestly and do not leak into another task. |
| `overflow_and_unavailable_history_are_explicit` | Retain: reserved continuity overflow and history backend failure are surfaced in separate projection flags. |
| `downstream_section_budget_never_cuts_a_question` | Retain: downstream section joining omits an indivisible question/answer when it cannot fit. |

| Function in `test_memory_formatting.py` | Decision and behavior |
| --- | --- |
| `every_memory_budget_keeps_whole_records` | Retain five budgets: bounded output, whole claims, and explicit omissions across several sizes. They are size variants of one behavior, not five separate features. |
| `v3_and_desktop_preserve_record_status_and_citations` | Retain: actual V3 context and desktop snapshot consumers carry disputed status and source citations. It does not establish candidate-loop integration. |
| `final_composed_prompt_does_not_recut_complete_memory_records` | Retain: final V3 prompt composition after a long preamble does not slice previously bounded memory claims. |
| `unstructured_oversized_memory_is_omitted_instead_of_sliced` | Retain: oversized unstructured prose is omitted, including an extremely small budget. |
| `nested_limits_and_tiny_remaining_space_never_emit_partial_claims` | Retain: repeated bounding and small remainder after a prior section avoid partial claims. |
| `request_tail_cannot_forge_a_memory_boundary_during_shortening` | Retain: a forged memory-like line in a later request is not treated as a durable memory record. |

No test was removed: similar whole-record checks operate at different
boundaries, and individual inspection did not establish a duplicate whose
deletion preserves those behaviors. Some tests rely on rendered strings rather
than an end-to-end model decision; a passing string check does not establish
that Alphonse will use the information correctly. Validation:
`.venv/bin/python -m pytest -q tests/test_memory_continuity.py
tests/test_memory_formatting.py` — **21 passed in 0.25 seconds**.

### Skill management and distribution: two files

On 14 November, individually reviewed **8 functions / 9 collected cases** in
`test_skill_management.py` (7 functions) and `test_skill_distribution.py`
(one function, clean/bytecode-polluted builds). These use synthetic skills in
temporary directories; a temporary pytest plugin redirects the default legacy
user root to disposable storage. They cover the existing package-management
contract, not an end-to-end new-loop skill execution.

| Function | Decision and behavior |
| --- | --- |
| `customize_conflict_update_export_uninstall` | Retain: customization, upstream conflict, explicit replacement, recovery copy, binary export, uninstall and preservation of colocated legacy data. Its broad workflow covers several state transitions. |
| `binary_metadata_and_path_authority` | Retain: binary editing denial, valid/invalid metadata edit, protected definition, traversal/absolute path and symlink refusal. |
| `export_refuses_conventional_private_state` | Retain: private file blocks export and replacement while remaining unchanged. |
| `ipc_management_authorization_and_readiness` | Retain: cross-actor and member IPC denial, install/update/customize and binary read, missing dependency projection, and binding retained across deletion. It checks IPC/store behavior, not whether the new loop invokes a skill. |
| `invalid_update_keeps_active_revision_and_bundled_updates_are_visible` | Retain: invalid upstream package leaves active revision unchanged and bundled changes are reported. |
| `harness_state_cannot_escape_by_symlink` | Retain: state-directory symlink prevents customization without creating a package. |
| `large_asset_import_and_legacy_link_diagnostics` | Retain: imported large binary remains non-text and a legacy symlink is reported as invalid. |
| `installed_distribution_includes_nested_skill_resources` | Retain both clean and bytecode-polluted source builds: sdist→wheel→install retains nested resources, executable mode, starter packages, and excludes bytecode. This checks packaging; it does not prove real household activation. |

No removal was justified: these checks cover different product entry points,
privacy boundaries or package lifecycle states. The two distribution cases
are variations of one packaging behavior, not two distinct product features.
Validation: `PYTHONPATH=/private/tmp:. .venv/bin/python -m pytest -q -p
alphonse_audit_isolation_plugin tests/test_skill_management.py
tests/test_skill_distribution.py` — **9 passed in 4.12 seconds**. The
isolation plugin is temporary and is not a product dependency or retained
test fixture.

### Memory migration and source contracts: two files

Individually reviewed **22 functions / 29 cases** in `test_memory_migration.py`
(9) and `test_memory_contracts.py` (13). All use synthetic data in temporary
or in-memory stores. These exercise shared memory infrastructure and do not
establish that the candidate loop can discover or use that infrastructure.

| Function in `test_memory_migration.py` | Disposition and specific value |
| --- | --- |
| `dry_run_is_read_only_and_apply_is_additive_idempotent` | Retain: original bytes, idempotence, private unverified provenance and candidate identity. |
| `partial_migration_adopts_identical_copy` | Retain: interrupted migration can reuse an identical archive without duplication. |
| `conflicts_fail_before_any_copy_and_preserve_both` | Retain: divergent archive aborts before mutations. |
| `receipt_never_guesses_changed_or_ambiguous_state` | Retain five cases: changed original/archive, missing original, owner mismatch and future receipt version. |
| `migration_rejects_symlink_sources_and_destinations` | Retain: symlinked input and archive destination cannot escape the scope. |
| `backup_restore_includes_wal_and_never_overwrites_new_data` | Retain: consistent WAL backup, separate restore and destination protection. |
| `corrupt_backup_and_unsafe_destinations_fail_closed` | Retain: nested backup rejection and tampered checkpoint refusal. |
| `restore_does_not_accept_checkpoint_changed_during_copy` | Retain: catches changes between validation and the copied snapshot. |
| `repeated_migration_preserves_canonical_events_knowledge_and_daily_views` | Retain: repeated migration does not duplicate canonical events or materialized daily views. |

| Function in `test_memory_contracts.py` | Disposition and specific value |
| --- | --- |
| `contract_roundtrip_versions_and_derived_audience` | Retain: serialization, audience restriction and invalid-version/visibility rejection. |
| `read_recall_and_assembly_share_identity_across_midnight_and_projects` | Retain: source identity is stable across transcript, recall and direct read. |
| `visibility_is_filtered_before_any_backend_content_is_rendered` | Retain: adversarial backend cannot leak a private record and duplicate shared results are collapsed. |
| `legacy_import_never_invents_date_speaker_or_channel` | Retain: idempotent import with honest missing provenance. |
| `additive_migration_preserves_original_events_and_ambiguous_imports` | Retain: old SQLite events survive restart without fabricated observation or unseen-state changes. |
| `rebuild_and_erasure_preserve_ids_remove_all_generated_copies` | Retain: owner-only erasure, projection cleanup and no resurrection on redelivery/restart. |
| `erasure_projection_failure_leaves_no_stale_generated_text` | Retain: deletion behavior when derived-view rebuild fails. |
| `recall_envelope_distinguishes_empty_and_unavailable_backends` | Retain: empty search is distinct from backend failure. |
| `derived_memory_rechecks_current_sources_after_erasure` | Retain: cached derived knowledge cannot outlive source erasure. |
| `task_evidence_projection_is_owner_scoped_and_preserves_attempts` | Retain: failed and successful attempts remain separate and owner-scoped. |
| `historical_reference_cannot_preserve_revoked_sharing` | Retain: a stale shared-source reference cannot override current revocation. |
| `recording_an_everyday_message_requires_no_project` | Retain: ordinary conversation has no mandatory project boundary. |
| `malformed_recipients_are_never_interpreted_as_grants` | Retain four malformed recipient types at contract, projection and stored-read boundaries. |

No test removal was justified: similar migration, visibility and provenance
checks use different entry points or failure states. The erasure-failure test
does not assert that a daily file existed before injecting the failure, so it
does not independently establish removal of an already written stale file.
Validation:
`.venv/bin/python -m pytest -q tests/test_memory_migration.py
tests/test_memory_contracts.py` — **29 passed in 0.25 seconds**. No household
data or model calls were used.

### Coding-skill configuration and delegation: two files

Individually reviewed **17 functions / 20 cases** in
`test_coding_skill_runtime.py` (4) and `test_coding_skill_service.py` (13).
These use synthetic packages and providers; they do not run a real container,
model or personal household program, and their V3 phase test does not establish
that the simpler loop can use the same capability.

| Function in `test_coding_skill_runtime.py` | Disposition and specific value |
| --- | --- |
| `factory_requires_explicit_immutable_image_and_persistent_household` | Retain: disabled without pinned image, shared provider/verifier runner, reject ephemeral workspace. |
| `factory_rejects_mutable_image_and_symlinked_root` | Retain: rejects mutable image and escaping workspace root. |
| `persistent_image_setting_is_validated` | Retain: configured digest roundtrip, rejection and clearing. |
| `misconfigured_optional_delegation_does_not_block_runtime` | Retain: unavailable delegation is surfaced without breaking unrelated runtime. |

| Function in `test_coding_skill_service.py` | Disposition and specific value |
| --- | --- |
| `fixture_provider_creates_verified_skill_that_can_be_used_and_replayed` | Retain: verify/install/use, idempotent replay and changed-package reconciliation. |
| `native_tool_is_only_registered_for_explicitly_configured_service` | Retain: registration, task activation, real fixture Bash use and mismatched store rejection. |
| `phase_builds_activates_and_uses_skill_for_original_task` | Retain: scripted V3 phase builds and uses capability for original task; not live model behavior. |
| `runtime_refresh_retains_explicitly_configured_coding_service` | Retain: configured tool survives runtime refresh. |
| `provider_and_verification_are_required_before_reserving_work` | Retain: provider absence and unauthorized actor reject the request. The test does not independently demonstrate a verifier-only failure. |
| `failed_or_revoked_work_never_publishes` | Retain four cases: invalid package, failed verification, cancellation and revoked access leave no installation. |
| `package_cannot_change_during_verification` | Retain: package mutation between generation and verification cannot publish. |
| `package_symlink_is_rejected_before_verification` | Retain: generated link is rejected and not installed. |
| `lost_publication_receipt_requires_reconciliation_and_no_replay` | Retain: installed effect with missing receipt requires reconciliation rather than repeat. |
| `reconcile_stops_both_containers_without_replaying_or_claiming_success` | Retain: unauthorized actor refused and both simulated containers stopped. |
| `reconcile_keeps_job_running_if_container_removal_unconfirmed` | Retain: unresolved cleanup does not become success. |
| `native_reconciliation_requires_owner_and_original_task` | Retain: successful native reconciliation targets original task; the test does not itself check a wrong owner or wrong target. |
| `unconfirmed_provider_cleanup_preserves_running_job_for_reconciliation` | Retain: uncertain cleanup prevents re-execution. |

No case was removed: apparent overlap uses different APIs, actors or lifecycle
states. The four-case failure test accepts several exception classes, so it
establishes no publication but not that each scenario raised its intended
reason. Validation: with a temporary pytest plugin redirecting the legacy user
root into disposable storage, `.venv/bin/python -m pytest -q
tests/test_coding_skill_runtime.py tests/test_coding_skill_service.py` —
**20 passed in 0.38 seconds**. No live account, container or household data was
used.

### Household users and deletion recovery: two files

On 14 November, individually reviewed **25 functions / 25 cases** in
`test_agent_v2_users.py` (13) and `test_user_deletion_recovery_order.py` (12).
These exercise user, project, filesystem and daemon infrastructure shared by
the current bridge; they do not demonstrate that the candidate loop integrates
with these lifecycle operations. All use disposable records and paths.

| User function suffix | Disposition and behavior |
| --- | --- |
| `user_timezone_persists_when_store_is_reopened` | Retain: timezone survives database reopening. |
| `timezone_can_be_configured_per_user_with_global_fallback` | Retain: distinct user override and fallback. |
| `automation_preferred_channel_copy_setting_defaults_off_and_persists` | Retain: default and persisted preference; no dispatch is exercised. |
| `onboarding_creates_uuid_admin_profile_and_context` | Retain: initial administrator identity, profile and state. |
| `workspace_root_is_required_state_and_migrates_user_data_idempotently` | Retain: workspace changes preserve profile and daily data across repeated migration; it checks a copy remains, not every crash point. |
| `workspace_migration_moves_managed_projects_and_keeps_imported_roots` | Retain: managed content moves while an imported external root stays in place. |
| `v2_address_mapping_resolves_without_v1_identity` | Retain: distinct inbound and outbound channel-to-user mappings. |
| `user_aliases_are_unique_and_limited_to_two` | Retain: normalization, ownership conflict and limit. |
| `non_admin_profile_projects_and_membership_are_isolated` | Retain: owner privacy, shared visibility and member write denial. |
| `onboarding_migrates_legacy_local_projects` | Retain: daemon transfers legacy project ownership during onboarding. |
| `hard_delete_removes_user_profile_projects_and_addresses` | Retain: daemon removes a user, owned project and profile. Despite its name, the test does not independently inspect the address table after deletion. |
| `daemon_approval_binds_request_and_allows_its_private_chat` | Retain: accepted access request updates the matching integration channel allowlist. |
| `hard_delete_requires_exact_confirmation_and_never_deletes_admin` | Retain: deletion of an admin is rejected. Despite its name, it does not try a member with incorrect confirmation; its broad `ValueError` does not distinguish reasons. |

| Deletion-recovery function suffix | Disposition and behavior |
| --- | --- |
| `project_cleanup_failure_preserves_owner_rows_for_retry` | Retain: failed project filesystem removal leaves project/user rows and file, then retry completes. |
| `profile_cleanup_failure_keeps_user_and_addresses_for_retry` | Retain: profile failure retains the user and address, followed by retry. |
| `project_symlink_escape_is_rejected_before_rows_or_external_data_change` | Retain: replacement project symlink cannot cause external deletion or row removal. |
| `profile_symlink_is_rejected_before_user_row_removal` | Retain: profile symlink protects external data and user row. |
| `interrupted_project_cleanup_keeps_rows_for_idempotent_retry` | Retain: interruption after project removal keeps rows and allows retry. |
| `project_record_inside_another_profile_does_not_delete_other_users_data` | Retain: invalid cross-owner project root is rejected without destroying either owner's data. |
| `deletion_order_works_when_user_and_project_stores_share_database` | Retain: shared-database deletion ordering and filesystem removal. |
| `interrupted_profile_cleanup_keeps_user_row_for_idempotent_retry` | Retain: interrupted profile removal retains user row and supports retry. |
| `parent_symlink_swap_during_project_removal_never_deletes_external_data` | Retain: parent replacement during removal does not follow the outside path. |
| `missing_inner_entry_does_not_mark_project_cleanup_complete` | Retain: an interrupted inner filesystem removal cannot be mistaken for completed cleanup. |
| `missing_users_root_is_not_treated_as_successful_profile_cleanup` | Retain: unavailable root retains the user and existing moved profile. |
| `platform_without_safe_fd_removal_fails_closed` | Retain: unsupported contained removal preserves project and user. |

No case or assertion was removed in this pass: superficially similar cleanup
tests exercise different failure points and retained records. Validation:
both files **25 passed in 0.55 seconds**, with a temporary pytest plugin
redirecting the default legacy users root and skill/config paths to disposable
storage. Explicitly configured fixture roots were preserved; this setup does
not verify the real default-root configuration. No live household data or model
was used, and this focused run was not the full suite.

### Inbound channel and communication routing: two files

On 14 November, individually reviewed **15 original functions / 15 cases** in
`test_agent_v2_communication_channel.py` (10) and
`test_agent_v2_communication_router.py` (5). These exercise shared messaging
with disposable queues and stores, not live provider delivery or a qualified
simpler-loop conversation.

| Channel function suffix | Disposition and behavior |
| --- | --- |
| `queue_message_requires_project_provenance` | Retain: queues the prompt and user with a project argument. Despite its name, the assertions do not inspect stored project provenance or verify rejection when absent. |
| `queue_message_accepts_unassociated_conversation_message` | Retain: ordinary conversation may have no project. |
| `queue_message_generates_timezone_aware_local_timestamp` | Retain: generated timestamp is timezone aware. |
| `queue_message_preserves_explicit_timestamp` | Retain: caller-provided timestamp survives queueing. |
| `selector_dequeues_by_user_project_id_and_tag` | Retain: combined selector chooses the correct queued message. |
| `queue_message_preserves_correlation_id_for_selector_lookup` | Retain: correlation ID both selects and survives in message. |
| `slash_command_detected_only_at_prompt_start` | Retain: command fields, channel defaults and literal mid-sentence/indented text. It also asserts the exact legacy command metadata covered by the removed case below. |
| `queue_message_accepts_explicit_channel_metadata` | Retain: provider, address, reply and thread metadata survive. |
| `legacy_slash_command_metadata_shape_is_still_available` | Remove: its only three metadata assertions duplicate the retained slash-command test with the same command and inputs. No asserted behavior is lost; the focused rerun validates the remaining coverage. |
| `blank_prompt_or_user_raises_value_error` | Retain: independently rejects blank prompt and user. |

| Communication-router function suffix | Disposition and behavior |
| --- | --- |
| `deliver_to_user_uses_recipient_address_and_relays_explicit_reply` | Retain: queued recipient address, labeled content and correlated inbound relay. Provider delivery is simulated by marking the thread delivered. |
| `scheduled_communication_retries_reuse_outbox_and_thread` | Retain: repeated scheduled request reuses one message, records recipient and occurrence, and separate subgoal gets another message; delivery is simulated. |
| `plain_reply_requires_exactly_one_open_thread_and_unknown_recipient_is_safe` | Retain: unknown recipient and a reply with one open thread. Despite its name, it does not try two open threads. |
| `recipient_resolution_supports_nickname_unaccented_name_and_rejects_ambiguity` | Retain: exact nickname, name normalization and ambiguous prefix. |
| `inbound_router_consumes_linked_reply_before_queueing_capd_work` | Retain: correlated reply is handled without starting new queued work. |

The duplicate case removal leaves **14 cases**, all passing in 0.11 seconds.
Validation used both files and the same disposable default-root, skills and
config isolation described above; it is not a full-suite run.

### Outbox idempotency and schema migration: two files

On 14 November, individually inspected **22 original test functions / 49
collected cases** in `test_outbox_idempotency_identity.py` (9 functions) and
`test_outbox_status_migration_safety.py` (13). They use synthetic data and
disposable SQLite stores. These guard the existing delivery and data-preservation
boundaries; they do not establish candidate-loop delivery, externally verified
receipts, or end-to-end stale-claim fencing.

| Test function in `test_outbox_idempotency_identity.py` | Decision |
| --- | --- |
| `idempotency_rejects_different_operation_without_mutation` | Consolidate the 12-case field/connection cross-product to 7: keep each of the six different changed fields and a supplied-connection metadata conflict. Both connection branches call the same `_validate_idempotent_result`; the retained transactional case exercises its distinct branch and the specialized metadata comparison. The five dropped pairings no longer independently test every field through the supplied-connection branch; a future branch-specific comparison would need additional coverage. |
| `repeated_identical_operation_returns_existing_delivery_state` | Retain existing delivered status and provider receipt for an identical retry. |
| `unrelated_constraint_failure_does_not_return_a_phantom_message` | Retain unrelated SQLite uniqueness failure and no fabricated row. |
| `in_memory_retry_ignores_temporary_table_shadow` | Retain idempotent retry lookup with an in-memory shadow table. |
| `in_memory_consumers_ignore_temporary_table_shadow` | Retain read, claim and delivery settlement against the real table despite forged shadow content. |
| `shared_connection_reclaim_ignores_temporary_table_shadow` | Retain shared-connection lease reclamation against the real table. |
| `reclaim_rejects_connection_from_other_outbox` | Retain file-database connection mismatch refusal. |
| `reclaim_requires_own_persistent_in_memory_connection` | Retain in-memory connection identity and correct empty reclaim. |
| `regenerated_ack_conflict_keeps_one_message_and_continues_task` | Retain current V3 bridge acknowledgement retry conflict and no false completion; fake inference/judgment is not delivery evidence. |

| Test function in `test_outbox_status_migration_safety.py` | Decision |
| --- | --- |
| `unknown_state_is_not_migrated_or_dispatched` | Retain unknown stored state refusal for both status-constraint variants; this exercises initialization and preserved state, not an actual external dispatch attempt. |
| `supported_legacy_states_and_required_indexes_survive_migration` | Retain supported legacy rows and required indexes. |
| `existing_unconfirmed_outcome_remains_unconfirmed` | Retain uncertainty rather than turning an uncertain send into pending work. |
| `future_status_schema_is_rejected_even_when_no_future_rows_exist` | Retain future schema refusal even with currently familiar rows. |
| `migration_failure_rolls_back_ddl_data_and_indexes` | Retain transaction rollback following injected post-rebuild failure. |
| `preflight_and_migration_hold_one_write_lock` | Retain concurrent writer exclusion while preflight and migration run. |
| `unrecognized_legacy_schema_objects_are_not_dropped` | Retain distinct index, trigger, dependent view/table, generated-column and constraint/schema extensions; no individual equivalence established. |
| `oldest_recorded_status_schema_and_missing_columns_upgrade` | Retain the oldest recorded sparse schema upgrade and delivered result. |
| `unsupported_sparse_legacy_store_is_not_altered_before_refusal` | Retain sparse unsupported store rollback, journal mode and error privacy. |
| `existing_store_without_status_is_rejected_without_mutation` | Retain missing required status column refusal. |
| `check_text_in_comment_or_literal_is_not_a_constraint` | Retain comment, SQL literal and identifier spoof cases with distinct parsing forms. |
| `schema_preflight_closes_file_connection_on_failure` | Retain file connection cleanup when schema preflight fails. |
| `successful_schema_init_keeps_shared_connection_settings` | Retain SQLite connection configuration and migration-table checks. |

The reduction removes **five collected cases**, not functions, and records the
branch-specific coverage deliberately lost above. Validation:
`.venv/bin/python -m pytest -q tests/test_outbox_idempotency_identity.py
tests/test_outbox_status_migration_safety.py` — **44 passed in 0.29 seconds**.
No provider, model, actual message delivery or household data was used. This was
not a full-suite run.

### Project-check policy and lifecycle: one file

On 14 November, individually inspected all **19 test functions / 53
collected cases** in `test_project_checks.py`. The service manages an
authorized, bounded proactive project check; these tests use disposable
identities, stores and projects without provider calls. This protects an
existing product capability, not evidence that the simpler loop has integrated
or qualified proactive behavior.

| Function | Disposition and distinct behavior |
| --- | --- |
| `test_persisted_default_disabled_and_explicit_authority` | Retain: disabled default, explicit activation, persistence and actor-scoped retrieval. |
| `test_no_implicit_source_audience_or_actor_authority` | Retain seven variants: reject changed resource, recipient, connection, grant, skill, actor and write scope. |
| `test_goal_edits_suspend_and_need_new_authorization` | Retain four variants: changed goal, scope, duration or direct file edit invalidates the old authorization. |
| `test_config_spoof_cannot_create_or_activate_policies` | Retain: a project config cannot manufacture or activate a stored policy. |
| `test_lifecycle_round_trip_fences_queued_work` | Retain three project status transitions that must not revive the old policy. |
| `test_pause_resume_cancel_and_end_never_revive_old_work` | Retain: policy pause/resume revisions, per-run cancellation and irreversible ending. |
| `test_dependency_changes_fence_and_do_not_expand_sharing` | Retain eight changes covering grants, connection, selected binding, ACL, actor, resource and membership; membership additionally checks recipients. |
| `test_projectless_queries_preserved_but_checks_require_accessible_project` | Retain: normal projectless binding use remains possible while a proactive check requires an accessible project. |
| `test_duration_and_explicit_ongoing_renewal` | Retain: expiration, explicit renewal and invalid missing end for finite duration. |
| `test_bounded_schema_rejects_unsafe_or_incomplete_inputs` | Retain 12 distinct schema boundaries; these check the common invalid-definition result, not full end-to-end scheduling. |
| `test_concurrent_policy_revision_has_one_winner` | Retain: two competing policy edits cannot both win. |
| `test_status_hopping_cannot_bypass_reauthorization` | Retain three preexisting states that must not return to paused without authorization. |
| `test_future_start_is_inactive_without_losing_authorization` | Retain: a future start blocks execution without silently revoking the active policy. |
| `test_project_duration_independently_limits_ongoing_policy` | Retain: project duration independently limits an ongoing check. |
| `test_revision_change_fences_query_and_duration_and_preserves_identity` | Retain the exercised query revision, stable ID, stale policy and conflict checks. Despite its name this function does not edit duration; duration is exercised by separate tests. |
| `test_empty_recipients_and_bounded_late_policy_are_explicit` | Retain: explicitly empty recipient set and bounded-late setting survive guard. |
| `test_removing_executing_member_invalidates_access` | Retain: removal of a member from a separately owned shared project revokes access. |
| `test_project_deletion_and_id_reuse_never_revive_queued_work` | Retain: delete/recreate the same ID cannot preserve old authority; this tests guard, not a real queue delivery. |
| `test_calendar_policy_requires_explicit_finite_offset_window` | Retain four absent or invalid window examples for calendar source policy. |

No deletion or consolidation was supported: similar variants target different
authority or lifecycle boundaries and the test names alone do not establish
duplication. Several negative cases accept either `PermissionError` or
`ValueError`, so their passing result alone does not establish the exact
rejection reason. Validation: `PYTHONPATH=/private/tmp:. .venv/bin/python -m
pytest -q -p alphonse_audit_isolation_step5 tests/test_project_checks.py` —
**53 passed in 1.89 seconds**. The temporary isolation plugin redirects legacy
user, skill and agent-config defaults to `tmp_path`; this was not a full-suite
run.

### Standard skill validation and starter packages: two files

On 14 November, individually inspected **18 test functions / 107 collected
cases** in `test_agent_v2_skills.py` (15) and
`test_household_starter_skills.py` (3). Tests use temporary skill stores; starter
selection cases use reviewed inputs and do not evaluate Jev's choices. These
test the installed standard-skill surface and V3 bridge, not complete
simpler-loop skill activation or real household workflows.

| Function | Disposition and distinct behavior |
| --- | --- |
| `test_package_validation_fixtures` | Retain 39 metadata/frontmatter examples: names and directory mismatch, description boundaries/types, optional metadata, normalization and unknown/missing fields. Two similar IDs are distinct underscore vs slash inputs. |
| `test_reference_validator_agrees_on_shared_rules` | Retain as an optional differential check against `skills_ref` for the cases flagged compatible. In this run all such cases were skipped because the optional reference validator was unavailable; they are not evidence it agreed. |
| `test_description_boundary_across_authoring_update_and_schema` | Retain four length cases spanning install schema, execution, authoring and update paths. |
| `test_authoring_rejects_wrong_description_types` | Retain four non-string value types through native authoring. |
| `test_install_replace_export_preserve_entire_package` | Retain package bytes and file modes, metadata preservation, conflict resolution and export behavior; the final equality check occurs after a rejected repeat export. |
| `test_package_copy_rejects_symlink_escape` | Retain separate install, replace and export entry points. |
| `test_listing_exposes_specification_operational_and_filesystem_diagnostics` | Retain distinct invalid package, oversized file and missing file diagnostics. |
| `test_ipc_metadata_diagnostics_export_and_authorization` | Retain Desktop-facing IPC package metadata, export and actor denial. |
| `test_skill_overview_does_not_inventory_packages_and_detail_is_scoped` | Retain overview avoids resource inventory, detail requests one package and rejects a stranger. |
| `test_package_directory_listing_includes_empty_and_excludes_symlinks` | Retain empty directories and linked-directory exclusion. |
| `test_metadata_numeric_key_is_rejected` | Retain non-string metadata-key rejection at installation. |
| `test_reference_validator_accepts_complete_package` | Retain optional reference package validation; skipped here with the unavailable validator. |
| `test_malformed_frontmatter_is_diagnosed` | Retain four malformed delimiter, structure and YAML examples. |
| `test_import_body_has_no_authoring_length_constraint` | Retain the distinction between imported and newly authored instruction limits. |
| `test_description_limits_count_unicode_characters` | Retain Unicode character boundary rather than byte count. |
| `test_clean_store_validates_starters_and_linked_resources` | Retain bundled defaults, link containment and readability. |
| `test_representative_selections_load_complete_composed_guidance` | Retain eight reviewed skill-selection combinations, including project-free activation and on-demand resources. These are explicit inputs, not model routing performance. |
| `test_third_party_package_roundtrip_without_registry_changes` | Retain third-party standard package install/resource/export without per-script registry. |

No deletion was justified: overlapping-looking checks exercise distinct entry
points, invalid types or package boundaries. The optional reference tests are
potential conformance coverage, but are not running in this environment and
their skip count should not be mistaken for passing validation. Validation:
`PYTHONPATH=/private/tmp:. .venv/bin/python -m pytest -q -p
alphonse_audit_isolation_step5 tests/test_agent_v2_skills.py
tests/test_household_starter_skills.py` — **72 passed, 35 skipped in 0.41
seconds**. This was a focused run, not the full suite.

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
