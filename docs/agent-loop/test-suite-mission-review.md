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

1. `tests/test_agent_v2_basic_intelligence.py` has two cases for the old
   `BasicIntelligenceProcessor` echo and its `/project new` placeholder. The
   repository search found no use of the class outside its own two tests and
   its definition/export; the sole installed command enters the daemon CLI.
   Neither case exercises the normal conversation path, actual command
   execution, dependable-loop behavior or a safety invariant. They are strong
   low-value candidates for removal. **Tradeoff:** the class is explicitly
   exported from the distributable Python package, so removing the tests gives
   up coverage of an importable placeholder; unknown external users cannot be
   ruled out. This is a product/API policy decision, not proof of zero risk.
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

Two identity-only tests were removed in this review. These candidates do not
justify deleting the old Python implementation or its exported symbols.
The remaining eight-file V2 group plus deterministic-replay and replay-harness
checks passed together: **56 passed in 1.11 seconds**. The CSV is the earlier
collection snapshot, so it still lists the two removed functions.
For each deletion, verify collection and focused tests and record the deliberate
loss; do not treat a reduction in count as an outcome in itself.

### Retirement and consolidation candidates requiring more work

- The 53 cases in the eight-file narrow V2-internal snapshot were individually
  inspected, including the two now removed. The remaining 51 include two
  placeholder API tests noted above and 49 tests with behavior or structure
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
