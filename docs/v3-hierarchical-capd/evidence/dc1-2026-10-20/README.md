# DC-1 deterministic invariant qualification — 2026-10-20

Issue: [#24](https://github.com/khabaznak/alphonse-agent/issues/24).
Protocol: [decision-context evaluation v2](../../13-decision-context-evaluation.md#dc-1--deterministic-context-and-control-invariants).
Result: **the deterministic invariant fixtures pass, with no invariant exceptions**.
This is synthetic DC-1 evidence, not a live discovery, response-quality, cost,
latency, held-out, or rollout decision. Projection defaults are unchanged.

The tested starting revision is `2af7eca` (the paired harness merged after the
issue's starting revision `95666c4`). The final run uses the local working tree.
[verification.json](verification.json) records source fingerprints, commands,
exact failure identities and individual dispositions. Other work appeared in the
shared checkout during qualification; it is identified separately in the manifest
and was not reverted or adopted by this change. The full-suite result describes
that checkout, not an isolated commit or a frozen paired live experiment.

## Results and failure injection

- Initial full suite: **36 failed, 905 passed**. All failure identities match the
  retained household-pilot baseline; none are waived as invariant exceptions.
- Focused qualification: **392 passed**.
- Final full suite: **2 failed, 1,006 passed**, **zero new failure identities**.
  Thirty-four original failure identities are resolved. The two remaining
  dispositions below are outside the DC-1 context/control and adapter checks.
- Red regression run: copy the new projection, outer-controller and artifact
  qualification tests into a disposable archive of `2af7eca`, then run the same
  selected tests against the original product code: **26 failed, 26 passed,
  72 deselected**. These 26 failures pass in the final checkout. The manifest
  retains their exact IDs; it does not infer qualification from a test count alone.
- All fixtures are synthetic. Artifact programs are disposable fixtures in
  temporary directories; no generated household artifact was copied into source.
  Jev and inference are stubbed, except that the Codex transport tests exercise
  the actual adapter with a synthetic subprocess result. No live provider is used.

## Invariant coverage

The following test paths are relative to the repository root. Parameterized
fixtures run over `admission`, `curation`, `planning`, `execution`, `review`, and
`response` wherever applicable. Review intentionally performs no historical recall;
its current typed evidence is supplied separately from its context projection.

1. **Request identity and deduplication.**
   `tests/test_agent_v3_projections.py`: `test_request_once_and_canonical_recall_deduplicated`,
   `test_dc1_identical_historical_text_keeps_distinct_event_identity` (five
   history-bearing purposes), and `test_atomic_budget_dedup_and_required_priority`.
   The current event is excluded by message identity, not text equality. Identical
   historical events remain distinct; recall/history copies share one source ID.
2. **Required record integrity and exact budgets.**
   `test_dc1_required_integrity_and_failure_before_any_provider` covers all six
   purposes, request/contract/steering/user constraints/global policy/skill tails,
   byte-token fallback metadata and zero Jev calls on required overflow.
   `test_dc1_exact_utf8_budget_boundary_counts_labels_and_separators` checks exact
   fit and one-byte overflow with multilingual input. Required records are atomic.
   `test_dc1_cancellation_and_unavailable_required_skill_fail_explicitly` prevents
   selected but missing skills from silently disappearing.
3. **Continuity, restart, channel changes and completed effects.**
   `tests/test_memory_continuity.py` tests whole exchanges, multiline steering,
   question/answer checkpoint persistence, owner-bound answers and failed evidence.
   `tests/test_agent_v3_phase_executor.py` tests restored completed subgoals and
   `test_dc1_cross_channel_checkpoint_does_not_repeat_completed_mutation`: a
   Telegram checkpoint resumes on Desktop with only verification remaining.
   `test_whole_continuity_exchange_is_atomic` prevents orphaned assistant replies.
4. **Visibility before providers and after access changes.**
   `test_dc1_visibility_rechecked_at_every_decision` covers all six purposes and
   spies on Jev inputs before and after sharing revocation, deletion, a channel
   change and a user switch. Relevant private content never reaches the spy.
   `tests/test_memory_recall.py` exercises native search/source expansion, live
   supporting-source checks, deleted derived records, malicious backend envelopes,
   unauthorized neighbors and traversal/symlink rejection.
5. **Recall survives curation while deterministic policy still applies.**
   `test_request_curation_cannot_hide_recall_tools` rejects every tool with Jev and
   retains search and expansion at both request and phase curation.
   `test_dc1_recall_preservation_does_not_bypass_phase_policy` verifies that the
   same tools remain excluded in a phase without recall authorization.
   `tests/test_agent_v3_tool_revealing.py` and executor tests cover prerequisite,
   unavailable integration, hidden-call and mutation-scope enforcement.
6. **Immutable contract and current typed evidence.**
   `tests/test_agent_v3_outer_controller.py`:
   `test_dc1_failed_or_historical_evidence_cannot_satisfy_contract` injects invalid
   success claims through both Jev and System Two. The contract remains unchanged,
   failed/current and historical references cannot satisfy it, and unrelated
   ledger text is absent. Existing scope and prepared-response freshness checks
   still pass. `test_dc1_fallback_review_receives_required_context_before_inference`
   covers unavailable and absent Jev; required steering reaches System Two and
   overflow fails before its call. Review budget exceptions cannot become fallback
   calls that discard the required context.
7. **Selected skill/artifact integrity and access.**
   `tests/test_agent_v3_artifact_resources.py`:
   `test_candidate_projection_keeps_complete_authorized_artifact_instructions`
   covers all six purposes, full long instruction tails, verified resource paths,
   argument schemas and explicit overflow. Existing private-artifact, ownership,
   sharing, selected-skill, disabled-resource and registry tests pass. Artifact IDs
   never become independently curated runtime tools.
8. **Recall status and malformed decisions.**
   `test_dc1_recall_statuses_survive_projection_without_success` injects empty,
   partial, unavailable and failed native responses. Coverage/limits survive and
   no success is fabricated. `test_dc1_malformed_recall_selection_retains_hits`
   injects unknown IDs, nulls, objects, strings and booleans. Missing, ambiguous
   and failed relevance judgments preserve discovery. Native recall tests retain
   distinct backend statuses, bounded expansion and redacted failures.
9. **Cancellation and terminal state.**
   Projection tests cancel before/during Jev, at final recall relevance and during
   direct-response inference. Outer tests cancel before/during review, during Act
   recommendation and with a prepared closure-only response. A persisted success
   closure also cannot bypass failed scope verification. No terminal success
   or later tool execution is permitted. Executor cancellation tests remain green.
10. **Flag-off path and call-time evidence.**
    `test_flag_off_keeps_legacy_ledger` runs all six purposes with projections off.
    `tests/test_decision_context_manifests.py` verifies call-time receipts, source
    ranges, omission reasons, failed calls, and owner-scoped source expansion.

## Baseline corrections and related issues

The per-identity dispositions are in the manifest. Corrections preserve stronger
assertions rather than deleting tests or restoring obsolete product behavior:

- Project fixtures now use `project_config.json`, explicit relevant-context
  selection, and explicit shared-project membership. They still test member
  isolation and denial of unauthorized edits. This exposed and fixed an undefined
  variable in the legacy project-context renderer.
- #20 and #21 are covered by supported config candidates and explicit admission
  selection. Legacy Markdown is not reintroduced into prompts.
- #19: session commands now describe their binding as applying to explicitly scoped
  project tasks. Ordinary conversation stays projectless. Synthetic TUI and Telegram
  tests verify the shared explicit-project session binding and the projectless path.
- #22: `pdca_task` is advisory for same-owner intake relevance, as documented by
  the existing intake implementation. The test now verifies both candidates are
  considered, only selected IDs are consumed, and the rejected request remains
  queued independently. Existing owner and authenticated-question boundaries stay
  enforced; relevance does not expand ownership.
- Legacy CAPD tests provide an explicit synthetic Jev reviewer/router. Missing
  acceptance inference must fail rather than report completion. Act routes initial
  contract creation to Plan; the real Codex adapter is still exercised at Plan.
  Prepared-response and verified-evidence completion guards remain asserted.
- Jev fixtures use current Score triage, separate context-selection/admission
  calls and exclusion of artifact tool IDs. IPC tests retain secret redaction and
  now expect friendly tool labels, attempt/action fields and the current validation
  timeout. The native Bash guidance assertion follows its current wording.
- The oversized memory fixture now contains distinct complete records, allowing
  the supported formatter to keep the journal path and omit an oversized archive
  atomically instead of expecting a sliced opaque claim.

Two original failures remain, without being counted as passed:

- `tests/test_project_provenance_waiting.py::test_new_conversation_creates_protected_home_project`
  expects ordinary conversation to create a protected home project. Current
  conversation-first ingress creates no project. This is a project lifecycle
  expectation outside DC-1; it does not establish a memory access failure.
- `tests/test_project_provenance_waiting.py::test_independent_message_receives_exact_turns_ahead_notice`
  stops at the same automatic-home-project expectation. The queue count assertion
  passes; its later notice assertion is not reached. Current notice emission also
  requires a project. Projectless queue-notice behavior remains unresolved and must
  not be advertised as verified by this report. It does not affect source access,
  required context, inference input, evidence verification or cancellation.

## Reproduction and limits

Run `.venv/bin/python -m pytest -q --tb=short` for the full suite. The manifest
lists the exact 22-file focused command, the red regression selector, individual
failure dispositions and SHA-256 fingerprints. All expected invariant failures
are explicit exceptions/cancelled or incomplete outcomes, never task success.

This report qualifies the enumerated deterministic fixtures without exceptions;
it does not prove every possible runtime schedule or semantic judgment. DC-2
through DC-5 and the broader engine rollout gates remain unassessed. In particular,
no live paired trials, human-reviewed answer quality or rollout acceptance are
manufactured from these tests.
