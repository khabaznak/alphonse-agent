# Evaluation-arm retirement audit — 14 November 2026

This audit supports #229; it is not a decision to retire the V3 bridge or proof
of a qualified simpler loop. Historical reports and results remain unchanged.

**Subsequent owner direction:** retain recorded results in Git, but do not
retain executable V2 or V3 loop implementations for historical replay. The
provisional replay-retention recommendations below describe the earlier audit;
they are superseded by the owner's decision. Migrate useful current gym cases
and shared helpers and remove the old implementation. Replace the active V3
conversation path before deleting that live bridge.

The follow-up individual pass records a disposition for all 97 current
top-level Python modules in `evaluation/`, in this report and the linked
[V3/memory](evaluation-arm-review-v3-memory.md),
[planning](evaluation-arm-review-planning.md),
[support](evaluation-arm-review-support.md),
[fixture](evaluation-arm-review-fixtures.md), and
[gym](evaluation-arm-review-gym.md) reviews. The two setup modules removed in
this pass are documented separately below. Decisions to retain are provisional:
revisit them as the bridge and shared dependencies are retired. This inventory
does not establish all transitive or external consumers, inspect every package
file, or validate every historical run.

## Removed: superseded direct-API operator setup

The candidate and V3 direct Responses API clock setup CLIs were superseded by
the owner's Codex-subscription direction. Both `--execute` branches already
unconditionally refused direct API execution. Their only in-repository
consumers were each other and their dedicated tests. Searches covered Python,
documentation, evidence manifests, project packaging and scripts. The dated
`test-inventory-2026-11.csv` names the removed tests as a historical snapshot;
it is not a current test list and was preserved. Both setup modules and their
setup-only tests were removed. Their former Python paths and
`python -m alphonse.agent_v2.evaluation.clock_pilot_setup` and
`python -m alphonse.agent_v2.evaluation.v3_clock_pilot_setup` invocations are
no longer available; historical Git revisions retain the code. This removes
unusable setup commands, not retained results or
the underlying offline experiment adapters. Other low-level Responses transport
code still exists and must not be treated as approved for live provider calls.

## Retained: reproducible legacy baseline and shared fixture code

`evaluation/deterministic_adapters.py` constructs the historical V2 PDCA and
V3 processors for the documented deterministic replay baseline. The current
Codex clock comparison also imports fixture utilities from that module. The
PDCA replay implementation and its prompt templates remain packaged under
`evaluation/legacy_pdca/`; deleting them would break that executable baseline.
Recorded replay is scripted and does not establish autonomous performance.

`evaluation/clock_responses_attempt.py` is no longer an approved live-provider
operator route, but its `_reason_category` helper is currently imported by the
Codex clock and prescription comparison modules. Related Responses adapters and
transport also have offline tests. They cannot be removed safely as one dead
arm without first separating the shared helpers and deciding which offline
comparisons remain useful. Retain their existing results and error cases.

The Codex subscription comparison modules are the current synthetic comparison
path; retaining them is not a claim that their runs qualify production use.
No provider was called in this audit. After removal, 67 focused deterministic
replay, synthetic clock/prescription comparison and retained offline adapter
checks passed with disposable loopback sockets; this is not a full-suite audit.

## Entry-point and comparison-arm inventory

This is a source-and-description inventory of operator entry points and named
comparison arms. It does not establish that every listed entry point still
runs, nor audit every helper, dependency or stored receipt. Names refer to
modules in `evaluation/`.

| Family | Entry points or named arms | Disposition and evidence |
| --- | --- | --- |
| Historical V2/V3 baseline | `replay`, `deterministic_adapters`, `legacy_pdca`, `simplified_loop`, `candidate_comparison` | Retain documented deterministic replay and shared fixtures; replacement requires an explicit replay/comparison decision. Scripted replay is not model-performance evidence. |
| Subscription synthetic comparisons | `codex_clock_comparison`, `codex_mail_comparison`, `codex_prescription_comparison`, `codex_candidate_chooser`, `external_codex_pilot`, `small_pilot_plan` | Retain current synthetic comparisons and planning paths. Each live provider call requires its applicable owner scope; external pilot setup itself does not dispatch. |
| Superseded Responses clock setup | `clock_pilot_setup`, `v3_clock_pilot_setup` | Removed as explained above; historical test-inventory rows remain a snapshot. |
| Responses-era supporting experiments | `clock_responses_attempt`, `v3_clock_responses_attempt`, `candidate_clock_responses`, `v3_clock_responses`, `v3_live_clock`, `v3_responses_adapter`, `offline_responses_candidate`, `responses_pilot_transport`, `clock_pilot_journal` | Retain for now: current Codex comparisons import shared reason classification from `clock_responses_attempt`, and offline wiring has tests. These historical low-level modules are not permission to call direct APIs. |
| Scripted three-arm and capability baselines | `offline_three_arm`, `read_only_development_manifest`, `prescription_three_arm`, `prescription_safety_three_arm`, `prescription_candidate_adapter`, `prescription_v3_adapter`, `traffic_processor_development`, `mutation_development`, `followthrough_candidate_adapter`, `followthrough_v3_adapter`, `offline_run_journal`, `synthetic_journal_summary` | Retain fixture/control-flow comparisons and development provenance. These do not show autonomous household capability or verified real-world effects. Reassess individually against retained fixtures and reports. |
| V3 rollout, memory and context | `lifecycle_qualification`, `isolated_v3_arm`, `context_pilot`, `context_pilot_live`, `memory_recall`, `memory_replay`, `memory_replay_live`, `memory_rollback`, `memory_scoring`, `paired_memory`, `pilot_readiness` | Retain while V3 remains the production bridge and rollback/replay remains relevant. Live-labeled modules are not proof that a call is currently authorized. Further individual review is needed. |
| Planning and resourcefulness | `planning_comparison`, `planning_independent_reads`, `planning_independent_reads_live`, `planning_jev_probe`, `planning_multifield`, `planning_multifield_live`, `resourceful_delegation`, `resourceful_live`, `resourceful_live_review`, `household_delegation`, `household_live`, `skill_reuse` | Retain synthetic probes pending outcome-by-outcome review. Some are scripted and some may call configured providers; a local budget does not authorize another call. |
| Browser and tactical-wire investigations | `browser_handoff_comparison`, `browser_sequence_live`, `tactical_wire_corpus` | Retain while historical V3 browser and sequencing behavior is relevant. Scripted fixtures and model-only trials are different evidence classes. |
| Isolation probe | `local_isolation_probe` | Retain as an offline macOS diagnostic; its selected-path and port checks do not establish a complete isolation boundary or authorize a live pilot. |

No additional whole arm or operator CLI is identified as safely removable in
this pass. Absence of a textual caller does not prove an operator entry point or
external consumer is unused. Standalone live-named diagnostics have distinct
comparison or isolation purposes and should not be removed solely for age or
lack of imports.

## Individual review: comparison and replay families

These are decisions for the current milestone, not permanent endorsements.
References to tests below show specific retained coverage; they do not mean a
model-based benchmark was rerun or that a test's existence alone proves value.

| Entry | Decision and concrete reason |
| --- | --- |
| `replay` | Retain the documented `05-rollout.md` replay entry point; `deterministic_adapters` imports its trace/case contract and `test_agent_v3_deterministic_replay.py` exercises it. |
| `deterministic_adapters` | Retain historical V2/V3 implementation and shared fixture utilities: current `codex_clock_comparison` imports `_Action`, `_RunState`, `_Scenario`, and `_registry`. |
| `legacy_pdca` | Retain for executable V2 replay; `deterministic_adapters.v2_runner` imports its processor and templates. This is historical code, not production V2 dispatch. |
| `simplified_loop` | Retain the paired evaluation contract consumed by `offline_run_journal` and prescription comparison adapters; see `test_simplified_loop_evaluation.py`. |
| `candidate_comparison` | Retain scripted candidate/V3 fixture adapters consumed by `offline_three_arm` and `read_only_development_manifest`; `test_single_loop_paired_adapter.py` exercises the candidate adapter. |
| `codex_clock_comparison` | Retain active synthetic clock comparison and shared utilities imported by the mail/prescription comparisons; `test_codex_clock_comparison.py` checks fixture plumbing, not autonomous performance. |
| `codex_mail_comparison` | Retain synthetic inbox comparison used by the gym; its distinct oracle and result paths are exercised by `test_codex_mail_comparison.py`. It does not connect to Gmail. |
| `codex_prescription_comparison` | Retain synthetic multi-source comparison and independent result checks exercised by `test_codex_prescription_comparison.py`; no clinical validation is implied. |
| `codex_candidate_chooser` | Retain gym adapter used by `codex_clock_comparison`; it shares decision construction with runtime and has targeted chooser tests. |
| `external_codex_pilot` | Retain external reference setup used by `external_codex_fixture_runner`; `test_external_codex_pilot.py` exercises preparation and inspection, not live dispatch. |
| `small_pilot_plan` | Retain non-dispatching five-case plan; `test_small_pilot_plan.py` checks the plan and its declared limitations. It is not a frozen held-out protocol. |

## Individual review: Responses-era support

The direct-API clock **operator setup** modules were removed above. The modules
below remain low-level experimental code; their presence is not live-call
authorization. Continuing ownership should revisit them when the historical
offline comparison is replaced or their reusable code is isolated.

| Entry | Decision and concrete reason |
| --- | --- |
| `clock_responses_attempt` | Retain for now: its `_reason_category` is imported by the active Codex clock and prescription comparisons. `test_clock_responses_attempt.py` also covers sanitized failure and usage handling. |
| `v3_clock_responses_attempt` | Retain provisionally: its repo consumer is `test_v3_live_clock.py`, which combines scripted actual V3 processing, projection modes, separately metered fake Jev, journal redaction, and unmetered failure. Lower-level tests overlap, but this combined assertion has not been replaced. |
| `candidate_clock_responses` | Retain for offline candidate wiring used by `clock_pilot_journal` and `clock_responses_attempt`; `test_candidate_clock_responses.py` checks the adapter. |
| `v3_clock_responses` | Retain the isolated offline V3 clock wiring exercised by `test_v3_clock_responses.py`; it is not an approved API launcher. |
| `v3_live_clock` | Retain its bounded V3 clock test harness: `test_v3_live_clock.py` covers cancellation/deadline behavior with injected providers and fake metered Jev. |
| `v3_responses_adapter` | Retain the offline V3 request adapter used by V3 clock harnesses and tested in `test_v3_responses_adapter.py`. |
| `offline_responses_candidate` | Retain documented offline candidate request wiring in `35-offline-responses-candidate-wiring.md` and its regression test; no live call is needed. |
| `responses_pilot_transport` | Retain for the above offline transport/ledger tests and injected local sender; its default-off design must not be confused with user authorization. |
| `clock_pilot_journal` | Retain offline synthetic journal integration and redaction checks in `test_clock_pilot_journal.py`; it is not a live comparison. |

## Individual review: scripted three-arm and capability baselines

| Entry | Decision and concrete reason |
| --- | --- |
| `offline_three_arm` | Retain the scripted three-arm runner used by `prescription_three_arm` and `offline_run_journal`; `test_offline_three_arm.py` covers the contrast. |
| `read_only_development_manifest` | Retain the selected-source integrity manifest documented in `28-manifest-integrity.md` and tested by `test_read_only_development_manifest.py`; it is not full environment attestation. |
| `prescription_three_arm` | Retain the cross-arm synthetic prescription runner; `test_prescription_v3_adapter.py` checks its V3 path alongside the candidate. |
| `prescription_safety_three_arm` | Retain the scripted safety comparison and its independent fixture checks in `test_prescription_safety_three_arm.py`; these are not medical guidance. |
| `prescription_candidate_adapter` | Retain because both prescription three-arm runners import it; `test_prescription_candidate_adapter.py` checks its synthetic candidate binding. |
| `prescription_v3_adapter` | Retain because both prescription three-arm runners import it and its process-isolated V3 path has dedicated tests. |
| `traffic_processor_development` | Retain the scripted synthetic traffic fixture used in `small_pilot_plan` and tested by `test_traffic_processor_development.py`; it does not measure real traffic lookup. |
| `mutation_development` | Retain the disposable-store mutation control and independent checks in `test_mutation_development.py`; it is not verified external-effect execution. |
| `followthrough_candidate_adapter` | Retain because `question_development` and `denied_followthrough_development` exercise this candidate path; see `test_followthrough_candidate_adapter.py`. |
| `followthrough_v3_adapter` | Retain the matching isolated V3 path used by the same question/denial comparisons; see `test_followthrough_v3_adapter.py`. |
| `offline_run_journal` | Retain the durable offline attempt journal used by synthetic-suite and clock runners; `test_offline_run_journal.py` checks the receipt contract. |
| `synthetic_journal_summary` | Retain the no-rerun journal summary described in `benchmark-protocol.md` and tested in `test_synthetic_journal_summary.py`; it summarizes caller-controlled fixture grades. |

Further retirement should trace concrete consumers and recorded baselines. A
reference search cannot prove absence of external Python callers, so removal
also changes the importable interface for those users. The individual pass
covers top-level modules, not every package file or transitive dependency.
