# Individual evaluation-arm review: fixtures, bridges and synthetic recovery

14 November 2026. This reviews 11 previously unclassified evaluation modules
for #229. **Active** below means used by current development or test code;
**historical** means preserving a past comparison or experiment. Neither label
means production-qualified. No real provider, household service or household
data was used, and no fixture or evidence was deleted. Repository searches do
not establish whether external users import the Python modules.

| Module in `alphonse/agent_v2/evaluation/` | Value, observed consumers and status | Disposition and condition for retirement |
| --- | --- | --- |
| `loopback_traffic_pilot.py` | Disposable portal and separate evaluator for synthetic traffic scenarios; imported by `external_codex_fixture_runner` and covered by [loopback tests](../../tests/test_loopback_traffic_pilot.py). **Active fixture** for exercising navigation/extraction in a controlled task. | **Retain.** Revisit when the external-agent task and oracle have an explicitly adopted replacement. The loopback service is not real traffic, a browser security sandbox or proof of user delivery. |
| `model_candidate_bridge.py` | Injected offline model-decision bridge with usage semantics; imported by `offline_responses_candidate` and covered by [bridge tests](../../tests/test_model_candidate_bridge.py) and HTTP processor fixtures. **Historical/offline comparison dependency.** | **Retain while those offline adapters remain.** Remove only after migrating their callers and preserving fixture outcomes. A transport's offline declaration is not network isolation, a spend limit or approval for live use. |
| `paired_oracles.py` | Source-range and outcome checks for decision-context paired fixtures, consumed by `paired_fixture`, `heldout_memory`, and [paired tests](../../tests/test_paired_memory.py). **Active shared oracle** for preserved comparisons. | **Retain.** Preserve fixture semantics and frozen outcome/source-range checks before replacing; a range observation or fixture-state oracle is not itself a semantic correctness judgment. |
| `planning_form_fixture.py` | Disposable two-field form with effect checks, used by `planning_multifield`, `planning_multifield_live`, and [fixture tests](../../tests/test_planning_form_fixture.py). **Active synthetic planning fixture.** | **Retain** until an equivalent scenario and effect oracle replace its callers. It is not a production browser backend or an authorization boundary. |
| `prescription_http_processors.py` | Shared read-only HTTP source and offline-injected processor adapters used by Codex clock comparison, clock adapters, and [HTTP processor tests](../../tests/test_prescription_http_processors.py). Also used by the appointment fixture. **Active shared synthetic adapter.** | **Retain** until shared `LocalSource` consumers and matched fixture behavior are migrated. It does not verify medical correctness, production access, or delivery; the prescription values are synthetic. |
| `prescription_safety_development.py` | Synthetic prescription cases and separately specified fixture answers, imported by `prescription_safety_three_arm` and [development tests](../../tests/test_prescription_safety_development.py). **Development oracle**, not a patient workflow. | **Retain** while its three-arm comparison and cases remain meaningful. Reassess with the gym corpus; preserve cases, expected outcomes, failures and limits when replacing. It is neither clinical validation nor held-out qualification. |
| `router_candidate_chooser.py` | Compatibility re-export of the shared chooser now located in `core/intelligence/v3/router_candidate_chooser.py`. Scoped search found no current in-repository import of the evaluation alias; runtime imports the core module. **Historical import compatibility**, not a separate loop or chooser. | **Retain for now; candidate for future consolidation.** Removal changes an importable Python path even though the core chooser remains. Verify external/API compatibility and document any migration before deleting; do not infer absence of outside callers from repository search. |
| `seeded_v3_resume.py` | Disposable seeded V3 checkpoint replay, used by `simplified_loop`, `mutation_development`, and [paired adapter tests](../../tests/test_single_loop_paired_adapter.py). **Historical V3 recovery baseline.** | **Retain** pending explicit baseline replacement and retirement of its consumers. It seeds prior progress; it does not prove recovery from an actual crash or a production effect. |
| `synthetic_delivery_settlement.py` | Single-database simulated effect settlement/recovery, used by synthetic lifecycle/expired-recovery tests and [claim-bound answer tests](../../tests/test_claim_bound_answer_delivery.py). **Active safety-regression fixture**, not runtime code. | **Retain** until equivalent reconciliation regressions exist and are independently reviewed. Fixture authority/effect rows are forgeable; never install or invoke it as a household recovery API. |
| `synthetic_staged_dispatch.py` | Disposable staged dispatch simulation used by [staging tests](../../tests/test_synthetic_staged_dispatch.py), synthetic lifecycle/recovery tests and claim-bound answer tests. **Active safety-regression fixture.** | **Retain** pending a reviewed replacement for the synthetic regressions. It does not send a message, authenticate an actor, or fence external I/O; its durable marker does not establish production delivery. |
| `synthetic_suite_journal.py` | Offline-synthetic per-attempt recording and normalization, imported by `clock_pilot_journal` and `synthetic_journal_summary`, with [journal tests](../../tests/test_synthetic_suite_journal.py). **Active development measurement helper.** | **Retain** while journal consumers and attempt-level failure/unknown accounting depend on it. Replace only with equivalent append/attempt preservation; an offline callback declaration is not a network sandbox, and a fixture pass is not an architecture pass. |

## Verification and observed regression

The focused synthetic and offline suite initially reported **244 passed, 2
failed**. Both failures were in [single-loop paired adapter tests](../../tests/test_single_loop_paired_adapter.py):
the fixture's simulated abrupt interruption inherited from `Exception`, so the
candidate's newer evidence-recording handler caught it before the fixture could
exercise restart. The observed result was a safe blocked state but **zero**
recorded restarts; it did not satisfy the frozen restart oracle. This was a
fixture incompatibility, not evidence of successful restart.

The test fixture's explicitly synthetic interruption now inherits from
`BaseException`, modeling abrupt process loss without weakening ordinary host
exception handling. The rerun passed: **246 passed** across the 12 focused test
files, including disposable loopback sockets. This is a same-process synthetic
interruption followed by reconstructed collaborators, not a power-loss test or
proof of production durability. No historical benchmark result was rewritten.

See [the evidence index](historical-benchmark-evidence-index.md) and
[arm audit](evaluation-arm-retirement-audit-2026-11.md) for related recorded
outcomes and the wider review scope.
