# Gym source-selection and telemetry module review

Reviewed on 14 November 2026 for #229. These 11 modules were not individually
listed in the initial evaluation-arm inventory. This is a source and direct
consumer review, not a run or proof of real-world performance. **Active** here
means a retained development fixture or telemetry helper with a present test or
consumer; it does not mean a case is held out, a live run is scheduled, or a
module is production code. Dated results remain historical observations with
their original source versions and limitations.

The source-selection exercises use invented records and a disposable loopback
service. Their expected answers and source-coverage checks are evaluated after
service shutdown; they do not establish clinical correctness, real household
access or delivery. Their `run` functions accept an injected candidate callback:
whether that callback calls a provider depends on the caller. No callback,
provider or live model was run in this review.

| Module | Mission value and observed direct consumers | Disposition |
| --- | --- | --- |
| `selective_sources_development` | Pharmacy development case asks the candidate to select relevant sources despite unrelated material. `selective_sources_contrast` imports its source definitions; `selective_sources_ambiguity` imports its catalog constants; `tests/test_selective_sources_development.py` exercises the fixture. | **Active development baseline:** retain for source-selection diagnosis and dependent fixtures. Prior public runs are development evidence, not a held-out score. |
| `selective_sources_contrast` | Uses the same service and descriptions but asks about a paused exercise program, making a formerly distracting source relevant. `tests/test_selective_sources_contrast.py` compares it with the pharmacy fixture. | **Active development contrast:** retain; it tests whether relevance changes with the goal, not a general knowledge or health outcome. |
| `selective_sources_ambiguity` | Provides a current prescription identifier with conflicting pharmacy values, testing a justified unknown or clarification. `tests/test_selective_sources_ambiguity.py` exercises the fixture. | **Active development negative case:** retain for uncertainty and source-grounding regressions. Prior results are public development observations, not clinical evidence. |
| `selective_sources_variants` | Supplies agreement, missing, superseded and resolution variants of the same source-selection task. `selective_sources_catalog_v2` reuses its cases and evaluator; `selective_sources_identity_v3` reuses its sources and evaluator; `tests/test_selective_sources_variants.py` covers it. | **Active development variants and shared fixture:** retain while dependent contrast cases and their baselines use it. Results cannot independently prove held-out generalization. |
| `selective_sources_catalog_v2` | Re-exports the variants and their evaluator with a more neutral, discoverable source description. Imported by conflict, identity and reference contrast modules and checked by `tests/test_selective_sources_catalog_v2.py`. | **Active shared development catalog:** retain because current fixtures depend on it. Its `v2` name identifies the fixture/catalog revision, not the retired V2 agent loop. |
| `selective_sources_conflict_contrast` | Paired case where another record does or does not resolve conflicting information. Imports the neutral catalog; covered by `tests/test_selective_sources_conflict_contrast.py`. | **Active development contrast:** retain to test source-based resolution versus truthful uncertainty. Its public variants are not a blind release benchmark. |
| `selective_sources_reference_contrast` | Paired records differ in which item an earlier conversation identifies as current; tests whether the answer follows that reference. Imports the neutral catalog and is covered by `tests/test_selective_sources_reference_contrast.py`. | **Active development contrast:** retain to test conversation-to-record linking and prevent a fixed-answer shortcut; historical run receipts remain separately labeled. |
| `selective_sources_identity_conflict` | Combines current-item identification with a conflict that may remain inconclusive. Imports the neutral catalog; `tests/test_selective_sources_identity_conflict.py` covers its cases. | **Active development contrast:** retain for combined identity, conflict and uncertainty behavior. Its two public cases do not establish overall reliability. |
| `selective_sources_identity_v3` | Follow-up diagnostic reusing the public variants/catalog; includes categorical answer-shape and question assessment helpers. `tests/test_selective_sources_identity_v3.py` imports those helpers. | **Active development diagnostic:** retain while its question assessment is used; `v3` denotes the diagnostic revision, not evidence that the production V3 loop is selected. Categorical assessment does not prove useful question delivery. |
| `selective_sources_appointment` | Paired synthetic service appointments and contacts test whether the current conversation reference determines which appointment and location to select. `tests/test_selective_sources_appointment.py` covers the fixture, and `docs/agent-loop/gym-decision-clock-2026-11.md` records its development use. | **Active development contrast:** retain as a non-prescription example of source selection. It does not connect to a real calendar, and its paired runs are not held-out qualification. |
| `token_usage_diagnostics` | Computes explicitly non-authoritative text-token estimates; imported by `codex_clock_comparison`, `codex_candidate_chooser`, `model_candidate_bridge`, `jev_attempt_meter`, and retained Responses-era adapters; covered by `tests/test_token_usage_diagnostics.py`. | **Active shared telemetry helper:** retain while those consumers need labeled estimates. Estimates exclude provider framing and hidden usage; they neither enforce a budget nor establish actual token use, cost or billing. |

The public development runs and receipts are discussed in
`docs/agent-loop/gym-decision-clock-2026-11.md` and
`docs/agent-loop/gym-source-selection-2026-11.md`. Preserve failed, unknown,
and later corrected observations with their provenance. No whole module is
recommended for removal in this pass: several are shared by other fixtures,
and each remaining case has a distinguishable source-selection or measurement
purpose. Reassess consolidation against actual case coverage and stored
baselines, not matching filenames alone. Searches cannot establish whether
external callers exist.
