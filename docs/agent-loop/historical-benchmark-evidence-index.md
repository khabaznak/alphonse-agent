# Historical benchmark and gym evidence index

Inventory for #229, 14 November 2026. This is a map of retained reports and
their limits, not a new evaluation, a release decision, or a claim that the
older V2/V3 loops have been removed. The owner chose the simpler loop as the
architecture to develop; the evidence below does not by itself qualify it for
normal conversations. Retain failures, interrupted attempts, unknown values,
manifests, source hashes, and historical fixtures when retiring evaluation code.
No provider calls or household data were used to assemble this index.

**Later owner direction:** executable historical V2 replay and the PDCA loop
have been removed locally as part of the move to one simpler loop. These
reports describe the original runs; the old implementation is available in
Git history rather than kept as a runnable baseline in the current tree.

## Loop comparisons and gym work

| Evidence | Kind and retained material | What it can and cannot establish |
| --- | --- | --- |
| [Gate A: 20 September 2026](../v3-hierarchical-capd/evidence/2026-09-20-gate-a/README.md) | Deterministic V2/V3 replay; [case report](../v3-hierarchical-capd/evidence/2026-09-20-gate-a/v3-comparison.md) and [raw JSON](../v3-hierarchical-capd/evidence/2026-09-20-gate-a/v3-comparison.json). | Historical developer opt-in passed 9/9 under scripted providers. Counts measure fixture orchestration, not live model quality, latency, delivery or production safety. |
| [Gate B offline replay: 1 October 2026](../v3-hierarchical-capd/evidence/2026-10-01-gate-b-offline-replay/README.md) | Ten-case deterministic replay; case-level [report](../v3-hierarchical-capd/evidence/2026-10-01-gate-b-offline-replay/v3-comparison.md) and [JSON](../v3-hierarchical-capd/evidence/2026-10-01-gate-b-offline-replay/v3-comparison.json). | Gate B was **not passed**: V2 5/10, V3 6/10; full suite then had 21 failures. The cross-channel reminder was a stub, not real delivery. Do not overwrite the failed baseline. |
| [Offline simpler-loop comparison, #218](../v3-hierarchical-capd/27-offline-comparison-evidence.md) | Scripted fixtures and corpus hashes; the report records results, but full per-run artifacts were not saved. | Harness and instrumentation check only. Different scripted providers and unavailable arms do not establish autonomous performance, provider cost, or delivered messages. Reproduction commands describe the recorded revision, not necessarily current imports. |
| [Subscription clock observations](../v3-hierarchical-capd/37-subscription-clock-first-observation.md) | Processor-level synthetic runs with signed-in Codex, including failed first attempts, later prompt changes and a corrected V3 response-tool fixture. | Retains a fairness defect in the original V3 fixture and small exploratory matched reruns. Do not pool different versions or infer a reliability rate. Recipient delivery and precise provider billing were not observed. |
| [Subscription inbox and prescription observations](../v3-hierarchical-capd/38-subscription-mail-prescription-observation.md) and [candidate recovery experiment](../v3-hierarchical-capd/38-candidate-recovery-experiment.md) | Synthetic source and post-run oracle results; recovery changes followed observations. | The one inbox attempt passed for each arm; initial prescription attempts failed. Recovery mechanisms differ between arms, so comparisons are not fully matched. No real mailbox, patient record, clinical judgment or user delivery was exercised. |
| [Gym cycle: multi-source and action format](gym-cycle-2026-11-13.md) | Nine separate subscription-backed synthetic attempts, including the baseline and unsuccessful prescription attempts; [categorical receipts](evidence/gym-cycle-2026-11-13/). | The two later prescription runs returned expected-looking answers but failed source coverage. Clock and inbox passed in these small samples. Raw model decisions, answer text, billing and recipient delivery are not captured. |
| [Gym: source selection](gym-source-selection-2026-11.md) | Self-review development attempts with a prescheduled repeated prescription case and a later final-source-hash rerun. | Three passes on the same tuned prescription case belong to the earlier source hash; the final-version rerun failed prescription and clock and passed inbox. These are development observations, not held-out reliability or causal proof. |
| [Gym: action schema, clock and source contrasts](gym-decision-clock-2026-11.md) | Prescription/clock/inbox plus selective-source, ambiguity, identity and appointment development cases; individual plans and receipts are in [the gym evidence directory](evidence/gym-cycle-2026-11-13/). | One tuned attempt per reported case generally cannot establish reliability. For example, the shared-policy appointment reruns read an unrelated note even though earlier appointment attempts did not. The bearer audit cannot prove who read a source or whether every read preceded answer formation. |

The gym directory contains separate start/final records and source inventories
where the applicable runner provided them. The associated reports specify which
plan and receipt group corresponds to each experiment. Some older reports state
that chooser logic was evaluation-only; use the dated [shared-policy follow-up](gym-decision-clock-2026-11.md#shared-chooser-policy-regression)
and current code to understand subsequent reuse. Do not relabel earlier runs as
results of the newer policy or source hash. Provider-reported token use is
unknown where the receipt says unknown; visible-text estimates are not bills.
The first subscription-clock aggregate did not retain individual V3 trajectories,
actual response text, or provider token totals. Those missing records remain
unknown and cannot be reconstructed from the summary.

## Related V3 and harness evidence

These older experiments measure different capabilities and are useful context,
not matched V2/V3/simpler-loop comparison arms:

- [Paired live harness smoke, including attempts 1–6](../v3-hierarchical-capd/evidence/paired-live-smoke-2026-10-19/README.md): each attempt has its own manifest, traces, gates and human-review sheet. Source and fixtures changed between attempts; failures, interruptions and the two failed reminder outcomes remain important. They must not be pooled into a single success rate.
- [Autonomous resourceful-delegation qualification](../v3-hierarchical-capd/evidence/resourceful-delegation-live-v1/README.md): 12/28 passed; qualification failed. Its [case results](../v3-hierarchical-capd/evidence/resourceful-delegation-live-v1/case-results.md), manifest, compressed runs and cost ledger preserve failures. This is the actual V3 processing path with synthetic stores, not a current simpler-loop or full daemon qualification.
- [Household delegation evidence](../v3-hierarchical-capd/evidence/household-delegation-2026-10-18/README.md), [memory baseline](../v3-hierarchical-capd/evidence/memory-baseline-v1/README.md), [memory live review and rescoring](../v3-hierarchical-capd/evidence/memory-live-2026-10-17/README.md), and [decision-context qualification artifacts](../v3-hierarchical-capd/evidence/dc5-2026-10-22/README.md) retain their own scope, revisions and limitations. Consult each original report rather than aggregating unrelated results.
- The broader [historical evidence tree](../v3-hierarchical-capd/evidence/) also retains memory search/recall/migration, lifecycle, context-pilot, issue-190 and earlier paired-harness artifacts. This index does not assert that every artifact is an autonomous benchmark or that every result is passing.

## Preservation and retirement decisions

1. Preserve the reports and receipts above, including all unsuccessful and
   superseded attempts and their source/configuration provenance. Do not edit
   recorded outcomes when code, prompts, or interpretation change; append a
   dated correction with its own evidence.
2. On 14 November the owner chose one congruent simpler loop and directed that
   executable V2 and V3 loops not be retained for historical replay. Git and
   the preserved reports retain historical structure and results. Extract useful
   engine-neutral gym cases and shared helpers, then remove the old replay and
   its dependencies. V3 remains the live bridge until the replacement handles
   the supported conversation path; this direction is not a claim that the
   replacement is already running or that legacy state recovery is complete.
3. Keep synthetic/scripted, real-model-on-fixture, and production evidence
   distinct. A prepared answer is not a delivered message; source access alone
   is not proof of semantic grounding; unknown usage is not zero cost.
4. [#216](https://github.com/khabaznak/alphonse-agent/issues/216) tracks actual
   stale-worker and recovery obligations. [#217](https://github.com/khabaznak/alphonse-agent/issues/217)
   and [#218](https://github.com/khabaznak/alphonse-agent/issues/218) cover the
   historical decision protocol and comparison. Their scope has been reconciled
   with the owner's direction; they remain open for outstanding reliability,
   measurement and qualification work, not to require old-loop replay.
5. The [held-out draft seal](heldout-pilot-seal-2026-11.md) is not itself a
   qualified held-out study. Preserve separation and do not expose hidden cases
   or imply that a development pass qualifies household use.
