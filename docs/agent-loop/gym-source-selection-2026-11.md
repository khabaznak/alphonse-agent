# Gym cycle: selecting relevant sources

Synthetic development evaluation, spanning 13–14 November 2026 in
America/Mexico_City. This continues [the preceding cycle](gym-cycle-2026-11-13.md),
whose last prescription attempt matched the answer but failed its specified
three-source coverage rule. No real household data, real Gmail, Jev or user
delivery was used.

## Change being evaluated

The experimental candidate can opt in to one budgeted, checkpointed pre-answer
self-review. When it first proposes an answer, the text is held back; the next
model decision sees the task, authorized observations and a general request to
consider whether another relevant authorized source might change the answer.
It may read, ask, or answer; the host does not give it the oracle's source list
or require it to read every source. Existing source authorization, cancellation,
freshness and budget controls continue to apply. The flag is enabled only by
the subscription-backed synthetic host and defaults off elsewhere. This is a
model self-check, not independent semantic verification or production authority.

## Recorded development results

All runs used the signed-in Codex subscription, requested `gpt-6-luna`, a
35-second per-call timeout and 90-second task deadline. `matched_provenance`
kept the prescription oracle's required-source list out of the candidate.
Results below use the post-run fixture oracle, not the processor's self-reported
completion. Provider token usage and actual delivery were not observed.

| Case | Result | Time | Model calls | Fixture reads |
| --- | --- | ---: | ---: | ---: |
| Prescription, first run | Pass | 34.02 s | 6 | 4 |
| Prescription, prescheduled repetition 2 | Pass | 31.78 s | 6 | 4 |
| Prescription, prescheduled repetition 3 | Pass | 36.94 s | 6 | 4 |
| Clock | Pass | 18.44 s | 4 | 1 |
| Synthetic inbox | Pass | 13.45 s | 3 | 1 |

Each prescription run recorded the expected answer and reads of the order,
start conversation and treatment status, plus the index. The two extra
prescheduled repetitions were declared in a separate plan before they ran.
The preceding one-attempt configuration failed the same coverage criterion and
used 3 calls in 16.94 s; its clock and inbox each passed with 2 calls. These
comparisons are useful development observations, not evidence of a population
reliability or latency change. This is 3/3 on the **same tuned development
case**, not three held-out cases. The fixture exposes just the index and the
three required source names, so reading all of them cannot demonstrate
selective relevance or resistance to needless reads.

Separate contrastive offline tests show that changing only treatment status or
only start-date availability changes the specified expected outcome. The
three-read prescription rule remains a conservative fixture criterion; an
explicit current status could support a different interpretation, so missing a
read alone does not establish an incorrect or unsafe answer. Input and expected
fixture values are separately declared but reside in the same comparison
module; independent authorship has not been established. The loopback audit
observes bearer-capability reads before evaluation, not who made them or whether
they preceded answer formation. Categorical traces do not label the review turn.

Every live attempt and the repetition plan have separate durable receipts under
`evidence/gym-cycle-2026-11-13/` with recorded source hashes and inventory
checks. The receipts contain categorical traces, not raw model decisions, and
their Python inventory is not a complete environment attestation. A subsequent
code review found cancellation could be observed before a new loop iteration
yet sources were still loaded and another reservation made. A per-iteration
cancellation check and regression were added **after** the runs above; those
results do not measure the exact final candidate source hash.

## Final-version rerun after cancellation fix

The three cases below were scheduled before dispatch and run once each with the
same model and limits against the new source hash. Every provider event returned;
no failed case was retried. The plan and all three receipts are preserved in the
same evidence directory under `source-selection-v2-*`.

| Case | Result | Time | Model calls | Fixture reads |
| --- | --- | ---: | ---: | ---: |
| Prescription | Fail: blocked on invalid decision; no answer | 38.89 s | 7 | 4 |
| Clock | Fail: source matched, answer did not | 18.49 s | 3 | 1 |
| Synthetic inbox | Pass | 11.85 s | 3 | 1 |

All three substantive prescription sources and the index were observed before
evaluation, but that is not a passing outcome without an answer. The categorical
decision trace was four tool actions, an invalid action, an answer, and another
invalid action. The prior format-repair allowance had already been consumed;
the runner did not retain the malformed actions or answer text. The clock's
incorrect answer was not retained either. These are distinct failures and must
not be explained as consequences of the cancellation fix: stochastic output and
other unmeasured factors prevent that attribution. In particular, the previous
3/3 prescription passes belong to the preceding source hash, not this one.

The results support further work on source discovery, bounded decision recovery
and answer accuracy. They do not establish that the self-review reliably improves
task success or generalizes beyond the development fixtures.

The separate held-out draft described in [the seal](heldout-pilot-seal-2026-11.md)
was not run or exposed for tuning. It is not qualification-grade: independent
review, executable scoring, same-user isolation and broader coverage remain
outstanding. The candidate remains unregistered for household conversations.
