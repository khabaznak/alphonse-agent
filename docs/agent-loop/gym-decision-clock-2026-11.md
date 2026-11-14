# Gym cycle: action format and clock diagnostics

Synthetic development evaluation, 14 November 2026 in America/Mexico_City.
This continues [the source-selection cycle](gym-source-selection-2026-11.md).
The prior final-version prescription run blocked on an invalid action after
using its one format repair, while the clock read its source but returned an
answer that failed the oracle. Neither raw malformed decisions nor the wrong
clock answer were retained, so their exact shapes and causes are unknown.

## Change and measurement

The synthetic Codex chooser now requests an explicit four-field action schema
from the CLI and checks returned action shape and allowed source names locally.
Invalid actions are not executed; the candidate's existing bounded format
repair remains in effect. The schema does not grant tool access or validate an
answer's truth. This change is confined to the evaluation chooser, not a
registered household processor.

The clock evaluator now records closed-set post-run diagnostics for the
time, date and timezone fields without preserving response text or supplying
the oracle to the candidate. An offline contrastive check exercises a UTC
instant in two timezones across a date boundary. Neither addition establishes
why the previous clock answer was wrong; no earlier receipt was changed.

The following three attempts were scheduled before dispatch and run once each
with signed-in Codex subscription, requested model `gpt-6-luna`, no Jev, a
35-second per-call timeout and 90-second task deadline. The provider accepted
the action schema; no failed case was retried. `matched_provenance` keeps the
prescription oracle's required-source list out of the candidate.

| Case | Post-run fixture oracle | Time | Model calls | Fixture reads |
| --- | --- | ---: | ---: | ---: |
| Prescription v2 | Pass: expected answer and specified three-source coverage | 32.49 s | 6 | 4 |
| Clock | Pass: source and exact answer matched | 16.02 s | 3 | 1 |
| Synthetic inbox | Pass: source and answer matched | 15.65 s | 3 | 1 |

These are single tuned development attempts. They do not establish a reliable
fix, a speed improvement, selective source relevance, real delivery or clinical
correctness. The prescription source catalog contains only the index and the
three sources used by its conservative coverage oracle. The source audit uses
a bearer capability and does not establish who performed the reads or whether
they all preceded answer formation. Fixture inputs and expected values are
separately declared but not independently authored or attested. The clock
diagnostic tells us this run matched, not which field failed previously.

The prespecified plan and all three exclusive, fsynced receipts are in
`evidence/gym-cycle-2026-11-13/decision-clock-v1-*`. They include source hashes,
an inventory digest, categorical traces and oracle results. The inventory is
not a complete environment or provider attestation. Provider token usage and
real recipient delivery remain unknown. The separately sealed held-out draft was
not run and is not yet qualification-grade. The candidate remains unregistered
for normal household conversations and the production release gates remain open.
# Selective-source development follow-up — 14 November

## Guidance experiment

### Paired source-selection contrast — 14 November

The chooser's general question guidance was clarified to consider source
descriptions and relevant observations, without implying that every source's
contents must be read. A separately scored development contrast uses the same
records and descriptions but asks about the exercise program, making the
previous distractor relevant. The two tasks were scheduled before dispatch,
with one subscription-backed attempt each, and the candidate received neither
answer keys nor required-source lists.

The pharmacy attempt answered correctly and read the prescription and
conversation records, without reading the exercise record or index: **2 reads,
4 model calls, 21.30 seconds**. The exercise attempt answered correctly and
read the exercise record without reading the other records or index: **1 read,
3 model calls, 13.76 seconds**. The plan and categorical receipts are
`evidence/gym-cycle-2026-11-13/selective-contrast-pair-*-v1.jsonl` and
`selective-contrast-pair-v1-plan.json`. Actual provider token use was unknown.
These are encouraging task-specific observations, not a demonstrated causal
effect of the wording change or an estimate of general reliability; the runs
were concurrent, not held out, and earlier failures remain part of the record.
Host trace identifies decisions and completed callbacks, while the bearer audit
cannot prove process attribution or when an answer was formed. No real user
delivery was assessed.

### Source-order diagnostic — 14 November

An opt-in synthetic trace now records whitelisted source selections, completed
source-read callbacks and whether the chooser had previously proposed an answer;
it retains neither model text nor record contents. In one new subscription-backed
attempt on the same development case, the candidate selected prescription,
exercise-program (`treatment-status`), and conversation in that order, all
before its first answer proposal. It then answered correctly (three reads, five
model calls, 26.53 seconds). This locates the extra selection in this attempt
before the self-review, rather than during the review; it does not establish
the model's reasoning or explain earlier attempts. The bearer service audit
does not independently identify the reader or determine when an answer was
formed; the order reported here is from host/chooser instrumentation. The
receipt is `evidence/gym-cycle-2026-11-13/selective-sources-order-v1.jsonl`.
This is one further development observation, not held-out evidence or proof
of an improved selection strategy. Provider token usage remained unknown.

After the first attempt below, the synthetic subscription chooser was updated
to prefer sources that can resolve a specific missing fact or material
uncertainty, skip clearly unrelated descriptions, and inspect uncertain
sources that could matter. This is an evaluation-only prompt change, not a
production source-selection mechanism. The same development case was run once
with the revised guidance: the answer was correct and both required records
were observed, but the distractor was still read. The optional index was not
read (three source reads, five model calls, 28.70 seconds). The previous
attempt had four reads including the index, but a single stochastic attempt in
each condition cannot establish an improvement or attribute the difference
to the guidance. The failure to avoid the distractor remains unresolved.

One attempt each on the existing prescription, clock and inbox development
cases passed their independent fixture criteria, with three, one and one
source reads respectively. The prescription coverage criterion remains a
conservative fixture requirement, not proof that every source is logically
necessary in every formulation. All four attempts were run concurrently with
the signed-in subscription provider, requesting `gpt-6-luna`; provider token
usage was not reported. The raw categorical receipts are
`selective-sources-guidance-v2.jsonl` and `source-guidance-v2-{prescription,clock,inbox}.jsonl`
in `evidence/gym-cycle-2026-11-13/`. The three regression receipts retain the
old runner's `decision-clock-v1` phase label despite using the revised chooser;
their recorded source hashes identify the code actually used. These are not
held-out cases and do not establish qualification, real delivery, attribution
of reads to a process, or timing of reads relative to answer formation.

## First selective-source attempt

The previous prescription catalog exposed only task-relevant substantive
records, so reading everything passed its source-coverage criterion. A new,
separate synthetic exercise asks which pharmacy belongs to the current
prescription. Conversation identifies the current prescription, a second
record maps prescriptions to pharmacies, and a third contains exercise-program
notes. The candidate sees the source descriptions but no answer key or
required-source list. An independent post-run key scores the answer and records
which sources were accessed. The five-read budget permits reading the extra
record without preventing completion.

One subscription-backed candidate attempt completed in **28.74 seconds**.
It gave the correct answer and the audit observed both required records, but it
also read the unrelated exercise-program record and the optional index: four
reads, six model calls. Thus this attempt shows successful information joining,
but does not demonstrate selective avoidance of the distractor. Provider token
usage was not reported; usage estimates and categorical traces are in
`evidence/gym-cycle-2026-11-13/selective-sources-v1.jsonl`. The source files
recorded in the receipt had unchanged hashes across the attempt.

This is one exploratory development example, not a held-out qualification or a
general reliability measurement. The loopback audit observes bearer-capability
access, not who made the reads or whether they preceded answer formation; real
delivery was not tested. The exercise does not explain *why* the extra source
was read. The next useful work is to diagnose and improve relevance selection,
then test on varied cases, including held-out examples kept from the model.
