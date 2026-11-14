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
