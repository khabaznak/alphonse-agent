# Experimental System One Check and Act

Status: implemented, disabled by default
Depends on: Stage 4 outer review and routing

## Purpose

Alphonse V3 can use TypeSafe.ai Jev as a fast System One decision engine for the
semantic part of outer Check. The same request returns an advisory route for Act.
This is an optional provider path; it does not replace deterministic authorization,
effect, cancellation, or completion invariants.

## Check boundary

For each still-pending acceptance criterion and each bounded successful evidence
entry, Alphonse sends one Noul question asking whether that exact evidence directly
demonstrates the criterion. A high-probability answer may attach only the evidence
reference named in that question. Midrange answers are treated as ambiguous and the
entire semantic review falls back to the existing System Two phase-review call.

The request contains bounded phase objectives, immutable acceptance criteria, and
short verified-evidence summaries. It does not include API keys, unrestricted project
files, memory ledgers, or chain-of-thought.

## Act boundary

The same Jev request asks a Choice question among `complete`, `continue`, `replan`,
`ask_user`, and `fail`. The result is deliberately asymmetric:

- It may conservatively withhold an otherwise verified completion and route to
  strategic replanning.
- It may turn deterministic continuation of an incomplete task into replanning.
- It cannot create completion, failure, or a user question.
- It cannot override cancellation, waiting, blocked-phase handling, mutation scope,
  or verification failures.

This keeps Jev useful as a fast judgment layer without making a probabilistic answer
the authority for irreversible or terminal state transitions.

## Configuration and failure behavior

Administrators configure the endpoint, model, API key, and probability thresholds in
Desktop Settings → System One. Enabling requires a successful live validation. The
API key is stored in the local settings database and is never returned through IPC.

System One is disabled by default. Connection errors, malformed responses, missing
answers, or ambiguous evidence cause a per-task fallback to System Two. The last
validated settings remain unchanged after a failed validation. Structured telemetry
records provider status, latency, model, usage, and routing metadata without logging
the key or request contents.

## Verification

- Exact evidence-reference mapping is tested.
- Ambiguous evidence is tested to fall back.
- Provider failure is tested to fall back.
- Act is tested to withhold completion but never manufacture success.
- IPC access is administrator-only and API-key masking is tested.
- V3 remains opt-in; this feature does not advance the rollout beyond Gate A.

## Implementation log

- 2026-09-20 — Added the Jev HTTP client and persistent settings, bounded atomic Noul
  review, advisory Choice routing, System Two fallback, masked admin IPC, desktop
  settings and validation, and structured telemetry. Full Python suite: 472 passed;
  desktop suite: 47 passed.
