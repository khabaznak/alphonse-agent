# Evaluator-only pilot cases: sealed draft

Prepared 2026-11-13 alongside the gym improvement cycle. This is a small
synthetic **pilot**, not the independently sequestered qualification corpus
specified in `docs/v3-hierarchical-capd/27-agent-architecture-decision-protocol.md`.
Four separately authored prescription-source variants and their answer keys are
stored in separate evaluator-owned local files outside this repository. Neither
the case details nor answer keys should be provided to the candidate developer or
model during development; the evaluator exposes only the source data for the
single case being executed. All data are invented. No model has been run against
this pilot as part of its preparation.

The hashes below freeze the exact draft input and oracle bytes; changes require a
new version and new recorded hashes, rather than overwriting a prior result.

- Inputs SHA-256: `1cd6ce02a8c1d022fa5f1debd5e69256cc3616831d123e57fc72e5e2e849c399`
- Oracle SHA-256: `3b4fcc5527b0a10b9a6e7662f81b167cdb24ef852c37a9bffecf9e61194f7214`

## Evaluation protocol

Freeze the candidate revision, prompt, provider/model, limits and analysis plan
before any pilot execution. Use a fresh disposable source environment per
attempt, retain every attempt including failures and missing receipts, and
calculate grades only after execution. For each case, score structured answer
correctness separately from observations of sufficient authorized sources.
Report unavailable, unknown, clarification or blocked outcomes separately; do
not treat prepared text or a fixture response as actual user delivery. No
test-specific hidden key, list of required source reads, or grader output may
be passed to the chooser. The case and oracle files use a **draft data format**;
an adapter and independently reviewed scoring code are still needed before a
run, and adding an adapter must not expose other cases or oracle material.

## Limitations and qualification gates

The files are outside the repository, but filesystem permissions do not isolate
processes running as the same user. The existing Codex CLI `read-only` sandbox
is not proof of secrecy. Independent authorship, host isolation, answer-key
sequestration, semantic quality, and executable oracle correctness are not yet
independently verified. If candidate implementers or a model have access to the
pilot's contents before prompts/code freeze, label its results exploratory and
replace it. The four cases cover only a narrow task family; they do not meet
the protocol's 24 independently authored scenarios, other required families,
matched arms or five-attempt schedule. This seal alone is not qualification,
an external benchmark, a provider run, or clinical validation.
