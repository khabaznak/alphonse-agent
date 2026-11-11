# Proposed synthetic live-comparison pilot (#218)

Status: **not ready to execute and not authorized**. This is a concrete pilot
proposal, not a measured benchmark or the held-out decision experiment. It
does not approve API calls, payments, access to household data, or publication.

## Scope for a future approval

Use exactly six fresh Codex CLI sessions: three attempts of one invented
traffic-ticket lookup and three of one invented prescription-memory task.
The external agent would be Codex CLI `0.156.1`, pinned to OpenAI
`gpt-6.1-sol` and a recorded reasoning setting, with no silent fallback.
The requested user ceiling would be **US $10 total incremental provider
charges** for the six attempts. No automatic reruns are included. The limit is a
proposed approval boundary, not currently enforceable by this implementation.
Run exactly the six attempts or stop early; a stopped, timed-out, blocked or
errored attempt remains in the report. This pilot would assess feasibility and
observability, not pass #217 or justify a production architecture decision.

The supplied material consists only of generated fictitious names, registration
identifiers, tickets, prescription instructions, start dates and timestamps.
The traffic task must use a local disposable portal with an independent
subject/result audit. The prescription task must begin with an earlier
synthetic conversation and discoverable records, not the answer pasted into
the final request; report a pasted-source control separately if later approved.
Send no real patient, vehicle, household, credential or mailbox information,
and do not contact real public or household services.

The six attempts exercise only the external Codex arm. The same cases and
boundaries must later be run for actual V3, the targeted V3 variant and the
single-loop candidate; this proposal does **not** authorize their model calls or
any Jev/TypeSafe calls. Missing comparator arms are unavailable, not successful
or absent from the record. A decision-grade comparison additionally needs the
separate, independently authored #217 held-out corpus and repetition plan.

## Execution and isolation prerequisites

None of the existing three-arm development adapters is ready for this run:
they use scripted decisions. Disposable loopback traffic and prescription
services now exist, including bearer capabilities, but they do not establish
process isolation, agent identity or complete attribution. The prescription
service is connected to offline-injected V3 and candidate processors only.
Before requesting final approval:

1. Implement an isolated loopback fixture transport that supplies only the
   task's synthetic information, exposes the same actions and read permissions
   to every arm, and records an evaluator-owned subject-scoped read/effect
   audit. Distinguish any browser and client capability differences. Include a
   real structured source-discovery path for the medication task.
2. Keep answer keys and grading rules outside the agent's accessible workspace,
   prompts, tools and other cases. Give each attempt a fresh environment and
   fixture instance, record actual arm order and predeclare limits. Verify
   source/effect observations independently of the agent's output. No prior
   Codex sessions, repository instructions, skill catalogs, personal files or
   connectors may be present. An isolated VM/container or equally restrictive
   execution boundary must be checked; setting a working directory or using an
   in-process allowlist alone does not establish read or network isolation.
3. Pin and verify the actual installed Codex executable, account/authentication
   mode, `gpt-6.1-sol` access, model identity returned at execution, tool
   availability, price schedule and usage visibility without printing secrets.
   `codex --version` only confirms the local CLI is installed; it does not
   prove account entitlement, provider charging or isolation. Record API vs
   ChatGPT billing and policy separately. If exact attribution is unavailable,
   report usage or costs as unknown rather than zero.
4. Implement and test a **pre-dispatch hard cost reservation** covering every
   possible request, retries, concurrent calls, input/output and applicable
   tool or other charges at verified prices, with remaining-cost accounting
   and fail-closed behavior on unknown prices or usage. Enforce this at a
   gateway or other boundary the CLI cannot bypass; no such gateway or full
   test exists yet. Kill/timeout alone does not cancel already-billed work.
   If an actual hard ceiling cannot be established, do not run under a promise
   of a strict US $10 maximum. A different, explicitly approved allowance for
   possible overshoot would be a separate choice, not an inferred approval.
5. Establish a dedicated project/account and network policy for only the
   approved provider and loopback fixture, with no inherited MCP servers or
   credentials except the isolated provider authentication mechanism. Run
   adversarial tests proving the agent cannot access household paths, host
   environment or unapproved network destinations. Treat failure or an
   inability to prove containment as a stop condition.
6. Freeze the protocol, fixture and oracle digests, executable/model/tool and
   permission settings, 3+3 order, timing boundaries, prices, budget control,
   retained traces and explicit approval before dispatch. Persist start and
   terminal records; an interrupted attempt has unknown outcome and is never
   silently retried. Audit first useful response, independent outcome,
   end-to-end time, user interventions, provider usage and complete costs.

Local retention proposed for approval: sanitized prompts, source text, tool
calls/results, traces and measurements in an access-controlled temporary
directory for 30 days, followed by deletion; never commit the raw traces.
The provider receives the synthetic task content and tool observations. Verify
the actual authenticated service's retention policy and account settings at
run time; local deletion and `--ephemeral` do not guarantee provider deletion.
For the OpenAI API, the published [data controls](https://developers.openai.com/api/docs/guides/your-data)
describe abuse-monitoring and endpoint-specific application-state retention;
those terms cannot be assumed to describe a ChatGPT-authenticated Codex session.

Official OpenAI documentation describes [Codex model selection](https://learn.chatgpt.com/docs/models)
and [non-interactive JSON output](https://learn.chatgpt.com/docs/non-interactive-mode).
The published [model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
lists API token prices, which must be rechecked against the actual service and
include applicable cached-input/cache-write rates and other charges. OpenAI's
[spend-limit documentation](https://developers.openai.com/api/docs/guides/spend-limits)
states project hard-limit enforcement is not instantaneous and recorded spend
can slightly exceed the setting. An account spend alert or hard project limit
alone therefore does not establish the proposed strict per-pilot cap.

## Current offline checks

These commands call no providers and do **not** start the pilot:

```sh
codex --version
codex exec --help
.venv/bin/python -m pytest tests/test_traffic_processor_development.py tests/test_prescription_safety_three_arm.py -q
```

The first two only inspect the installed client. The pytest command exercises
public synthetic scripted fixtures. A live launch command will be documented
only after the fixture interface, isolation and enforceable cost control exist;
there is currently no safe command to supply for the proposed live pilot.
