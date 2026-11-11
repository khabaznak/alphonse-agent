# Coding-agent delegation for reusable skills

Status: proposal; no implementation or rollout claim

## Prototype status

An injectable `CodingSkillService` now coordinates durable job identity, a coding
provider, an independent verifier and installation through `SkillStore`. A
conditional native tool is enabled only when an immutable, explicitly configured
container image is supplied. A disposable live run used real Codex inside a
non-root container, verified a skill in a separate network-disabled container,
installed it, and activated and ran it through native Bash in the original
synthetic task. Sandbox probes denied the mounted credential and network access;
container exit and cancellation stopped tested detached descendants. No real-site
or autonomous household-task qualification has been performed.

The earlier host-only Codex prototype is not production-ready: a sandboxed process
could detach and keep writing after its parent exited. The container-backed runner
must confirm container removal before verification/publication; uncertainty preserves
the running job for explicit actor-scoped reconciliation. Credentials are mounted
read-only and copied only into container tmpfs. The build container has network
access for Codex itself while its generated commands are denied network access;
the independent verifier has no container network. A user-supplied brief must not
contain credentials or private household records. Host Docker and image integrity
remain trusted infrastructure, and container isolation is not a guarantee against
kernel or runtime vulnerabilities.

The container runs non-root with dropped capabilities, no-new-privileges, a read-only
root, bounded processes and memory, and a modified Docker seccomp profile that
allows the namespace syscalls required for Codex's nested sandbox. This widens the
default syscall policy. The runtime probes the nested sandbox before each build;
if Docker or isolation fails, no skill is published. An interrupted or ambiguous
job is never blindly rerun. Reconcile both named job containers and inspect any
possible publication before starting another operation. Questions, broader runtime
readiness and autonomous live qualification remain follow-up work.

### Configuring the local container

Build the bundled image with Docker and select its exact immutable image ID in the
daemon environment (do not supply a mutable image tag):

```sh
docker build -t alphonse-codex-sandbox:0.156.1 -f alphonse/agent_v2/coding_sandbox/Dockerfile alphonse/agent_v2/coding_sandbox
docker image inspect --format '{{.Id}}' alphonse-codex-sandbox:0.156.1
```

Run `alphonse configure-codex-sandbox <sha256:...>` with that ID, then restart
the daemon. Alternatively, set `ALPHONSE_CODEX_SANDBOX_IMAGE_ID` in the daemon
environment; it overrides the stored setting. Use `alphonse configure-codex-sandbox
off` to disable the stored setting. The daemon requires a persistent configured
household workspace, a reachable Docker daemon, a non-root host identity, and
an existing Codex `auth.json` under `CODEX_HOME` (or `~/.codex`). Optional
`ALPHONSE_CODEX_DOCKER_BIN` selects Docker; `ALPHONSE_CODEX_CODING_MODEL` selects
the Codex model. The image must be built from the bundled Dockerfile by a trusted
administrator; an immutable ID does not by itself prove image provenance.
Without the image setting, the native coding tool is not registered. Docker and
the image are not automatically downloaded or installed by Alphonse.

## Outcome

Alphonse owns a household request from start to finish. When a reusable software
capability would help, it can commission a coding agent to create or repair a
standard skill, evaluate the result, activate the skill and continue the original
task. A successful build is an intermediate result, not completion of the user's
request. An existing suitable skill should be reused or maintained when appropriate;
not every task requires a new skill.

The Jalisco traffic-check request is a representative acceptance case: given the
user's authorized vehicle data, check each vehicle at the official portal, explain
the actual results and, if requested, arrange a recurring check. A repeatable skill
should distinguish an explicit no-debt response from an inaccessible page, changed
form, CAPTCHA, ambiguous response or other failure. A scheduled check is a separate
verified effect; skill installation or a successful one-time check does not prove
that a schedule was created. The site and its requirements must be verified at
execution time, not assumed from this example.

## Observations from the household conversation

A read-only review of the relevant conversation and selected task checkpoints
found the following. These are observations, not a complete causal analysis.
Private vehicle data and household file contents are intentionally omitted.

- Alphonse asked the user to approve a notes directory after being asked to suggest
  one. It then created a plan and a vehicle-data template. Subsequent exchanges
  show inconsistent interpretations of the data and repeated requests for guidance.
- Several resumptions ended before useful work with an invalid execution plan. Two
  inspected checkpoints reported `inspection_handoff_requires_action_or_question`
  and `inspection_handoff_requires_source_resolution_before_mutation`; each recorded
  one plan-validation recovery attempt. Other inspected checkpoints ended with a
  generic `v3_runtime_error:ValueError`; this alone does not identify their cause.
- A response treated submission of the lookup form as outside a read-only scope;
  the user then explicitly asked Alphonse to proceed. The transcript contains
  inconsistent language about whether a submission happened. It does not establish
  verified portal results or a working automation.
- The user said they independently validated the supplied vehicle data on the site.
  That does not establish the current state of the local file, the website, or a
  reusable skill. Do not copy, modify or commit the household program or data to
  this repository while investigating.

Separately, the repository's historical live resourceful-delegation qualification
reported 12/28 passing trials and did not qualify live websites. See
[the retained report](v3-hierarchical-capd/evidence/resourceful-delegation-live-v1/README.md).
That evaluation is not a replay of this household session and does not prove its
cause. Investigate current fixes and baseline behavior before attributing the
observed failures to any one component.

## Architecture and boundaries

Follow the [standard skills contract](standard-skills-contract.md),
[script execution](skill-script-execution.md), [bindings](skill-bindings.md),
[management](skill-management.md) and [maintenance](skill-maintenance.md).
Correcting an earlier proposal: there is no new artifact registration or per-script
registry. The deliverable is a standard `SKILL.md` package with optional ordinary
scripts and supporting resources, installed in user-owned skill storage. Live
household data, credentials, runtime state and logs remain separate. Discovery and
execution use existing skill and native-tool controls.

Use two cooperating layers:

1. **Skill-building workflow.** Alphonse frames an outcome, checks existing skills,
   assembles relevant prior attempts and sanitized evidence, defines acceptance
   checks and chooses when to delegate. It can request a new package or a revision
   to an existing package. System Two writes the brief and decides next steps;
   any Jev questions are bounded relevance/progress judgments, not permission.
2. **Native coding-job capability.** A product-owned service invokes a configured
   provider, tracks a durable job, returns a structured receipt and supports
   cancellation. A provider adapter initially targets Codex, with an interface
   that allows another coding agent later. The existing Codex inference transport
   is for model inference in a temporary directory; it must not be assumed to be
   an agentic coding-job integration merely because it invokes the Codex CLI.

The workflow must continue in an ordinary conversation without requiring a project.
Install/publish authority, package access, native effects, bindings and task
cancellation are checked deterministically; selection or model claims grant no
permission. An authorized user request defines the work's scope. Ask only when a
required decision or authorization is genuinely missing; use existing consent and
normal policy for in-scope actions, including the data sent to an external provider
or website. A lookup's POST method alone does not establish that it changes records;
classify the actual operation and disclosure under the existing policy.

## Job contract and lifecycle

- **Brief:** original objective and acceptance criteria, authorized actor and task,
  scope and allowed effects, relevant evidence/failures, package intent (new or
  expected existing revision), required interfaces and resource limits. Include only
  necessary sanitized data. Never put credentials or private household records in
  the reusable package or prompt; use synthetic data for initial development.
- **Workspace:** stage the proposed package in a job-specific directory outside
  installed packages and household data. This is an organizational boundary, not
  a security sandbox. Before claiming restrictions on filesystem, environment or
  network, implement and verify a real process/credential isolation boundary for
  the supported provider and host. If the selected provider cannot satisfy required
  restrictions, fail with a clear setup/actionable reason. Do not silently run with
  broader access.
- **Job states:** queued, running, awaiting essential input, succeeded, failed,
  cancelled and interrupted/reconciliation-needed. Persist actor, task, provider,
  operation ID, workspace, timestamps and bounded, redacted progress/receipt.
  Resume by inspecting durable state; do not blindly repeat an uncertain job or
  publication. Timeouts and cancellation must terminate the process tree where
  supported and prevent late publication; recheck authorization before each
  consequential step. Record what is known when termination cannot be confirmed.
- **Receipt:** proposed package location and digest, change summary, instructions,
  prerequisites, declared effects, checks actually run and their results, known
  limitations and any required human interaction. Provider assertions are evidence
  to inspect, not proof of correctness or authorization. Limit retained output and
  redact sensitive input; treat generated instructions and external pages as
  untrusted content, not policy.
- **Validation and publication:** validate package format and containment, review
  resources and effects, perform relevant independent behavior checks using fixtures,
  check environment readiness and bindings, and compare expected revision before
  publication. Use the existing package management/maintenance paths where their
  semantics apply; identify and implement any missing staged-install capability
  rather than bypassing authorization, conflict handling, provenance or rollback.
  A failed or cancelled draft leaves the active revision unchanged. Never silently
  install dependencies or change bindings. In a fresh activation, load the installed
  revision and run the requested operation under normal native controls, verifying
  its actual result before responding. Repair failures through a bounded iteration,
  carrying diagnostic evidence forward.

## Delivery slices

1. **Diagnose and reproduce the reliability gap.** Turn the invalid inspection
   handoffs, continuation ambiguity, justified lookup submission and no-progress
   behavior into minimal disposable regressions. Inspect the existing validation,
   recovery and task-continuity paths; fix confirmed product defects independently
   of delegation. Preserve legitimate effect checks and distinguish historical
   evidence from current state.
2. **Prove the end-to-end contract with a fake provider.** Implement the minimal
   job lifecycle and a provider test double that writes a standard skill into a
   disposable staging directory. Exercise creation, validation, installation,
   fresh discovery, use, and original-task completion without real household data.
   Include failure, cancellation, duplicate operation, stale revision, revoked
   access, malicious paths, and interrupted publication cases.
3. **Add one real coding-agent adapter.** Explicitly configure the executable,
   authentication, model, supported isolation and resource limits. Verify actual
   subprocess behavior, progress, timeout, cancellation, unavailable provider and
   recovery from restart. Provider integration must not assume the inference
   adapter grants coding capabilities or an isolation guarantee.
4. **Qualify the butler workflow.** Run an autonomous evaluation with a local
   simulated traffic site and separate cases for existing-skill repair, missing
   dependencies, CAPTCHA and changed results. Measure completed original requests,
   correct unknown/failure handling, reuse in a fresh conversation and unnecessary
   user interventions. Independently verify effects rather than trusting job output.
5. **Pilot with the household.** With the applicable authorization and a working
   isolation/data path, inspect the actual portal and existing household work in
   place. Start from sanitized tests; use the existing private data only through an
   authorized binding or explicit, narrowly scoped input. Verify each actual result
   against the official site, then verify any requested schedule and delivery
   separately. Report real obstacles rather than asserting a working checker.

## Decisions to resolve during implementation

- Which installed provider and host isolation mechanism can enforce the required
  filesystem, network and credential boundary? What are the actual guarantees on
  each supported platform?
- Which existing job/task persistence and package installation APIs can support
  staging, recovery and atomic publication without adding a parallel registry?
- How should essential provider questions be relayed and answered within the
  originating conversation, including restart and cancellation?
- What evidence is sufficient to trust a site's negative result, and how will a
  changed site trigger repair without misreporting an empty result?
- What limits on retries, runtime and provider spending should be configurable,
  and what user-facing behavior follows when they are reached?
