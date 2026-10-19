# Household delegation: demonstrate the useful behavior first

The household owner clarified the priority on 2026-10-18: Alphonse must carry a
delegated responsibility through to its requested outcome, make unresolved failures
visible, recover resourcefully, and retain useful improvements. A household computer
with an administrator is still a multi-user system: other members use channels such
as Telegram. Deployment size is not a reason to narrow the product's purpose.

Memory redesign is valuable when it enables that delegation. Prompt reduction alone
does not establish success. This diagnostic pilot exercises recognizable tasks
before the repeated, paired rollout evaluation in protocol v2. No numeric gate is
relaxed, and no feature default is changed.

## Five responsibilities

The versioned synthetic corpus is
[`tests/fixtures/household_delegation/v1.json`](../../tests/fixtures/household_delegation/v1.json).
The [v2 corpus](../../tests/fixtures/household_delegation/v2.json) preserves those
cases and explicitly asks the learning skill to prefer the verified working route.
The v1 learning oracle already required that preference although the request only
asked for a reusable workflow. V2 makes that requirement visible to the agent;
earlier failed v1 scores remain unchanged. This is an exploratory corpus revision,
not a relaxation of the frozen 22-case memory benchmark or its rollout thresholds.

1. Anniversary recall: find a wedding-celebration dinner record from last month,
   beyond the recent conversation window, then answer from that evidence.
2. Medication reminders: resolve the requester's mother and her recorded reminder
   instructions, create the correct recurring reminder for her registered Telegram
   address, verify the persisted message/schedule, and queue its due fixture message.
   The medication is invented; the model is not asked to prescribe or change it.
3. Recovery: when the usual sensor route fails, discover and use a working live
   alternative, then report the fresh reading.
4. Visible failure: when every available sensor route fails, explain that a fresh
   reading could not be obtained. An honest limitation is distinct from obtaining
   the measurement; both are reported explicitly.
5. Reusable repair: recover a working sensor route, present a complete instruction-only
   skill draft, obtain a predeclared synthetic user's approval, install into the
   disposable skill repository, and have a new task select and reuse that skill.
   This tests retention of reusable instructions, not autonomous product-code repair.

Skills are the primary reusable unit. Improvement can update their instructions,
but it can also repair an existing artifact whose implementation is not yielding
the expected result. The skill supplies the expertise for diagnosis, modification
and verification; the artifact remains in its registered storage with its stable
identity. A combination of skills may contribute to that repair. This initial
instruction-only pilot does not establish that fuller repair loop; it remains a
separate responsibility to exercise. See [skill-owned resources](../skill-artifact-resources.md).

## Runtime and effect verification

The harness uses the real `HierarchicalCAPDProcessor`, its admission, planning,
tactical execution, Jev review and outer routing, plus real SQLite conversation,
knowledge, question, scheduling and identity stores. It has no real integrations.
Providers receive actual runtime inputs, never the oracle, expected answer or full
corpus. Jev supplies bounded decisions; System Two composes plans, queries, calls,
responses and skill instructions. No prewritten query or plan is supplied.

Native search, source expansion, scheduling, delivery modification, questions,
responses and skill installation execute normally against disposable stores. Bash
is replaced by an exact command allowlist that implements only the fixture sensor;
arbitrary commands never enter a shell. Skill files remain disposable fixtures.
The due-reminder check runs the actual scheduler and daemon outbox-enqueue method,
without running a Telegram sender. A fixture outbox entry demonstrates correct
routing and body composition, not a real provider delivery acknowledgement.

The outcome oracle reads stored schedules and checks the recipient, local time,
daily recurrence, preserved medication message and actual fixture outbox entry.
It checks real sensor attempts/results rather than accepting a completion label.
Recall diagnostics inspect successful inference inputs for delivered source text;
full call-time manifests are retained. These short-source diagnostics are not the
stage/range oracle required for the larger rollout protocol. Answers and traces
still require independent human review.

Adapter revisions are retained separately. Initial trials omitted the knowledge,
task-evidence and ledger adapters, causing artificial partial/unavailable coverage;
adapter v2 connects real empty fixture stores. Initial inference transport encoded
runtime JSON as a string inside another JSON object; malformed inner JSON was a
transport problem. Adapter v3 uses direct JSON, the runtime's parser and any supplied
output schema. Earlier attempts remain evidence of diagnostic limitations, not a
matched comparison with v3. These distinctions must not be hidden by aggregating
all trials into a rollout pass rate.

Adapter v4 gives each new or resumed task its own execution deadline while retaining
the workflow's total latency and call limit. A synthetic approval and a fresh task
are separate task admissions; time spent obtaining approval must not exhaust the
next task's budget. The scheduler uses the corpus clock for task creation and an
explicit due time for fixture dispatch. Earlier whole-workflow deadline failures
are retained, including a run that installed and selected the repair but was
cancelled before its final answer.

## Repairs motivated by the pilot

- Reminders can name a verified registered recipient while the requester retains
  ownership and the origin for failure notifications. The destination is resolved
  again when due; an inactive recipient cannot silently become a reminder sent to
  the requester. Creation returns a read-back of the actual stored schedule,
  recipient and message so the agent can verify its work.
- Terminal scheduler-dispatch and outbound-delivery failures notify the original
  requester. Notifications are idempotent per occurrence, and the original request
  channel is retained separately from the recipient's destination.
- A prepared response records the work evidence available when it was composed.
  Later work invalidates that response and requires a refreshed final answer. The
  original recovery trial read 24°C but retained an earlier "no reading" response;
  the task's success label did not make that useful or correct.
- Jev evaluates the original subgoal's completion separately from model-authored
  action questions. Reading CLI help cannot stand in for obtaining a measurement.
  An ambiguous operational-stage judgment cannot be converted to success by a
  generic output-present predicate. A response-only stage retains ambiguity
  fallback for message preparation, with content and mission verification in outer
  acceptance review. A later recall diagnostic exposed the need for this distinction.
- Read-only `local_fallback` stages permit a bounded alternative without requiring
  a new outer plan after every failed read. Three failed alternatives return to
  strategy. Unknown partial external mutations still require outer recovery.
- A separate bounded Jev judgment tracks whether an attempt adds meaningful new
  evidence. Three consecutive attempts without demonstrated progress return to
  strategy, including differently worded empty searches. Identical incomplete
  results remain a separate deterministic stop condition.
- Provider/composition exceptions persist a terminal task state and preserve prior
  effects. Failure explanations include bounded successful-result excerpts as well
  as errors, and a deterministic fallback remains available if explanation inference
  fails. Named secret fields are redacted; task arguments/commands are not added.
- Authenticated question answers are appended as typed resumption receipts and
  imported into later V3 phases. The legacy tool-call record alone was insufficient
  for V3 completion review after an approval. Receipts do not rewrite the original
  waiting event or manufacture an external effect. They retain the scope of the
  original question's phase so completion review can verify their provenance.
- Act checks whether authenticated follow-up work is fulfilled even when the
  original acceptance criteria omitted that later work.
  Acknowledging approval cannot substitute for installing the approved skill.
  Acknowledgment-only variations do not reset the outer no-progress counter.
  This guard applies to resumed authenticated answers, not ordinary factual
  questions. A diagnostic recall run exposed an overly broad version that
  repeatedly rejected a qualified factual answer. The final guard is scoped to
  the approval-provenance defect it addresses, with a regression for that distinction.
- Skill installation reads back the approved definition and returns verification;
  an identical retry verifies the existing skill, while a different definition at
  the same name fails rather than overwriting it.
- Planning criteria describe the requested outcome; applicable memory supplies
  missing people, destinations and operating details for the current request.
  Recall guidance explains how to broaden restrictive literal entity filters.
  Instruction-only drafts belong in user-facing text, followed by the normal
  approval and native installation workflow.
- Instruction-only skills use ordinary native capabilities. A skill ID must never
  be supplied as an artifact ID; only registered artifacts use that parameter and
  their verified resource manifests.

These are general product changes. None embeds a fixture answer, dinner venue,
medication, sensor command or household identity in production behavior.

## Reproduce and assess

The live command explicitly reads only the configured Jev provider credentials and
configured System Two model selection. Inference transport disables host tools,
apps, plugins, memories and local skills; the harness controls all fixture effects.

```bash
python -m alphonse.agent_v2.evaluation.household_delegation \
  --corpus tests/fixtures/household_delegation/v2.json \
  --output /tmp/household-pilot-new.json --live --arm candidate
```

`--case` selects one diagnostic case; `--arm both` runs legacy/candidate assembly
against separate fixture state. Output destinations must be new. Live pilot defaults
limit inference to 24 calls per case and use a cooperative six-minute deadline per
new or resumed task;
an in-flight provider request can outlast that deadline. A limit or provider
error remains a failed/aborted attempt, not a retry silently substituted for it.
Later attempts record source hashes and the corpus hash; actual prompts/responses, Jev decisions, usage,
timings, task state and effect read-back are retained. Missing provider usage remains
unavailable. No live household conversations, calendars, reminders or artifact
programs are used.

The synthetic approval in the learning scenario authorizes only its disposable
fixture skill. It is not a human approval of production changes or independent
review of the evaluation. A later explicit owner decision is still required for
default rollout. Broad held-out coverage, repeated paired behavior, performance,
interrupted-task rollback and independent review remain pending.

Retained exploratory attempts and their dispositions are indexed in the
[pilot evidence](evidence/household-delegation-2026-10-18/README.md). They span code
and adapter revisions and must not be pooled into a success rate.
