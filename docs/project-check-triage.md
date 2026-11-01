# Bounded project-check triage (#115)

The daemon runs acquisition and then `ProjectTriage` through the existing project
check worker. No account connection creates a policy, schedule, subscription or
monitor. The acquisition-only service remains independently usable/testable for
provider adapter work. The daemon always supplies the triage stage, including
when System One is disabled: missing/disabled/unvalidated Jev leaves candidates
pending and the check partial. There is no wholesale LLM scan fallback.

## Reviewed question contract

`project_triage.py` owns `project-check/1`, its SHA-256 question digest, and fixed
thresholds. Changes to instructions, rubrics, thresholds or routing require review
and a version bump. Each evidence item has four independent questions submitted
in one parallel Jev request: Noul project relevance, Noul actionable commitment,
Choice changed/unchanged/unknown, and Score urgency 0–2. Project scope/goals/query
are fixed context. Source MIME is normalized before triage, tagged untrusted and
kept separate. No source text supplies instructions, tool arguments or authority.
There is no fabricated prior comparison: absent comparison is explicitly unknown.

Strict finite numeric validation rejects booleans, numeric strings, NaN/infinity,
out-of-range values, incomplete/non-normalized distributions, unknown choices and
inconsistent expected urgency scores. Relevance/commitment use 0.2/0.8 cutoffs;
Choice and urgency confidence require 0.8. Midrange or malformed answers stay
pending. Relevant commitments, explicit changes and immediate obligations route
bounded interpretation to System Two. Unknown change cannot become unchanged.
System Two owns extraction, prose, strategy and concrete native tool arguments.

## Budgets, persistence and failure

One call handles at most 16 items/64 independent questions, 4 KiB normalized content
per item and 64 KiB serialized state including goals and source metadata. Content
is never silently clipped; oversized content and unexpanded attachments stay
pending. Large goals or an oversized batch fail atomically into pending results.
Use smaller acquisition budgets or an explicitly authorized bounded interpretation
workflow for such candidates; do not expand the retrieval scope automatically.

Each call reserves the existing policy's model-call budget before I/O. This path
has zero implicit retries and a transport deadline capped at 10 seconds and the
remaining window/runtime. Existing worker leases, cancellation, policy revision,
project lifecycle, exact binding/grant/resource/audience checks remain authoritative.
They are rechecked after inference; lease generations fence receipt persistence.
An in-flight remote call cannot be recalled, but cannot authorize further work.

Source pages/cursors remain independent of processing. Receipts bind the exact
run/evidence ID, content digest and source version. Resolved IDs alone enter the
classification checkpoint. A crash between receipts and that checkpoint is
reconciled without another inference. Ambiguity, outage and saturation produce
durable pending receipts and a partial check, never irrelevant/no-match. Subsequent
checks may reconsider their newly acquired evidence; this version does not retry
pending decisions in a terminal occurrence or automatically enqueue LLM recovery.

Acquisition no-match retains its existing invariant (zero acquired items). A fully
classified, complete-coverage all-irrelevant result stays quiet at publication while
preserving acquired item counts and a successful acquisition outcome. Partial or
pending results are published through the existing scoped event route for bounded
interpretation. No second event store, scheduler, queue or outbox is introduced.

Receipt telemetry contains contract version/digest, opaque batch ID, status,
question count, latency and a numeric allowlist of provider usage/cost fields.
No excerpts, prompts, answers, exception strings, addresses or credential values
enter telemetry. Batch metadata is shared by its per-evidence receipts: deduplicate
by `batch_id` before totaling cost. Unreported provider billing stays unknown;
zero/absent usage must not be interpreted as a free failed call.

## Managed account capability exposure

Activated V3 skills receive `SkillBindings.account_capabilities`, an advisory
manifest of the authenticated actor's explicit role choices and currently usable
operation/resource constraints for the exact project (or projectless request).
The method intersects grant and binding operations and checks installed skill,
ACL, connection state, scopes, credential-backend health and current grant before
exposing metadata. It exposes no credentials, configuration or other members'
choices. Revoked/unavailable choices disappear on the next assembly.

Execution remains the existing trusted managed adapter `use_account` path, with
checks before credential delivery and after provider I/O. Model-forged operations,
resources, recipients or project identities confer no access. A prompt manifest
is never a bearer token. Managed credentials cannot flow through legacy unrestricted
Bash bindings; ordinary Bash remains trusted host execution, not an OS sandbox.
Existing family-calendar package and native reveal/execution guards are reused.
There is no new Calendar tool or per-script/per-account tool registry. Concrete
provider adapter rollout and household skill implementations remain separate work.

## Synthetic calibration and validation

The held-out synthetic routing vectors in `tests/test_project_triage.py` exercise
11 cases: explicit commitment, unrelated evidence, unchanged information, absent
comparison, ambiguous relevance, correction, immediate obligation and probability
boundaries 0.79/0.8/0.21/0.2. All expected routes pass at the fixed 0.2/0.8 cutoffs;
midrange uncertainty never becomes a negative. Separate invalid-score/distribution
cases reject malformed certainty. These calibrate deterministic routing against
synthetic posteriors, not the semantic accuracy of a live Jev model. No live model
or private account access was performed or claimed.

A 33-item saturation fixture with a two-call policy verifies exactly two 64-question
calls, 32 resolved receipts, one durable pending item and partial coverage. This
bounds cost without losing evidence. Oversized state fails before inference;
attachments/oversized items, disabled Jev and outages retain pending evidence.
Scoped receipt, cancellation/revocation and crash-recovery tests cover persistence.
Existing V3 reveal and skill/Bash suites verify hidden calls and selected-skill
requirements; managed adapter tests reject forged calls before callbacks run.
