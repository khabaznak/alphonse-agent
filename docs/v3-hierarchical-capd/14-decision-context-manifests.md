# Call-time decision context manifests

Implemented for issue #17, 2026-10-18. The default context selection and the
opt-in issue #16 projection remain unchanged; this adds observability to both.

V3 processing captures a version 1 manifest at the application decision boundary:
System Two's inference router and Jev's bounded decision calls. Each logical call
gets a unique `decision_id`, including repeated and failed calls. System Two
passes that ID through inference metadata and telemetry. Transport retries belong
to the same logical decision. `returned` means the provider returned, not that its
answer passed semantic or deterministic checks.

The manifest is checkpointed before the provider call (`started`) and after its
return/failure. It identifies purpose, builder/schema version, timestamp, status,
actual input sections, included source IDs/types, selection reasons, source clips,
whole-record omissions, tool IDs and image asset IDs. Capturing a manifest neither
authorizes effects nor changes a task's terminal state. Jev still supplies only
bounded judgments; System Two composes calls.

## Both context builders carry provenance

The default path carries inventories with the rendered strings it already builds.
Conversation continuity identifies canonical sources and task-working records;
ledger input identifies complete knowledge records or an explicitly named
unstructured memory snapshot; global/project/skill guidance has its own identity.
The formatting helpers preserve those inventories when joining, selecting or
clipping sections. Prompt bytes and selection order stay the same. Source identity
is not guessed by parsing the final prompt or by reading later task state.

Default-path source IDs include `request:<message-id>`, `task-working:<task-id>:…`,
canonical conversation/knowledge IDs, `global:…`, `project:…` and selected skill
IDs. Task-working and input-field snapshots are explicitly different from canonical
memory sources. Each remaining application input field has a named inventory;
prompt framing is accounted for separately from the dynamic sources it surrounds.
There are no aggregate-only new calls on the default path.

With `ALPHONSE_V3_CONTEXT_PROJECTIONS=1`, the existing opt-in builder supplies
its token-budgeted included/omitted source inventory. Its atomic required records,
recall judgments, authorization, skill/artifact instructions and rollout flag
are unchanged. Both paths bind deep-copied receipts to the exact rendered inputs
at call composition. Serving a work log never rebuilds context. A previously built
projection is never attributed to an unrelated input.

## Omissions, clipping and counts

Omissions distinguish `irrelevant`, `duplicate`, `over_budget`, and `unavailable`.
Additional selection reasons identify continuity windows and deterministic review
filters. Unavailable adapters have no known visible record IDs: availability uses
an adapter identifier without enumerating inaccessible records. Authorization runs
before source selection and inventories; unauthorized records are never listed as
included or omitted candidates.

Whole-record/exchange budget omissions are distinguished from source clips.
Default project/skill bounds and Jev candidate/evidence-summary clips preserve
character ranges into the original bounded source. Review captures the exact
pending required criteria and actual evidence supplied, with explicit exclusions
for evidence outside the review window or not eligible as successful evidence.
System Two review separately inventories the full contract, phase, evidence and
valid-evidence-reference filter. Its supplied evidence may include failed records;
being present in a review input does not make a record verified evidence.

Counts use the documented conservative UTF-8 byte fallback, not provider billing
token counts. The manifest shows each actual application section's total, per-source
counts/ranges, formatting overhead, and applicable selection limits. Counts exclude
provider framing and image tokens. Image asset IDs remain visible as input metadata.
Source ranges refer to the field/source representation, not model token offsets.
Structured Jev candidate fields also show field-specific ranges for title/content
or name/description clips. Typed section inventories can include nested source
references; section totals are authoritative and nested totals are not additive.

Selected and loaded skill IDs and missing/over-budget skill records are explicit.
Recall coverage and results are captured when recall runs. The default path reports
`not_performed` for automatic recall and identifies its ledger-tail strategy, rather
than pretending that a ledger snapshot is a fresh search. Native recall remains
available normally. Neither builder promotes skill artifacts into tool curation.

## Work-log surfaces and access

Desktop separates **Decision context**, **Execution audit**, and **Task state
diagnostics**. The existing admin-owned Desktop event gate remains in place.
A2UI carries the structured captured manifests and their human-readable inventory.
TUI users open that same inventory with **Ctrl+W**; only events owned by the current
user enter its view. Both readers display captured inputs after later task changes.

The local `decision_context` IPC method requires an explicit administrator actor
and an owned task checkpoint. Without a source ID it returns the captured
manifests or an explicit legacy result. With decision/source IDs it first checks
membership in that captured inventory, then delegates to native memory source
expansion, which rechecks current access. Deleted/revoked and unknown sources do
not expose content. Another user's checkpoint is indistinguishable from a missing
checkpoint. Desktop and TUI label expansion as a **current source read**, not the
historical text delivered to the model. Expansion is bounded and has no neighbors.
Non-memory sources remain inventory entries.

Older logs without receipts say no call-time manifest was captured. Older
aggregate-only receipts remain readable and explicitly retain their legacy label;
they are not retrospectively upgraded to source-complete inventories.

## Retention and privacy

No raw prompts, source excerpts, rendered context, question text, model prose,
image bytes, or credentials are added to manifests. Exact rendered context has
**no persistent debug retention**: it exists transiently while building/sending
inputs, and the processing capture scope is discarded afterward. There is no debug
context archive to expire. Metadata follows the existing task-checkpoint and
work-log event lifecycles. Source expansion returns authorized content on demand
without adding it to a manifest or telemetry. Only the System Two decision ID is
added to inference telemetry; inventories do not enter broad telemetry. Existing
task diagnostics and execution audit retain their separate meanings and policies.

## Verification

Synthetic public-call tests exercise default admission, planning, request/phase
curation, Jev completion review, and System Two review. They compare captured
section/source ranges and counts with the inputs actually passed to each client,
then change task state and verify that checkpoint/IPC/A2UI readers preserve the
original inventories. A Textual test mounts the actual TUI reader and verifies
owner filtering and captured-manifest display. Additional tests cover opt-in
receipts, skill clipping/unavailability, memory budget omissions, failed/repeated
calls, scope isolation, private-source exclusion, source revocation and missing
source equivalence, admin/owner checks, legacy labels and Desktop rendering.

No context default switch or issue #18 rollout gate is claimed by this change.

Validation on 2026-10-18: 103 focused Python tests pass, 59 Desktop tests pass,
and Desktop production build, Python compilation and whitespace checks pass.
The complete Python suite reports 782 passed and 36 failed. A clean unchanged-HEAD
comparison reports 767 passed and the identical 36 failure identities; no new or
removed failures were observed. The repository-wide suite is not fully green.
