# Bounded conversation and active-task continuity

Issue #15, under #10; builds on #12's canonical source contracts.
Implementation candidate, 2026-10-17. Live rollout gates remain in
[the evaluation protocol](07-memory-evaluation.md).

`core/memory/transcript.py` projects at most three recent exchanges within a
character budget. A user record starts an exchange; subsequent assistant records
belong to it until the next user record. Consecutive user messages each form a
record-sized exchange. Date, channel, project and memory-session changes never
split or discard a record. The store fetch is owner scoped and capped at 30
records; deterministic access filtering and canonical-ID deduplication precede
selection. The projection is a small immediate window, not semantic task selection
or complete historical recall. It can contain unrelated recent exchanges, but
completed tasks cannot accumulate without bound. Older context requires recall.

Current-task messages get priority over the historical window. New checkpoints
persist structured conversation records so multiline steering stays whole. Old
Markdown checkpoints remain readable as opaque records, including any old prefix
before subsequently appended structured messages. The newest canonical user turn
can replace an identical current turn; earlier repeated short answers remain
separate events. Task-local working IDs and canonical IDs remain distinct.

History fits as whole exchanges or is omitted. After a historical exchange fails
to fit, older exchanges are withheld too, avoiding an older answer presented as
the immediate antecedent. Current steering fits as whole messages, newest first.
Oversized current records are explicitly omitted rather than sliced. Tiny budgets
may not fit even the omission notice; metadata still reports missing records.
`conversation_sources` now describes only canonical records actually included in
full. `conversation_projection` describes source IDs/reasons for historical
omissions, included/omitted task-local record IDs, required recall, fetch failure,
and a possibly incomplete 30-record store window. It does not claim a full-history
coverage scan or resolve ambiguous pronouns by guessing.

`core/memory/continuity.py` projects only the current checkpoint's objective,
acceptance constraints, actual pending/answered question, and evidence status with
stable task-evidence references. It reserves 9,000 characters before optional
conversation/project/memory sections. Overflow omits whole records and sets
`active_continuity_projection.reserved_overflow`; omitted text remains in the
checkpoint. This extractive projection uses no generated prose, invented outcomes,
or imported historical permission. Failed and successful execution statuses retain
their original meaning; the ordinary deterministic effect/authorization/terminal
checks remain authoritative. It is not a claim that a successful call verified its
effects. System One admission and final context assembly omit whole opaque records
or sections rather than slicing conversation/goal text downstream.

Question resolution retains the question and normalized answer together in the
checkpoint independently of plan compaction. Channel delivery metadata is excluded
from the continuity projection. Reopening the SQLite question store and changing
the memory session do not lose the pair. Existing explicit question expiry still
applies; midnight itself has no effect.

## Task selection and resumption

The existing runtime boundaries remain authoritative:

1. An authenticated explicit question ID (or bound provider reply ID) selects only
   a pending question assigned to that respondent. With no target, exactly one
   pending question may resume; several return an ambiguity response and park all
   tasks. Neither project nor newest timestamp breaks this tie.
2. The selected question loads its exact persisted task ID, preserving the original
   objective, constraints and audit evidence. Its answer is checkpointed before
   requeue. A different channel can deliver the answer.
3. An authorized active correlation routes a correlated response. The same owner's
   active project can nominate steering; Jev's existing bounded intake decision
   determines relevance before consumption, within deterministic eligibility.
4. Other messages enter normal admission with the bounded immediate conversation
   window. Historical context never revives a completed task or copies its mutation
   authority. Ambiguous language can require a new clarification. There is no new
   heuristic that picks one of several persisted tasks by lexical similarity.

## Validation and limits

### Retry-originating payloads (#128, 2026-10-31)

`core/memory/retry_context.py` supplements the immediate tail for a current request
that Jev judges to depend on earlier context. It considers at most 200 owner-scoped
events and applies visibility checks before constructing at most 64 adjacent-exchange
windows. Selection previews share a 48,000-character budget; user requests and
confirmation directives receive space before the remaining substantive extraction.
Repeated failure reports cannot consume the budget before the original request.

Source selection separates three bounded judgments. Independent Noul questions
classify each window's substantive source material, with only that window in its
state (up to eight concurrent requests). Confident substantive windows enter a
Choice comparison explicitly asking for the payload **and its subsequent user
review/confirmation** for the current request. This distinguishes the complete pair
from overlapping windows with only its payload or approval, and offers `none` and
`ambiguous`. A final independent Noul verifies the chosen referent. Every positive
gate uses the unchanged configured yes threshold (0.8); uncertainty at any gate
cannot be converted into acceptance by winning a comparison. Source identity does
not require inventing uncertain clinical fields or deciding authorization/completion.
Malformed judgments fail closed. Decision metadata retains confidence/status only.

Previews are for selection only: the selected sources are reloaded whole before
use. Disconnected qualifying episodes remain ambiguous rather than being merged by
topical similarity.

The binding stores a current-request/steering fingerprint and canonical source IDs,
never cached prescription contents or imported mutation authority. Every prompt
rechecks source access and existence. Missing antecedents withhold the whole bundle
rather than displaying approval without the payload it approved. Selected sources
fit whole within 16,000 characters/32 records; otherwise context reports the limit
and requires targeted recall or a concrete source question. Current attachments
take precedence over historical source selection. New steering invalidates the
binding. The default path and the opt-in projection both supply source receipts
to admission, tool curation, planning, execution and review; projection inputs are
required records and therefore cannot be silently evicted by optional history.
An unresolved or revoked binding cannot fall back to writing an older similar
record: Plan must produce one specific source/change question, with its native
capability retained only within runtime exposure. The executor rechecks bound
sources before each action, including resumed phases, and blocks an already
planned operation if the originating payload becomes unavailable.

This selects task parameters for the current authorized retry. It does not resume
an old failed task, copy its authorization, prove that a record was written, or
modify any skill/artifact program. Jev selects bounded sources only; System Two
composes plans and operations, and deterministic ownership, scope, cancellation
and effect checks remain authoritative.

Synthetic regressions place a reviewed October prescription and its confirmation
behind forty failed retries, alongside an older September treatment and another
owner's private records. Both context paths recover the October source and retain
its uncertainty; a disposable SQLite transaction and fresh read-back preserve the
older record and create the requested new one. The regressions also cover restart,
revocation, ambiguity, fresh attachments, steering changes and whole-source limits.
No synthetic test calls a live provider or writes household records.

An explicitly approved read-only live diagnostic on 2026-10-31 recovered the exact
original payload and confirmation IDs and reloaded both complete records in two
trials. Source-form confidence was 0.96/0.96, comparative choice 0.81/0.82, and
independent source support 0.92/0.92, at the unchanged 0.8 threshold. The diagnostic
used a read-only SQLite connection, submitted no Alphonse task and changed zero
rows. This validates source recovery only: live database mutation/read-back and
owner acceptance remain pending. Household source contents are not included here.

Eleven new synthetic tests cover complete exchange omissions and manifests,
repeated short answers, retry/pronoun antecedents, channel and midnight changes,
multiline and legacy checkpoints, tiny budgets, task restart/plan compaction,
multiple pending tasks, owner isolation, failed evidence, and explicit overflow.

The version 4 [offline candidate](evidence/memory-continuity-v4/results.json)
retains the frozen version 1 corpus and version 1–3 evidence. Necessary-source
recall is unchanged on every existing case. No live inference or external action
was performed. This is not evidence of end-to-end pronoun interpretation or
permission safety by a live model. Implicit recall beyond the window remains
incomplete; a recent irrelevant record may still be present. Source-linked
extractive working state avoids introducing a new generative summarizer.

Full Python verification: 718 passed, 36 failed. All 36 failure identities also
occur against an archive of unchanged HEAD `1d20b7a` (707 passed, 36 failed).
The candidate introduces 11 passing tests and no new full-suite failures. Existing
failures include legacy project-context expectations, V2 CAPD/Act provider setup,
IPC/UI contracts, and prior Jev admission/intake expectations; they are not waived
rollout gates. Replay preserves necessary-source recall, reduces aggregate
irrelevant included sources from 83 to 29, and increases the aggregate prompt
estimate from 3,287 to 3,776 tokens because explicit task-state/provenance labels
add overhead. Token reduction and live answer quality are not claimed.
