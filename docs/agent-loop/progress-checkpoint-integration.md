# Progress checkpoints: integration status

`progress_checkpoint.py` is an **inactive, experimental policy**. The owner
loop separately offers an opt-in, trusted-local Desktop preview for
counter-based checkpoints and durable continuation. It includes an optional
Jev observation-review hook, but normal conversations cannot currently supply
the necessary trusted provider-disclosure authorization; their progress status
therefore remains unknown. Synthetic tests use explicitly authorized fixture
data and predetermined judgments. A judgment is not proof of an effect or
permission to continue.

The preview preserves the original task, records a task-scoped Desktop choice,
and reclaims quiescent work under a fresh claim after a restart. Uncertain or
in-flight work is not replayed. Cancellation is durably recorded before it is
acknowledged. A disposable test kills an isolated daemon process while paused,
starts a new process, and confirms a prior Bash write is not repeated. A
separate real-model exercise used a fake progress judge and synthetic owner
approval; neither run qualifies the packaged Desktop or live Jev. In trusted-local mode,
same-UID processes, including Bash, can impersonate the Desktop; this is not a
release-grade owner authority boundary. The setting is disabled by default.

Further qualification and progress-based behavior require more than calling
`decide` from the loop:

1. **Metrics and evidence.** Configure an explicit metric, interval and
   calibrated judgment thresholds. Build a bounded, privacy-conscious projection
   of actual settled actions and their results; use a separately evaluated Jev
   judgment for semantic progress. Unknown or unavailable judgments remain
   unknown. Concrete blockers and alternative-route availability require trusted
   evidence and cannot be inferred from a progress score. Retain independent
   hard safety and provider limits.
2. **Pause and continuation.** The local preview pauses at a quiescent
   boundary and binds the explicit Desktop choice to the task, checkpoint,
   metric and bounded window. The peer check is a same-UID development
   boundary, not proof of an unforgeable owner decision. Further qualification
   must cover real Desktop delivery and loss at each handoff.
3. **Recover without replay.** The preview revalidates saved evidence and
   acquires a fresh claim for approved paused or quiescent resumed work. It
   rejects unknown usage, in-flight actions and unresolved effects. Extend
   coverage across real-model tasks, cancellation and changed permissions;
   do not silently repeat a dispatched action. Ordinary clarification remains
   a separate new conversation turn.
4. **Retry and validation.** The retry gate requires separately established
   operation identity, safe repeat semantics, authorization, bounded attempts
   and a positive backoff deadline. Exercise duplicate replies, restarts at each
   transition, cancellation, revocation, uncertain delivery/effects, expired
   grants, stale evidence and unavailable judgment in Dojo and integration
   tests. Test real-model behavior only after those deterministic tests pass.

The current preview asks at a configurable counter checkpoint. Existing work
window ceilings remain in place and a user's approval increases the bounded
window. The optional review evaluates only newly observed evidence since the
previous checkpoint; unavailable, stale, ambiguous, or unapproved evidence
remains unknown. It has not been validated against a live Jev provider and
must not be described as an evaluated production progress decision.
