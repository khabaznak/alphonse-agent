# Task communication — issue #31

On October 29, 2026, GitHub's native blocked-by relationships and issue bodies
confirmed that #67 and #9 were completed. #68's question work was also completed.
This implementation builds on their existing task activity and question paths.

## Reproductions and fixes

The acknowledgement prompt already prohibited completion claims, but generated
text was delivered without review. Injecting `Done, I've updated the record.`,
`Listo, ya quedó actualizado.`, or `All set!` reproduced premature completion
wording through admission, the real SQLite outbox and conversation recording.
All three regression cases fail against baseline `05a054ae` and pass here.

System Two still composes a short acknowledgement in the user's language. A
bounded Jev Noul judgment now checks whether it is appropriate before execution.
Rejected, ambiguous, malformed, unavailable or oversized candidates are omitted;
work continues. No replacement success template is generated. Cancellation is
checked before and after review. Accepted acknowledgements retain the existing
one-time admission and idempotent delivery behavior. Direct conversational replies
still use the existing direct-response route.

A separate lifecycle delivery defect reproduced through `V2Daemon.run_once`:
a parked task with a prepared response could publish it while awaiting input.
That regression also fails against the baseline. Final-response outbox projection
now requires the task's existing `completed` state. Preparing prose is not task
completion. Acceptance/evidence verification, permissions and cancellation rules
remain unchanged; this gate only controls delivery after those rules have run.

Terminal failure wording receives a separate bounded Jev review, including
prepared closure explanations. It cannot reclassify evidence or change task state.
Unreviewed or misleading wording falls back to an explicit failure, preserves
partial work and tells the user to inspect the failed action and request a retry
when its blocker is resolved. Failure notifications retain the matching task ID,
origin, owner and correlation; unrelated task explanations are not reused.
Queued ordinary conversation now receives the existing waiting notice even without
a selected project, and that notice carries the admitted message/task ID.

Desktop reuses #9's renderer, activity identity, independent task logs and terminal
cleanup. Admitted, queued, planned, waiting and blocked activity has explicit
labels and no execution spinner. Blocked activity points to its blocker and asks
for the needed input/action. Existing failed/cancelled/interrupted guidance stays
reachable in the work log. #68's question identity and answer/continuation path
are preserved.

## Verification

- Admission: immediate conversational reply, one acknowledgement, safe intention,
  English/Spanish premature success, unavailable/ambiguous review and cancellation
  during review. SQLite delivery and history are checked.
- Completion delivery: pending, running, waiting-user, failed and cancelled tasks
  cannot publish a prepared final response; completed tasks can. The parked case
  also runs through the daemon's real delivery path.
- Failure: evidence-based explanation, partial successful observations, secret
  redaction, rejected success wording, unavailable review, task/owner identity,
  daemon terminal delivery and no silent disappearance.
- Existing regression suites: deterministic outer completion and verified-effect
  guards; authorized delegated reminder delivery to the intended member exactly
  once; killswitch cancellation; Desktop IPC question answer/resume exactly once,
  stale questions and unauthorized respondents; task activity projection/render.
- The call-input manifest test includes the new acknowledgement review and checks
  the actual serialized Jev input. Probability tests reject non-finite, invalid,
  boolean and uncertain answers.

Isolated final validation: 1,211 Python tests pass; 103 Desktop tests pass; the
TypeScript/Vite production build passes with the existing bundle-size advisory.
Three Python failures reproduce unchanged on an untouched baseline export:
`test_runtime_wires_extraction_to_current_router_and_jev_without_project` attempts
to write the configured real user memory path outside the sandbox, and
`test_new_conversation_creates_protected_home_project` plus
`test_independent_message_receives_exact_turns_ahead_notice` still expect automatic
home-project creation. These are separate baseline/test-isolation issues, not
communication fixes.

Tests use synthetic inference and Jev responses with real runtime/store/IPC paths.
They establish control flow and delivery contracts, not live-provider calibration
or native WebKit endurance. Daily-use acceptance remains in #78.
