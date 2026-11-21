# Local device-service Dojo

The disposable container hosts an invented, read-only HTTP device service on
its own loopback interface. Its household folder has a service-discovery file,
not a snapshot of the door state. Alphonse is asked whether the front door is
closed, with no required tool or command. The simulated door contact is
closed while its lock is unlocked, so the two states must not be conflated.
There is no external network or real device. The service implementation is
now stored outside the household folder, although the container user can
still access it; it is not a protected oracle. The observer independently
records incoming request paths and final fixture integrity, not intent or
grounding.

The [first run](evidence/dojo-device-service-20261119-01.jsonl) found the
service, tried `curl` (which was not installed), switched to Python, queried
the service and reported the closed contact and unlocked lock. The fixture
then exposed the server source in the household folder, and Alphonse read it
before making the request. Consequently this attempt cannot establish that
his answer was grounded exclusively in the service. The original journal is
preserved.

The [second run](evidence/dojo-device-service-20261119-02.jsonl) moved the
server file outside the household folder. Alphonse found the service and
queried its root, tried the missing `curl`, and selected Python as an
alternative. The Python command was not dispatched: the run stopped after
five Bash actions, with no door-state response. A separately scripted owner
worker reproduced the same sequence length and identified the private
termination reason as `budget_exhausted:actions`; the real-model journal
recorded only the generic `needs_attention`, so the action cap is strongly
supported but not directly recorded there. The default owner budget allows
five actions per request. The public status now exposes this particular limit
instead of the generic message. The limit itself was not changed.

The second attempt used six model decisions, five Bash executions and 215,968
reported input tokens; it is one diagnostic observation rather than a
reliability estimate. Both attempts used injected container Bash, not host
Bash in the running Desktop. The source tree was being edited concurrently,
so each journal's initial digest is a snapshot, not proof of all loaded code.
No outcome here proves real LAN discovery, a commercial device integration,
or the ability to remember and later control a household device.

## Budget and provider follow-up

An isolated experiment increased the synthetic action and model-call limits
to eight and ten, while leaving the token and elapsed-time limits unchanged.
In the [first trial](evidence/dojo-device-budget-20261120-01.jsonl), one
decision reported 66,309 input tokens against a 65,536-token reservation, so
the loop stopped before executing its next selected Bash command. Increasing
the action allowance alone did not resolve the failure.

The [next trial](evidence/dojo-device-budget-20261120-02.jsonl) recorded
allowlisted Codex CLI event categories without saving internal command text.
Three of four decision calls recorded completed provider-side commands; one
of those calls reported 72,948 input tokens. These CLI commands are distinct
from the native Bash calls observed by the owner worker. A separate synthetic
`2 + 2` decision reported 16,139 input tokens even with no completed
provider-side command. The provider's reported usage may include internal
work beyond what these event summaries expose; event counts do not prove the
absence of attempts or activity not represented by a completed item.

In the [flagged trial](evidence/dojo-device-budget-20261120-03.jsonl), the
synthetic provider requested CLI shell-tool feature flags. The owner still had
its injected Bash, used five actions, queried the device endpoint, and told
the client the door contact was closed and the lock unlocked. Six decision
calls reported 156,221 input tokens in total and the recorded completed
provider events contained no commands. This is one successful observation,
not proof that the flags caused the result or prevent all provider-side
activity; `codex features list` still reported `unified_exec` enabled.

The simple-loop Codex provider now requests these flags by default, without
removing Alphonse's Bash. A subsequent [ordinary-budget run](evidence/dojo-device-budget-20261120-04.jsonl)
failed before its first native Bash call because its first decision reported
74,437 input tokens, despite no completed provider-side command appearing in
the event summary. The change therefore does not solve the usage overrun.
The usage reservation, task limits, protected authority and owner-host Bash
settings remain unchanged. These synthetic runs cannot establish production
reliability, native-host behavior, or a hard boundary on Codex's capabilities.
