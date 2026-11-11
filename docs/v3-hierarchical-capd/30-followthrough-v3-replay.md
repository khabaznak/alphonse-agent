# V3 follow-through development replay

`alphonse.agent_v2.evaluation.followthrough_v3_adapter` drives the real V3
processor with the existing synthetic follow-through fixture. Its optional
context-projection variant runs in a separate process with that option enabled.
Both variants use synthetic Jev judgments, scripted source actions and a
preselected response. The worker receives no fixture oracle input. These are
development wiring checks, not tests of model autonomy or real delivery.

Run the checks from the repository root:

```sh
.venv/bin/python -m pytest tests/test_followthrough_v3_adapter.py -q
```

In these scripted checks, both V3 variants read the unavailable primary and
the authorized backup, then return the supplied correct response. The optional
post-processor fixture bridge simulates a delivered message. Both variants also
stop a pre-cancelled task without fixture reads or delivery attempts. The
processor returns `Task cancelled.`; the adapter preserves that literal and
normalizes only that known cancellation reason for the fixture oracle.

For the missing-identity case, the script reads the primary and supplies a
question as the desired final response. Both variants instead end failed before
that response, with the actual reason: "A required criterion relies on
unavailable or unusable execution evidence. A bounded review could not establish
a recovery objective." No simulated message is sent and the fixture reports no
handoff. This is an observed failure of this scripted replay, not a claim about
what a live model would do with the same information.

Every available result is marked `decision_qualified: false`, product delivery
and real-time deadlines are unassessed, and the fixture clock does not advance.
An empty action script is unavailable rather than a pass. Failed source
authorization attempts are retained in the fixture audit. The bridge runs only
after V3 returns and is not a notification transport or evidence of receipt by
a real user. No external provider, private records or household effects are
used. These cases do not cover restart, mutation reconciliation, or production
authorization.
