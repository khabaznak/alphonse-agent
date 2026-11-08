# Browser fill sequence prototype (issue 190, incomplete)

This prototype accepts one fill followed by an observation and reconciliation,
using the existing executor for every step. The proposal is ephemeral. It is not
a general browser plan, cannot submit, and does not establish task completion.
The browser service continues to check the live actor grant, session identity,
phase scope and prerequisites. The executor checks cancellation and validates
and reveals each action again. A failed fill or reconciliation leaves the
existing uncertain-effect recovery path in charge.

## Disposable comparison

From the repository root, run:

```sh
python -c 'import json; from alphonse.agent_v2.evaluation.browser_handoff_comparison import run_case; print(json.dumps([run_case(control=False, changed=c, sequence=s) for c in (False, True) for s in (False, True) for _ in range(3)], indent=2))'
```

Three local runs per arm with the unchanged form independently verified both
fields. The existing continued-phase control used seven scripted selector calls;
the sequence used three, including the final no-action selection in both cases.
Both made six native calls and three observations, with no repeated observations.
The local elapsed ranges were approximately 8.5–11.5 ms and 8.5–8.7 ms,
respectively. These timings exclude provider latency and are not a performance
claim. Neither arm submitted the form or qualified completion. In changed-form
runs neither dispatched a fill, and the attempted effect stayed unresolved.

These are scripted selector calls, not actual LLM calls. The observation count
is unchanged, so the requested reduction in observations is not demonstrated.
The comparison also does not yet cover retries, submission/result qualification,
or a real browser. Do not check off issue 190 or merge this prototype on the
strength of these results. A broader matched evaluation and review of the plan's
state-change and recovery semantics remain necessary.
