# Independent self-correction development cases

`dojo_self_correction_independent` adds a Portuguese correction case and a
separate English control with a genuine later owner change and a consistent
assistant recap. The facts are synthetic, older decisions are outside the
recent-turn window, and no tool route is required. The evaluator's rubric stays
separate from the goal. The separate model provider may have access to fixture
code, so these cases are not guaranteed blind.

After timeline projection, the first [Portuguese attempt](evidence/dojo-independent-portuguese-20261125-01.json)
reported the recorded decision and confirmation but omitted the assistant's
contradictory recap. Its self-correction criterion failed. The [control](evidence/dojo-independent-later-change-20261125-01.json)
reported the later choice and did not invent an error. Each used one search and
two model calls, took 9.980 and 12.699 seconds, and reported 31,794/115 and
49,838/155 input/output tokens, respectively.

After adding a general explanation of recorded speaker roles, the [second
Portuguese attempt](evidence/dojo-independent-portuguese-20261125-02.json)
reported the earlier choice and explicitly acknowledged its own mistaken recap;
it did not mention the later owner confirmation, so the full recall criterion
remains incomplete. The [second control](evidence/dojo-independent-later-change-20261125-02.json)
reported the later choice without inventing an error. Each used one search and
two model calls, took 15.586 and 13.997 seconds, and reported 49,695/236 and
49,955/126 input/output tokens, respectively. Both responses were acknowledged
by the test client. These are development attempts, not reliability estimates or
proof that the presentation change caused the answers. The same fixture was
used in the follow-up. Source completeness and grounding were not independently
established; the installed Desktop was not exercised.

Run a fresh case with:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_self_correction_independent \
  /tmp/dojo-independent.json --case portuguese-correction --model MODEL
```

Use `later-change-control` and a separate journal for the control. Review each
answer against the full rubric; marker matches do not establish meaning.
