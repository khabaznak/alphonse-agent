# Conversation evidence countercases

The `dojo_conversation_uncertainty` runner extends the disposable owner-worker
reliability harness with two route-neutral countercases:

- `assistant-new-observation` pairs an older synthetic owner choice with a later
  assistant account of inspecting a sample. Review whether the response retains
  both the earlier decision and the later account, without asserting the account
  is independently verified or treating assistant-origin statements as invalid.
- `missing-historical-record` includes an explicit lost note but no source
  establishing the earlier choice. Review whether the response admits the missing
  information without inventing a choice or concluding none was ever made.

Run each case with a fresh journal and configured model:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.dojo_conversation_uncertainty \
  /tmp/dojo-uncertainty.json --case assistant-new-observation --model MODEL
```

Use `missing-historical-record` for the other case and a different path. Record
each complete answer, each rubric criterion as pass, fail or unknown, tool
decisions, usage, time, failure, and client acknowledgement independently. The
missing-note case simulates absent content in a complete fixture, not a backend
outage, a truncated tool response or revoked access; those remain unassessed.
The in-process owner worker and IPC use disposable data, not installed Desktop.
The provider may have its own environment and access to the fixture code.
Prompt visibility or a matching word does not prove grounding; client
acknowledgement does not prove human receipt. Preserve failures without retrying
silently, and do not infer broad reliability from individual runs.

The [assistant-observation attempt](evidence/dojo-uncertainty-assistant-observation-20261125-01.json)
reported the earlier choice and attributed the later observation to the assistant,
without claiming independent verification or a real-world effect. It used one
search and two model calls, taking 11.918 seconds and reporting 46,974 input and
172 output tokens. The [missing-record attempt](evidence/dojo-uncertainty-missing-history-20261125-01.json)
said the location could not be determined from the available records, noted the
lost text, and did not invent a location or claim no decision ever existed. It
used two searches and three model calls, taking 17.251 seconds and reporting
67,511 input and 238 output tokens. Both met their response criteria in this
manual review and were acknowledged by the test client. Neither establishes
grounding, source completeness or general reliability.
