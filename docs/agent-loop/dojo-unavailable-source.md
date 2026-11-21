# Unavailable source Dojo

An owner asks for the last inspection's water pressure and points to a report.
In the disposable fixture the report is a dangling link; a separate technician
note is available elsewhere in the household records. The owner asks Alphonse
to look for another source if the report is unavailable, without specifying a
tool, command or alternate location. No real household files or devices are
accessed.

In the [real-model attempt](evidence/dojo-unavailable-source-20261119-01.jsonl),
Alphonse checked the requested path, observed that it was unavailable, searched
the synthetic household folder and read the technician note. He reported the
note's 43 psi reading and the source's date, and explained that the current
report was unavailable. The owner client acknowledged the reply; the fixture's
final contents and broken-link state matched their initial state. These response
and recovery observations were reviewed from the trace, not automatically
graded. Final hashes do not prove no temporary modification occurred.

This is one attempt through a synthetic owner worker with injected container
Bash. It does not establish host Bash availability, handling of network outages,
or general reliability. Source files in the broader working tree were being
edited concurrently; the journal's tree digest is a starting snapshot, not an
attestation of all loaded code. The external model provider has a separate
environment. The attempt used three model decisions and reported 106,998 input
and 522 output tokens across them; these numbers describe the provider usage,
not just the size of the tool registry.

For a compact factual view of a recorded attempt, run:

```
python -m alphonse.agent_v2.evaluation.dojo_attempt_summary JOURNAL
```

The summary preserves missing or unassessed results, and reports model usage,
tool choices, process returns, owner status, effects, and fixture integrity
separately. It does not decide whether the answer was correct or infer a Dojo
pillar score from a tool call.
