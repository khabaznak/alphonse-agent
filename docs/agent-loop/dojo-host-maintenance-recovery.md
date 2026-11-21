# Trusted-host maintenance report recovery

An owner asks for a past inspection's incoming water pressure. The referenced
report is unavailable, and an alternative technician record sits elsewhere in
the same synthetic folder. The owner asks Alphonse to find another source if
necessary and keep the records unchanged. The prompt does not prescribe tools,
commands, or the name of the alternative record.

This runs the real simple owner worker and its Desktop-style Unix-socket
conversation ingress using synthetic storage and **native host Bash** in the
explicit trusted local mode. The fixture is disposable, but the Bash process
has the macOS user's normal host and network access. It is not the installed
Desktop, an OS sandbox, or a household authorization test. The separately
invoked Codex inference provider has its own execution environment.

The evaluator independently checks the alternative note's content hash and the
dangling report symlink after execution. It records only bounded indications
that the reply mentions the synthetic measurement, unit and source, and that
synthetic source text appeared in Bash output; it does not record shell commands
or raw output. Literal markers do not establish reply semantics or grounding.
The checkpoint's process observations are stored in map iteration order, which
is **not guaranteed to be execution order**. Final hashes do not prove no
temporary changes were made.

The [first attempt](evidence/dojo-host-maintenance-recovery-20261121-01.json)
produced an acknowledged reply containing the measurement, unit and technician
markers; both returned Bash processes had exit code zero and the fixture was
unchanged. Its instrumentation did not check whether the synthetic source text
appeared in process output. Its limited vocabulary did not detect an
unavailability marker; that is not a semantic assessment.

After adding source-output indicators, the [second attempt](evidence/dojo-host-maintenance-recovery-20261121-02.json)
observed the synthetic technician note and measurement in Bash output. The
acknowledged reply contained the expected measurement, unit, technician and
unavailability markers; the fixture was unchanged. Three Bash observations
included exit codes zero, one and zero, in unspecified order. The answer's full
meaning and causal grounding are unassessed. These are two development
observations, not a reliability estimate.

Reproduce only in a trusted local environment where unrestricted host execution
is explicitly authorized, with a configured model and a new journal path:

```sh
python -m alphonse.agent_v2.evaluation.owner_dojo_trusted_host \
  --case host-unavailable-maintenance-report --model MODEL \
  --journal /private/tmp/dojo-host-recovery.json --accept-trusted-host
```
