# Offline comparison journal

`alphonse.agent_v2.evaluation.offline_run_journal` runs the existing three
scripted development arms and writes one exclusive JSONL receipt. It does not
run an external agent or a live provider. Use only synthetic inputs; receipts
can include fixture text and recorded actions.

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.offline_run_journal run \
  tests/fixtures/single_loop_comparison/read_only_v2.json /tmp/alphonse-offline-run.jsonl
.venv/bin/python -m alphonse.agent_v2.evaluation.offline_run_journal inspect \
  /tmp/alphonse-offline-run.jsonl
```

Choose a new output path for each run. An existing file is never overwritten,
resumed or retried. Both commands return nonzero if a completed run contains an
unsuccessful or unavailable arm; inspection also returns nonzero for an
incomplete or damaged run. The read-only fixture currently has expected failing
denial diagnostics, so a nonzero status is expected.

The receipt starts with a unique run ID, the corpus digest, arm revisions,
selected harness source hashes, Python and platform details, and the complete
planned schedule. Each started attempt and its finished result receive separate
UTC timestamps. Finished attempts also contain a process-monotonic elapsed
duration. The runner flushes and syncs each record and syncs the parent directory
once after exclusive creation. If that directory sync fails, it aborts before
starting any arm; the already-created partial receipt must be inspected and a
new path chosen for a subsequent independent run.

Inspection reconstructs finished outcomes and distinguishes started but
unfinished attempts (outcome **unknown**) from those never started. A missing
run-finished record leaves the run incomplete. A truncated line, duplicate,
out-of-order record or mismatched run ID invalidates the receipt. No interrupted
attempt is automatically repeated, and an attempted or queued communication is
not proof of delivery.

A process can die between a tool event and the attempt's terminal record, so
partial in-attempt traces and effects may be unknown. Sync calls reduce the
window in which acknowledged records can be lost, but cannot guarantee
hardware, filesystem or power-loss behavior. The journal does not enforce an
OS/network sandbox, retain the entire runtime dependency set, attest that the
recorded configuration was actually used, anchor its own content independently,
or establish user-facing performance, production safety or delivery. These
scripted fixtures are development diagnostics, not the #217 held-out study or
#218's decision-grade comparison.

Journal completeness means that every scheduled attempt has a structurally
consistent terminal record; it is not benchmark qualification. Inspection marks
qualification `unassessed` even when all recorded fixture checks pass. It
rejects empty schedules and terminal records whose availability, pass status,
oracle checks and instrumentation fields are missing or inconsistent. These
schema checks cannot establish that the records are authentic or truthful.
