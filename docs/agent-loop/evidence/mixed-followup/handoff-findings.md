# Mixed follow-up handoff verification

November 27, 2026, Mexico City. The corrected owner loop delivered both dispositions through Unix IPC. The work log recorded the blocked reminder before handing off the independent repository review. The test client acknowledged the reply.

## Observed outcomes

- The final live-model variant read the clock, attempted the reminder, then selected the handoff. The work log recorded that validation blocked the reminder. A separate receiving process persisted exactly one acceptance for the pinned fixture repository and proposal operation.
- The reminder readback was empty. The repository file hashes matched before and after, and the subtitle test still failed. The acceptance receipt covered a pending proposal. Patch generation and verification were outside this fixture.
- The worker brief contained the operation ID, worker name, target, pinned revision, and requested operation. It contained neither synthetic privacy marker nor the full owner turn.
- The scripted provider exercised the same order and produced the same dispositions. The original live prompt led the model to hand off first and attempt the reminder afterward. Both outcomes were retained.
- An interim reminder-first run reserved the handoff and failed before dispatch. During implementation the new guard assumed the optional steering table already existed. Checking for that table before reading steering fixed the issue. The failed attempt remains recorded.

## Receipts and scope

- [Final live reminder-first](live-reminder-first-02.json) took 40.273 seconds. Six calls reported 101,055 input and 597 output tokens.
- [Scripted order control](scripted-handoff.json) took 1.835 seconds. Its six decisions and token counts were simulated.
- [Original live prompt](live-handoff.json) took 41.765 seconds. Six calls reported 103,415 input and 579 output tokens.
- [Interim failed live attempt](live-reminder-first.json) took 30.429 seconds. Five calls reported 84,526 input and 422 output tokens.

The live model controlled the parent loop in the live attempts. Gmail and repository contents were synthetic. A separate disposable process stood in for Codex and acknowledged the operation. The fixture verifies handoff acceptance and accurate reporting. Live Codex worker transport, result delivery, applied changes, and private digest reference resolution remain unassessed. Monetary cost was unavailable.

The focused backend suite passed 147 tests with local Unix socket access. The scripted regression lives in `tests/test_dojo_mixed_followup.py`.

## Real Codex worker

The [live parent and worker run](live-real-codex-worker-reminder-first-04.json)
took 48.347 seconds. The reminder remained blocked. The live parent sent one
bound handoff to a separate Codex CLI in a disposable copy. The verifier saw
the fixture test fail on the original revision and pass with the proposed
patch. The original repository remained unchanged. Both Desktop reply
deliveries were acknowledged. This exercise supplies no household worker
configuration or general repository access.

From the repository root, run the deterministic control with:

```sh
PYTHONPATH=.:tests .venv/bin/python docs/agent-loop/evidence/mixed-followup/run_baseline.py \
  /tmp/alphonse-mixed-control.json --scripted --handoff --reminder-first
```

For a live run, omit `--scripted` and add `--real-worker`. The parent and
worker require authenticated Codex CLIs. Supply `--parent-cli` and
`--worker-cli` when the configured executables are in different locations.
Each run creates its own repository, provider fixture and database.
