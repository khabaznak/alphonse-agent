# Memory migration supporting evidence — 2026-10-18

This is synthetic migration/recovery evidence for issue #18, not a frozen paired
live experiment and not a pass of DC-0 through DC-5. Implementation and procedure:
[Memory migration and recovery](../../15-memory-migration.md).

`validation.json` retains full-suite counts and the unchanged failure identities.
Tests ran against working-tree changes on the recorded source HEAD. Baseline:
782 passed, 36 failed. Candidate: 795 passed, the same 36 failed; zero new failure
identities. Focused context, migration, continuity, recall, knowledge, contract and
manifest coverage: 144 passed. The failing suite remains a rollout blocker;
unchanged failures are not waived. Their detailed runtime/obsolete-expectation
disposition remains outstanding before DC-1 can pass.

The 13 new migration tests exercise dry runs and repeated additive copies, interrupted
copy adoption, collision/changed/deleted resources, version/owner mismatch,
symlink rejection, WAL recovery, corruption and mutation during restore, retention
of newly written compatible live data, and unchanged canonical Spanish conversation
events and daily views across a local date boundary. Restores use a new disposable
directory; no live installation is overwritten. The tests create no knowledge from
unattributed summaries. They do not establish interrupted-task engine rollback.

Reproduce from the repository root:

```bash
python -m pytest -q tests/test_memory_migration.py \
  tests/test_agent_v3_projections.py tests/test_memory_continuity.py \
  tests/test_memory_recall.py tests/test_memory_knowledge.py \
  tests/test_memory_contracts.py tests/test_decision_context_manifests.py
python -m pytest -q --disable-warnings
```

The existing 22-case offline memory replay and both arms of the five-case
context-only pilot also executed successfully. Those use fixed-query/scripted
diagnostics and cannot establish live source discovery, verified task completion,
independent review or performance gates. They were not promoted into new historical
baselines. No provider calls, household state reads, generated artifact adoption,
feature-default changes or GitHub mutations occurred.

Outstanding qualifying validation remains the full isolated runtime adapters,
stage-specific oracles, extension/held-out fixtures, repeated paired live trials
with consistent usage/call/latency accounting, independent human review,
interrupted/resumed-task rollback and explicit owner rollout decision. The earlier
provider failures remain retained in their original pilot package. This work
neither diagnoses them through another live run nor establishes a semantic-recall
gap that warrants a vector backend.
