# Stage 5: Compatibility, observability, evaluation, and rollout

Status: not started  
Depends on: Stages 1 through 4

## Objective

Make hierarchical CAPD safely operable, measurable, restartable, and reversible before
it becomes the default engine.

## Engine compatibility

Support an explicit engine setting during rollout:

```text
tactical_v2
hierarchical_v3
```

The selected engine is stamped onto a task at ingestion and persisted. A configuration
change must not switch an already-queued or active task to another engine.

## Checkpoint and queue compatibility

- Existing V2 tasks finish under V2 unless explicitly migrated through a tested tool.
- Legacy messages without an engine stamp use the configured compatibility default.
- V3 checkpoint schema versions are validated before resume.
- Unknown future schema versions fail visibly and preserve the checkpoint.
- Resuming never repeats a completed non-idempotent side effect.

## Observability

Record structured events for:

- Outer CAPD transition.
- Phase creation and completion.
- Subgoal activation and completion.
- Capability and concrete-tool reveal decisions.
- Tactical action start/result.
- Budget consumption.
- Local recovery and strategic escalation.
- Mutation authorization and actual affected targets.
- Verification and terminal routing.
- Model calls, prompt estimates, latency, and provider/model profile.

Do not log secrets, unrestricted file contents, or hidden chain-of-thought.

## Desktop and IPC

Expose enough structured state for a nested progress view:

```text
Phase: Complete solar project
  done  Locate record
  done  Update exact record
  doing Verify final state
```

IPC additions should be versioned and backward compatible. Existing clients must not
break when V3 fields are absent or ignored.

## Evaluation harness

Build a replayable corpus from sanitized real failures and synthetic boundary cases.
Every case includes:

- Initial message and authorized project fixture.
- Relevant memory/session fixture.
- Available tool/integration fixture.
- Expected allowed and forbidden effects.
- Expected questions or final response properties.
- Token, model-call, tool-call, and latency measurements.

Run V2 and V3 on the same corpus where semantics permit. Store machine-readable
results and a concise human review report.

## Rollout gates

### Gate A: developer opt-in

- Unit and integration suites pass.
- Checkpoint/resume and cancellation pass fault injection.
- No unauthorized mutation in the evaluation corpus.

### Gate B: shadow planning

- V3 plans and reveal decisions are recorded without executing side effects.
- Compare chosen phases/tools with V2 outcomes and human expectations.
- Fix tool-recall and scope issues before live execution.

### Gate C: selected-project execution

- Enable V3 only for explicitly selected projects/users.
- Provide immediate engine fallback for new tasks.
- Do not migrate an active phase automatically.

### Gate D: default with rollback

- Correctness and silent-failure rates are no worse than V2.
- Simple-task model calls, token usage, and latency improve materially.
- Operational dashboards and logs can diagnose failed phases.
- V2 remains available for a defined stabilization period.

## Implementation checklist

- [ ] Add engine selection to settings with safe defaults.
- [ ] Stamp engine and schema version at message ingestion.
- [ ] Preserve engine selection across queue and checkpoint persistence.
- [ ] Add V2/V3 compatibility loaders and explicit failure behavior.
- [ ] Add structured phase/subgoal/action logs and metrics.
- [ ] Extend daemon status and IPC progress payloads.
- [ ] Add functional nested progress to the desktop client.
- [ ] Build the replay/evaluation harness and initial corpus.
- [ ] Add fault injection for tool failure, restart, timeout, cancellation, and steering.
- [ ] Measure prompt/tool-schema token estimates per inference.
- [ ] Add administrator-visible V3 enablement and rollback controls.
- [ ] Document operational recovery for stuck or incompatible tasks.
- [ ] Complete each rollout gate with a dated evidence report.

## Required regression scenarios

- [ ] Solar-project completion uses one outer phase, exact mutation, verified read-back,
      and a warm response without unrelated edits.
- [ ] A known LG capability is selected without attempting unrelated Home Assistant
      tools when project evidence routes to LG.
- [ ] A prior medical treatment is retrieved from authorized memory/artifacts before
      asking the user to repeat it.
- [ ] An OCR failure is reported as an infrastructure/extraction failure, not proof
      that the source is unreliable.
- [ ] A nonzero command exit is a failed action and cannot prove success.
- [ ] A hundreds-of-kilobytes legacy memory ledger does not exceed the configured
      bounded memory projection.
- [ ] Steering interrupts tactical execution without sending a stale response.

## Exit criteria

- V3 can be enabled, observed, and disabled safely.
- Queue and checkpoint behavior is deterministic across restarts.
- Desktop and daemon expose actionable nested progress and failures.
- The replay corpus shows no correctness or authorization regression.
- Measured simple-task efficiency improves enough to justify default rollout.
- A rollback procedure has been exercised, not merely documented.

## Open decisions

- Exact numerical thresholds for token, latency, and reliability rollout gates.
- Length of the V2 stabilization/fallback period.
- Whether shadow planning runs for every opted-in task or a sampled subset.
- Retention policy for detailed tactical action telemetry.

## Implementation log

- No implementation entries yet.

