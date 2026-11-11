# Decision experiment manifest integrity

`alphonse.agent_v2.evaluation.decision_manifest` supplies pre-registration
plumbing for #217 and #218. It does not run agents or determine whether an
experiment is valid. A manifest records the protocol, corpus and independent
oracle digests; every required arm and its configuration or an explicit reason
it is unavailable; and a deterministic paired schedule. Available arms must pin
source, prompt, tool catalog and authorization policy digests, provider/model,
runtime, decoding, limits, timeout, timezone, environment, network and
capabilities. `validate_manifest` recomputes the plan and schedule, compares
actual supplied bytes and configurations, and returns unresolved blockers.

The manifest digest must be recorded in an independently controlled,
timestamped location **before** exposure or runs. A digest stored alongside a
manifest is a useful checksum but is not an independent anchor: both can be
replaced. The code cannot establish that caller-supplied files exhaust all
runtime inputs or that executing models and settings match those described;
execution must capture and independently verify its actual environment. Nor
can a digest establish corpus coverage, independent authorship, secrecy,
review, approval or permission to transmit data. These always remain explicit
blockers from the validator. A mismatch invalidates a confirmatory run.

`tests/fixtures/single_loop_comparison/prescription_schedule_development_plan_v1.json`
is an example pre-registration of the **existing open development fixture**,
not a confirmatory corpus. All four arms are explicitly unavailable. It binds
three public cases, five repeated schedule entries per arm, and the draft
protocol. Its corpus and oracle digests intentionally refer to the same public
fixture: that fixture keeps answers in its JSON and offers no evaluator/arm
sequestration. No agent run or outcome is implied. Its recorded digest was
`b5418647af53a1dfe173f2de160254b7089e001312c75875f45a978e696ec229`
at creation; the source file and this document are not an independent anchor.
Changing the protocol, corpus, plan or arm configuration invalidates this
snapshot. Creating a valid held-out run additionally requires genuinely
sequestered cases and oracles, matched working arms, execution authorization,
coverage and statistical review, and a fresh independently anchored manifest.

## Three-arm read-only development snapshot

`tests/fixtures/single_loop_comparison/read_only_v2_development_plan_v1.json`
records a prospective offline plan for the existing three synthetic read,
denial and cancellation cases. Its arms are the actual V3 fixture processor
with projections off, the same processor with projections on, and the
unregistered single-loop fixture. The provider and Jev actions are scripted;
the external arm is explicitly unavailable. The corpus contains both the
script and the oracle, and the V3 and candidate scripts differ. The snapshot
is neither a completed run nor an autonomous model comparison. The candidate
has synthetic budget limits but no enforced wall-clock process timeout; its
configuration records `unbounded`. This is allowed only for development plans.

The command below checks the recorded historical snapshot against its draft
protocol, fixture, selected source files and machine configuration. It matches
only the exact pre-commit working tree and Git HEAD used to create it; a later
commit or source edit requires a fresh development snapshot:

```sh
.venv/bin/python -m alphonse.agent_v2.evaluation.read_only_development_manifest verify \
  tests/fixtures/single_loop_comparison/read_only_v2_development_plan_v1.json \
  --expected-digest e217d4cf0b3534b9b7fd8b7f7f5abd52d5194cff4d6f9aacfcc8b2d7d949fb46
```

The inventory hashes all Python files in `alphonse/agent_v2`, including
untracked files, plus `pyproject.toml` and `requirements.txt`; the plan also
pins the current Git HEAD, Python and platform metadata, adapter/script bytes,
and explicit arm flags. This is intentionally conservative about code changes
but incomplete: it does not pin installed package contents, non-Python package
data, host environment, native libraries or actual running process identity.
The Python/platform values make verification machine-specific. A mismatch
requires a fresh plan, not modification of this one. Create one at a new path
with the same module's `freeze NEW_PATH`, then independently anchor the printed
digest. The repository-local digest above is merely a reproduction checksum,
not an independent timestamp or provenance guarantee. The plan's randomized
schedule records intended order, not runs; any comparison receipt must record
its actual order and all outcomes separately.
