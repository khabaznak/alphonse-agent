# Standard skill script execution

Implemented for [issue #35](https://github.com/khabaznak/alphonse-agent/issues/35),
under the [standard skills contract](standard-skills-contract.md).

A package needs only standard `SKILL.md` instructions and its ordinary resources.
Scripts are never registered separately or exposed as tools. Jev selects relevant
skills/native capabilities; System Two follows instructions and composes commands.
The native registry, phase prerequisites and effect policy still gate dispatch.

## Invocation

For a selected installed skill, use `native.bash` with `skill_id`:

```json
{
  "skill_id": "skill:sample",
  "runtime": ".venv/bin/python",
  "required_files": ["scripts/update.py", "/explicit/user/config.json"],
  "command": "\"$ALPHONSE_SKILL_RUNTIME\" \"scripts/update.py\" --config /explicit/user/config.json"
}
```

These are optional call arguments, not package metadata or a proprietary manifest.
`command` remains the only required Bash argument. An executable shell script can
be invoked as `scripts/run.sh`; a non-executable Python script can be invoked with
its documented interpreter. Quote every path that can contain spaces.

The harness re-reads the selected installation through the skill store and checks
an authenticated actor. Its absolute package root becomes cwd and
`ALPHONSE_SKILL_ROOT`, independently of task project and daemon cwd. An explicit
cwd is relative to that root (or an absolute directory within it). Traversal and
symlink escape from package-relative cwd and prerequisite files are rejected.
Selection does not grant execution permission. Discovery/access/revision lifecycle
work remains owned by #34; this path uses the installed store and task selection,
not a parallel package registry or a new access model.

`runtime` resolves a PATH executable, an explicit absolute host executable, or a
package-relative executable. It exports `ALPHONSE_SKILL_RUNTIME` without invoking
anything automatically. Relative runtime parents must remain inside the package;
a contained virtual environment may link its interpreter to a host executable.
Missing/non-executable runtimes fail before dispatch with setup guidance.
`required_files` checks readable files relative to cwd, or explicitly supplied
absolute host configuration paths. Missing files fail before dispatch; no files
or environments are created and no dependencies are installed by resolution.
Absolute configuration paths are explicit host prerequisites, not managed data
bindings or an authorization grant; account/data binding policy belongs to #36.

Executable availability is not dependency readiness. System Two must run the
package's documented dependency/version check with that runtime before a mutation
(for example `"$ALPHONSE_SKILL_RUNTIME" -c 'import required_module'`). Missing
modules/version conflicts remain nonzero command evidence with stderr. Report the
missing dependency and required setup, or choose an authorized repair route;
never assume another Python environment works. Requirements need not be encoded
in metadata, and Alphonse does not attempt to parse arbitrary prose into checks.

## Evidence, cancellation and resume

The same invocation service records arguments, resolved cwd, exit status, stdout,
stderr, duration, timeout and cancellation. Nonzero exits, timeout and cancellation
are failures; a favorable Jev review cannot turn them into completion. Bash never
opts into automatic tactical retries. Cancellation before dispatch runs nothing;
cancellation during execution terminates the process group, including children,
and preserves partial output. Partial writes may remain after failure/cancellation.

Plan prerequisite checks, mutation and readback in one coherent phase. A zero exit
status alone is not proof of a household effect. Inspect the written file, query
the updated record or read the remote state, and attach that evidence to acceptance
review. Generic Bash cannot infer arbitrary command semantics or provide exactly
once external effects.

Completed subgoals and their evidence survive checkpoints. Resume at missing
readback instead of executing a completed write again. A recorded shell mutation
with the same command, skill/artifact, cwd and runtime cannot be blindly replayed
inside the phase, including a planned/running checkpoint whose effects are
uncertain. The executor blocks with a reconciliation diagnostic before dispatch.
Read back existing effects, then form an explicit recovery plan if another write
is necessary. This conservative guard is not shell equivalence analysis: commands
with different syntax and cross-phase recovery still need effect reconciliation,
and external idempotency keys should be used when the target supports them.

## Actual host access and compatibility

Bash is trusted host execution, **not a filesystem sandbox**. Resolver containment
protects the paths it supplies, not arbitrary shell code. Commands can access the
files, credentials/environment, network and processes available to Alphonse's OS
identity. Phase descriptions and cwd do not isolate household members at OS level.

Ordinary calls without skill/artifact IDs retain existing project/host cwd behavior,
explicit cwd, output limits and optional timeout. Legacy `artifact_id` still uses
its catalog authorization/readiness and timeout checks; it cannot be combined with
`skill_id`. No household program or database is copied into product source.

## Verification

`tests/test_skill_script_execution.py` creates disposable packages and data for:
spaces in paths, explicit and ordinary interpreters, project/no-project and daemon
cwd independence, prerequisite availability, contained interpreter links, traversal
and symlink rejection, unselected/unavailable packages, dependency/command failures,
timeouts, cancellation of children, readback, checkpoint resume and uncertain write
replay rejection. Existing native Bash, skill, phase executor, reveal and outer
completion tests exercise compatibility and authorization boundaries.
