# Loop retirement surface audit

Review for #229 against `e5595474` and the accompanying working-tree changes.
This is a source, packaging and focused-test audit, not a production release
or proof that the currently running household daemon has this revision.
Historical results and legacy state remain intact.

| Surface | Finding and disposition |
| --- | --- |
| Desktop | No engine selector remains. The System One panel no longer claims V3 task routing. The PDCA activity widget is no longer offered; a previously saved placement remains visible as unavailable and removable. In owner-only mode, a locally tracked pending request is labeled pending instead of inferring an old phase or idle from absent legacy events. This indicator does not recover requests after a Desktop restart. The desktop guide identifies current owner-only and scheduler limits. |
| TUI | The terminal UI is retired; the `alphonse` CLI still manages the daemon. Stored `tui` channel identities and restricted notice review are compatibility and recovery state, not a V2 loop. |
| Daemon and routing | The daemon entry point starts the fenced owner simple-loop worker; old engine/schema stamps do not dispatch to it. The worker does not start legacy scheduled/event workers, integration listeners or ordinary outbox delivery. Generic host and compatibility routes still exist and are not evidence of a second registered loop. |
| Configuration | The only selectable engine is `single_loop_experimental`. The compatibility settings store presents that value without rewriting historical engine/project preferences on a no-op save. Persisted task stamps and database names are not silently reinterpreted. |
| Documentation | Current entry-point guides describe the limited runtime; older V2/V3 implementation notes are marked historical. Dated evidence is preserved and may refer to removed modules or commands. |
| Packaging and imports | Removed unused acceptance prompts and a V3-only Responses adapter. The remaining task-state prompt is included in distributions and exercised after sdist-to-wheel installation. No LangChain/LangGraph dependency or direct import of the removed V3 package was found in the scoped source search; unknown external consumers are not covered. |
| Tests | Focused Desktop, settings, distribution, owner daemon, claim and recovery tests exercise specific boundaries; their existence is not complete release qualification or proof of real-world effects. |
| Rollback and recovery | The [transition inventory](retirement-transition-2026-11-15.md) records the observed boundaries, proposed sequence and unresolved cross-version, backup and effect-reconciliation work. A binary switch or old backup alone is not a qualified rollback. |

Some surviving names such as `agent_v2`, `v2_`, `pdca_task` and legacy engine
stamps are current storage or compatibility contracts. Removing or relabeling
them without migration could damage saved work; their presence does not prove a
retired processor is runnable. Historical evaluation helpers may retain V3 in a
name without implementing the V3 loop.

The broader #229 criteria for migrating shared capabilities, reconciling old or
uncertain work, and qualifying the release remain separate from this audit.
See #230 for simpler-loop capability and gym work.

## Follow-up V3 retirement audit

The follow-up source review after #239 and #240 checked the active intelligence
package, daemon routing, engine settings, Python packaging, dependencies and
remaining V3-named source references. The V3 implementation package is absent;
the fresh-interpreter runtime test now checks that the package is not importable
as well as checking that its former processor modules are not loaded. The router
admits only `single_loop_experimental` with integer schema version 1 and a
registered owner processor. Engine settings return only that engine, while
preserving old settings and task stamps. These are source and disposable-test
findings, not an inspection of an installed or running household process.

No direct `core.intelligence.v3` import or LangChain/LangGraph dependency was
found in the scoped Python source and dependency manifests. The sole packaged
intelligence template is the shared task-state template; it includes a legacy
PDCA field, not a runnable processor. Some remaining V3-named paths have distinct
purposes and should not be removed just to erase the name:

- Browser and question guards recognize old metadata to prevent unsafe replay
  or disclosure. Old task and queue stamps remain recovery evidence.
- Skill-selection metadata, resource receipts, memory continuity and outbox
  references can describe retained state or shared behavior. They need explicit
  data and capability migration before their persisted keys can change.
- Historical evaluation fixtures and evidence may refer to a V3 arm or recorded
  results without containing the retired executable loop.

A focused disposable run of runtime registration, engine settings and schema
boundaries, supersession inspection/hold, and queue identity tests passed 96
tests. This supports the routing and preservation observations, not a complete
V3-retirement acceptance: the owner worker still does not start general legacy
scheduler, event, integration-listener or outbox workers; shared capability and
old task/effect recovery are not fully migrated. See the transition inventory
for the separate migration, effect reconciliation and rollback limitations.

The combined disposable Python suite covered 151 tests across engine settings,
runtime, migration, owner recovery and restart, daemon IPC, claim fencing, TUI
identity compatibility and installed distributions. The migration tests were
rerun after the final test-fixture adjustment. These are local working-tree
results; they are not evidence that the installed app contains the changes.
The Desktop suite passed 210 tests, and its frontend build passed.
