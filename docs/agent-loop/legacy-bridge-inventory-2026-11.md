# Legacy loop bridge inventory (2026-11-12)

Working inventory for #222, based on the integration branch at `246d3d19`.
It is not a complete removal plan or a release qualification. The chosen simple
loop is experimental; V3 remains the production bridge, limited to security and
critical fixes until the replacement is qualified. The `agent_v2` package and
`v2_` database prefix alone do not identify disposable V2 loop code.

| Surface | Current evidence | Transition obligation |
| --- | --- | --- |
| Runtime and ingestion | `runtime.py` constructs `EngineRoutingProcessor` with `HierarchicalCAPDProcessor`; `core/messages/channel.py` stamps new messages V3/schema 3. The candidate is not registered. | Preserve explicit engine/schema and actor identity through the whole lifecycle; keep the candidate default-off until #228 and #227. |
| Settings and UI | `intelligence_engine_settings.py` reads a stored `tactical_v2` preference as V3 for new work; daemon settings IPC remains. TUI no longer offers V2. | Audit all clients and old settings before removing settings and selectors; changing a preference must not migrate a saved task. |
| Saved and queued work | `TaskState.from_dict` and `from_message` interpret absent/blank engine stamps as V2; the router rejects V2 and unknown engines. V3 checkpoint loading and question/scheduled handoff occur in `core/core.py` and `core/questions.py`. | Define visible, explicit recovery for old tasks and uncertain effects; do not erase history or silently replay work in another engine. Test queued, resumed, scheduled and malformed states end to end. |
| Questions, delivery, scheduling | `core/questions.py`, `core/io/`, `services/scheduled_worker.py`, `services/event_worker.py`, and `daemon.py` own shared persistence, dispatch and delivery state; V3 has loop-specific resumption and scheduled-delivery metadata. | Separate shared lifecycle from V3 assumptions; preserve question identity, actor and actual delivery receipts. Prepared or queued is not delivered. |
| Fencing and recovery | `core/messages/supersession_inspection.py` and `supersession_hold.py` contain experimental machinery; `CoreLoop` refuses runtime claim-fencing mode. #216 remains open. | Complete end-to-end stale-worker fencing, uncertain-effect reconciliation and publication/queue ownership before rollout; do not infer safety from the experimental components. |
| Loop code and shared capabilities | V2 PDCA has moved from `core/intelligence/pdca/` to `evaluation/legacy_pdca/` for historical replay; the unused V2 `basic.py` placeholder has been removed. V3 is under `core/intelligence/v3/`, alongside experimental candidate code. Tools, skills, actor authorization, memory, storage and integrations are shared infrastructure. | Classify dependencies before deleting anything; preserve bindings, privacy, household data and recovery state. |
| Evaluations and evidence | `evaluation/` includes historical V2/V3 and candidate adapters; `docs/v3-hierarchical-capd/` retains exploratory results. | Keep historical evidence labeled and failures intact. Scripted fixtures do not establish autonomous performance or real delivery. |

The V2 retirement tests in `tests/test_agent_v3_engine_rollout.py` exercise
settings, stamps and router rejection. They do not establish safe user-visible
recovery of all previously queued or checkpointed tasks. The core loop catches
router failures as processing failures. The daemon now treats an unregistered
engine or mismatched schema as non-retryable and prepares a notice that the
saved state needs review; it does not claim that review or delivery occurred.
`tests/test_legacy_bridge_daemon.py` exercises unstamped, V2, unknown and
embedded V2 checkpoint messages through the durable queue, core router and
daemon: no V3 dispatch, one failed attempt, retained queue record and a failure
notice queued in the outbox. This is not a recovery UI or a delivery receipt. Scheduled
occurrences, authoritative stored checkpoints, question resumes and failure
notice delivery still require end-to-end verification; memory setup precedes
router validation for other engines and must not be treated as generally fenced
by this regression. For the normal `EngineRoutingProcessor`, V2-stamped,
unstamped, unknown-engine and mismatched-schema tasks are now rejected before
task memory and conversation history are loaded; daemon prework and the
active-task callback still precede this check. The isolated historical replay
processor remains separately usable.

Independently unused V2-only code may be removed before the broader release
after its consumers, exports, replay, and legacy-state recovery are checked.
Final removal of the V3 bridge and a single-loop production transition require
the relevant #224–#228 work, reconciliation of #216–#218, a release decision,
and a data-preserving recovery and rollback plan in #229. No household rollout
or legacy data deletion is authorized by this inventory.

## Targeted V2 removal audit — 14 November

On the owner's direction to continue V2 retirement, removed `basic.py` and the
`BasicIntelligenceProcessor` export. This retired placeholder had no remaining
repository callers and did not execute real tasks. Its removal changes the
distributable Python API: external imports of that class will fail, and
repository searches cannot rule out unknown external users. No shared runtime,
V3 router, stored state, history, or deterministic PDCA replay was removed.

The PDCA package is no longer selected for normal task dispatch, but the
historical `evaluation/deterministic_adapters.py` still constructs
`PDCAIntelligenceProcessor`, and repository tests directly exercise its nodes
and replay. Its processor, graph, nodes and package exports therefore cannot
be removed under an assumption that replay and public imports stay intact.
An inventory of module-level private functions in the PDCA package found no
definition with zero references in repository Python source and tests; that
search does not prove each reference is live or discover dynamic external
callers. No code was removed in this pass. To retire PDCA, first decide whether
to preserve the executable historical replay baseline, replace it, or retain
only its already-recorded evidence; separately decide the public API policy.

## Isolate replay imports — 14 November

The shared intelligence package no longer re-exports the retired PDCA processor
or graph helpers. Historical replay imports the retained implementation directly
and only for its V2 arm. A fresh interpreter importing `TaskState` does not load
the PDCA package. This removes three public package-level Python imports; callers
of those aliases must migrate. The internal legacy PDCA module and historical
V2/V3 replay remain for now. This does not retire V3.

## Move the V2 loop out of the active core — 14 November

The retained PDCA implementation now lives in `evaluation/legacy_pdca/`, and
the historical replay adapter and its tests import it there. No active runtime
path imports the old implementation. The former
`alphonse.agent_v2.core.intelligence.pdca` Python import paths are removed;
external callers using them must migrate or pin an earlier version. Historical
replay is still executable, and recorded evidence is unchanged. This is a
packaging and namespace transition, not final deletion of replay or retirement
of the active V3 bridge; pending task-state recovery remains a separate task.

The old PDCA planning, check, act and criteria-review templates also now live
with the historical evaluation package. Its planner resolves its own template
directory, and package-data configuration includes those files. The active
shared acceptance and task-state templates remain in place. Focused template,
planner and replay tests pass; this move does not itself retire replay.
For commit `91643e88`, a wheel built from a clean `git archive` contained the
legacy processor and all five moved templates. A disposable target installation
loaded the replay planner and rendered its template from the installed wheel;
this verifies packaging for that path, not full installed runtime qualification.

Some remaining names containing `pdca` are persisted or active shared
contracts, not references to the removed V2 processor. In particular,
`routing_disposition="pdca_task"` is consumed by the current queue and V3, and
scheduled-task `delivery_mode="pdca"` still selects deferred agent work.
Their labels can be migrated later only with compatibility for persisted tasks;
deleting or renaming the values alone would disrupt the active bridge. Shared
task-state and processor docstrings no longer describe that state as a future
V2 graph.
