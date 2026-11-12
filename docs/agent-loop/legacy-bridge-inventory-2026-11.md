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
| Loop code and shared capabilities | `core/intelligence/pdca/` and `basic.py` remain; V3 is under `core/intelligence/v3/`, alongside experimental candidate code. Tools, skills, actor authorization, memory, storage and integrations are shared infrastructure. | Classify dependencies before deleting anything; preserve bindings, privacy, household data and recovery state. |
| Evaluations and evidence | `evaluation/` includes historical V2/V3 and candidate adapters; `docs/v3-hierarchical-capd/` retains exploratory results. | Keep historical evidence labeled and failures intact. Scripted fixtures do not establish autonomous performance or real delivery. |

The V2 retirement tests in `tests/test_agent_v3_engine_rollout.py` exercise
settings, stamps and router rejection. They do not establish safe user-visible
recovery of all previously queued or checkpointed tasks. The core loop catches
router failures as processing failures, but the full daemon/queue and delivery
behavior for each legacy state still needs the end-to-end audit in #222.

Removal requires #224–#228, reconciliation of #216–#218, a qualified release
decision, and an explicit data-preserving recovery and rollback plan in #229.
No household rollout or legacy data deletion is authorized by this inventory.
