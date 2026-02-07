# Message Pipeline (Current)

This is the current end-to-end pipeline for user messages through Atrium/Alphonse, based on the code as of today. It lists the main path, the core decision points, and known legacy/parallel paths that can cause inconsistencies.

## Primary Flow (API/Web UI/CLI)
1. **UI -> HTTP**
   - The UI posts to Alphonse via `/agent/message`.
   - Entry: `interfaces/http/main.py` (calls `_post_alphonse_message()`).

2. **HTTP -> Signal**
   - `/agent/message` in `alphonse/infrastructure/api.py` wraps the message in an `api.message_received` signal.
   - Payload includes `text`, `args`, `channel`, `metadata`, `timestamp`, etc.

3. **Signal -> Bus**
   - `ApiGateway.emit_and_wait()` uses `ApiSense.emit()` to enqueue a bus signal.
   - `ApiSense` enforces `ALPHONSE_API_TOKEN` if set.

4. **Bus -> Heart**
   - `Heart.run()` blocks on the bus, updates runtime, and hands the signal to the DDFSM.

5. **DDFSM -> Action**
   - `api.message_received` transitions to `handle_incoming_message` (seeded in `nervous_system/seed.sql`).

6. **Intent Pipeline**
   - `IntentPipeline.handle()` instantiates `HandleIncomingMessageAction` and executes it.

7. **HandleIncomingMessageAction**
   - Loads/updates conversation state.
   - Handles pending interactions.
   - Builds the cortex state.
   - Calls `invoke_cortex()`.
   - Executes reply plans and any additional plans.

8. **Cortex Graph**
   - `ingest_node` -> `intent_node` -> `catalog_slot_node` (for some intents) -> `respond_node`.
   - If intent is unknown or needs clarification, emits `response_key` like `clarify.intent` or `generic.unknown`.

9. **Response Rendering**
   - `ResponseComposer` renders via prompt templates (prompt DB) or safe fallbacks.

10. **Plan Execution**
   - `PlanExecutor` executes scheduling and other actions via extremities (e.g., scheduler).

11. **Timers & Reminder Delivery**
   - When timers fire, `handle_timer_fired` renders and dispatches reminders via extremities.

## Known Legacy / Parallel Paths (Potential Regression Sources)
1. **Telegram Extremity Direct Handling**
   - Removed. Telegram now routes through the normalized sense adapter + DDFSM.

2. **Legacy Intent Detector Toggle**
   - `ALPHONSE_USE_LEGACY_INTENT_DETECTOR=true` routes cortex intent detection through the legacy LLM detector.

## Common Response Keys & Fallbacks
- `clarify.intent` and `generic.unknown` render via safe fallbacks if no prompt template is found.
- Example fallback: “I’m not sure what you mean yet. What would you like to do?”

## Files Touched (Reference)
- `interfaces/http/main.py`
- `alphonse/infrastructure/api.py`
- `alphonse/infrastructure/api_gateway.py`
- `alphonse/agent/nervous_system/senses/api.py`
- `alphonse/agent/nervous_system/senses/bus.py`
- `alphonse/agent/heart.py`
- `alphonse/agent/nervous_system/ddfsm.py`
- `alphonse/agent/nervous_system/seed.sql`
- `alphonse/agent/cognition/intentions/intent_pipeline.py`
- `alphonse/agent/actions/handle_incoming_message.py`
- `alphonse/agent/cortex/graph.py`
- `alphonse/agent/cognition/response_composer.py`
- `alphonse/agent/cognition/prompt_store.py`
- `alphonse/agent/cognition/plan_executor.py`
- `alphonse/agent/actions/handle_timer_fired.py`
- `alphonse/agent/extremities/scheduler_extremity.py`
