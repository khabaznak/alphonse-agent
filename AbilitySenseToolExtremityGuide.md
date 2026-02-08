# Ability, Sense, Tool, Extremity Guide

## Canonical Concepts

### Sense
- Purpose: perceive external/internal events and emit normalized inbound signals.
- Owns: ingestion, polling/subscription, source-specific extraction.
- Does not own: business decisions, user-facing wording, tool orchestration.

### Tool
- Purpose: perform a concrete operation with explicit inputs/outputs.
- Owns: deterministic execution (clock read, scheduling, API call, DB write).
- Does not own: intent resolution, channel formatting.

### Skill (Ability)
- Purpose: execute an intent by orchestrating state + tools.
- Owns: intent-level behavior and decisioning for one capability.
- Does not own: transport/adapters.

### Extremity
- Purpose: deliver normalized outbound messages to a destination channel/system.
- Owns: channel-specific formatting/delivery mechanics.
- Does not own: intent logic.

## Runtime Flow
- `Sense -> Intent Detection -> Ability -> Tool(s) -> Plan/Response -> Extremity`

## Current Code Anchors
- Senses: `/Users/alex/Code Projects/atrium-server/alphonse/agent/nervous_system/senses/`
- Tools: `/Users/alex/Code Projects/atrium-server/alphonse/agent/tools/`
- Abilities registry: `/Users/alex/Code Projects/atrium-server/alphonse/agent/cognition/abilities/registry.py`
- Intent execution path: `/Users/alex/Code Projects/atrium-server/alphonse/agent/cortex/graph.py`
- Channel extremities/adapters: `/Users/alex/Code Projects/atrium-server/alphonse/agent/io/`
- Normalized contracts: `/Users/alex/Code Projects/atrium-server/alphonse/agent/io/contracts.py`

## How To Add A New Sense
1. Create a new sense module in `/Users/alex/Code Projects/atrium-server/alphonse/agent/nervous_system/senses/`.
2. Define `key`, `signals`, `start()`, `stop()`, and emit `Signal(...)` on bus.
3. Normalize incoming payload shape before emission (prefer IO adapters where applicable).
4. Add tests for emission behavior and payload correctness.

Checklist:
- [ ] Emits correct signal type.
- [ ] Includes correlation id when available.
- [ ] Can start/stop cleanly.

## How To Add A New Tool
1. Create tool module under `/Users/alex/Code Projects/atrium-server/alphonse/agent/tools/`.
2. Keep function signatures explicit and side effects limited.
3. Register tool in `/Users/alex/Code Projects/atrium-server/alphonse/agent/tools/registry.py`.
4. Add unit tests for success/error paths.

Checklist:
- [ ] Input/Output contract documented.
- [ ] No channel-specific logic.
- [ ] Tool registered in default tool registry.

## How To Add A New Skill (Ability)
1. Add intent spec in `/Users/alex/Code Projects/atrium-server/alphonse/agent/cognition/intent_catalog.py`.
2. Implement ability executor in `/Users/alex/Code Projects/atrium-server/alphonse/agent/cortex/graph.py` (or a dedicated ability module if moved later).
3. Register ability in `AbilityRegistry` with required tools declared.
4. If needed, produce plans for `PlanExecutor`.
5. Add regression tests for:
- intent routing
- ability response/plan generation
- fallback/clarify behavior

Checklist:
- [ ] Intent exists and is enabled.
- [ ] Ability registered with declared tools.
- [ ] End-to-end test passes.

## How To Add A New Extremity
1. Add adapter in `/Users/alex/Code Projects/atrium-server/alphonse/agent/io/`.
2. Implement `deliver(NormalizedOutboundMessage)`.
3. Register in `/Users/alex/Code Projects/atrium-server/alphonse/agent/io/registry.py`.
4. Ensure no business logic leaks into adapter.
5. Add delivery-path tests.

Checklist:
- [ ] Reads only normalized outbound payload.
- [ ] Handles missing/invalid target safely.
- [ ] Registered in IO registry.

## Design Rules
- Keep orchestration in abilities, not in senses/extremities.
- Keep tools reusable across abilities.
- Keep channel-specific behavior in extremities only.
- Prefer explicit state/metadata over hidden heuristics.

## Definition of Done For A New Capability
- [ ] New intent added.
- [ ] Ability implemented + registered.
- [ ] Required tool(s) implemented/registered.
- [ ] Inbound/outbound contracts preserved.
- [ ] Tests added and passing.
