# Ability, Sense, Tool, Extremity Guide

## Terminology Standard
- `Skills`:
  - Markdown-based operational instructions used by coding agents (for example `SKILL.md` files).
  - Human-authored guidance, not Alphonse runtime executables.
- `Abilities`:
  - Alphonse runtime capabilities bound to intents.
  - Implemented via code and/or JSON ability specs.
  - Executed through `intent -> ability -> tools`.

## Canonical Concepts

### Sense
- Purpose: perceive external/internal events and emit normalized inbound signals.
- Owns: ingestion, polling/subscription, source-specific extraction.
- Does not own: business decisions, user-facing wording, tool orchestration.
Example: `LocationSense` captures address/location input and delegates normalization to `GeocoderTool`.

### Tool
- Purpose: perform a concrete operation with explicit inputs/outputs.
- Owns: deterministic execution (clock read, scheduling, API call, DB write).
- Does not own: intent resolution, channel formatting.
Example: `GeocoderTool` converts an address to lat/lng via Google Maps.

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

## Onboarding Model

### Primary Onboarding (Out-Of-Box)
- Purpose: bootstrap Alphonse on first run with the first admin user.
- Scope: global system bootstrap + first chat principal.
- Trigger: no system bootstrap marker exists.
- Result:
  - captures first admin display name,
  - stores bootstrap completion marker,
  - stores bootstrap admin principal id.

### Secondary Onboarding (Subsequent Users)
- Purpose: onboard additional household users after primary bootstrap is complete.
- Scope: per-user/per-channel profile capture and linkage.
- Trigger: known system bootstrap, but incoming principal has no profile/onboarding completion.
- Result:
  - captures user display name and defaults,
  - stores per-principal onboarding state.

### State Ownership
- Global bootstrap state: system-scope preferences (single source of truth).
- Per-user onboarding state: channel/person principal preferences.
- Onboarding + location operational records: persisted in dedicated nerve-db tables.

## Onboarding + Location Data Model

### `onboarding_profiles`
- Purpose: resumable onboarding progress per principal.
- Key fields:
  - `principal_id` (PK)
  - `state` (`not_started|awaiting_name|operational|in_progress|completed`)
  - `primary_role` (for example `admin`)
  - `next_steps_json`
  - `resume_token`
  - `completed_at`

### `location_profiles`
- Purpose: stable places per principal (home/work/other).
- Key fields:
  - `location_id` (PK)
  - `principal_id` (FK principals)
  - `label` (`home|work|other`)
  - `address_text`
  - `latitude`, `longitude`
  - `source`, `confidence`, `is_active`

### `device_locations`
- Purpose: ephemeral location samples from devices (for example AlphonseLink).
- Key fields:
  - `id` (PK)
  - `principal_id` (optional FK principals)
  - `device_id`
  - `latitude`, `longitude`, `accuracy_meters`
  - `source`, `observed_at`, `metadata_json`

## API + CLI Surfaces

### API
- `/agent/onboarding/profiles`
- `/agent/locations`
- `/agent/device-locations`
- `/agent/tool-configs`

### CLI
- `alphonse.agent.cli onboarding ...`
- `alphonse.agent.cli locations ...`
- `alphonse.agent.cli tool-configs ...`

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
