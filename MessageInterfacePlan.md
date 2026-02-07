# Message IO Contract + Transition Plan

This document defines a channel-agnostic messaging contract for Alphonse and a staged plan to migrate existing channel-specific code (Telegram, CLI, etc.) to the new interface.

## Goals
- Normalize inbound messages into a single canonical shape before they enter the core.
- Normalize outbound messages from the core into a single canonical shape before they exit.
- Keep channel-specific formatting (buttons, embeds, markdown quirks) out of the core.
- Allow multiple channels (Telegram, Discord, WhatsApp, Teams, SSH, CLI, UI, Push) without core changes.

---

## Proposed Message IO Contract

### Inbound (Sense -> Core)
All inbound events should be normalized into this shape before entering the DDFSM/heart:

```python
@dataclass(frozen=True)
class NormalizedInboundMessage:
    text: str
    channel_type: str              # e.g. telegram, discord, cli, api, ui
    channel_target: str | None     # chat_id, channel_id, user handle, etc.
    user_id: str | None
    user_name: str | None
    timestamp: float               # unix seconds
    correlation_id: str | None
    metadata: dict[str, object]    # channel-specific raw data (not for core logic)
```

Notes:
- `metadata` is for channel adapters only. Core should not branch on it.
- If the adapter cannot provide a field, set it to `None`.

### Outbound (Core -> Extremity)
Core emits a canonical message intent that channel extremities render:

```python
@dataclass(frozen=True)
class NormalizedOutboundMessage:
    message: str
    channel_type: str              # requested channel
    channel_target: str | None     # where to send it
    audience: dict[str, str]       # {kind: person|system, id: ...}
    correlation_id: str | None
    metadata: dict[str, object]    # optional hints, not channel-specific formatting
```

Notes:
- `message` is plain text from the core.
- `metadata` should only contain hints (tone, locale, urgency). No channel-specific markup.

---

## Current Hardcoded Areas (Targets for Refactor)
1. **Telegram direct handling bypassing DDFSM**
   - Action: Make Telegram a pure adapter that emits `NormalizedInboundMessage`.

2. **Legacy HandleMessageAction pipeline**
   - Action: Removed in favor of the unified pipeline.

3. **Action payloads with channel-specific fields**
   - Example: `direct_reply` in `handle_incoming_message.py`
   - Action: Replace with canonical outbound message + adapter rendering.

---

## Transition Plan (Phased)

### Phase 1: Define Contract + Adapter Interfaces
- Add `NormalizedInboundMessage` and `NormalizedOutboundMessage` dataclasses.
- Introduce adapter base classes:
  - `SenseAdapter`: produces `NormalizedInboundMessage`.
  - `ExtremityAdapter`: consumes `NormalizedOutboundMessage`.
- Add channel registry to map `channel_type -> adapter`.

Deliverable:
- New module: `alphonse/agent/io/contracts.py` (or similar).

### Phase 2: Normalize Inbound Flow
- Update all inbound entrypoints to produce `NormalizedInboundMessage`.
  - API sense
  - CLI sense
  - Telegram sense
- Update DDFSM signals to carry normalized payload only.

Deliverable:
- Incoming payloads are consistent across channels.

### Phase 3: Normalize Outbound Flow
- Update action results to emit `NormalizedOutboundMessage`.
- Ensure the coordinator and extremities accept only normalized outputs.

Deliverable:
- No channel-specific formatting in the core.

### Phase 4: Migrate Telegram
- Use Telegram adapters for inbound/outbound message normalization.
- Ensure no direct interpretation bypass exists.

Deliverable:
- Telegram uses the same pipeline as all other channels.

### Phase 5: Deprecate Legacy Pipeline
- HandleMessageAction already removed.
- Remove legacy LLM intent detector if unused.

Deliverable:
- Single canonical flow through `HandleIncomingMessageAction` + cortex.

---

## Risks and Mitigations
- **Risk:** Channel-specific UX regressions (buttons, formatting).
  - Mitigation: Keep channel formatting in extremities only.
- **Risk:** Message metadata needed by core is lost.
  - Mitigation: Explicitly surface required fields in the contract, keep metadata for adapters.

---

## Suggested Next Steps
1. Decide the module path for the new contract (e.g. `alphonse/agent/io/`).
2. Draft the adapter interfaces and registry.
3. Start with Telegram (largest surface area), then CLI and API.
