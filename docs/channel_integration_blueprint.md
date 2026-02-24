# Channel Integration Blueprint (Telegram as Reference)

This document defines the reusable scaffolding for integrating a communication channel into Alphonse (Telegram today, Discord/others next).

## Objectives

- Keep ingestion, planning, tool execution, and delivery contracts stable across channels.
- Let the LLM decide **what** to do, while tools deterministically enforce **how** it is done.
- Keep channel security and authorization deterministic and auditable.

## Architecture Layers

1. Integration adapter (provider API boundary)
- Example: `alphonse/agent/extremities/interfaces/integrations/telegram/telegram_adapter.py`
- Responsibilities:
  - Poll/webhook provider events
  - Normalize raw provider events into stable signal payloads
  - Apply early access checks (reject/accept/invite flow)
  - Emit bus signals only after policy checks

2. Sense normalization
- Example: `alphonse/agent/nervous_system/senses/telegram.py`, `alphonse/agent/io/telegram_channel.py`
- Responsibilities:
  - Convert provider-specific payloads into `telegram.message_received` style normalized payloads
  - Preserve structured metadata (`content_type`, `contact`, `chat_type`, reply metadata)

3. Cortex + planner
- Example: `alphonse/agent/cortex/nodes/plan.py`, `alphonse/agent/cortex/task_mode/pdca.py`
- Responsibilities:
  - Decide tool calls from ToolSpec catalog
  - Execute tools through runtime registry
  - Consume canonical tool responses and continue/recover

4. Tool runtime + ToolSpec SSOT
- Runtime registry: `alphonse/agent/tools/registry.py`
- Prompt/catalog registry: `alphonse/agent/tools/registry2.py`
- Renderer: `alphonse/agent/cognition/tool_catalog_renderer.py`
- Responsibilities:
  - Runtime: real callable tools
  - ToolSpec: prompt-visible tool contract (single source of truth)

5. Policy + persistence
- Examples:
  - `alphonse/agent/nervous_system/telegram_chat_access.py`
  - `alphonse/agent/nervous_system/telegram_invites.py`
  - `alphonse/agent/nervous_system/user_service_resolvers.py`
- Responsibilities:
  - Access control, invite approval, identity linkage, and audit-safe state updates

## Tool Contract Rules (Required)

All new channel-related tools must:

- Be registered in runtime registry (`registry.py`)
- Be declared in ToolSpec registry (`registry2.py`)
- Return canonical shape:
```json
{
  "status": "ok|failed",
  "result": {},
  "error": null,
  "metadata": {}
}
```
- Enforce authorization/validation deterministically in tool code (not in prompt logic)

## Deterministic Security Pattern

Use this pattern for sensitive actions (user onboarding, permissions, deletion, etc.):

1. LLM chooses tool call.
2. Tool validates caller identity from execution state/context.
3. Tool verifies authorization from DB (for example `is_admin`).
4. Tool executes deterministic side effects only if authorized.
5. Tool returns explicit `permission_denied` (or equivalent) on failure.

This allows flexible planning while keeping enforcement deterministic.

## Channel Access Model Pattern

Recommended DB-backed channel policy model:

- Pending invite table (who asked, status, context)
- Access table (chat/channel id, type, owner, status, policy)
- Resolver mapping table (internal user ↔ provider user id)

Telegram implementation reference:

- `telegram_pending_invites`
- `telegram_chat_access`
- `user_service_resolvers`

## Integration Sandbox Root Contract

For all file-producing integrations (Telegram, Discord, WhatsApp, Slack, etc.):

1. Default sandbox paths must live under Alphonse workdir root.
2. Channel folders should branch from a shared integration root (for example `.../dumpster/sandboxes/<channel_or_asset_alias>`).
3. `/tmp` or `/private/tmp` should only be used when explicitly configured via environment override.
4. Tool code must resolve paths through sandbox aliases, not hardcoded absolute paths.

Current default policy implementation:

- Workdir-backed root preferred (`dumpster/sandboxes`)
- Fallback root: `/tmp/alphonse-sandbox`
- Override env: `ALPHONSE_SANDBOX_ROOT`

## Structured Content Payload Pattern

Do not create separate signal envelopes for every content type.

Use same inbound signal type with structured payload additions:

- `content_type`: `text|contact|text+contact|...`
- content object when present (`contact`, `attachment`, etc.)
- preserve original provider payload for audit/debug

This keeps downstream contracts stable and extendable.

## Timed Follow-up Pattern

For proactive workflows (for example onboarding greeting):

- Tool schedules a timed signal via store/scheduler APIs
- Use small delay (for example +10s) to avoid blocking current turn
- Keep payload deterministic (target, message, locale/context)

Reference:

- `alphonse/agent/nervous_system/timed_store.py`
- `alphonse/agent/nervous_system/timed_scheduler.py`

## Implementation Checklist for a New Channel (Discord, etc.)

1. Build integration adapter and emit normalized inbound signal.
2. Add channel IO adapter for outbound delivery.
3. Add channel config loader/env wiring.
4. Add invite/access policy persistence and enforcement.
5. Add resolver mapping between internal users and channel user IDs.
6. Add/extend tools needed by channel-specific workflows.
7. Register tools in both runtime registry and ToolSpec registry.
8. Ensure tool catalog renderer includes those tools.
9. Add tests:
  - ingestion normalization
  - policy allow/deny/invite behavior
  - admin-gated tool behavior
  - timed follow-up behavior
  - outbound blocked when unauthorized

## Testing Standard

At minimum include:

- Unit tests for tool deterministic gates (admin/non-admin)
- Unit tests for policy resolution and chat/channel authorization
- Integration-style tests for signal flow and adapter behavior
- Regression tests for canonical response shape and required fields

## Telegram Features Implemented with This Blueprint

- DB-backed inbound access enforcement (private/group semantics)
- Invite flow and approval provisioning to active access
- Owner-linked group enforcement path
- Structured contact payload ingestion
- Admin-gated user onboarding/removal tools:
  - `user_register_from_contact`
  - `user_remove_from_contact`
- Proactive timed intro scheduling after successful onboarding

## Notes for Future Channels

- Keep provider specifics in adapter layer only.
- Avoid prompt-only security logic.
- Prefer shared generic primitives (resolver mapping, invite/access tables, timed signals, ToolSpec registry).
- Add channel-specific policy exceptions only where unavoidable.
