# Message IO Contract

This document defines the normalized message contract between Alphonse core logic and channel adapters.

## Goals
- Keep core reasoning channel-agnostic.
- Normalize inbound messages before entering the core pipeline.
- Normalize outbound messages before delivery.
- Let adapters own channel-specific transport and formatting.

## Core Types
Source: `alphonse/agent/io/contracts.py`

### `NormalizedInboundMessage`
- `text: str`
- `channel_type: str`
- `channel_target: str | None`
- `user_id: str | None`
- `user_name: str | None`
- `timestamp: float`
- `correlation_id: str | None`
- `metadata: dict[str, Any]`

### `NormalizedOutboundMessage`
- `message: str`
- `channel_type: str`
- `channel_target: str | None`
- `audience: dict[str, str]`
- `correlation_id: str | None`
- `metadata: dict[str, Any]`

## Adapter Interfaces
Source: `alphonse/agent/io/adapters.py`

### Sense Adapter
- Protocol: `SenseAdapter`
- Method: `normalize(payload: dict[str, Any]) -> NormalizedInboundMessage`

### Extremity Adapter
- Protocol: `ExtremityAdapter`
- Method: `deliver(message: NormalizedOutboundMessage) -> None`

### Registry
- `AdapterRegistry` stores channel adapters.
- Default registration lives in `alphonse/agent/io/registry.py`.

## Runtime Flow
1. A sense receives channel payload.
2. The sense adapter normalizes payload into `NormalizedInboundMessage`.
3. Core processing runs (cortex, plans, policy, narration orchestrator).
4. Outbound narration orchestrator produces `NormalizedOutboundMessage`.
5. Extremity adapter delivers to target channel.

## API Contract (`/agent/message`)
Source: `alphonse/infrastructure/api.py`

### Request
```json
{
  "text": "Hello",
  "args": {},
  "channel": "webui",
  "timestamp": 1730000000,
  "metadata": {"user_name": "Alex"},
  "correlation_id": "optional"
}
```

### Response
```json
{
  "message": "...",
  "data": {}
}
```

## Web Outbound Stream (`/agent/events`)
Source: `alphonse/infrastructure/api.py`

- Method: `GET`
- Query: `channel_target` (default: `webui`)
- Media type: `text/event-stream`
- Event payload format:

```text
data: {"message":"...","data":{},"channel_target":"webui","correlation_id":"..."}
```

This stream is populated by `WebExtremityAdapter` in `alphonse/agent/io/web_channel.py`.

## Where To Extend
- Add/modify adapters under `alphonse/agent/io/`.
- Register new channels in `alphonse/agent/io/registry.py`.
- Keep channel-specific formatting out of core logic.
