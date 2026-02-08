# Hume Integration Plan (Audio/Video Emotion + Audio Output)

## Objective
- Integrate Hume for:
  - Emotion interpretation from audio/video.
  - Expressive audio output (TTS).
- Keep Alphonse core channel-agnostic and preserve normalized inbound/outbound contracts.
- Enable incremental rollout with low risk to current Telegram/WebUI/CLI behavior.

## Current Fit in This Repo
- IO contract already supports normalization:
  - `alphonse/agent/io/contracts.py`
  - `NormalizedInboundMessage`
  - `NormalizedOutboundMessage`
- Outbound delivery is centralized:
  - `alphonse/agent/cognition/narration/outbound_narration_orchestrator.py`
  - `alphonse/agent/cognition/plan_executor.py`
  - `alphonse/agent/io/web_channel.py`
- API/WebUI bridge already exists:
  - `alphonse/infrastructure/api.py`
  - SSE stream at `/agent/events`

## Product Scope (MVP -> v2)

### MVP (Phase 1)
- Add asynchronous audio/video emotion analysis from uploaded media files.
- Add optional TTS generation for a text response, delivered to WebUI as an audio asset.
- Keep interaction request/response based (no full-duplex live voice yet).

### Phase 2
- Add real-time emotion streaming from microphone/webcam sessions (WebSocket pipeline).
- Add low-latency streaming TTS for faster playback start.

### Phase 3
- Add full speech-to-speech conversational mode (EVI) as an optional channel mode.

## Architecture Decisions

### 1) Do not break normalized message contracts
- Keep `NormalizedInboundMessage` and `NormalizedOutboundMessage` as-is.
- Carry modality payloads in `metadata`, for example:
  - Inbound `metadata`: `media_url`, `media_kind`, `mime_type`, `duration_sec`, `capture_source`.
  - Outbound `metadata`: `audio_url`, `audio_mime`, `audio_duration_ms`, `emotion_summary`.

### 2) Add Hume client module (provider boundary)
- New module:
  - `alphonse/agent/cognition/providers/hume.py`
- Responsibilities:
  - `analyze_media(...)` using Expression Measurement API.
  - `synthesize_speech(...)` using Octave TTS.
  - Optional streaming helpers for later phases.

### 3) Add explicit plan types for multimodal work
- Extend `alphonse/agent/cognition/plans.py` with:
  - `ANALYZE_EMOTION_MEDIA`
  - `SYNTHESIZE_AUDIO`
- Keep execution in `PlanExecutor`:
  - `alphonse/agent/cognition/plan_executor.py`

### 4) Keep channel decoration in extremities
- Core cognition produces normalized output with optional `audio_url` and `emotion_summary`.
- Channel adapters decide delivery formatting:
  - WebUI can render player/cards.
  - Telegram/CLI can stay text-only initially.

## Implementation Plan

## Phase 1A: Provider + Config
- Add env/config:
  - `HUME_API_KEY`
  - `HUME_BASE_URL` (default `https://api.hume.ai`)
  - `ALPHONSE_HUME_ENABLED` (`true/false`)
  - `ALPHONSE_HUME_TTS_ENABLED` (`true/false`)
  - `ALPHONSE_HUME_EMOTION_ENABLED` (`true/false`)
- Add provider implementation:
  - `alphonse/agent/cognition/providers/hume.py`
- Add provider health check utility.

### Acceptance Criteria
- Feature flags disabled => no behavior change.
- Missing key => clean warning path, no crash.

## Phase 1B: Emotion analysis (async media)
- Add API endpoint for WebUI-triggered analysis:
  - `POST /agent/analyze-media` (new in `alphonse/infrastructure/api.py`)
  - Input: `media_url` or uploaded file reference, `media_kind` (`audio|video`), optional `channel_target`.
- Pipeline:
  - API signal -> action -> emits `ANALYZE_EMOTION_MEDIA` plan.
  - `PlanExecutor` calls Hume Expression Measurement REST.
  - Result normalized into an `emotion_summary` payload.
  - Delivery via `WebExtremityAdapter` through SSE event stream (`/agent/events`).

### Acceptance Criteria
- A valid media request returns a summary payload to web events.
- Errors are structured and user-safe.

## Phase 1C: TTS audio output (non-streaming first)
- Add optional TTS request path:
  - `POST /agent/tts` or augment communicate flow with `output.audio=true`.
- Pipeline:
  - Create `SYNTHESIZE_AUDIO` plan.
  - `PlanExecutor` calls Hume Octave TTS.
  - Store output in temporary/public asset location.
  - Deliver normalized outbound message with:
    - `message` (text transcript)
    - `metadata.audio_url`
    - `metadata.audio_mime`
- Keep WebUI primary consumer for MVP.

### Acceptance Criteria
- Text-to-audio works for WebUI and returns playable audio URL.
- Existing text-only flows remain unchanged.

## Phase 1D: Persistence + observability
- Add DB storage for emotion artifacts (minimal):
  - `analysis_id`, `channel`, `target`, `source_type`, `top_emotions`, `raw_ref`, `created_at`.
- Add logs/traces for:
  - Hume request id, latency, success/failure, correlation id.

### Acceptance Criteria
- Emotion outputs are queryable for diagnostics.
- Trace entries correlate request -> plan -> delivery.

## UI Integration Contract (for Web UI agent)
- Existing:
  - `POST /agent/message`
  - `GET /agent/events?channel_target=<id>`
- New (MVP):
  - `POST /agent/analyze-media`
    - Request:
      - `channel: "webui"`
      - `channel_target: "<session-or-user-id>"`
      - `media_url` or `upload_id`
      - `media_kind: "audio" | "video"`
    - Response/event payload:
      - `type: "emotion.analysis.completed"`
      - `emotion_summary: { top: [...], valence?: number, arousal?: number, notes?: string }`
  - `POST /agent/tts`
    - Request:
      - `text`
      - `voice_id` (optional)
      - `channel_target`
    - Response/event payload:
      - `type: "tts.completed"`
      - `audio_url`
      - `audio_mime`
      - `text`

## Security, Privacy, and Policy Guardrails
- Never store raw audio/video by default unless explicit retention is enabled.
- Redact API keys and sensitive metadata from logs.
- Add retention policy for generated assets (TTL cleanup).
- Add user consent checks before emotion analysis in UI flows.
- Add policy gate for high-stakes inference usage (no diagnostic/medical claims).

## Testing Strategy
- Unit tests:
  - Hume client request/response mapping.
  - Plan executor behavior for new plan types.
  - WebExtremity event payload shape.
- Integration tests:
  - `/agent/analyze-media` end-to-end with mocked Hume responses.
  - `/agent/tts` end-to-end and SSE delivery.
- Regression tests:
  - Existing Telegram/CLI/Web text flows unchanged.

## Rollout Strategy
- Step 1: Behind feature flags in non-prod.
- Step 2: Enable for WebUI test users only.
- Step 3: Observe latency/error rates and adjust retries/timeouts.
- Step 4: Expand to broader users and add streaming enhancements.

## Open Questions
- Should emotion analysis be user-triggered only (MVP), or automatic for all uploaded voice/video?
- Preferred storage strategy for generated audio:
  - local temp files
  - object storage (recommended for scale)
- Do we want one default voice or per-family-member voice preferences in settings?

## References (Official Hume docs)
- Expression Measurement overview: https://dev.hume.ai/docs/expression-measurement/overview
- Expression Measurement REST (batch jobs): https://dev.hume.ai/docs/expression-measurement/rest
- Expression Measurement WebSocket (real-time): https://dev.hume.ai/docs/expression-measurement/websocket
- TTS overview (Octave): https://dev.hume.ai/docs/text-to-speech-tts/overview
- Voice overview: https://dev.hume.ai/docs/voice/overview
- EVI overview: https://dev.hume.ai/docs/speech-to-speech-evi/overview
