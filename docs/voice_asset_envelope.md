# Voice Asset Envelope (MVP)

This project accepts provider messages through the same incoming envelope path (`api.message_received` -> IO normalization -> Heart action -> Cortex).

## Envelope shape

Use this shape for audio from `webui` (and future providers):

```json
{
  "provider": "webui",
  "channel": "webui",
  "target": "webui",
  "text": "",
  "content": {
    "type": "asset",
    "assets": [
      { "asset_id": "ASSET_UUID", "kind": "audio" }
    ]
  },
  "controls": {
    "audio_mode": "local_audio"
  },
  "metadata": {}
}
```

`controls.audio_mode`:
- `none`: text-only response
- `local_audio`: text response plus local TTS playback via `local_audio_output.speak`

## Asset ingestion

Upload binary files through:
- `POST /agent/assets` (multipart form-data)

Response:
- `asset_id`
- `mime`
- `bytes`
- `sha256`

The backend stores files in sandbox storage and registers metadata in Nerve-DB. The LLM only receives `asset_id`.

## STT tool

Tool: `stt_transcribe`

Input:
- `asset_id` (required)
- `language_hint` (optional, e.g. `es-MX`)

Output:
- `text`
- `segments` (optional)

The tool resolves file location internally from Nerve-DB + sandbox alias, so raw filesystem paths are not exposed to the model.

## Pattern for future senses

Reuse the same envelope + asset registry for:
- OCR: `kind=image` + `ocr_extract(asset_id)`
- Vision: `kind=image` + `analyze_image(asset_id, prompt)`
- PDF parsing: `kind=document` + `extract_pdf_text(asset_id)`

Keep all of them asset-id based and policy-gated in tool availability.
