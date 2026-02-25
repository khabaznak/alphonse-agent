# Alphonse Tools Inventory

This inventory is aligned with the current ToolSpec single source of truth in:
- `/Users/alex/Code Projects/atrium-server/alphonse/agent/tools/registry2.py`

Runtime registrations are wired in:
- `/Users/alex/Code Projects/atrium-server/alphonse/agent/tools/registry.py`

## Planner-Callable Tools

| Tool | Signature (agent call) | Scope | Description | Suggested Functionality | Safety / Confirm |
|---|---|---|---|---|---|
| `askQuestion` | `askQuestion(question)` | `planning, clarification` | Ask user one clear question | Missing required user data | `low / no` |
| `get_time` | `get_time()` | `time, planning` | Get current time | Time reference, scheduling | `low / no` |
| `create_reminder` | `create_reminder(ForWhom, Time, Message)` | `time, reminders` | Create reminder | User reminder requests | `medium / no` |
| `get_my_settings` | `get_my_settings()` | `context, settings` | Get runtime settings | Locale/time/tone-sensitive tasks | `low / no` |
| `get_user_details` | `get_user_details()` | `context, identity` | Get user/channel details | Identity/context before actions | `low / no` |
| `local_audio_output_speak` | `local_audio_output_speak(text, voice?, blocking?, volume?)` | `audio, output` | Local TTS output | Spoken output when requested | `low / no` |
| `stt_transcribe` | `stt_transcribe(asset_id, language_hint?)` | `audio, transcription` | Transcribe audio asset | Incoming audio transcript | `medium / no` |
| `telegram_get_file_meta` | `telegram_get_file_meta(file_id)` | `telegram, files` | Resolve Telegram file metadata | Before download/transcription | `medium / no` |
| `telegram_download_file` | `telegram_download_file(file_id, sandbox_alias?, relative_path?)` | `telegram, files` | Download Telegram file | Prepare file for downstream tools | `high / no` |
| `transcribe_telegram_audio` | `transcribe_telegram_audio(file_id, language?, sandbox_alias?)` | `telegram, audio, transcription` | Download+transcribe Telegram audio | Direct Telegram voice handling | `high / no` |
| `analyze_telegram_image` | `analyze_telegram_image(file_id?, prompt?, sandbox_alias?)` | `telegram, image, analysis` | Download+analyze Telegram image | Telegram inbound image semantics | `high / no` |
| `vision_analyze_image` | `vision_analyze_image(sandbox_alias, relative_path, prompt?)` | `vision, image, analysis` | Analyze sandbox image with dedicated vision model | Receipts, notes, package checks, object ID | `medium / no` |
| `job_create` | `job_create(name, description, schedule, payload_type, payload, timezone?, domain_tags?, safety_level?, requires_confirmation?, retry_policy?, idempotency?, enabled?)` | `automation, jobs, productivity` | Create scheduled job | Recurring/background automation | `medium / no` |
| `job_list` | `job_list(enabled?, domain_tag?, limit?)` | `automation, jobs, productivity` | List jobs | Review configured automations | `low / no` |
| `job_pause` | `job_pause(job_id)` | `automation, jobs, control` | Pause job | Temporarily disable automation | `medium / no` |
| `job_resume` | `job_resume(job_id)` | `automation, jobs, control` | Resume job | Re-enable automation | `medium / no` |
| `job_run_now` | `job_run_now(job_id)` | `automation, jobs, control` | Trigger immediate run | Test/force scheduled job | `medium / no` |
| `job_delete` | `job_delete(job_id)` | `automation, jobs, control` | Delete job | Remove automation | `high / yes` |
| `terminal_sync` | `terminal_sync(command, cwd?, timeout_seconds?)` | `ops, terminal, automation` | Policy-constrained terminal execution | Controlled terminal-like ops | `high / yes` |
| `terminal_async` | `terminal_async(command, cwd?, timeout_seconds?, sandbox_alias?)` | `ops, terminal, automation` | Submit terminal command for async execution | Long-running command workflows | `high / yes` |
| `terminal_async_command_status` | `terminal_async_command_status(command_id)` | `ops, terminal, automation` | Poll status/output for async command | Read async execution result | `low / no` |
| `user_register_from_contact` | `user_register_from_contact(display_name?, role?, relationship?, contact_user_id?, contact_first_name?, contact_last_name?, contact_phone?)` | `identity, admin, onboarding` | Register/update user from shared contact | Admin-led onboarding | `medium / no` |
| `user_remove_from_contact` | `user_remove_from_contact(contact_user_id?)` | `identity, admin, access-control` | Deactivate registered user | Admin-led removal | `high / yes` |

## Runtime/Internal Aliases

These are registered for compatibility/internal routing and are not distinct business tools:
- `clock`
- `createTimeEventTrigger`
- `schedule_event`

## Notes

- Tool schemas used by the planner are generated dynamically from ToolSpec via `/Users/alex/Code Projects/atrium-server/alphonse/agent/cognition/tool_schemas.py`.
- Tool catalog markdown for prompts is rendered from ToolSpec via `/Users/alex/Code Projects/atrium-server/alphonse/agent/cognition/tool_catalog_renderer.py`.
