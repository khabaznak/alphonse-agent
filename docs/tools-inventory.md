# Alphonse Tools Inventory

This inventory is aligned with the current ToolSpec single source of truth in:
- `/Users/alex/Code Projects/atrium-server/alphonse/agent/tools/registry2.py`

Runtime registrations are wired in:
- `/Users/alex/Code Projects/atrium-server/alphonse/agent/tools/registry.py`

## Planner-Callable Tools

| Tool | Signature (agent call) | Scope | Description | Suggested Functionality | Safety / Confirm |
|---|---|---|---|---|---|
| `askQuestion` | `askQuestion(question)` | `planning, clarification` | Ask user one clear question | Missing required user data | `low / no` |
| `getTime` | `getTime()` | `time, planning` | Get current time | Time reference, scheduling | `low / no` |
| `createReminder` | `createReminder(ForWhom, Time, Message)` | `time, reminders` | Create reminder | User reminder requests | `medium / no` |
| `getMySettings` | `getMySettings()` | `context, settings` | Get runtime settings | Locale/time/tone-sensitive tasks | `low / no` |
| `getUserDetails` | `getUserDetails()` | `context, identity` | Get user/channel details | Identity/context before actions | `low / no` |
| `local_audio_output.speak` | `local_audio_output.speak(text, voice?, blocking?, volume?)` | `audio, output` | Local TTS output | Spoken output when requested | `low / no` |
| `stt_transcribe` | `stt_transcribe(asset_id, language_hint?)` | `audio, transcription` | Transcribe audio asset | Incoming audio transcript | `medium / no` |
| `telegramGetFileMeta` | `telegramGetFileMeta(file_id)` | `telegram, files` | Resolve Telegram file metadata | Before download/transcription | `medium / no` |
| `telegramDownloadFile` | `telegramDownloadFile(file_id, sandbox_alias?, relative_path?)` | `telegram, files` | Download Telegram file | Prepare file for downstream tools | `high / no` |
| `transcribeTelegramAudio` | `transcribeTelegramAudio(file_id, language?, sandbox_alias?)` | `telegram, audio, transcription` | Download+transcribe Telegram audio | Direct Telegram voice handling | `high / no` |
| `analyzeTelegramImage` | `analyzeTelegramImage(file_id?, prompt?, sandbox_alias?)` | `telegram, image, analysis` | Download+analyze Telegram image | Telegram inbound image semantics | `high / no` |
| `vision_analyze_image` | `vision_analyze_image(sandbox_alias, relative_path, prompt?)` | `vision, image, analysis` | Analyze sandbox image with dedicated vision model | Receipts, notes, package checks, object ID | `medium / no` |
| `scratchpad_create` | `scratchpad_create(title, scope?, tags?, template?)` | `productivity, caregiving, memory` | Create scratchpad doc | Start durable notes/log/plans | `medium / no` |
| `scratchpad_append` | `scratchpad_append(doc_id, text)` | `productivity, caregiving, memory` | Append timestamped block | Immutable progress updates | `medium / no` |
| `scratchpad_read` | `scratchpad_read(doc_id, mode?, max_chars?)` | `productivity, caregiving, memory` | Read bounded scratchpad content | Inspect working context safely | `low / no` |
| `scratchpad_list` | `scratchpad_list(scope?, tag?, limit?)` | `productivity, caregiving, memory` | List scratchpad docs | Find candidate docs | `low / no` |
| `scratchpad_search` | `scratchpad_search(query, scope?, tags_any?, limit?)` | `productivity, caregiving, memory` | Search scratchpad docs | Keyword retrieval of prior notes | `low / no` |
| `scratchpad_fork` | `scratchpad_fork(doc_id, new_title?)` | `productivity, caregiving, memory` | Fork scratchpad doc | Major rewrite preserving history | `medium / no` |
| `job_create` | `job_create(name, description, schedule, payload_type, payload, timezone?, domain_tags?, safety_level?, requires_confirmation?, retry_policy?, idempotency?, enabled?)` | `automation, jobs, productivity` | Create scheduled job | Recurring/background automation | `medium / no` |
| `job_list` | `job_list(enabled?, domain_tag?, limit?)` | `automation, jobs, productivity` | List jobs | Review configured automations | `low / no` |
| `job_pause` | `job_pause(job_id)` | `automation, jobs, control` | Pause job | Temporarily disable automation | `medium / no` |
| `job_resume` | `job_resume(job_id)` | `automation, jobs, control` | Resume job | Re-enable automation | `medium / no` |
| `job_run_now` | `job_run_now(job_id)` | `automation, jobs, control` | Trigger immediate run | Test/force scheduled job | `medium / no` |
| `job_delete` | `job_delete(job_id)` | `automation, jobs, control` | Delete job | Remove automation | `high / yes` |
| `python_subprocess` | `python_subprocess(command, timeout_seconds?)` | `ops, maintenance, system` | Execute subprocess command | Admin maintenance fallback | `critical / yes` |
| `terminal_execute` | `terminal_execute(command, cwd?, timeout_seconds?)` | `ops, terminal, automation` | Policy-constrained terminal execution | Controlled terminal-like ops | `high / yes` |
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
