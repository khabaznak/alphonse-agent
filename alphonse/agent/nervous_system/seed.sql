-- Minimal FSM seed data (safe to re-run).

INSERT OR IGNORE INTO states (key, name, description, is_terminal, is_enabled)
VALUES
  ('idle', 'Idle', 'Ready and waiting', 0, 1),
  ('dnd', 'Do Not Disturb', 'Reduced notifications', 0, 1),
  ('error', 'Error', 'Error state', 0, 1),
  ('onboarding', 'Onboarding', 'Initial setup', 0, 1),
  ('shutting_down', 'Shutting Down', 'Preparing to stop', 1, 1);

INSERT OR IGNORE INTO senses (key, name, description, source_type, enabled, owner)
VALUES
  ('system', 'System', 'Core system impulses', 'system', 1, NULL),
  ('timer', 'Timer Sense', 'Emits timer signals', 'system', 1, NULL),
  ('telegram', 'Telegram Sense', 'Receives Telegram messages', 'service', 1, NULL),
  ('cli', 'CLI Sense', 'Receives CLI messages', 'system', 1, NULL),
  ('api', 'API Sense', 'Receives API messages', 'service', 1, NULL),
  ('terminal', 'Terminal Sense', 'Emits terminal command updates', 'system', 0, NULL),
  ('terminal_executor', 'Terminal Executor', 'Executes terminal commands asynchronously', 'system', 0, NULL);

INSERT OR IGNORE INTO signals (key, name, source, description, is_enabled)
VALUES
  ('shutdown_requested', 'Shutdown Requested', 'system', 'Shutdown requested', 1),
  ('action.succeeded', 'Action Succeeded', 'system', 'Action completed successfully', 1),
  ('action.failed', 'Action Failed', 'system', 'Action execution failed', 1),
  ('telegram.message_received', 'Telegram Message Received', 'telegram', 'Incoming Telegram message', 1),
  ('cli.message_received', 'CLI Message Received', 'cli', 'Incoming CLI message', 1),
  ('api.message_received', 'API Message Received', 'api', 'Incoming API message', 1),
  ('api.status_requested', 'API Status Requested', 'api', 'API status request', 1),
  ('api.timed_signals_requested', 'API Timed Signals Requested', 'api', 'API timed signals request', 1),
  ('timer.fired', 'Timer Fired', 'timer', 'Scheduled timer fired', 1),
  ('timed_signal.fired', 'Timed Signal Fired', 'timer', 'Scheduled timed signal fired', 1),
  ('terminal.command_updated', 'Terminal Command Updated', 'terminal', 'Terminal command updated', 1),
  ('terminal.command_executed', 'Terminal Command Executed', 'terminal_executor', 'Terminal command executed', 1),
  ('telegram.invite_requested', 'Telegram Invite Requested', 'telegram', 'Telegram chat invite awaiting approval', 1);

INSERT OR IGNORE INTO timed_signals (
  id, trigger_at, timezone, status, fired_at, signal_type, payload, target, origin, correlation_id
)
VALUES (
  'daily_report',
  datetime('now'),
  NULL,
  'pending',
  NULL,
  'timed_signal',
  '{"kind":"daily_report"}',
  NULL,
  'system',
  'daily_report'
);

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  10,
  1,
  NULL,
  'shutdown',
  1,
  'global shutdown'
FROM states s1
JOIN signals sig ON sig.key = 'shutdown_requested'
JOIN states s2 ON s2.key = 'shutting_down';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  30,
  1,
  NULL,
  'handle_incoming_message',
  1,
  'telegram message received'
FROM states s1
JOIN signals sig ON sig.key = 'telegram.message_received'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  30,
  1,
  NULL,
  'handle_telegram_invite',
  1,
  'telegram invite requested'
FROM states s1
JOIN signals sig ON sig.key = 'telegram.invite_requested'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  30,
  1,
  NULL,
  'handle_incoming_message',
  1,
  'cli message received'
FROM states s1
JOIN signals sig ON sig.key = 'cli.message_received'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  30,
  1,
  NULL,
  'handle_incoming_message',
  1,
  'api message received'
FROM states s1
JOIN signals sig ON sig.key = 'api.message_received'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  30,
  1,
  NULL,
  'handle_status',
  1,
  'api status request'
FROM states s1
JOIN signals sig ON sig.key = 'api.status_requested'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  30,
  1,
  NULL,
  'handle_timed_signals',
  1,
  'api timed signals request'
FROM states s1
JOIN signals sig ON sig.key = 'api.timed_signals_requested'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  20,
  1,
  NULL,
  'handle_timer_fired',
  1,
  'timer fired'
FROM states s1
JOIN signals sig ON sig.key = 'timer.fired'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  20,
  1,
  NULL,
  'handle_timer_fired',
  1,
  'timed signal fired'
FROM states s1
JOIN signals sig ON sig.key = 'timed_signal.fired'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  5,
  1,
  NULL,
  'handle_action_failure',
  1,
  'default action failure'
FROM states s1
JOIN signals sig ON sig.key = 'action.failed'
JOIN states s2 ON s2.key = 'error';

INSERT OR IGNORE INTO transitions (
  state_id,
  signal_id,
  next_state_id,
  priority,
  is_enabled,
  guard_key,
  action_key,
  match_any_state,
  notes
)
SELECT
  s1.id,
  sig.id,
  s2.id,
  5,
  1,
  NULL,
  NULL,
  1,
  'default action success'
FROM states s1
JOIN signals sig ON sig.key = 'action.succeeded'
JOIN states s2 ON s2.key = 'idle';

INSERT OR IGNORE INTO channels (channel_id, channel_type, person_id, address, is_enabled, priority)
VALUES
  ('api-default', 'api', NULL, 'api', 1, 0),
  ('cli-default', 'cli', NULL, 'cli', 1, 0);

INSERT OR IGNORE INTO plan_kinds (plan_kind, description, is_enabled)
VALUES
  ('greeting', 'Harmless greeting response', 1),
  ('unknown', 'Clarifying response for unclear intent', 1),
  ('create_reminder', 'Schedule a reminder', 1),
  ('send_message', 'Send a message', 1);

INSERT OR IGNORE INTO plan_kind_versions (plan_kind, plan_version, json_schema, example, is_deprecated)
VALUES
  (
    'greeting',
    1,
    '{"type":"object","required":["plan_kind","plan_version","plan_id","correlation_id","created_at","source","actor","intent_confidence","requires_confirmation","questions","intent_evidence","payload"],"properties":{"plan_kind":{"const":"greeting"},"plan_version":{"type":"integer","const":1},"plan_id":{"type":"string"},"correlation_id":{"type":"string"},"created_at":{"type":"string"},"source":{"type":"string"},"actor":{"type":"object","required":["channel"],"properties":{"person_id":{"type":["string","null"]},"channel":{"type":"object","required":["type","target"],"properties":{"type":{"type":"string"},"target":{"type":"string"}}}}},"intent_confidence":{"type":"number"},"requires_confirmation":{"type":"boolean"},"questions":{"type":"array","items":{"type":"string"}},"intent_evidence":{"type":"object","required":["reminder_verbs","time_hints","quoted_spans","score"],"properties":{"reminder_verbs":{"type":"array","items":{"type":"string"}},"time_hints":{"type":"array","items":{"type":"string"}},"quoted_spans":{"type":"array","items":{"type":"string"}},"score":{"type":"number"}}},"payload":{"type":"object","required":["language","text"],"properties":{"language":{"type":["string","null"]},"text":{"type":["string","null"]}}}}}',
    '{"plan_kind":"greeting","plan_version":1,"plan_id":"uuid","correlation_id":"uuid","created_at":"2026-01-31T08:00:00Z","source":"telegram","actor":{"person_id":null,"channel":{"type":"telegram","target":"123"}},"intent_confidence":0.2,"requires_confirmation":false,"questions":[],"intent_evidence":{"reminder_verbs":[],"time_hints":[],"quoted_spans":[],"score":0},"payload":{"language":"es","text":"Hola"}}',
    0
  ),
  (
    'unknown',
    1,
    '{"type":"object","required":["plan_kind","plan_version","plan_id","correlation_id","created_at","source","actor","intent_confidence","requires_confirmation","questions","intent_evidence","payload"],"properties":{"plan_kind":{"const":"unknown"},"plan_version":{"type":"integer","const":1},"plan_id":{"type":"string"},"correlation_id":{"type":"string"},"created_at":{"type":"string"},"source":{"type":"string"},"actor":{"type":"object","required":["channel"],"properties":{"person_id":{"type":["string","null"]},"channel":{"type":"object","required":["type","target"],"properties":{"type":{"type":"string"},"target":{"type":"string"}}}}},"intent_confidence":{"type":"number"},"requires_confirmation":{"type":"boolean"},"questions":{"type":"array","items":{"type":"string"}},"intent_evidence":{"type":"object","required":["reminder_verbs","time_hints","quoted_spans","score"],"properties":{"reminder_verbs":{"type":"array","items":{"type":"string"}},"time_hints":{"type":"array","items":{"type":"string"}},"quoted_spans":{"type":"array","items":{"type":"string"}},"score":{"type":"number"}}},"payload":{"type":"object","required":["user_text","reason"],"properties":{"user_text":{"type":"string"},"reason":{"type":"string"},"suggestions":{"type":"array","items":{"type":"string"}}}}}}',
    '{"plan_kind":"unknown","plan_version":1,"plan_id":"uuid","correlation_id":"uuid","created_at":"2026-01-31T08:00:00Z","source":"telegram","actor":{"person_id":null,"channel":{"type":"telegram","target":"123"}},"intent_confidence":0.1,"requires_confirmation":false,"questions":["¿Puedes aclarar?"],"intent_evidence":{"reminder_verbs":[],"time_hints":[],"quoted_spans":[],"score":0},"payload":{"user_text":"hola","reason":"no_intent"}}',
    0
  ),
  (
    'create_reminder',
    1,
    '{"type":"object","required":["plan_kind","plan_version","plan_id","correlation_id","created_at","source","actor","intent_confidence","requires_confirmation","questions","intent_evidence","payload"],"properties":{"plan_kind":{"const":"create_reminder"},"plan_version":{"type":"integer","const":1},"plan_id":{"type":"string"},"correlation_id":{"type":"string"},"created_at":{"type":"string"},"source":{"type":"string"},"actor":{"type":"object","required":["channel"],"properties":{"person_id":{"type":["string","null"]},"channel":{"type":"object","required":["type","target"],"properties":{"type":{"type":"string"},"target":{"type":"string"}}}}},"intent_confidence":{"type":"number"},"requires_confirmation":{"type":"boolean"},"questions":{"type":"array","items":{"type":"string"}},"intent_evidence":{"type":"object","required":["reminder_verbs","time_hints","quoted_spans","score"],"properties":{"reminder_verbs":{"type":"array","items":{"type":"string"}},"time_hints":{"type":"array","items":{"type":"string"}},"quoted_spans":{"type":"array","items":{"type":"string"}},"score":{"type":"number"}}},"payload":{"type":"object","required":["target","schedule","message"],"properties":{"target":{"type":"object","required":["person_ref"],"properties":{"person_ref":{"type":"object","required":["kind"],"properties":{"kind":{"type":"string"},"id":{"type":["string","null"]},"name":{"type":["string","null"]}}}}},"schedule":{"type":"object","required":["timezone"],"properties":{"timezone":{"type":"string"},"trigger_at":{"type":["string","null"]},"rrule":{"type":["string","null"]},"time_of_day":{"type":["string","null"]}}},"message":{"type":"object","required":["text"],"properties":{"language":{"type":["string","null"]},"text":{"type":"string"}}},"delivery":{"type":"object","properties":{"channel_type":{"type":["string","null"]},"priority":{"type":["string","null"]}}},"idempotency_key":{"type":["string","null"]}}}}}',
    '{"plan_kind":"create_reminder","plan_version":1,"plan_id":"uuid","correlation_id":"uuid","created_at":"2026-01-31T08:00:00Z","source":"telegram","actor":{"person_id":null,"channel":{"type":"telegram","target":"123"}},"intent_confidence":0.8,"requires_confirmation":false,"questions":[],"intent_evidence":{"reminder_verbs":["recuérdale"],"time_hints":["mañana"],"quoted_spans":["Recuérdale"],"score":0.8},"payload":{"target":{"person_ref":{"kind":"person_id","id":"adrian"}},"schedule":{"timezone":"UTC","trigger_at":"2026-02-01T08:00:00Z","rrule":null,"time_of_day":"morning"},"message":{"language":"es","text":"Tomar medicina"},"delivery":{"channel_type":"telegram","priority":"normal"},"idempotency_key":"telegram:123"}}',
    0
  ),
  (
    'send_message',
    1,
    '{"type":"object","required":["plan_kind","plan_version","plan_id","correlation_id","created_at","source","actor","intent_confidence","requires_confirmation","questions","intent_evidence","payload"],"properties":{"plan_kind":{"const":"send_message"},"plan_version":{"type":"integer","const":1},"plan_id":{"type":"string"},"correlation_id":{"type":"string"},"created_at":{"type":"string"},"source":{"type":"string"},"actor":{"type":"object","required":["channel"],"properties":{"person_id":{"type":["string","null"]},"channel":{"type":"object","required":["type","target"],"properties":{"type":{"type":"string"},"target":{"type":"string"}}}}},"intent_confidence":{"type":"number"},"requires_confirmation":{"type":"boolean"},"questions":{"type":"array","items":{"type":"string"}},"intent_evidence":{"type":"object","required":["reminder_verbs","time_hints","quoted_spans","score"],"properties":{"reminder_verbs":{"type":"array","items":{"type":"string"}},"time_hints":{"type":"array","items":{"type":"string"}},"quoted_spans":{"type":"array","items":{"type":"string"}},"score":{"type":"number"}}},"payload":{"type":"object","required":["target","message"],"properties":{"target":{"type":"object","required":["person_ref"],"properties":{"person_ref":{"type":"object","required":["kind"],"properties":{"kind":{"type":"string"},"id":{"type":["string","null"]},"name":{"type":["string","null"]}}}}},"message":{"type":"object","required":["text"],"properties":{"language":{"type":["string","null"]},"text":{"type":"string"}},"delivery":{"type":"object","properties":{"channel_type":{"type":["string","null"]},"priority":{"type":["string","null"]}}}}}}}',
    '{"plan_kind":"send_message","plan_version":1,"plan_id":"uuid","correlation_id":"uuid","created_at":"2026-01-31T08:00:00Z","source":"telegram","actor":{"person_id":null,"channel":{"type":"telegram","target":"123"}},"intent_confidence":0.6,"requires_confirmation":false,"questions":[],"intent_evidence":{"reminder_verbs":[],"time_hints":[],"quoted_spans":["envia"],"score":0.6},"payload":{"target":{"person_ref":{"kind":"person_id","id":"adrian"}},"message":{"language":"es","text":"hola"},"delivery":{"channel_type":"telegram","priority":"normal"}}}',
    0
  );

INSERT OR IGNORE INTO plan_executors (plan_kind, plan_version, executor_key, min_agent_version)
VALUES
  ('greeting', 1, 'actions.execute_greeting_v1', NULL),
  ('unknown', 1, 'actions.execute_unknown_v1', NULL),
  ('create_reminder', 1, 'actions.execute_create_reminder_v1', NULL),
  ('send_message', 1, 'actions.execute_send_message_v1', NULL);
