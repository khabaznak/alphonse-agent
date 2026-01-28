-- Minimal FSM seed data (safe to re-run).

INSERT OR IGNORE INTO states (key, name, description, is_terminal, is_enabled)
VALUES
  ('observing', 'Observing', 'Collecting signals', 0, 1),
  ('interpreting', 'Interpreting', 'Interpreting inputs', 0, 1),
  ('acting', 'Acting', 'Executing actions', 0, 1),
  ('shutdown', 'Shutdown', 'Terminal state', 1, 1);

INSERT OR IGNORE INTO senses (key, name, description, source_type, enabled, owner)
VALUES
  ('system', 'System', 'Core system impulses', 'system', 1, NULL),
  ('intent', 'Intent Detector', 'Interprets incoming intents', 'service', 1, NULL);

INSERT OR IGNORE INTO signals (key, name, source, description, is_enabled)
VALUES
  ('time_tick', 'Time Tick', 'system', 'Periodic tick signal', 1),
  ('intent_received', 'Intent Received', 'intent', 'New intent detected', 1),
  ('shutdown_requested', 'Shutdown Requested', 'system', 'Shutdown requested', 1);

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
  100,
  1,
  NULL,
  NULL,
  0,
  'baseline transition'
FROM states s1
JOIN signals sig ON sig.key = 'time_tick'
JOIN states s2 ON s2.key = 'observing'
WHERE s1.key = 'observing';

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
  50,
  1,
  NULL,
  'begin_interpretation',
  0,
  'intent received'
FROM states s1
JOIN signals sig ON sig.key = 'intent_received'
JOIN states s2 ON s2.key = 'interpreting'
WHERE s1.key = 'observing';

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
  50,
  1,
  NULL,
  'begin_acting',
  0,
  'interpretation complete'
FROM states s1
JOIN signals sig ON sig.key = 'time_tick'
JOIN states s2 ON s2.key = 'acting'
WHERE s1.key = 'interpreting';

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
JOIN states s2 ON s2.key = 'shutdown';
