PRAGMA foreign_keys = ON;

-- Optional but recommended for a local operational DB:
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

----------------------------------------------------------------------
-- 1) STATES
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS states (
  id            INTEGER PRIMARY KEY,
  key           TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  description   TEXT,
  is_terminal   INTEGER NOT NULL DEFAULT 0,
  is_enabled    INTEGER NOT NULL DEFAULT 1,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

----------------------------------------------------------------------
-- 2) SIGNALS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
  id            INTEGER PRIMARY KEY,
  key           TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  source        TEXT NOT NULL DEFAULT 'system',
  description   TEXT,
  is_enabled    INTEGER NOT NULL DEFAULT 1,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

----------------------------------------------------------------------
-- 2.5) SENSES (ADMIN/OBSERVABILITY ONLY)
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS senses (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  key           TEXT NOT NULL UNIQUE,
  name          TEXT NOT NULL,
  description   TEXT,
  source_type   TEXT NOT NULL,
  enabled       INTEGER NOT NULL DEFAULT 1,
  owner         TEXT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
  CHECK (enabled IN (0,1))
) STRICT;

----------------------------------------------------------------------
-- 2.6) TIMED SIGNALS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS timed_signals (
  id              TEXT PRIMARY KEY,
  trigger_at      TEXT NOT NULL,
  next_trigger_at TEXT,
  rrule           TEXT,
  timezone        TEXT,
  status          TEXT NOT NULL DEFAULT 'pending',
  fired_at        TEXT,
  attempt_count   INTEGER NOT NULL DEFAULT 0,
  signal_type     TEXT NOT NULL,
  payload         TEXT,
  target          TEXT,
  origin          TEXT,
  correlation_id  TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
  CHECK (status IN ('pending', 'fired', 'cancelled', 'error'))
) STRICT;

----------------------------------------------------------------------
-- 2.6.1) PENDING PLANS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pending_plans (
  pending_id    TEXT PRIMARY KEY,
  person_id     TEXT,
  channel_type  TEXT NOT NULL,
  correlation_id TEXT,
  plan_json     TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending',
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  expires_at    TEXT,
  CHECK (status IN ('pending', 'confirmed', 'cancelled', 'expired'))
) STRICT;

----------------------------------------------------------------------
-- 2.7) FSM TRACE
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fsm_trace (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  ts              TEXT NOT NULL DEFAULT (datetime('now')),
  correlation_id  TEXT,
  state_before    TEXT,
  signal_type     TEXT,
  transition_id   INTEGER,
  action_key      TEXT,
  state_after     TEXT,
  result          TEXT NOT NULL,
  error_summary   TEXT
) STRICT;

----------------------------------------------------------------------
-- 2.7.1) PLAN REGISTRY
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS plan_kinds (
  plan_kind    TEXT PRIMARY KEY,
  description  TEXT,
  is_enabled   INTEGER NOT NULL DEFAULT 1,
  created_at   TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

CREATE TABLE IF NOT EXISTS plan_kind_versions (
  plan_kind     TEXT NOT NULL,
  plan_version  INTEGER NOT NULL,
  json_schema   TEXT NOT NULL,
  example       TEXT,
  is_deprecated INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (plan_kind, plan_version)
) STRICT;

CREATE TABLE IF NOT EXISTS plan_executors (
  plan_kind        TEXT NOT NULL,
  plan_version     INTEGER NOT NULL,
  executor_key     TEXT NOT NULL,
  min_agent_version TEXT,
  PRIMARY KEY (plan_kind, plan_version)
) STRICT;

CREATE TABLE IF NOT EXISTS plan_instances (
  plan_id             TEXT PRIMARY KEY,
  plan_kind           TEXT NOT NULL,
  plan_version        INTEGER NOT NULL,
  correlation_id      TEXT,
  status              TEXT NOT NULL,
  actor_person_id     TEXT,
  source_channel_type TEXT NOT NULL,
  source_channel_target TEXT,
  intent_confidence   REAL,
  payload             TEXT NOT NULL,
  intent_evidence     TEXT,
  original_text       TEXT,
  created_at          TEXT NOT NULL
) STRICT;

----------------------------------------------------------------------
-- 2.7.1) TELEGRAM UPDATES
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS telegram_updates (
  update_id   INTEGER PRIMARY KEY,
  chat_id     TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

----------------------------------------------------------------------
-- 2.8) IDENTITY REGISTRY
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS persons (
  person_id     TEXT PRIMARY KEY,
  display_name  TEXT NOT NULL,
  relationship  TEXT,
  timezone      TEXT,
  is_active     INTEGER NOT NULL DEFAULT 1
) STRICT;

CREATE TABLE IF NOT EXISTS groups (
  group_id  TEXT PRIMARY KEY,
  name      TEXT NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 1
) STRICT;

CREATE TABLE IF NOT EXISTS person_groups (
  person_id TEXT NOT NULL,
  group_id  TEXT NOT NULL,
  PRIMARY KEY (person_id, group_id)
) STRICT;

CREATE TABLE IF NOT EXISTS channels (
  channel_id   TEXT PRIMARY KEY,
  channel_type TEXT NOT NULL,
  person_id    TEXT,
  address      TEXT NOT NULL,
  is_enabled   INTEGER NOT NULL DEFAULT 1,
  priority     INTEGER NOT NULL DEFAULT 100
) STRICT;

CREATE TABLE IF NOT EXISTS communication_prefs (
  prefs_id            TEXT PRIMARY KEY,
  scope_type          TEXT NOT NULL,
  scope_id            TEXT NOT NULL,
  language_preference TEXT,
  tone                TEXT,
  formality           TEXT,
  emoji               TEXT,
  verbosity_cap       TEXT,
  quiet_hours_start   INTEGER,
  quiet_hours_end     INTEGER,
  allow_push          INTEGER NOT NULL DEFAULT 1,
  allow_telegram      INTEGER NOT NULL DEFAULT 1,
  allow_web           INTEGER NOT NULL DEFAULT 1,
  allow_cli           INTEGER NOT NULL DEFAULT 1,
  model_budget_policy TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS presence_state (
  person_id     TEXT PRIMARY KEY,
  in_meeting    INTEGER NOT NULL DEFAULT 0,
  location_hint TEXT,
  updated_at    TEXT
) STRICT;

----------------------------------------------------------------------
-- 3) TRANSITIONS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transitions (
  id              INTEGER PRIMARY KEY,
  state_id        INTEGER NOT NULL,
  signal_id       INTEGER NOT NULL,
  next_state_id   INTEGER NOT NULL,
  priority        INTEGER NOT NULL DEFAULT 100,
  is_enabled      INTEGER NOT NULL DEFAULT 1,
  guard_key       TEXT,
  action_key      TEXT,
  match_any_state INTEGER NOT NULL DEFAULT 0,
  notes           TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (state_id)      REFERENCES states(id)   ON DELETE CASCADE,
  FOREIGN KEY (signal_id)     REFERENCES signals(id)  ON DELETE CASCADE,
  FOREIGN KEY (next_state_id) REFERENCES states(id)   ON DELETE CASCADE,
  CHECK (priority >= 0),
  CHECK (is_enabled IN (0,1)),
  CHECK (match_any_state IN (0,1))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_transitions_state_signal_priority
  ON transitions(state_id, signal_id, priority);

CREATE INDEX IF NOT EXISTS ix_transitions_lookup
  ON transitions(state_id, signal_id, is_enabled, match_any_state, priority);

CREATE INDEX IF NOT EXISTS ix_transitions_signal
  ON transitions(signal_id);

CREATE INDEX IF NOT EXISTS ix_transitions_next_state
  ON transitions(next_state_id);

----------------------------------------------------------------------
-- 4) SIGNAL QUEUE
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signal_queue (
  id            INTEGER PRIMARY KEY,
  signal_id     TEXT NOT NULL,
  signal_type   TEXT NOT NULL,
  payload       TEXT NOT NULL,
  source        TEXT,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  processed_at  TEXT,
  error         TEXT,
  durable       INTEGER NOT NULL DEFAULT 1
) STRICT;

CREATE INDEX IF NOT EXISTS ix_signal_queue_type
  ON signal_queue(signal_type, created_at);

CREATE UNIQUE INDEX IF NOT EXISTS ux_signal_queue_signal_id
  ON signal_queue(signal_id);
