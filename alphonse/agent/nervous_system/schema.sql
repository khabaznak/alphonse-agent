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
  attempts        INTEGER NOT NULL DEFAULT 0,
  last_error      TEXT,
  signal_type     TEXT NOT NULL,
  payload         TEXT,
  target          TEXT,
  origin          TEXT,
  correlation_id  TEXT,
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
  CHECK (status IN ('pending', 'processing', 'fired', 'failed', 'cancelled', 'error', 'skipped', 'dispatched'))
) STRICT;

----------------------------------------------------------------------
-- 2.6.1) PRINCIPALS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS principals (
  principal_id   TEXT PRIMARY KEY,
  principal_type TEXT NOT NULL,
  channel_type   TEXT,
  channel_id     TEXT,
  display_name   TEXT,
  created_at     TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
  CHECK (principal_type IN ('person', 'channel_chat', 'household', 'office', 'system'))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_principals_channel_unique
  ON principals (channel_type, channel_id)
  WHERE channel_type IS NOT NULL AND channel_id IS NOT NULL;

----------------------------------------------------------------------
-- 2.6.2) PREFERENCES
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS preferences (
  preference_id TEXT PRIMARY KEY,
  principal_id  TEXT NOT NULL REFERENCES principals(principal_id) ON DELETE CASCADE,
  key           TEXT NOT NULL,
  value_json    TEXT NOT NULL,
  source        TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_principal_key
  ON preferences (principal_id, key);

----------------------------------------------------------------------
-- 2.6.2.1) PROMPT TEMPLATES
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_templates (
  id            TEXT PRIMARY KEY,
  key           TEXT NOT NULL,
  locale        TEXT NOT NULL,
  address_style TEXT NOT NULL,
  tone          TEXT NOT NULL,
  channel       TEXT NOT NULL,
  variant       TEXT NOT NULL,
  policy_tier   TEXT NOT NULL,
  template      TEXT NOT NULL,
  enabled       INTEGER NOT NULL DEFAULT 1,
  priority      INTEGER NOT NULL DEFAULT 0,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_prompt_templates_key_enabled
  ON prompt_templates (key, enabled);

CREATE INDEX IF NOT EXISTS idx_prompt_templates_match
  ON prompt_templates (key, locale, address_style, tone, channel, variant, policy_tier);

CREATE UNIQUE INDEX IF NOT EXISTS ux_prompt_templates_selectors
  ON prompt_templates (key, locale, address_style, tone, channel, variant, policy_tier);

----------------------------------------------------------------------
-- 2.6.2.2) PROMPT VERSIONS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prompt_versions (
  id           TEXT PRIMARY KEY,
  template_id  TEXT NOT NULL,
  version      INTEGER NOT NULL,
  template     TEXT NOT NULL,
  changed_by   TEXT NOT NULL,
  change_reason TEXT,
  created_at   TEXT NOT NULL,
  FOREIGN KEY (template_id) REFERENCES prompt_templates(id) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_prompt_versions_template
  ON prompt_versions (template_id, version);

----------------------------------------------------------------------
-- 2.6.2.3) INTENT SPECS (CATALOG)
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intent_specs (
  intent_name         TEXT PRIMARY KEY,
  category            TEXT NOT NULL,
  description         TEXT NOT NULL,
  examples_json       TEXT NOT NULL,
  required_slots_json TEXT NOT NULL,
  optional_slots_json TEXT NOT NULL,
  default_mode        TEXT NOT NULL,
  risk_level          TEXT NOT NULL,
  handler             TEXT NOT NULL,
  enabled             INTEGER NOT NULL DEFAULT 1,
  intent_version      TEXT NOT NULL,
  origin              TEXT NOT NULL,
  parent_intent       TEXT,
  created_at          TEXT NOT NULL,
  updated_at          TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_intent_specs_enabled
  ON intent_specs (enabled, intent_name);

----------------------------------------------------------------------
-- 2.6.2.4) ABILITY SPECS (RUNTIME)
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ability_specs (
  intent_name    TEXT PRIMARY KEY,
  kind           TEXT NOT NULL,
  tools_json     TEXT NOT NULL,
  spec_json      TEXT NOT NULL,
  enabled        INTEGER NOT NULL DEFAULT 1,
  source         TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ability_specs_enabled
  ON ability_specs (enabled, intent_name);

----------------------------------------------------------------------
-- 2.6.2.1) CAPABILITY GAPS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capability_gaps (
  gap_id           TEXT PRIMARY KEY,
  created_at       TEXT NOT NULL,
  principal_type   TEXT,
  principal_id     TEXT,
  channel_type     TEXT,
  channel_id       TEXT,
  correlation_id   TEXT,
  user_text        TEXT,
  intent           TEXT,
  confidence       REAL,
  missing_slots    TEXT,
  reason           TEXT NOT NULL,
  status           TEXT NOT NULL,
  resolution_notes TEXT,
  metadata         TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_capability_gaps_status
  ON capability_gaps (status);
CREATE INDEX IF NOT EXISTS idx_capability_gaps_created
  ON capability_gaps (created_at);

----------------------------------------------------------------------
-- 2.6.2.2) GAP PROPOSALS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gap_proposals (
  id                          TEXT PRIMARY KEY,
  gap_id                      TEXT NOT NULL,
  created_at                  TEXT NOT NULL,
  status                      TEXT NOT NULL,
  proposed_category           TEXT NOT NULL,
  confidence                  REAL,
  proposed_next_action        TEXT NOT NULL,
  proposed_intent_name        TEXT,
  proposed_clarifying_question TEXT,
  notes                       TEXT,
  language                    TEXT,
  reviewer                    TEXT,
  reviewed_at                 TEXT,
  CHECK (status IN ('pending', 'approved', 'rejected', 'dispatched'))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_gap_proposals_status
  ON gap_proposals (status);
CREATE INDEX IF NOT EXISTS idx_gap_proposals_gap
  ON gap_proposals (gap_id);

----------------------------------------------------------------------
-- 2.6.2.3) GAP TASKS (DISPATCH QUEUE)
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS gap_tasks (
  id          TEXT PRIMARY KEY,
  proposal_id TEXT NOT NULL,
  type        TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'open',
  created_at  TEXT NOT NULL,
  payload     TEXT,
  CHECK (status IN ('open', 'done')),
  CHECK (type IN ('plan', 'investigate', 'fix_now'))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_gap_tasks_status
  ON gap_tasks (status);
----------------------------------------------------------------------
-- 2.6.3) PENDING PLANS
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
-- 2.8) CORTEX SESSIONS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cortex_sessions (
  chat_id     TEXT PRIMARY KEY,
  state_json  TEXT NOT NULL,
  updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
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

----------------------------------------------------------------------
-- 5) LAN PAIRING (ALPHONSE LINK)
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pairing_codes (
  code        TEXT PRIMARY KEY,
  expires_at  TEXT NOT NULL,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
) STRICT;

CREATE TABLE IF NOT EXISTS paired_devices (
  device_id       TEXT PRIMARY KEY,
  device_name     TEXT,
  paired_at       TEXT NOT NULL,
  allowed_scopes  TEXT NOT NULL,
  armed           INTEGER NOT NULL DEFAULT 0,
  armed_at        TEXT,
  armed_by        TEXT,
  armed_until     TEXT,
  token_hash      TEXT,
  token_expires_at TEXT,
  last_seen_at    TEXT,
  last_status     TEXT,
  last_status_at  TEXT
) STRICT;

----------------------------------------------------------------------
-- 6) PAIRING REQUESTS + AUDIT
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pairing_requests (
  pairing_id   TEXT PRIMARY KEY,
  device_name  TEXT,
  challenge    TEXT,
  otp_hash     TEXT,
  status       TEXT NOT NULL,
  expires_at   TEXT NOT NULL,
  approved_via TEXT,
  approved_at  TEXT,
  created_at   TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS delivery_receipts (
  receipt_id   TEXT PRIMARY KEY,
  run_id       TEXT,
  pairing_id   TEXT,
  stage_id     TEXT,
  action_id    TEXT,
  skill        TEXT,
  channel      TEXT,
  status       TEXT NOT NULL,
  details_json TEXT,
  created_at   TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS audit_log (
  id            TEXT PRIMARY KEY,
  event_type    TEXT NOT NULL,
  correlation_id TEXT,
  payload_json  TEXT,
  created_at    TEXT NOT NULL
) STRICT;

----------------------------------------------------------------------
-- 7) HABITS + PLAN RUNS
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS habits (
  habit_id           TEXT PRIMARY KEY,
  name               TEXT NOT NULL,
  trigger            TEXT NOT NULL,
  conditions_json    TEXT NOT NULL,
  plan_json          TEXT NOT NULL,
  version            INTEGER NOT NULL,
  enabled            INTEGER NOT NULL DEFAULT 1,
  created_at         TEXT NOT NULL,
  updated_at         TEXT NOT NULL,
  success_count      INTEGER NOT NULL DEFAULT 0,
  fail_count         INTEGER NOT NULL DEFAULT 0,
  last_success_at    TEXT,
  last_fail_at       TEXT,
  menu_snapshot_hash TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS plan_runs (
  run_id         TEXT PRIMARY KEY,
  habit_id       TEXT,
  plan_id        TEXT NOT NULL,
  trigger        TEXT NOT NULL,
  correlation_id TEXT NOT NULL,
  status         TEXT NOT NULL,
  resolution     TEXT,
  resolved_via   TEXT,
  started_at     TEXT NOT NULL,
  ended_at       TEXT,
  state_json     TEXT,
  scheduled_json TEXT,
  plan_json      TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_habits_trigger_enabled ON habits (trigger, enabled);
CREATE INDEX IF NOT EXISTS idx_plan_runs_correlation ON plan_runs (correlation_id);

----------------------------------------------------------------------
-- 8) INTENT LIFECYCLE
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intent_lifecycle (
  signature_key     TEXT PRIMARY KEY,
  intent_name       TEXT NOT NULL,
  category          TEXT NOT NULL,
  state             TEXT NOT NULL,
  first_seen_at     TEXT NOT NULL,
  last_seen_at      TEXT NOT NULL,
  usage_count       INTEGER NOT NULL DEFAULT 0,
  success_count     INTEGER NOT NULL DEFAULT 0,
  correction_count  INTEGER NOT NULL DEFAULT 0,
  last_mode_used    TEXT,
  last_outcome      TEXT,
  trust_score       REAL,
  opt_in_automated  INTEGER NOT NULL DEFAULT 0
) STRICT;

----------------------------------------------------------------------
-- 9) HABIT LIFECYCLE
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS habit_lifecycle (
  habit_id               TEXT PRIMARY KEY,
  intent_signature_key   TEXT NOT NULL,
  trigger_type           TEXT NOT NULL,
  trigger_definition     TEXT NOT NULL,
  target                 TEXT NOT NULL,
  lifecycle_state        TEXT NOT NULL,
  autonomy_level_override REAL,
  created_at             TEXT NOT NULL,
  last_executed_at        TEXT,
  execution_count        INTEGER NOT NULL DEFAULT 0,
  success_count          INTEGER NOT NULL DEFAULT 0,
  failure_count          INTEGER NOT NULL DEFAULT 0,
  paused                 INTEGER NOT NULL DEFAULT 0,
  user_opt_in            INTEGER NOT NULL DEFAULT 0,
  audit_required         INTEGER NOT NULL DEFAULT 1
) STRICT;
