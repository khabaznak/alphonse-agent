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
