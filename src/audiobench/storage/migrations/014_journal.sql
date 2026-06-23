CREATE TABLE IF NOT EXISTS journal_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    subsystem TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    trace_id TEXT,
    span_id TEXT,
    parent_span_id TEXT,
    message TEXT,
    metadata TEXT,
    duration_ms REAL,
    session_id TEXT,
    process TEXT,
    source TEXT DEFAULT 'live'
);

CREATE INDEX IF NOT EXISTS idx_se_ts ON system_events(ts);
CREATE INDEX IF NOT EXISTS idx_se_subsystem_ts ON system_events(subsystem, ts);
CREATE INDEX IF NOT EXISTS idx_se_entity_ts ON system_events(entity_type, entity_id, ts);
CREATE INDEX IF NOT EXISTS idx_se_trace ON system_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_se_level_ts ON system_events(level, ts);
CREATE INDEX IF NOT EXISTS idx_se_session_ts ON system_events(session_id, ts);

CREATE INDEX IF NOT EXISTS idx_se_elevated
    ON system_events(ts)
    WHERE level IN ('WARN', 'ERROR', 'CRITICAL');

CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    subsystem TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    labels TEXT
);

CREATE INDEX IF NOT EXISTS idx_sm_sub_met_ts ON system_metrics(subsystem, metric, ts);

CREATE TABLE IF NOT EXISTS managed_processes (
    name TEXT PRIMARY KEY,
    state TEXT,
    pid INTEGER,
    started_at TEXT,
    stopped_at TEXT,
    restart_count INTEGER DEFAULT 0,
    last_exit_code INTEGER,
    last_error TEXT,
    updated_at TEXT
);
