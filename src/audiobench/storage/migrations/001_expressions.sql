CREATE TABLE IF NOT EXISTS expressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    source_type VARCHAR(64) NOT NULL,
    source_id INTEGER,
    session_type VARCHAR(64),
    session_id INTEGER,
    speaker VARCHAR(64),
    inference_confidence FLOAT,
    inference_status VARCHAR(32),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_expressions_source_type ON expressions (source_type);
CREATE INDEX IF NOT EXISTS ix_expressions_source_id ON expressions (source_id);
CREATE INDEX IF NOT EXISTS ix_expressions_session_id ON expressions (session_id);
CREATE INDEX IF NOT EXISTS ix_expressions_created_at ON expressions (created_at);
