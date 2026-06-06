-- SQLite does not support DROP CONSTRAINT, so we recreate the table.
-- This migration removes the UNIQUE constraint from content_hash globally.
-- Application-level dedup by (source_type, content_hash) is used instead.

CREATE TABLE IF NOT EXISTS expressions_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    content_hash VARCHAR(64),
    source_type VARCHAR(64) NOT NULL,
    source_id INTEGER,
    session_type VARCHAR(64),
    session_id INTEGER,
    speaker VARCHAR(64),
    inference_confidence FLOAT,
    inference_status VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO expressions_new (
    id, content, content_hash, source_type, source_id, 
    session_type, session_id, speaker, inference_confidence, 
    inference_status, created_at
)
SELECT 
    id, content, content_hash, source_type, source_id, 
    session_type, session_id, speaker, inference_confidence, 
    inference_status, created_at
FROM expressions;

DROP TABLE expressions;
ALTER TABLE expressions_new RENAME TO expressions;

CREATE INDEX ix_expressions_content_hash ON expressions (content_hash);
CREATE INDEX ix_expressions_source_type ON expressions (source_type);
CREATE INDEX ix_expressions_source_id ON expressions (source_id);
CREATE INDEX ix_expressions_session_id ON expressions (session_id);
CREATE INDEX ix_expressions_created_at ON expressions (created_at);
CREATE INDEX ix_expressions_type_hash ON expressions (source_type, content_hash);
