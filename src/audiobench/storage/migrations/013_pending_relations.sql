CREATE TABLE IF NOT EXISTS pending_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_expression_id INTEGER NOT NULL REFERENCES expressions(id) ON DELETE CASCADE,
    to_expression_id_hint INTEGER NOT NULL,
    to_source_type VARCHAR(64),
    relation_type VARCHAR(64) NOT NULL DEFAULT 'explicit',
    raw_ref TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_pending_relations_to_id ON pending_relations (to_expression_id_hint);
