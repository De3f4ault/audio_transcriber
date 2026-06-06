CREATE TABLE IF NOT EXISTS expression_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_expression_id INTEGER NOT NULL,
    to_expression_id INTEGER NOT NULL,
    relation_type VARCHAR(64) NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_by VARCHAR(64) DEFAULT 'system',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(from_expression_id) REFERENCES expressions(id) ON DELETE CASCADE,
    FOREIGN KEY(to_expression_id) REFERENCES expressions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_expression_relations_from_expression_id ON expression_relations (from_expression_id);
CREATE INDEX IF NOT EXISTS ix_expression_relations_to_expression_id ON expression_relations (to_expression_id);
CREATE INDEX IF NOT EXISTS ix_expression_relations_relation_type ON expression_relations (relation_type);
