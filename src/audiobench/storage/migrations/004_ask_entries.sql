CREATE TABLE IF NOT EXISTS ask_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    token_count INTEGER DEFAULT 0,
    question_expression_id INTEGER,
    answer_expression_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(log_id) REFERENCES ask_logs(id) ON DELETE CASCADE,
    FOREIGN KEY(question_expression_id) REFERENCES expressions(id) ON DELETE SET NULL,
    FOREIGN KEY(answer_expression_id) REFERENCES expressions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS ix_ask_entries_log_id ON ask_entries (log_id);
