CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL UNIQUE,
    narrative TEXT NOT NULL,
    drift_phases TEXT DEFAULT '[]',
    key_insights TEXT DEFAULT '[]',
    open_threads TEXT DEFAULT '[]',
    refined_title VARCHAR(256),
    expression_id INTEGER,
    generated_by VARCHAR(128) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE,
    FOREIGN KEY(expression_id) REFERENCES expressions(id) ON DELETE SET NULL
);
