CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title VARCHAR(512) NOT NULL DEFAULT 'Untitled Note',
    body TEXT DEFAULT '',
    status VARCHAR(16) DEFAULT 'draft',
    expression_id INTEGER REFERENCES expressions(id) ON DELETE SET NULL,
    audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_notes_status ON notes (status);
CREATE INDEX IF NOT EXISTS ix_notes_audio_file_id ON notes (audio_file_id);
CREATE INDEX IF NOT EXISTS ix_notes_expression_id ON notes (expression_id);
