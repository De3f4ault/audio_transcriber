-- Migration: Works table and relations
-- Down-migration is not supported in sqlite for dropping columns easily without table rebuild, so this is up-only.

CREATE TABLE IF NOT EXISTS works (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_works_author ON works (author);

-- Add work_id to audio_files
ALTER TABLE audio_files ADD COLUMN work_id INTEGER REFERENCES works(id);
CREATE INDEX IF NOT EXISTS idx_af_work_id ON audio_files (work_id);

