-- Migration 019: Add name to study_projects and session_number/conversation_id to study_sessions

ALTER TABLE study_projects ADD COLUMN name TEXT;

ALTER TABLE study_sessions ADD COLUMN session_number INTEGER NOT NULL DEFAULT 1;
ALTER TABLE study_sessions ADD COLUMN conversation_id INTEGER REFERENCES chat_conversations(id) ON DELETE SET NULL;
