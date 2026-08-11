-- Migration 031: Add session_summary column to search_sessions
-- Stores an AI-generated executive summary of the session (nullable, on-demand).
-- summary_generated_at tracks when it was last regenerated.

ALTER TABLE search_sessions ADD COLUMN session_summary TEXT;
ALTER TABLE search_sessions ADD COLUMN summary_generated_at TIMESTAMP;
