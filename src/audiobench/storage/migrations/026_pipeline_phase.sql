-- Add failure_reason and attempt_count to transcriptions
ALTER TABLE transcriptions ADD COLUMN failure_reason VARCHAR(256);
ALTER TABLE transcriptions ADD COLUMN attempt_count INTEGER DEFAULT 0;
ALTER TABLE transcriptions ADD COLUMN updated_at DATETIME;

-- Create index for faster sweep queries
CREATE INDEX IF NOT EXISTS ix_transcriptions_pipeline_phase ON transcriptions(status);

-- Backfill existing completed transcriptions to 'complete' instead of 'completed'
UPDATE transcriptions SET status = 'complete' WHERE status = 'completed';
