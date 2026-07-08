-- Add work_id to expressions
ALTER TABLE expressions ADD COLUMN work_id INTEGER REFERENCES works(id);
CREATE INDEX IF NOT EXISTS idx_expr_work_id ON expressions (work_id);
