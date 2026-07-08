CREATE INDEX IF NOT EXISTS idx_expr_type_ts ON expressions (source_type, created_at DESC);
