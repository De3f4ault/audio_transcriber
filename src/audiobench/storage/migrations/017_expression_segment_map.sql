-- Create the expression_segment_map bridge table
-- This links semantic expressions (often multiple sentences) to their exact raw transcript segments.

CREATE TABLE IF NOT EXISTS expression_segment_map (
    expression_id INTEGER NOT NULL REFERENCES expressions(id) ON DELETE CASCADE,
    segment_id INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
    PRIMARY KEY (expression_id, segment_id)
);

-- Indexes for fast traversal in both directions
CREATE INDEX IF NOT EXISTS idx_esmap_expression ON expression_segment_map(expression_id);
CREATE INDEX IF NOT EXISTS idx_esmap_segment ON expression_segment_map(segment_id);
