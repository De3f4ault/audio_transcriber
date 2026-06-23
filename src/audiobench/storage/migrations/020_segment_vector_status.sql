-- Migration 020: Track whether each audio segment has been embedded into the
-- LanceDB segment_vectors table.
--
-- vector_indexed = 0 → not yet embedded (default for existing + new rows)
-- vector_indexed = 1 → embedded and up-to-date in segment_vectors
--
-- The daemon's background sweep and the 'audiobench db embed-segments' backfill
-- command both read this column to find work and set it to 1 on completion.
-- Any segment update that changes the text should reset this to 0 so the
-- vector is re-embedded with fresh content.

ALTER TABLE segments ADD COLUMN vector_indexed INTEGER NOT NULL DEFAULT 0;
