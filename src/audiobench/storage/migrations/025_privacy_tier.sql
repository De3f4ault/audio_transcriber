-- Migration 025: Privacy Tier System
--
-- Adds a privacy_tier column to both segments and expressions tables.
-- This is the foundational schema change for the security layer.
--
-- Tier 0 = Public      (default — audiobooks, podcasts, technical notes)
-- Tier 1 = Relational  (conversations with other enrolled speakers)
-- Tier 2 = Intimate    (owner voiceprint match or --sensitive flag)
-- Tier 3 = Override    (manually flagged at highest clearance)
--
-- The column defaults to 0 on all existing rows, so nothing breaks.
-- DDM (Dynamic Data Masking) is applied at the render layer, not here.
-- LanceDB vector search always queries the full corpus (no WHERE filter).

ALTER TABLE segments ADD COLUMN privacy_tier INTEGER NOT NULL DEFAULT 0;
ALTER TABLE expressions ADD COLUMN privacy_tier INTEGER NOT NULL DEFAULT 0;

-- Index for fast tier-based lookups (e.g. auditing, retroactive re-tagging)
CREATE INDEX IF NOT EXISTS idx_segments_privacy_tier ON segments(privacy_tier);
CREATE INDEX IF NOT EXISTS idx_expressions_privacy_tier ON expressions(privacy_tier);
