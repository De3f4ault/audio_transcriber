-- 027_graph_role.sql
-- Track 4: Add graph_role discriminator column to expressions, clean up historical
-- self-link and duplicate relation rows, then add unique constraint on relations.
--
-- RUN ORDER IS MANDATORY:
--   Step 1: Add graph_role column (schema extension)
--   Step 2: Clean up historical violations (must precede unique constraint)
--   Step 3: Add unique constraint (safe only once table is violation-free)
--
-- DEPLOYMENT SEQUENCE: stop daemon → run migration → verify → restart daemon.
-- The daemon must not hold an open connection during steps 2–3 to prevent a
-- concurrent sweep write from landing in the cleanup window before the constraint
-- is live.

-- Step 1: graph_role column on expressions.
-- NULL = pre-Track-4 row or non-sweep-origin expression (chat, memoir, bookmark, etc.)
-- Sweep-created rows carry: sweep_document (T1), sweep_passage (T2), sweep_chunk (T3).
-- Future consumers must treat NULL as "role unknown / not a tiered sweep node."
ALTER TABLE expressions ADD COLUMN graph_role VARCHAR(16);
CREATE INDEX ix_expressions_graph_role ON expressions(graph_role);

-- Step 2a: Delete self-link rows (from_id == to_id).
-- These cause live display corruption in _show_surrounding() (dot_commands.py):
-- sibling traversal queries relations pointing to the parent, and a self-link row
-- appears as a sibling of itself, producing doubled display output.
-- Confirmed count before migration: 5 rows, all expression #13416 accumulated
-- across 4 daemon restarts between 2026-07-27 and 2026-07-30.
DELETE FROM expression_relations
WHERE from_expression_id = to_expression_id;

-- Step 2b: Collapse duplicate (from_id, to_id, relation_type) triples to earliest row.
-- Confirmed: 490 duplicate groups, all structurally identical (weight=1.0,
-- created_by='system' across every group — zero distinct values). Earliest row
-- by id is authoritative; no data is discarded.
-- Confirmed count before migration: 1,739 duplicate rows removed, 533 rows remain.
DELETE FROM expression_relations
WHERE id NOT IN (
    SELECT MIN(id)
    FROM expression_relations
    GROUP BY from_expression_id, to_expression_id, relation_type
);

-- Step 3: Unique constraint on expression_relations.
-- Safe only after step 2 eliminates all existing violations.
-- Enforces idempotent link() behaviour at the DB level: concurrent duplicate
-- inserts silently no-op via on_conflict_do_nothing in the repository layer.
CREATE UNIQUE INDEX uq_expression_relation_edge
    ON expression_relations(from_expression_id, to_expression_id, relation_type);
