-- Migration 028: Purge orphan NULL-role expressions
--
-- Context (2026-07-31):
-- Track 4 (Migration 027) introduced the tiered graph topology:
--   T1 sweep_document → T2 sweep_passage → T3 sweep_chunk (LanceDB only)
--
-- Before Track 4, all expressions were written with graph_role = NULL and
-- were indexed flat into LanceDB under the old single-tier path.
--
-- After Track 4, transcripts are re-swept through the tiered path, producing
-- sweep_document / sweep_passage / sweep_chunk nodes.  The old NULL-role rows
-- for those same transcripts are now dead residue — they were never written to
-- LanceDB (the old flat sweep was replaced), and they never will be, because
-- the tiered path is now authoritative.
--
-- These orphan rows cause UnindexedExpressionRecovery to fire on EVERY boot:
-- it sees NULL-role expressions whose IDs are absent from LanceDB, flags their
-- transcripts as unindexed, resets is_indexed=0, triggers a re-sweep that
-- produces sweep_chunk rows… and on the next restart the NULL rows are still
-- there, so the cycle repeats.
--
-- Fix: delete NULL-role expressions that belong to transcripts which already
-- have sweep_chunk children.  These transcripts have been fully migrated to
-- the tiered path; their NULL rows serve no purpose and cause the boot loop.
--
-- NULL-role rows belonging to transcripts that do NOT have sweep_chunk nodes
-- are genuine pre-Track-4 legacy rows that were successfully written to LanceDB
-- under the old flat path.  Those are left untouched.

DELETE FROM expression_relations
WHERE from_expression_id IN (
    SELECT e.id
    FROM expressions e
    WHERE e.graph_role IS NULL
      AND e.source_type = 'audio_transcript'
      AND e.source_id IN (
          SELECT DISTINCT source_id
          FROM expressions
          WHERE graph_role = 'sweep_chunk'
            AND source_type = 'audio_transcript'
      )
)
OR to_expression_id IN (
    SELECT e.id
    FROM expressions e
    WHERE e.graph_role IS NULL
      AND e.source_type = 'audio_transcript'
      AND e.source_id IN (
          SELECT DISTINCT source_id
          FROM expressions
          WHERE graph_role = 'sweep_chunk'
            AND source_type = 'audio_transcript'
      )
);

DELETE FROM expressions
WHERE graph_role IS NULL
  AND source_type = 'audio_transcript'
  AND source_id IN (
      SELECT DISTINCT source_id
      FROM expressions
      WHERE graph_role = 'sweep_chunk'
        AND source_type = 'audio_transcript'
  );
