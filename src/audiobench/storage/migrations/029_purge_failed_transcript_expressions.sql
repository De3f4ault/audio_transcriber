-- Purge orphaned NULL-role expressions for transcripts that failed or are empty
-- These were test artifacts (e.g. from test_lifecycle_dummy.wav) that never
-- generated sweep_chunk nodes and continually get picked up by startup recovery.

DELETE FROM expressions
WHERE source_type = 'audio_transcript'
  AND graph_role IS NULL
  AND source_id IN (
      SELECT id FROM transcriptions
      WHERE status = 'failed'
         OR full_text IS NULL
         OR full_text = ''
  );
