-- Backfill chapter_id for segments
-- Assigns a chapter_id to a segment if the segment falls within the chapter's time bounds
-- and belongs to the same transcription.

UPDATE segments
SET chapter_id = (
    SELECT c.id
    FROM chapters c
    JOIN transcriptions t ON t.audio_file_id = c.audio_file_id
    WHERE t.id = segments.transcription_id
      AND segments.start_time >= c.start_time
      AND segments.start_time < c.end_time
    LIMIT 1
)
WHERE chapter_id IS NULL;
