-- 021_unified_jobs.sql
-- Creates the `all_jobs` view that merges background CLI jobs (jobs table)
-- with daemon-queued jobs (job_queue table) into a single unified feed.
--
-- Column map:
--   id         — row id within its source table
--   source     — 'jobs' or 'job_queue' so consumers know which table to write back to
--   label      — human-readable display label (command or file_path)
--   engine     — transcription engine name (NULL for CLI jobs)
--   status     — shared status vocabulary: running/done/failed/pending/processing
--   started_at — when the row was created / submitted
--   ended_at   — completion timestamp (NULL for job_queue rows, which have no ended_at)

CREATE VIEW IF NOT EXISTS all_jobs AS

    SELECT
        id,
        'jobs'          AS source,
        COALESCE(audio_file, command) AS label,
        NULL            AS engine,
        status,
        started_at,
        ended_at
    FROM jobs

    UNION ALL

    SELECT
        id,
        'job_queue'     AS source,
        file_path       AS label,
        engine,
        status,
        created_at      AS started_at,
        NULL            AS ended_at
    FROM job_queue;
