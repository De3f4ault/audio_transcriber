-- Migration 030: Search Session Identity
-- Creates persistent storage for search sessions, per-query history, and
-- retrieved fragments — enabling segment-overlap detection, synthesis
-- carryforward, and chat import of research threads.
--
-- NOTE: Fragment text is DENORMALIZED by design. segment_id is not a stable
-- reference in this system: re-transcription replaces segment rows entirely
-- (e.g. Bill Hicks coarse chunks → fine-grained chunks). Storing fragment_text
-- ensures session records show exactly what the user saw, not what the current
-- segment says after a later re-index. This is a correctness decision, not a
-- storage optimization.

-- One row per REPL invocation (or standalone command invocation).
-- Always created on entry, always persisted — no explicit /save required.
CREATE TABLE IF NOT EXISTS search_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,                              -- auto-generated from first query (~60 chars)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    query_count INTEGER DEFAULT 0,
    preset TEXT DEFAULT 'balanced'           -- initial preset for the session
);

-- One row per search within a session (S key, or standalone invocation).
CREATE TABLE IF NOT EXISTS search_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES search_sessions(id) ON DELETE CASCADE,
    sequence_num INTEGER NOT NULL,           -- 1-indexed position within session
    query_text TEXT NOT NULL,
    preset TEXT NOT NULL,                    -- preset used for this specific query
    synthesis_text TEXT,                     -- LLM synthesis output (NULL if offline/failed)
    synthesis_failed INTEGER DEFAULT 0,      -- 1 if synthesis was attempted but failed
    synthesis_error TEXT,
    hyde_document TEXT,                      -- HyDE generation if used
    retrieval_time_seconds REAL,
    synthesis_time_seconds REAL,
    total_time_seconds REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, sequence_num)
);

-- One row per fragment retrieved per search query.
-- Enables segment-ID set intersection across searches (offline-safe overlap detection).
CREATE TABLE IF NOT EXISTS search_query_fragments (
    query_id INTEGER NOT NULL REFERENCES search_queries(id) ON DELETE CASCADE,
    segment_id INTEGER NOT NULL,             -- FK to segments.id (not enforced — segments are mutable)
    source_file TEXT,                        -- denormalized from FusedResult.source_file
    rank INTEGER NOT NULL,                   -- 1-indexed display position
    rrf_score REAL,
    stream_contributions TEXT,               -- JSON: [["colbert", 3], ["dense", 1]]
    start_time REAL,
    end_time REAL,
    fragment_text TEXT NOT NULL,             -- full text as displayed (denormalized — see above)
    PRIMARY KEY (query_id, segment_id)
);

CREATE INDEX IF NOT EXISTS idx_sqf_segment ON search_query_fragments(segment_id);
CREATE INDEX IF NOT EXISTS idx_sq_session ON search_queries(session_id);
CREATE INDEX IF NOT EXISTS idx_ss_created ON search_sessions(created_at);
