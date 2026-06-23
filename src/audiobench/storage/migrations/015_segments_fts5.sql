-- Create the FTS5 virtual table using the external content pattern
-- We only index the 'text' column and use the porter stemmer.
CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='id',
    tokenize='porter'
);

-- Triggers to keep the FTS index synchronized with the 'segments' table
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments
BEGIN
    INSERT INTO segments_fts(rowid, text)
    VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments
BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments
BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text)
    VALUES (new.id, new.text);
END;

-- Rebuild the index completely to handle existing data cleanly.
-- The 'delete-all' command deletes all data from the FTS table,
-- and then the rebuild command rebuilds it from the content table.
INSERT INTO segments_fts(segments_fts) VALUES ('rebuild');
