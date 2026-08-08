-- Migration 032: YouTube Integration

-- Add youtube_video_id to audio_files
ALTER TABLE audio_files ADD COLUMN youtube_video_id VARCHAR(11) NULL;
CREATE UNIQUE INDEX ix_audio_files_youtube_video_id ON audio_files (youtube_video_id);

-- Create youtube_channels table
CREATE TABLE youtube_channels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL UNIQUE,
    channel_id  TEXT NOT NULL,
    title       TEXT,
    resolved_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
