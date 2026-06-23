-- Migration 017: Study Projects Schema

CREATE TABLE IF NOT EXISTS study_projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS study_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    chapter_ids TEXT NOT NULL, -- JSON list of chapter IDs
    memoir_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    closed_at DATETIME,
    FOREIGN KEY(project_id) REFERENCES study_projects(id) ON DELETE CASCADE,
    FOREIGN KEY(memoir_id) REFERENCES expressions(id) ON DELETE SET NULL
);
