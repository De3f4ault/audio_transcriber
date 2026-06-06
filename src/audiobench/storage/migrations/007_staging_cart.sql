CREATE TABLE IF NOT EXISTS staging_cart (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_file_id INTEGER NOT NULL UNIQUE,
    engine VARCHAR(64) DEFAULT 'gemini',
    model_name VARCHAR(64) DEFAULT 'medium',
    speed_preset VARCHAR(64) DEFAULT 'balanced',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(audio_file_id) REFERENCES audio_files(id) ON DELETE CASCADE
);
