CREATE TABLE IF NOT EXISTS job_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path VARCHAR(1024) NOT NULL,
    engine VARCHAR(64),
    model_name VARCHAR(64),
    speed_preset VARCHAR(64),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
