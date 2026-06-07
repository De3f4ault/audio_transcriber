import sqlite3
from pathlib import Path

db_path = Path("/home/de3f4ault/Desktop/Projects/audiobench/data/transcriptions.db")
conn = sqlite3.connect(db_path)

try:
    # Add column WITHOUT UNIQUE
    conn.execute("ALTER TABLE expressions ADD COLUMN content_hash VARCHAR(64);")
    conn.commit()
    print("Added content_hash column.")
except Exception as e:
    print("Add column error:", e)

# Hash existing
try:
    import hashlib

    cursor = conn.execute("SELECT id, content FROM expressions WHERE content_hash IS NULL")
    rows = cursor.fetchall()

    updates = []
    for row_id, content in rows:
        if content:
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            updates.append((h, row_id))

    if updates:
        for h, row_id in updates:
            # We don't have the unique index yet, so this will succeed for all
            conn.execute("UPDATE expressions SET content_hash = ? WHERE id = ?", (h, row_id))
        conn.commit()
        print(f"Hashed {len(updates)} existing expressions.")
except Exception as e:
    print("Hashing error:", e)

try:
    # Remove duplicates before creating unique index
    conn.execute("""
        DELETE FROM expressions 
        WHERE id NOT IN (
            SELECT MIN(id) 
            FROM expressions 
            GROUP BY content_hash
        ) AND content_hash IS NOT NULL
    """)
    conn.commit()
    print("Deleted duplicate expressions.")
except Exception as e:
    print("Delete duplicates error:", e)

try:
    # Add UNIQUE INDEX
    conn.execute("CREATE UNIQUE INDEX ix_expressions_content_hash ON expressions(content_hash);")
    conn.commit()
    print("Added unique index.")
except Exception as e:
    print("Index error:", e)

conn.close()
