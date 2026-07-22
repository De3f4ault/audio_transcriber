import shutil
import sqlite3
from pathlib import Path

db_path = Path("/home/de3f4ault/Desktop/Projects/audiobench/data/transcriptions.db")
conn = sqlite3.connect(db_path)
conn.execute("UPDATE transcriptions SET is_indexed = 0")
conn.commit()
conn.close()

lancedb_path = Path("/home/de3f4ault/Desktop/Projects/audiobench/data/lancedb")
if lancedb_path.exists():
    shutil.rmtree(lancedb_path)
    print("LanceDB directory deleted.")

print("Successfully reset is_indexed to 0 and cleared LanceDB.")
