import sqlite3
from pathlib import Path

kaggle_files = [
    "04 Experiences in a Concentration Camp_3.mp3",
    "11 Logotherapy in a Nutshell_3.mp3",
    "Cant Hurt Me - David Goggins.mp3",
    "Thus Spoke Zarathustra Penguin Classics.mp3",
    "03 Experiences in a Concentration Camp_2.mp3",
    "Beyond Good and Evil Penguin Classics.m4b",
    "Bill Gates - Source Code.m4b",
    "No Excuses.m4b",
    "12 Post Script to 1984 Edition - The Case for a Tragic Optimism.mp3",
    "09 Logotherapy in a Nutshell_1.mp3",
    "Cameron Hanes - Endure.m4b",
    "Poor Charlies Almanack - The Essential wit and wisdom of Charles T. Munger.m4b",
    "05 Experiences in a Concentration Camp_4.mp3",
    "Tao Te Ching The Essential Translation of the Ancient Chinese Book of the Tao.m4b",
    "08 Experiences in a Concentration Camp_7.mp3",
    "07 Experiences in a Concentration Camp_6.mp3",
    "Meditations Penguin Classics.m4b",
    "Cameron Hanes - Undeniable.m4b",
    "01 Preface by Gordon W. Allport Preface to 1984 Edition.mp3",
    "02 Experiences in a Concentration Camp_1.mp3",
    "06 Experiences in a Concentration Camp_5.mp3",
    "Fundamentals of Software Architecture (2nd Edition)_ A Modern Engineering Approach.m4b",
    "Never Split the Difference - Chris Voss.m4a",
    "10 Logotherapy in a Nutshell_2.mp3",
    "Napoleon Hill - Earl Nightingale Reads Think and Grow Rich.mp3",
    "The_Tao_of_Charlie_Munger.mp3",
    "The Prince Penguin Classics Niccolo Machiavelli.mp3",
    "The_Warren_Buffett_Way_(3rd_Edition).mp3"
]

db_path = "data/transcriptions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Checking transcription status for each Kaggle file:")
untranscribed = []
transcribed = []

# Get all audio files and their completed transcriptions
cursor.execute("SELECT id, file_name FROM audio_files")
db_audio_files = cursor.fetchall()

for kfile in kaggle_files:
    normalized_k = " ".join(kfile.split()).lower()
    
    # Find all matching audio file IDs in DB
    matching_fids = []
    for fid, db_name in db_audio_files:
        normalized_db = " ".join(db_name.split()).lower()
        if normalized_k in normalized_db or normalized_db in normalized_k:
            matching_fids.append(fid)
            
    # Check if any of these fids have a completed transcription
    has_completed_tx = False
    details = ""
    for fid in matching_fids:
        cursor.execute("SELECT id, status FROM transcriptions WHERE audio_file_id = ? AND status = 'completed'", (fid,))
        txs = cursor.fetchall()
        if txs:
            has_completed_tx = True
            details = f"DB Audio ID: {fid}, Trans ID: {txs[0][0]}"
            break
            
    if has_completed_tx:
        transcribed.append((kfile, details))
    else:
        untranscribed.append(kfile)

print("\n--- TRANSCRIBED FILES ---")
for f, info in transcribed:
    print(f"✓ {f} ({info})")

print("\n--- UNTRANSCRIBED FILES ---")
for f in untranscribed:
    print(f"✗ {f}")
