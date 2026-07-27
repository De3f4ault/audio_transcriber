import time
import threading
import resource
import sys
from pathlib import Path

from audiobench.daemon.client import DaemonClient
from audiobench.storage.repository import TranscriptionRepository
from audiobench.transcribe.transcription_result import Transcript, Segment
from audiobench.core.db_session import get_session
from audiobench.storage.models import TranscriptionRecord

def get_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def spam_daemon(stop_event, stats):
    client = DaemonClient()
    while not stop_event.is_set():
        try:
            client.ping()
            stats["queries"] += 1
        except Exception as e:
            stats["errors"] += 1

def monitor_memory(stop_event, mem_history):
    while not stop_event.is_set():
        mem_history.append(get_rss_mb())
        time.sleep(0.05)

def main():
    print("=== Testing Live Session Saving Under Daemon Load ===")
    client = DaemonClient()
    try:
        if not client.ping():
            print("Daemon not running, starting...")
            client._ensure_daemon_running()
            time.sleep(2)
    except Exception:
        print("Starting daemon...")
        client._ensure_daemon_running()
        time.sleep(2)

    stop_event = threading.Event()
    stats = {"queries": 0, "errors": 0}
    mem_history = []

    print("Launching 10 daemon saturation threads...")
    threads = []
    for _ in range(10):
        t = threading.Thread(target=spam_daemon, args=(stop_event, stats), daemon=True)
        t.start()
        threads.append(t)

    monitor_t = threading.Thread(target=monitor_memory, args=(stop_event, mem_history), daemon=True)
    monitor_t.start()
    threads.append(monitor_t)

    time.sleep(1.0)  # Allow saturation to stabilize
    baseline_rss = get_rss_mb()
    print(f"Baseline RSS before session: {baseline_rss:.2f} MB")

    # Simulate generating a large live transcript over a simulated session
    print("Simulating live session transcript generation...")
    segments = [
        Segment(id=i, start=i*2.0, end=(i+1)*2.0, text=f"This is live transcription sentence number {i} with some details.", speaker="SPEAKER_00")
        for i in range(100)
    ]
    transcript = Transcript(
        text=" ".join(s.text for s in segments),
        segments=segments,
        language="en",
        duration_seconds=200.0,
        model_name="base",
    )

    print("Executing upstream DB init and save_live_session under continuous daemon load...")
    start_time = time.perf_counter()
    from audiobench.core.db_engine import init_db
    init_db()
    repo = TranscriptionRepository()
    tx_id = repo.save_live_session(transcript)
    elapsed = time.perf_counter() - start_time
    after_save_rss = get_rss_mb()

    time.sleep(0.5)  # Let monitor capture post-save memory

    stop_event.set()
    for t in threads:
        t.join(timeout=0.5)

    max_rss = max(mem_history) if mem_history else after_save_rss
    delta_rss = max_rss - baseline_rss

    print("\n--- Test Results ---")
    print(f"Saved Live Session TX ID: #{tx_id}")
    print(f"Execution Time (init_db + save_live_session): {elapsed*1000:.2f} ms")
    print(f"Baseline RSS: {baseline_rss:.2f} MB")
    print(f"Peak RSS during session: {max_rss:.2f} MB")
    print(f"RSS Delta: +{delta_rss:.2f} MB")
    print(f"Daemon queries spammed during test: {stats['queries']} (errors: {stats['errors']})")

    # Verification checks
    with get_session() as session:
        record = session.query(TranscriptionRecord).get(tx_id)
        print(f"DB Record is_indexed: {record.is_indexed} (Expected: 0)")
        if record.is_indexed != 0:
            print(f"❌ FAILURE: Expected is_indexed=0, got {record.is_indexed}")
            sys.exit(1)

    if delta_rss > 200.0:
        print(f"❌ FAILURE: Memory increased by {delta_rss:.2f} MB (threshold 200 MB). Possible foreground model loading!")
        sys.exit(1)

    if elapsed > 1.0:
        print(f"❌ FAILURE: Save took {elapsed:.2f}s (threshold 1.0s). Possible synchronous daemon blocking!")
        sys.exit(1)

    print("\n✅ SUCCESS: save_live_session under daemon saturation completed rapidly with zero memory spike and is_indexed=0!")

if __name__ == "__main__":
    main()
