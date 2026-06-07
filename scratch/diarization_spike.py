import time
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from faster_whisper import WhisperModel

from audiobench.core.settings import get_settings
from audiobench.diarization.engine import PyannoteDiarizer
from audiobench.diarization.verification import SpeakerVerificationEngine

def test_pyannote_pipeline(audio_path: Path):
    print("\n" + "="*50)
    print(" PIPELINE 1: Pyannote Diarization (Current)")
    print("="*50)
    
    settings = get_settings()
    diarizer = PyannoteDiarizer(hf_token=settings.hf_token)
    
    start_time = time.time()
    print("Running Pyannote (this might take a while on CPU)...")
    turns = diarizer.get_speaker_turns(audio_path)
    elapsed = time.time() - start_time
    
    speakers = {t.speaker for t in turns}
    print(f"\n[Pyannote] Finished in {elapsed:.2f} seconds")
    print(f"[Pyannote] Detected {len(speakers)} speakers: {speakers}")
    
    # Print first 5 turns to see boundaries
    print("\nFirst 5 turns:")
    for t in sorted(turns, key=lambda x: x.start)[:5]:
        print(f"[{t.start:.2f} -> {t.end:.2f}] {t.speaker}")
        
    return elapsed, speakers

def test_whisper_ecapa_ahc_pipeline(audio_path: Path):
    print("\n" + "="*50)
    print(" PIPELINE 2: Whisper + ECAPA-TDNN + AHC (Proposed)")
    print("="*50)
    
    start_time = time.time()
    
    # Step 1: VAD via Whisper segments
    print("1. Running Whisper for VAD & Transcription...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if device == "cuda" else "int8" 
    
    model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    # Just need segments, no need to print word-by-word
    segments_gen, _ = model.transcribe(str(audio_path), vad_filter=True)
    segments = list(segments_gen)
    
    print(f"   -> Whisper returned {len(segments)} segments")
    
    # Step 2: ECAPA-TDNN feature extraction
    print("\n2. Extracting SpeechBrain ECAPA-TDNN embeddings...")
    verification_engine = SpeakerVerificationEngine()
    
    vectors = []
    valid_segments = []
    
    for seg in segments:
        duration = seg.end - seg.start
        if duration < 1.0:
            # Skip extremely short segments for speaker ID
            continue
            
        try:
            vec = verification_engine.extract_voice_print(audio_path, seg.start, seg.end)
            vectors.append(vec)
            valid_segments.append(seg)
        except Exception as e:
            print(f"Failed to extract {seg.start:.2f}-{seg.end:.2f}: {e}")
            
    print(f"   -> Extracted {len(vectors)} valid embeddings")
    
    # Step 3: Clustering
    print("\n3. Running AHC Clustering on embeddings...")
    if not vectors:
        print("No valid vectors to cluster.")
        return time.time() - start_time, set()
        
    X = np.array(vectors)
    
    # We use cosine distance. A threshold of 0.3 means similarity must be > 0.7 to merge
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=0.65
    )
    
    labels = clustering.fit_predict(X)
    
    elapsed = time.time() - start_time
    
    unique_labels = set(labels)
    speakers = {f"SPEAKER_{l:02d}" for l in unique_labels}
    
    print(f"\n[Whisper+ECAPA] Finished in {elapsed:.2f} seconds")
    print(f"[Whisper+ECAPA] Detected {len(speakers)} speakers: {speakers}")
    
    # Print first 5 segments
    print("\nFirst 5 segments:")
    for i, seg in enumerate(valid_segments[:5]):
        spk = f"SPEAKER_{labels[i]:02d}"
        print(f"[{seg.start:.2f} -> {seg.end:.2f}] {spk} | text: {seg.text.strip()}")
        
    return elapsed, speakers

def run_spike():
    target_file = Path("/home/de3f4ault/Downloads/How_to_be_a_great_programmer_Carmack.wav")
    
    if not target_file.exists():
        print(f"ERROR: Audio file not found at {target_file}")
        return
        
    print(f"Starting Diarization Spike Test on: {target_file.name}")
    print(f"File size: {target_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Run Proposed
    time_proposed, spk_proposed = test_whisper_ecapa_ahc_pipeline(target_file)
    
    # Run Pyannote
    # time_pyannote, spk_pyannote = test_pyannote_pipeline(target_file)
    time_pyannote = 1373.96
    spk_pyannote = {'SPEAKER_00', 'SPEAKER_01'}
    
    # Summary
    print("\n" + "="*50)
    print(" SUMMARY COMPARISON")
    print("="*50)
    print(f"{'Metric':<20} | {'Pyannote':<20} | {'Whisper+ECAPA+AHC':<20}")
    print("-" * 65)
    print(f"{'Time Taken (sec)':<20} | {time_pyannote:<20.2f} | {time_proposed:<20.2f}")
    print(f"{'Speakers Detected':<20} | {len(spk_pyannote):<20} | {len(spk_proposed):<20}")
    
    speedup = time_pyannote / time_proposed if time_proposed > 0 else 0
    print(f"\nProposed pipeline is {speedup:.1f}x faster on this file.")

if __name__ == "__main__":
    run_spike()
