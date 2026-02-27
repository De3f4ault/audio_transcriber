# Architecture

Technical reference for the AudioBench codebase.

---

## Module Diagram

```
┌─────────────────────────────────────────────────────────┐
│                        CLI Layer                        │
│  cli/main.py ── commands, progress display, formatting  │
│  cli/theme.py ── colors, panels, Rich console setup     │
└──────────────────────────┬──────────────────────────────┘
                           │ on_phase(), on_segment()
┌──────────────────────────▼─────────────────────────────┐
│                     Pipeline Layer                     │
│  core/pipeline.py ── orchestrates the full workflow    │
│    load → convert → transcribe → filter → save         │
└────────┬──────────────────┬────────────────┬───────────┘
         │                  │                │
┌────────▼────────┐ ┌──────▼───────┐ ┌───────▼──────────┐
│  Audio Loader   │ │    Engine    │ │     Storage      │
│  core/ffmpeg.py │ │  engines/    │ │  storage/        │
│  FFmpeg probe,  │ │  whisper_    │ │  database.py     │
│  convert, filter│ │  engine.py   │ │  repository.py   │
└─────────────────┘ │  faster-     │ └──────────────────┘
                    │  whisper     │
                    │  batched +   │
                    │  sequential  │
                    └──────────────┘
```

---

## Data Flow

```
Input file (m4a/mp3/wav/...)
    │
    ▼
┌─ FFmpeg ────────────────────────────────────────────┐
│  1. Probe: extract codec, duration, sample rate     │
│  2. Convert: → 16kHz mono WAV (+ optional filters) │
└────────────────────────────────┬────────────────────┘
                                 │
    ▼
┌─ Cache Check ──────────────────────────────────────┐
│  SHA-256 hash of input file                         │
│  If match found in DB → return cached transcript    │
└────────────────────────────────┬───────────────────┘
                                 │ (cache miss)
    ▼
┌─ Whisper Engine ───────────────────────────────────┐
│  faster-whisper (CTranslate2 backend)              │
│  Batched mode: BatchedInferencePipeline            │
│  Sequential mode: WhisperModel.transcribe()        │
│                                                     │
│  For each segment:                                  │
│    → Skip low-confidence (avg_logprob < -1.5)      │
│    → collapse_repetitions() + fix_broken_words()   │
│    → progress_callback(pct)                         │
│    → on_segment(segment)                            │
└────────────────────────────────┬───────────────────┘
                                 │
    ▼
┌─ Output ───────────────────────────────────────────┐
│  Formatter: txt / srt / vtt / json                  │
│  → File or terminal                                 │
│  → SQLite database (always, for history/search)     │
└────────────────────────────────────────────────────┘
```

---

## Configuration Precedence

```
CLI flags (--language en, --fast, -m small)
    ↓ overrides
Environment variables (AUDIOBENCH_LANGUAGE=en)
    ↓ overrides
.env file (AUDIOBENCH_LANGUAGE=en)
    ↓ overrides
Default values (settings.py)
```

Managed by Pydantic Settings (`src/audiobench/config/settings.py`).

---

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `TranscriptionPipeline` | `core/pipeline.py` | Orchestrates load → transcribe → save |
| `WhisperEngine` | `engines/whisper_engine.py` | faster-whisper wrapper (batched + sequential) |
| `AudioBenchSettings` | `config/settings.py` | Pydantic settings with env var binding |
| `TranscriptionRepository` | `storage/repository.py` | SQLAlchemy CRUD for transcription records |
| `PhaseTracker` | `cli/main.py` | Rich Live progress display |
| `Transcript` / `Segment` / `Word` | `core/models.py` | Pydantic data models |

---

## Speed Presets (Internal)

Resolved in `settings.py`:

| Preset | `beam_size` | `batch_size` | `temperature` | `condition_on_previous_text` |
|--------|-------------|--------------|---------------|------------------------------|
| fast | 1 | 8 | `0` | `False` |
| balanced | 3 | 4 | `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` | `False` |
| accurate | 5 | 1 | `[0, 0.2, 0.4, 0.6, 0.8, 1.0]` | `True` |

---

## Database Schema

SQLite via SQLAlchemy. Single table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment ID |
| `file_hash` | String | SHA-256 of input file (for cache lookup) |
| `file_name` | String | Original filename |
| `language` | String | Detected/forced language |
| `duration_seconds` | Float | Audio duration |
| `segment_count` | Integer | Number of segments |
| `word_count` | Integer | Total word count |
| `transcript_json` | Text | Full transcript as JSON (segments + words) |
| `created_at` | DateTime | Timestamp |
