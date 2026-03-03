# Architecture

Technical reference for the AudioBench codebase.

---

## Module Diagram

```
┌─────────────────────────────────────────────────────────┐
│                        CLI Layer                        │
│  cli/commands/           ── 29 registered commands        │
│  cli/repl.py             ── interactive shell + dot-cmds  │
│  cli/plugins.py          ── user plugin loader            │
│  cli/custom_group.py     ── fuzzy command matching        │
│  cli/helpers.py          ── PhaseTracker, file helpers    │
│  cli/theme.py            ── colors, panels, Rich console  │
└──────────────────────────┬──────────────────────────────┘
                           │ on_phase(), on_segment()
┌──────────────────────────▼─────────────────────────────┐
│                     Pipeline Layer                     │
│  core/pipeline.py ── orchestrates the full workflow    │
│    load → filter → convert → transcribe → save          │
└────────┬──────────────────┬────────────────┬───────────┘
         │                  │                │
┌────────▼────────┐ ┌──────▼───────┐ ┌───────▼──────────┐
│  Audio Loader   │ │    Engine    │ │     Storage      │
│  core/ffmpeg.py │ │  engines/    │ │  storage/        │
│  FFmpeg probe,  │ │  whisper_    │ │  database.py     │
│  convert, filter│ │  engine.py   │ │  repository.py   │
│  analyze, chain │ │  faster-     │ └──────────────────┐
│  builder        │ │  whisper     │
└─────────────────┘ │  batched +   │
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
| `PhaseTracker` | `cli/helpers.py` | Progress display (Live → streaming transition) |
| `ReplSession` | `cli/repl.py` | Interactive shell state, context tracking |
| `Transcript` / `Segment` / `Word` | `core/models.py` | Pydantic data models |
| `AudioBenchGroup` | `cli/custom_group.py` | Fuzzy command matching + suggestions |

---

## Audio Processing Pipeline

`build_filter_chain()` in `core/ffmpeg.py` encodes the optimal filter ordering:

```
highpass (200Hz) → [arnndn | afftdn] → silenceremove → loudnorm
```

Rules:

1. **highpass** always first when any cleaning is active
2. **arnndn** (neural) supersedes **afftdn** (spectral) — no double denoising
3. **silenceremove** after denoise, before normalization
4. **loudnorm** always last (EBU R128, I=-16 LUFS)

The RNNoise model (`bd.rnnn`, ~293 KB) is auto-downloaded on first `--denoise` use to `~/.audiobench/models/rnnoise/`.

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

---

## REPL Architecture

The interactive shell (`cli/repl.py`) provides:

- **Context tracking**: `set_context(id)` loads a transcript record and makes all dot-commands operate on it
- **ID injection**: Commands like `show`, `ask`, `summarize` auto-inject the context ID when none is given
- **Variable expansion**: `$last`, `$id` expand to the current context ID
- **Dot-commands**: `.stats`, `.segments`, `.find`, `.play`, `.edit`, `.next`, `.prev`
- **Fuzzy matching**: Typos like `.sarch` suggest `.search`
- **History persistence**: Command history saved to `~/.audiobench/repl_history`

---

## Plugin Architecture

Plugins are Python files in `~/.audiobench/plugins/` loaded via `cli/plugins.py`:

1. `discover_plugins()` — Scans directory for `.py` files (ignores `_` prefixed)
2. `load_plugin()` — Uses `importlib.util` to load each module
3. `register_plugins()` — Calls `register(cli)` if defined, or auto-registers Click commands

Plugins load **after** all built-in commands, so they can extend or override functionality.

---

## PhaseTracker Display Architecture

The `PhaseTracker` in `cli/helpers.py` uses a two-mode approach:

1. **Live mode** — During loading/converting, Rich Live shows animated spinners (uses `transient=True`)
2. **Streaming mode** — When the first transcript segment arrives, Live stops (frame vanishes), phases print statically at the top, and each segment prints below via `console.print()`. Text grows downward in real-time.
