# 🎙️ Audio Transcriber

 **offline** audio transcriber powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

> Runs entirely on your machine — no API keys, no cloud, no data leaving your laptop.

**Python 3.10+** · **MIT License** · **CPU-optimized** (`int8` quantization)

---

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/de3f4ault/audio_transcriber.git
cd audio_transcriber

# 2. Install
make install        # creates venv + installs deps
source venv/bin/activate

# 3. Transcribe
transcriber transcribe meeting.m4a
```

The first run downloads the model (~1.5 GB) to `~/.transcriber/models/`. Subsequent runs start faster.

---

## Installation

### Using Make (recommended)

```bash
make install          # Base install
make dev              # Dev install (linting, tests, editable)
```

### Manual

```bash
python -m venv venv
source venv/bin/activate    # or: source venv/bin/activate.fish
pip install -e .
```

### System Dependency

**FFmpeg** is required for audio conversion:

```bash
# Arch
sudo pacman -S ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

## CLI Commands

### `transcribe` — Core Transcription

```bash
# Print transcript to terminal
transcriber transcribe meeting.m4a

# Save as SRT subtitle file
transcriber transcribe meeting.m4a -f srt

# Auto-detect format from output filename
transcriber transcribe meeting.m4a -o notes.srt

# Batch: transcribe all m4a files, save to ./out/
transcriber transcribe *.m4a -o ./out/

# Fast preset (less accurate, ~2x speed)
transcriber transcribe --fast lecture.mp3

# Accurate preset (slower, best quality)
transcriber transcribe --accurate interview.wav

# Pipe-friendly raw output
transcriber transcribe -q meeting.m4a | grep "keyword"

# Inspect file metadata without transcribing
transcriber transcribe --check recording.m4a
```

#### Options

| Flag | Short | Description |
|------|-------|-------------|
| `--format` | `-f` | Output format: `txt`, `srt`, `vtt`, `json` |
| `--output` | `-o` | Output path (file or directory) |
| `--language` | `-l` | Language code (e.g. `en`, `sw`). Default: auto-detect |
| `--model` | `-m` | Model: `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `--fast` | |  Fast preset: beam=1, batch=8 |
| `--balanced` | |  Balanced preset: beam=3, batch=4 (default) |
| `--accurate` | |  Accurate preset: beam=5, sequential |
| `--prompt` | | Guide model with context (e.g. `"Conversation in Swahili and English"`) |
| `--enhance` | | Apply noise reduction + normalization filters |
| `--filter` | | Custom ffmpeg audio filter graph |
| `--no-cache` | | Re-transcribe even if cached |
| `--no-timestamps` | | Disable word-level timestamps |
| `--quiet` | `-q` | Raw output only (for piping) |
| `--check` | | Show file metadata only, no transcription |

---

### `history` — View Past Transcriptions

```bash
transcriber history            # Show last 20 transcriptions
transcriber history --limit 50 # Show last 50
```

### `search` — Full-Text Search

```bash
transcriber search "keyword"
transcriber search "yoga" --limit 5
```

### `export` — Re-export to Another Format

```bash
transcriber export 3 -f vtt           # Export ID #3 as VTT
transcriber export 3 -f srt -o sub/   # Save to sub/ directory
```

### `delete` — Remove from History

```bash
transcriber delete 3          # Delete transcription #3
transcriber delete --all      # Delete all history
```

### `download` — Pre-download Models

```bash
transcriber download large-v3-turbo   # Download for offline use
transcriber download small            # Smaller, faster model
```

### `info` — System Info & Settings

```bash
transcriber info
```

Shows: Python version, device (CPU/CUDA), model, compute type, storage paths, database size, and all configuration values.

---

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| `txt` | `.txt` | Plain text, default |
| `srt` | `.srt` | Subtitles (most video players) |
| `vtt` | `.vtt` | Web subtitles (HTML5 `<track>`) |
| `json` | `.json` | Programmatic access, word-level data |

---

## Configuration

Settings are loaded in priority order:

1. **CLI flags** (highest priority)
2. **Environment variables** (prefixed with `TRANSCRIBER_`)
3. **`.env` file** in project root
4. **Defaults**

Copy `.env.example` to get started:

```bash
cp .env.example .env
```

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBER_MODEL_NAME` | `large-v3-turbo` | Whisper model size |
| `TRANSCRIBER_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `TRANSCRIBER_COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (CUDA), `float32` |
| `TRANSCRIBER_LANGUAGE` | *(empty)* | Auto-detect. Set to `en`, `sw`, `fr`, etc. |
| `TRANSCRIBER_SPEED_PRESET` | `balanced` | `fast`, `balanced`, `accurate` |
| `TRANSCRIBER_BATCH_SIZE` | `4` | Batch inference size (1–16) |
| `TRANSCRIBER_CPU_THREADS` | `0` | CPU threads (`0` = auto-detect) |
| `TRANSCRIBER_OUTPUT_FORMAT` | `txt` | Default output format |
| `TRANSCRIBER_LOG_LEVEL` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Speed Presets

| Preset | Beam Size | Batch Size | Temperature | Condition on Previous | Use Case |
|--------|-----------|------------|-------------|----------------------|----------|
| `--fast` | 1 | 8 | 0 (no fallback) | No | Quick drafts, long recordings |
| `--balanced` | 3 | 4 | Fallback chain | No | Daily use (default) |
| `--accurate` | 5 | 1 (sequential) | Fallback chain | Yes | Important recordings |

---

## Directory Layout

```
~/.transcriber/              ← App data directory
├── models/                  ← Downloaded Whisper models (~1.5 GB each)
│   └── models--Systran--faster-whisper-large-v3-turbo/
└── (future: cache, exports)

./                           ← Project root
├── transcriptions.db        ← SQLite database (history, search)
├── .env                     ← Your configuration (gitignored)
└── .env.example             ← Configuration template
```

---

## Project Structure

```
audio_transcriber/
├── cli/                     ← Command-line interface
│   ├── main.py              ← CLI commands (transcribe, history, search, etc.)
│   └── theme.py             ← Rich console styling, colors, panels
├── src/transcriber/
│   ├── config/
│   │   ├── settings.py      ← Pydantic settings (env vars, .env, defaults)
│   │   └── logging_config.py
│   ├── core/
│   │   ├── pipeline.py      ← Orchestrates: load → convert → transcribe → save
│   │   ├── ffmpeg.py        ← FFmpeg integration (conversion, probing, filters)
│   │   ├── filters.py       ← Text quality filters (repetition, broken words)
│   │   └── models.py        ← Pydantic data models (Segment, Transcript, Word)
│   ├── engines/
│   │   └── whisper_engine.py ← faster-whisper integration (batched + sequential)
│   ├── output/
│   │   ├── base.py          ← Formatter registry
│   │   ├── txt.py / srt.py / vtt.py / json_fmt.py
│   │   └── (output formatters)
│   └── storage/
│       ├── database.py      ← SQLAlchemy engine + session
│       └── repository.py    ← CRUD operations on transcriptions
├── Makefile                 ← Build targets (install, test, lint, transcribe)
├── pyproject.toml           ← Project metadata + dependencies
├── requirements.txt         ← Pinned dependencies
└── .env.example             ← Configuration template
```

---

## Make Targets

```bash
make help              # Show all targets
make install           # Install base dependencies
make dev               # Install with dev dependencies (editable)
make test              # Run test suite with coverage
make lint              # Run ruff + mypy
make format            # Auto-format with black + ruff
make clean             # Remove build artifacts
make transcribe FILE=audio.m4a           # Quick transcribe
make transcribe-srt FILE=audio.m4a       # Transcribe → SRT
make history           # View transcription history
make search Q="word"   # Search transcriptions
make info              # Show system info
make download MODEL=large-v3-turbo       # Download model
```

---

## Verbose / Debug Mode

```bash
transcriber -v transcribe meeting.m4a     # Verbose (INFO logs)
transcriber --debug transcribe meeting.m4a # Debug (all logs)
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
