# Usage Guide

Real-world workflows for the AudioBench CLI.

---

## 1. Basic Transcription

### Print to terminal

```bash
audiobench transcribe meeting.m4a
```

The transcript prints directly to your terminal. Nothing is saved to a file, but it **is** cached in the database for later retrieval via `history` or `search`.

### Save to file

```bash
# Auto-detect format from filename
audiobench transcribe meeting.m4a -o meeting.srt

# Explicitly set format (saves as meeting.srt next to the input file)
audiobench transcribe meeting.m4a -f srt

# Save to a specific directory
audiobench transcribe meeting.m4a -o ~/Documents/transcriptions/
```

### Output format priority

| Flags | Result |
|-------|--------|
| `-o meeting.srt` | Format from extension → SRT |
| `-f srt` (no `-o`) | `<stem>.srt` next to input |
| `-o ./out/` (directory) | `./out/<stem>.<default-format>` |
| Neither `-o` nor `-f` | Print to terminal |

---

## 2. Batch Transcription

Transcribe multiple files at once using glob patterns:

```bash
# All m4a files in current directory
audiobench transcribe *.m4a

# Save all to a directory
audiobench transcribe *.m4a -o ./transcriptions/

# All audio files in a specific folder
audiobench transcribe ~/Music/recordings/*.m4a -f srt
```

A batch summary table is shown at the end with per-file stats.

---

## 3. Multilingual & Code-Switching

### Auto-detect language

By default, the model auto-detects the spoken language:

```bash
audiobench transcribe swahili_speech.m4a
```

### Force a language

```bash
audiobench transcribe interview.m4a --language sw     # Swahili
audiobench transcribe lecture.m4a --language fr        # French
```

### Code-switching (mixed languages)

Use `--prompt` to guide the model when speakers switch between languages:

```bash
audiobench transcribe conversation.m4a \
  --prompt "Conversation in Swahili and English, with occasional Sheng slang"
```

The prompt doesn't force a language — it gives the model context to handle transitions better.

---

## 4. Speed vs Accuracy

### Presets

```bash
audiobench transcribe --fast lecture.mp3       # Quick draft, lower quality
audiobench transcribe meeting.m4a              # Balanced (default)
audiobench transcribe --accurate interview.wav # Best quality, slowest
```

### What the presets change

| | Fast | Balanced | Accurate |
|-----|------|----------|----------|
| Beam size | 1 | 3 | 5 |
| Batch size | 8 | 4 | 1 (sequential) |
| Temperature | 0 (no fallback) | Fallback chain | Fallback chain |
| Context conditioning | No | No | Yes |
| Speed | ~2x faster | Default | ~0.5x slower |

### Choose a different model

Smaller models are faster but less accurate:

```bash
audiobench transcribe lecture.mp3 -m small     # ~461 MB, fastest
audiobench transcribe lecture.mp3 -m medium    # ~1.5 GB, good balance
audiobench transcribe lecture.mp3              # large-v3-turbo (default)
```

---

## 5. Audio Enhancement

### Built-in noise reduction

```bash
audiobench transcribe noisy_recording.m4a --enhance
```

Applies FFmpeg filters: high-pass at 200Hz, noise gate, and loudness normalization.

### Custom FFmpeg filters

```bash
audiobench transcribe recording.m4a --filter "highpass=f=300,lowpass=f=3000"
```

### Inspect file before transcribing

```bash
audiobench transcribe --check recording.m4a
```

Shows codec, duration, sample rate, channels, bitrate — without starting transcription.

---

## 6. History Management

### View past transcriptions

```bash
audiobench history              # Last 20
audiobench history --limit 50   # Last 50
```

Shows ID, filename, language, duration, word count, and timestamp for each.

### Search by content

```bash
audiobench search "yoga"
audiobench search "important meeting" --limit 10
```

Full-text search across all stored transcriptions.

### Re-export in a different format

```bash
audiobench export 3 -f vtt              # Export ID #3 as WebVTT
audiobench export 3 -f json -o data/    # As JSON to data/ directory
```

### Delete

```bash
audiobench delete 3          # Delete transcription #3
audiobench delete --all      # Wipe all history
```

---

## 7. Piping & Scripting

### Quiet mode for piping

```bash
# Grep for specific words
audiobench transcribe -q meeting.m4a | grep "deadline"

# Pipe to a file
audiobench transcribe -q meeting.m4a > transcript.txt

# Word count
audiobench transcribe -q meeting.m4a | wc -w
```

The `-q` / `--quiet` flag suppresses all UI and prints only the raw transcript.

---

## 8. Caching

Transcriptions are cached automatically by file hash. Re-transcribing the same file returns instantly:

```bash
audiobench transcribe meeting.m4a         # First run: full transcription
audiobench transcribe meeting.m4a         # Second run: instant cache hit
```

Force a fresh transcription:

```bash
audiobench transcribe --no-cache meeting.m4a
```

---

## 9. Pre-downloading Models

Download models ahead of time for offline use:

```bash
audiobench download large-v3-turbo     # Default model (~1.5 GB)
audiobench download small              # Smaller model (~461 MB)
```

Models are stored in `~/.audiobench/models/`.

---

## 10. Configuration

### Using `.env`

```bash
cp .env.example .env
# Edit .env with your preferred defaults
```

### Using environment variables

```bash
AUDIOBENCH_LANGUAGE=en audiobench transcribe meeting.m4a
AUDIOBENCH_SPEED_PRESET=fast audiobench transcribe *.m4a
```

### Check current settings

```bash
audiobench info
```

---

## Troubleshooting

### "FFmpeg not found"

Install FFmpeg:
```bash
sudo pacman -S ffmpeg       # Arch
sudo apt install ffmpeg     # Ubuntu/Debian
```

### First run is slow

The model (~1.5 GB) downloads on first use. Pre-download with:
```bash
audiobench download large-v3-turbo
```

### Poor accuracy

1. Try `--accurate` preset
2. Use `--prompt` to provide context
3. Use `--enhance` for noisy recordings
4. Try forcing the language with `--language en`

### Out of memory

Use a smaller model:
```bash
audiobench transcribe -m small recording.m4a
```
