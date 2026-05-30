"""LLM Prompts centralized for both CLI and Web UI usage."""

# --- Gemini Engine Prompts ---

GEMINI_TRANSCRIPTION_PROMPT = """\
Transcribe the following audio accurately and completely.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The transcribed text for this segment.",
      "words": [
        {"word": "The", "start": 0.0, "end": 0.3},
        {"word": "transcribed", "start": 0.35, "end": 0.9}
      ]
    }
  ]
}

Rules:
- Split into natural segments (sentences or clauses, ~5-15 seconds each).
- Include word-level timestamps if possible.
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_DIARIZATION_PROMPT = """\
Transcribe the following audio accurately and completely, identifying each speaker.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "<ISO 639-1 code>",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The transcribed text for this segment.",
      "speaker": "Speaker 1",
      "words": [
        {"word": "The", "start": 0.0, "end": 0.3},
        {"word": "transcribed", "start": 0.35, "end": 0.9}
      ]
    }
  ]
}

Rules:
- Identify each distinct speaker and label them consistently (Speaker 1, Speaker 2, etc.).
- Start a new segment when the speaker changes OR at natural sentence boundaries.
- Split into natural segments (sentences or clauses, ~5-15 seconds each).
- Include word-level timestamps if possible.
- Detect the spoken language automatically.
- Preserve the original language — do NOT translate unless asked.
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_TRANSLATE_PROMPT = """\
Transcribe the following audio and translate everything to English.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The translated English text for this segment.",
      "words": []
    }
  ]
}

Rules:
- Translate ALL speech to English.
- Split into natural segments (sentences or clauses).
- Return raw JSON only. No explanation, no markdown fences.
"""

GEMINI_DIARIZATION_TRANSLATE_PROMPT = """\
Transcribe the following audio, translate everything to English, and identify each speaker.

Return ONLY a valid JSON object with this exact structure (no markdown, no fences):
{
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "The translated English text for this segment.",
      "speaker": "Speaker 1",
      "words": []
    }
  ]
}

Rules:
- Identify each distinct speaker and label them consistently (Speaker 1, Speaker 2, etc.).
- Start a new segment when the speaker changes OR at natural sentence boundaries.
- Translate ALL speech to English.
- Split into natural segments (sentences or clauses).
- Return raw JSON only. No explanation, no markdown fences.
"""
