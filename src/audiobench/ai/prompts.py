"""Prompt templates for AI transcript analysis.

Each template takes a transcript text and returns a formatted prompt
suitable for LLM processing.
"""

from __future__ import annotations

# System prompt for all transcript analysis tasks
TRANSCRIPT_SYSTEM = (
    "You are an expert assistant that analyzes audio transcripts. "
    "Be concise, clear, and actionable. Use bullet points where appropriate. "
    "Do not include preamble or pleasantries."
)


def summarize(transcript: str) -> str:
    """Bullet-point summary of a transcript."""
    return (
        "Summarize the following transcript into concise bullet points. "
        "Focus on key topics, decisions, and conclusions.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def action_items(transcript: str) -> str:
    """Extract action items from a meeting transcript."""
    return (
        "Extract all action items, tasks, and commitments from this transcript. "
        "For each, identify WHO is responsible and WHAT they need to do. "
        "Format as a numbered list.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def rewrite(transcript: str) -> str:
    """Clean up disfluencies and improve readability."""
    return (
        "Rewrite this transcript to be clear and readable. "
        "Remove filler words (um, uh, like), fix grammar, "
        "and improve sentence structure. Preserve the original meaning.\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


def translate_text(transcript: str, target_language: str) -> str:
    """Translate transcript to a target language."""
    return (
        f"Translate the following text to {target_language}. "
        "Maintain the original formatting and paragraph structure.\n\n"
        f"TEXT:\n{transcript}"
    )


def qa(transcript: str, question: str) -> str:
    """Answer a question about a transcript."""
    return (
        f"Answer the following question based on the transcript below. "
        "If the answer is not in the transcript, say so.\n\n"
        f"QUESTION: {question}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )


# ── Chat Mode Prompts ──────────────────────────────────────

CHAT_SYSTEM = (
    "You are an expert AI assistant for AudioBench. "
    "You have access to the user's transcribed audio content shown below. "
    "Answer questions, summarize, extract insights, find action items, or "
    "help analyze the content. Be concise and reference specific parts of "
    "the transcripts when relevant. If asked about something not in the "
    "transcripts, say so clearly.\n\n"
    "{transcript_context}"
)

CHAT_SYSTEM_NO_CONTEXT = (
    "You are an expert AI assistant for AudioBench. "
    "The user has not loaded any transcripts yet. You can still chat freely, "
    "but suggest using /load <ID> to add transcript context. "
    "Be concise and helpful."
)

TITLE_PROMPT = (
    "Based on this short conversation, generate a title of at most 6 words. "
    "Reply with ONLY the title, no quotes, no punctuation at the end.\n\n"
    "User: {first_message}\n"
    "Assistant: {first_response}"
)
