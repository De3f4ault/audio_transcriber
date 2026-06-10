"""Service for generating semantic titles and auto-renaming audio files."""

import json
import logging
import re
import threading
from pathlib import Path

from google import genai

from audiobench.core.db_session import get_session
from audiobench.core.settings import get_settings
from audiobench.storage.models import AudioFileRecord, TranscriptionRecord

logger = logging.getLogger("audiobench.rename_service")


def generate_and_apply_title(tx_id: int, force_gemini: bool = False) -> tuple[bool, str]:
    """
    Generate a semantic title for a transcription and rename the source file.
    
    Args:
        tx_id: The ID of the transcription record.
        
    Returns:
        tuple[bool, str]: (Success boolean, Message or new filename).
    """
    settings = get_settings()

    with get_session() as session:
        tx_record = session.query(TranscriptionRecord).get(tx_id)
        if not tx_record:
            return False, f"Transcription #{tx_id} not found."
            
        audio_record = session.query(AudioFileRecord).get(tx_record.audio_file_id)
        if not audio_record:
            return False, f"No audio file linked to transcription #{tx_id}."
            
        old_path = Path(audio_record.file_path)
        if not old_path.exists():
            return False, f"Original file missing on disk: {old_path}"

        # Fetch first ~15 minutes of segments to capture more nuance and context
        # Sorting by segment_index just in case
        segments = sorted(tx_record.segments, key=lambda s: s.segment_index)
        intro_segments = [s for s in segments if s.end_time <= 900][:150]
        intro_text = " ".join([s.text for s in intro_segments]).strip()

    if not intro_text:
        return False, "Not enough text to generate a title."

    prompt = (
        "You are an expert audio librarian and archivist. "
        "Analyze the following transcript excerpt (the first 15 minutes of the audio) "
        "to understand the core topic, nuances, and main subject being discussed.\n"
        "Generate a comprehensive, highly semantic, and descriptive title for this audio file. "
        "The title should be 4 to 8 words long and capture the true essence of the content.\n"
        "Do NOT use quotes, special characters, or prefixes like 'Title:'. Output ONLY the raw title text.\n\n"
        f"{intro_text}"
    )

    new_title = ""
    # PRIMARY: Try Ollama (Qwen or default clean model) first
    if not force_gemini:
        try:
            from audiobench.chat.providers.ollama_provider import OllamaClient
            ollama = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.clean_model,
            )
            if ollama.is_available():
                # think=False because we just want a fast 3-5 word title
                response = ollama.chat([{"role": "user", "content": prompt}], think=False)
                new_title = response.get("content", "").strip()
            else:
                raise ValueError("Ollama not available or not running")
        except Exception as e:
            logger.warning("Ollama auto-rename failed/unavailable (%s), falling back to Gemini", e)

    if not new_title:
        # FALLBACK: Try Gemini
        try:
            if settings.gemini_api_key:
                client = genai.Client(api_key=settings.gemini_api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                if response and response.text:
                    new_title = response.text.strip()
            else:
                raise ValueError("No Gemini API key configured for fallback")
        except Exception as fallback_e:
            logger.warning("Gemini fallback also failed: %s", fallback_e)

    if not new_title:
        # BOTH FAILED! Tag the file for later retry.
        _add_pending_tag(tx_id)
        return False, "Failed to connect to LLM APIs. Tagged with pending_auto_rename."

    # Clean up title for filesystem
    new_title = re.sub(r"[^\w\s-]", " ", new_title).strip()
    new_title = re.sub(r"\s+", " ", new_title)

    if not new_title or len(new_title.split()) > 15:
        return False, f"LLM generated an invalid title: {new_title}"

    new_filename = f"{new_title}{old_path.suffix}"
    new_path = old_path.parent / new_filename

    # Handle collisions
    counter = 1
    while new_path.exists() and new_path != old_path:
        new_path = old_path.parent / f"{new_title}_{counter}{old_path.suffix}"
        new_filename = new_path.name
        counter += 1

    if new_path != old_path:
        try:
            old_path.rename(new_path)
        except OSError as e:
            return False, f"Filesystem rename failed: {e}"

        # Update DB (both audio file and transcription)
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if tx_record:
                # Update transcription record's own filename cache
                tx_record.file_name = new_filename
                
                # Update parent audio record
                if tx_record.audio_file_id:
                    audio_record = session.query(AudioFileRecord).get(tx_record.audio_file_id)
                    if audio_record:
                        audio_record.file_name = new_filename
                        audio_record.file_path = str(new_path)
                        
                        # Remove 'pending_auto_rename' tag if it exists
                        _remove_pending_tag(audio_record)

            session.commit()

        logger.info("Renamed file to %s", new_filename)
        return True, new_filename
    else:
        # Same filename? Just remove the tag if present.
        with get_session() as session:
            tx_record = session.query(TranscriptionRecord).get(tx_id)
            if tx_record and tx_record.audio_file_id:
                audio_record = session.query(AudioFileRecord).get(tx_record.audio_file_id)
                if audio_record:
                    _remove_pending_tag(audio_record)
            session.commit()
            
        return True, new_filename


def _add_pending_tag(tx_id: int) -> None:
    """Add the pending_auto_rename tag to the parent AudioFileRecord."""
    with get_session() as session:
        tx_record = session.query(TranscriptionRecord).get(tx_id)
        if tx_record and tx_record.audio_file_id:
            audio_record = session.query(AudioFileRecord).get(tx_record.audio_file_id)
            if audio_record:
                try:
                    tags_list = json.loads(audio_record.tags) if audio_record.tags else []
                except Exception:
                    tags_list = []
                if "pending_auto_rename" not in tags_list:
                    tags_list.append("pending_auto_rename")
                    audio_record.tags = json.dumps(tags_list)
        session.commit()


def _remove_pending_tag(audio_record: AudioFileRecord) -> None:
    """Remove the pending_auto_rename tag from an AudioFileRecord in-memory."""
    try:
        tags_list = json.loads(audio_record.tags) if audio_record.tags else []
    except Exception:
        tags_list = []
    if "pending_auto_rename" in tags_list:
        tags_list.remove("pending_auto_rename")
        audio_record.tags = json.dumps(tags_list)


def spawn_auto_naming(tx_id: int) -> None:
    """Spawn a background thread to generate a title and rename the file."""
    def _run() -> None:
        try:
            success, msg = generate_and_apply_title(tx_id)
            if not success:
                logger.warning("Background auto-rename failed for #%d: %s", tx_id, msg)
        except Exception as e:
            logger.warning("Background auto-rename thread crashed for #%d: %s", tx_id, e)

    thread = threading.Thread(target=_run, name=f"rename-{tx_id}", daemon=True)
    thread.start()
    logger.info("Spawned background auto-rename thread for #%d", tx_id)
