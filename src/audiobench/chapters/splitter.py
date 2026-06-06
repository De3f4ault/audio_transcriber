"""Chapter audio splitter."""

from __future__ import annotations

import subprocess
from pathlib import Path

from audiobench.chapters.cue_parser import ChapterInfo
from audiobench.core.logger_factory import get_logger

logger = get_logger("chapters.splitter")


class ChapterSplitter:
    """Splits audio files into chapters using ffmpeg -c copy."""

    def split(
        self, source: Path, chapters: list[ChapterInfo], output_dir: Path, fmt: str = "wav"
    ) -> list[Path | None]:
        """Split an audio file into chapter chunks.

        Uses `-c copy` to extract chunks losslessly and quickly without re-encoding,
        unless the target format is different from the source and requires conversion.

        Args:
            source: Path to the original audio file.
            chapters: List of ChapterInfo objects.
            output_dir: Directory to save the chunks.
            fmt: Target audio format (e.g., 'wav', 'mp3', 'flac').

        Returns:
            List of Paths to the generated chunk files. If a chapter was skipped
            (e.g., ghost chapter), that slot will contain None.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[Path | None] = []

        for chap in chapters:
            if chap.is_ghost:
                logger.debug("Skipping ghost chapter %d: %s", chap.index, chap.title)
                results.append(None)
                continue

            # Make a safe filename
            safe_title = "".join(c for c in chap.title if c.isalnum() or c in " -_")
            filename = f"{chap.index:03d}_{safe_title.replace(' ', '_')[:50]}.{fmt}"
            output_path = output_dir / filename

            if output_path.exists() and output_path.stat().st_size > 0:
                logger.debug("Chapter %d chunk already exists: %s", chap.index, output_path.name)
                results.append(output_path)
                continue

            logger.info("Extracting chapter %d: %s", chap.index, chap.title)

            # Construct ffmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-v",
                "error",
                "-i",
                str(source),
                "-ss",
                str(chap.start_time),
                "-to",
                str(chap.end_time),
            ]

            # If the format is the same as the source, we can just copy
            source_ext = source.suffix.lower().lstrip(".")
            if source_ext == fmt or (source_ext in ("m4a", "m4b", "mp4") and fmt in ("m4a", "aac")):
                cmd.extend(["-c", "copy"])
            elif fmt == "wav":
                cmd.extend(["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"])

            cmd.append(str(output_path))

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                results.append(output_path)
            except subprocess.CalledProcessError as e:
                logger.error("Failed to split chapter %d: %s\n%s", chap.index, e, e.stderr)
                results.append(None)
                if output_path.exists():
                    output_path.unlink()

        return results
