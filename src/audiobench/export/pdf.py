import platform
import sys
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, WARNING, console


class PDFExporter:
    """Renders transcripts to high-fidelity PDFs using Jinja2 and WeasyPrint."""

    def __init__(self):
        self._check_dependencies()
        self.template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))

    def _get_os_install_command(self) -> str:
        """Pragmatically detect the OS and return the exact setup command for missing C-libraries."""
        system = platform.system()

        if system == "Darwin":
            return "brew install pango cairo glib"
        elif system == "Linux":
            # Try to read /etc/os-release to be more specific
            try:
                with open("/etc/os-release") as f:
                    os_release = f.read().lower()
                    if "ubuntu" in os_release or "debian" in os_release:
                        return "sudo apt install libpango-1.0-0 libcairo2"
                    elif "arch" in os_release or "manjaro" in os_release:
                        return "sudo pacman -S pango cairo"
                    elif "fedora" in os_release:
                        return "sudo dnf install pango cairo"
            except Exception:
                pass
            # Generic Linux fallback
            return "sudo apt install libpango-1.0-0 libcairo2   (or your distro's equivalent)"
        else:
            return "Download and install GTK3 for Windows (which includes Cairo and Pango)"

    def _check_dependencies(self) -> None:
        """Lazy load WeasyPrint and gracefully catch missing system libraries."""
        try:
            import weasyprint  # noqa: F401
        except ImportError:
            console.print(f"  [{WARNING}]Missing Python package:[/] weasyprint")
            console.print(f"  [{DIM}]Run:[/] pip install weasyprint jinja2")
            sys.exit(1)
        except OSError as e:
            # This happens when dlopen() fails to load Cairo or Pango
            console.print(f"\n  [{BOLD}][{WARNING}]System Dependency Missing[/][/]")
            console.print("  AudioBench uses WeasyPrint for professional PDF generation.")
            console.print(
                "  This requires the [b]Cairo[/] and [b]Pango[/] graphics libraries to be installed on your system."
            )
            console.print()
            console.print(f"  [{DIM}]Error details:[/] {e}")
            console.print()

            cmd = self._get_os_install_command()
            console.print(f"  [{BOLD}]To fix this on your system, run:[/]")
            console.print(f"    [{ACCENT}]{cmd}[/]\n")
            sys.exit(1)

    def _format_time(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS."""
        secs = int(seconds)
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def export_transcript(self, rec: dict, output_path: str | Path) -> None:
        """Compile context, render Jinja2 template, and generate PDF."""
        import weasyprint

        # 1. Prepare data context
        file_name = rec.get("file_name", f"Transcript #{rec['id']}")

        # Format segments with human-readable timestamps
        segments = rec.get("segments", [])
        for seg in segments:
            seg["start_formatted"] = self._format_time(seg.get("start", 0))
            seg["end_formatted"] = self._format_time(seg.get("end", 0))

            # Map raw speaker IDs (e.g. SPEAKER_00) to names if a mapping exists
            # For this simple exporter, we just rely on what is in the segment,
            # or if the user used map_speakers during diarization, the names are already in the DB.
            if "speaker" in seg and "speaker_name" not in seg:
                seg["speaker_name"] = seg["speaker"]

        # Parse dates
        date_transcribed = ""
        if rec.get("created_at"):
            try:
                dt = datetime.fromisoformat(rec["created_at"])
                date_transcribed = dt.strftime("%B %d, %Y at %I:%M %p")
            except Exception:
                date_transcribed = str(rec["created_at"])

        # Parse chapters
        chapters = rec.get("chapters", [])
        chapter_map = {c["id"]: c["title"] for c in chapters}

        context = {
            "file_name": file_name,
            "date_generated": datetime.now().strftime("%Y-%m-%d"),
            "date_transcribed": date_transcribed,
            "word_count": f"{rec.get('word_count', 0):,}",
            "engine": rec.get("engine", "whisper"),
            "model": rec.get("model_name", "unknown"),
            "segments": segments,
            "chapter_map": chapter_map,
            # We don't have a DB field for AI summary yet natively stored on the transcript record
            # in standard AudioBench unless implemented via dot-commands saving.
            # So summary will be empty for now.
            "summary": rec.get("summary", ""),
        }

        # 2. Render HTML
        template = self.env.get_template("pdf_template.html")
        html_content = template.render(**context)

        # 3. Generate PDF
        # We wrap HTML in WeasyPrint's HTML class and call write_pdf
        pdf_file = weasyprint.HTML(string=html_content).write_pdf()

        # 4. Save to disk
        out_path = Path(output_path)
        out_path.write_bytes(pdf_file)

        return out_path
