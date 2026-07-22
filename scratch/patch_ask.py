import re

with open("src/audiobench/cli/commands/chat.py") as f:
    content = f.read()

replacement = """
    if interactive_mode:
        from audiobench.cli.wizard import prompt_transcription, prompt_string
        import sys
        try:
            if not transcript_id:
                transcript_id = prompt_transcription("Select a transcript to query")
            if not log and not question:
                question = prompt_string("What is your question?")
        except KeyboardInterrupt:
            sys.exit(0)

    if not transcript_id:
        console.print(error_panel("Usage: audiobench ask [ID] [QUESTION]\\nOr use --interactive"))
        sys.exit(1)

    if not log and not question:
        console.print(error_panel("Usage: audiobench ask [ID] [QUESTION]\\nOr use --interactive"))
        sys.exit(1)

    # Fetch transcript
    init_db()
    repo = TranscriptionRepository()
    record = repo.get_by_id(transcript_id)
    if not record:
        console.print(error_panel("Not found", f"Transcript #{transcript_id} not found"))
        return
        
    audio_file_id = record.get("audio_file_id")

    if log:
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import AskLog, AskEntry
        from rich.table import Table
        
        with get_session() as session:
            ask_log = session.query(AskLog).filter_by(audio_file_id=audio_file_id).first()
            if not ask_log or not ask_log.entries:
                console.print(f"[dim]No ask log entries found for audio file #{audio_file_id}.[/]")
                return
                
            table = Table(title=f"Ask Log for Audio #{audio_file_id}")
            table.add_column("Date", style="dim")
            table.add_column("Model", style="blue")
            table.add_column("Question", style="green")
            table.add_column("Answer")
            
            for entry in ask_log.entries:
                table.add_row(
                    entry.created_at.strftime("%Y-%m-%d %H:%M"),
                    entry.model_name,
                    entry.question,
                    entry.answer[:100] + ("..." if len(entry.answer) > 100 else "")
                )
            
            console.print(table)
        return
"""

content = re.sub(
    r'    if interactive_mode:.*?if not record:\n        console\.print\(error_panel\("Not found", f"Transcript #\{transcript_id\} not found"\)\)\n        return',
    replacement.strip(),
    content,
    flags=re.DOTALL,
)

with open("src/audiobench/cli/commands/chat.py", "w") as f:
    f.write(content)
