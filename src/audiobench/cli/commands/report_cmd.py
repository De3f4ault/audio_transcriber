"""CLI command for generating knowledge dossiers."""

import json

import click
from rich.console import Console

from audiobench.core.db_session import get_session
from audiobench.core.logger_factory import get_logger
from audiobench.memory.enums import SourceType
from audiobench.storage.models import (
    AskLog,
    AudioFileRecord,
    BookmarkRecord,
    ChatConversation,
    ConversationSummary,
    ExpressionRecord,
    ExpressionRelation,
    TranscriptionRecord,
)

logger = get_logger("cli.report")
console = Console()


@click.command()
@click.argument("audio_file_id", type=int)
def report(audio_file_id: int) -> None:
    """Generate a full knowledge dossier for an audio file."""

    with get_session() as session:
        # 1. Metadata
        audio = session.query(AudioFileRecord).filter_by(id=audio_file_id).first()
        if not audio:
            console.print(f"[red]Error: Audio file #{audio_file_id} not found.[/red]")
            return

        console.print(f"# Knowledge Dossier: {audio.file_name}\n")
        console.print("## 1. Metadata")
        console.print(f"- **ID**: {audio.id}")
        console.print(f"- **Filename**: {audio.file_name}")
        console.print(f"- **Duration**: {audio.duration_seconds}s")
        console.print(f"- **Created**: {audio.created_at}")

        # 2. Transcripts
        transcriptions = (
            session.query(TranscriptionRecord).filter_by(audio_file_id=audio_file_id).all()
        )
        console.print("\n## 2. Transcripts")
        if not transcriptions:
            console.print("No transcripts found.")
        else:
            for tx in transcriptions:
                console.print(f"### Transcript #{tx.id} ({tx.model_name})")
                console.print(tx.full_text)
                console.print("\n---\n")

        # 3. Ask Log
        console.print("\n## 3. Ask Log")
        ask_log = session.query(AskLog).filter_by(audio_file_id=audio_file_id).first()
        if not ask_log or not ask_log.entries:
            console.print("No ask log entries found.")
        else:
            for entry in sorted(ask_log.entries, key=lambda e: e.created_at):
                console.print(f"**Q:** {entry.question}")
                console.print(f"**A:** {entry.answer}\n")

        # 4. Chat Sessions
        console.print("\n## 4. Chat Sessions")
        all_convs = session.query(ChatConversation).all()
        related_convs = []
        for conv in all_convs:
            try:
                tids = json.loads(conv.transcript_ids)
                if any(tx.id in tids for tx in transcriptions):
                    related_convs.append(conv)
            except Exception:
                continue

        if not related_convs:
            console.print("No related chat sessions found.")
        else:
            for c in related_convs:
                console.print(
                    f"- **Session #{c.id}**: {c.title} ({c.message_count} messages, {c.created_at})"
                )

        # 5. Session Summaries & 6. Open Threads
        console.print("\n## 5. Session Summaries & Open Threads")
        found_summary = False
        for c in related_convs:
            summary = session.query(ConversationSummary).filter_by(conversation_id=c.id).first()
            if summary:
                found_summary = True
                console.print(f"### Session #{c.id}: {summary.refined_title}")
                console.print(f"**Narrative:** {summary.narrative}\n")

                try:
                    threads = json.loads(summary.open_threads)
                    if threads:
                        console.print("**Open Threads:**")
                        for t in threads:
                            console.print(f"- {t.get('question')} ({t.get('context')})")
                except Exception:
                    pass
                console.print("\n")

        if not found_summary:
            console.print("No session summaries found.")

        # 7. System Inferences
        console.print("\n## 7. System Inferences")
        # Find all transcript chunk expressions for this file
        tx_exprs = []
        for tx in transcriptions:
            # get Tier 1
            t1 = (
                session.query(ExpressionRecord)
                .filter_by(source_type=SourceType.AUDIO_TRANSCRIPT.value, source_id=tx.id)
                .first()
            )
            if t1:
                tx_exprs.append(t1)
                # get Tier 2 and Tier 3 using relations? They also have source_id=tx.id usually?
                # Actually chunks have source_id=tx.id too?
                # Note: Chunks are stored as AUDIO_TRANSCRIPT in repository.py
                chunks = (
                    session.query(ExpressionRecord)
                    .filter_by(source_type=SourceType.AUDIO_TRANSCRIPT.value, source_id=tx.id)
                    .all()
                )
                tx_exprs.extend(chunks)

        inference_ids = set()
        for expr in tx_exprs:
            rels = session.query(ExpressionRelation).filter_by(to_expression_id=expr.id).all()
            for r in rels:
                src_expr = (
                    session.query(ExpressionRecord).filter_by(id=r.from_expression_id).first()
                )
                if src_expr and src_expr.source_type == SourceType.SYSTEM_INFERENCE.value:
                    inference_ids.add(src_expr.id)

        if not inference_ids:
            console.print("No system inferences found.")
        else:
            for inf_id in inference_ids:
                inf = session.query(ExpressionRecord).filter_by(id=inf_id).first()
                if inf:
                    console.print(
                        f"- [#{inf.id}] ({inf.inference_confidence}) {inf.content} [{inf.inference_status}]"
                    )

        # 8. Bookmarks
        console.print("\n## 8. Bookmarks")
        bookmarks = (
            session.query(BookmarkRecord)
            .filter_by(audio_file_id=audio_file_id)
            .order_by(BookmarkRecord.timestamp)
            .all()
        )
        if not bookmarks:
            console.print("No bookmarks found.")
        else:
            for b in bookmarks:
                time_str = f"{b.timestamp:.1f}s"
                if b.end_timestamp:
                    time_str += f" - {b.end_timestamp:.1f}s"
                console.print(f"- **{time_str}** {b.name}: {b.notes or ''}")
