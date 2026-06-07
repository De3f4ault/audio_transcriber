import re

with open("src/audiobench/cli/commands/chat.py") as f:
    content = f.read()

replacement = """    if summary is not None:
        init_db()
        from audiobench.core.db_session import get_session
        from audiobench.storage.models import ConversationSummary
        from rich.panel import Panel
        from rich.markdown import Markdown

        with get_session() as session:
            record = session.query(ConversationSummary).filter_by(conversation_id=summary).first()
            if not record:
                console.print(error_panel("Not found", f"Session summary for conversation #{summary} not found."))
                sys.exit(1)

            md = f"**Narrative**: {record.narrative}\\n\\n"
            md += f"**Key Insights**: {record.key_insights}\\n\\n"
            md += f"**Open Threads**: {record.open_threads}\\n\\n"
            
            console.print(Panel(Markdown(md), title=f"Session Summary #{summary}", border_style="blue"))
        sys.exit(0)

    if delete_id is not None:"""

content = re.sub(r"    if delete_id is not None:", replacement, content)

with open("src/audiobench/cli/commands/chat.py", "w") as f:
    f.write(content)
