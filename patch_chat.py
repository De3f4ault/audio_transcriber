import re

with open("src/audiobench/cli/commands/chat.py") as f:
    content = f.read()

new_decorators = """@click.option(
    "--chapter",
    type=int,
    default=None,
    help="Chat with a specific chapter",
)
@click.option(
    "--summary",
    type=int,
    default=None,
    help="View the session memoir for a conversation",
)
def chat(
    transcript_ids: tuple[int, ...],
    model: str | None,
    temperature: float,
    search_query: str | None,
    recent: int | None,
    resume_id: int | None,
    list_chats: bool,
    delete_id: int | None,
    think: bool,
    chapter: int | None,
    summary: int | None,
) -> None:"""

content = re.sub(
    r'@click\.option\(\n    "--chapter",\n    type=int,\n    default=None,\n    help="Chat with a specific chapter",\n\)\ndef chat\(\n    transcript_ids: tuple\[int, \.\.\.\],\n    model: str \| None,\n    temperature: float,\n    search_query: str \| None,\n    recent: int \| None,\n    resume_id: int \| None,\n    list_chats: bool,\n    delete_id: int \| None,\n    think: bool,\n    chapter: int \| None,\n\) -> None:',
    new_decorators,
    content,
)

with open("src/audiobench/cli/commands/chat.py", "w") as f:
    f.write(content)
