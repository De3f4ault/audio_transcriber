"""Interactive wizard helpers for guided CLI flows."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from audiobench.cli.display.theme import ACCENT, BOLD, DIM, WARNING, console

T = TypeVar("T")


def prompt_menu(
    title: str,
    options: list[tuple[str, str, T]],
    default_idx: int | None = None,
) -> T:
    """Show a numbered menu and prompt the user for a choice.

    Args:
        title: The menu title.
        options: List of tuples (label, description, value).
        default_idx: 0-indexed default option if the user just presses Enter.

    Returns:
        The selected value.
    """
    console.print(f"\n  [{BOLD}]{title}[/]")
    for i, (label, desc, _) in enumerate(options):
        idx = i + 1
        default_marker = "  ← default" if default_idx == i else ""
        desc_padded = f"{desc:<25}"
        console.print(
            f"    [{ACCENT}][{idx}][/] {label:<15} [{DIM}]{desc_padded}[/]{default_marker}"
        )

    while True:
        try:
            choice = input("  → ").strip()
            if not choice and default_idx is not None:
                return options[default_idx][2]
                
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(options):
                    return options[idx - 1][2]
                    
            console.print(f"  [{WARNING}]Invalid choice. Please enter a number from 1 to {len(options)}.[/]")
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Wizard cancelled.[/]")
            raise
        except EOFError:
            raise KeyboardInterrupt


def prompt_bool(
    prompt: str,
    default: bool = False,
) -> bool:
    """Prompt for a yes/no answer.

    Args:
        prompt: The question to ask.
        default: Default answer if empty.

    Returns:
        True for yes, False for no.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    console.print(f"\n  {prompt}  [{DIM}]{suffix}[/]")
    
    while True:
        try:
            choice = input("  → ").strip().lower()
            if not choice:
                return default
            if choice in ("y", "yes"):
                return True
            if choice in ("n", "no"):
                return False
            console.print(f"  [{WARNING}]Please answer 'y' or 'n'.[/]")
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Wizard cancelled.[/]")
            raise
        except EOFError:
            raise KeyboardInterrupt


def prompt_string(
    prompt: str,
    default: str = "",
    validator: Callable[[str], bool] | None = None,
    validation_msg: str = "Invalid input.",
) -> str:
    """Prompt for a string input.

    Args:
        prompt: The prompt text.
        default: Default if empty.
        validator: Optional function to validate the input.
        validation_msg: Error message if validation fails.

    Returns:
        The user's string.
    """
    console.print(f"\n  {prompt}")
    
    while True:
        try:
            choice = input("  → ").strip()
            if not choice:
                choice = default
                
            if validator and not validator(choice):
                console.print(f"  [{WARNING}]{validation_msg}[/]")
                continue
                
            return choice
        except KeyboardInterrupt:
            console.print(f"\n  [{DIM}]Wizard cancelled.[/]")
            raise
        except EOFError:
            raise KeyboardInterrupt


def prompt_transcription(title: str = "Select a transcription", limit: int = 15) -> int:
    """Prompt the user to select a recent transcription.

    Args:
        title: Title of the menu.
        limit: Number of recent transcriptions to show.

    Returns:
        The selected transcription ID.
    """
    from audiobench.storage.repository import TranscriptionRepository
    from audiobench.core.db_engine import init_db
    
    init_db()
    repo = TranscriptionRepository()
    recent = repo.get_history(limit=limit)
    
    if not recent:
        console.print(f"  [{WARNING}]No transcriptions found in history.[/]")
        raise KeyboardInterrupt
        
    options = []
    for rec in recent:
        file_name = rec.get("file_name", "Unknown")
        # Truncate long file names
        if len(file_name) > 30:
            file_name = file_name[:27] + "..."
            
        tx_id = rec.get("id")
        created = rec.get("created_at", "")[:16]
        desc = f"ID: {tx_id}  |  {created}"
        options.append((file_name, desc, tx_id))
        
    return prompt_menu(title, options)


def prompt_file(title: str = "Enter file path") -> str:
    """Prompt the user for a valid file path."""
    import os
    
    def validate_file(p: str) -> bool:
        p = os.path.expanduser(p)
        return os.path.isfile(p)
        
    path = prompt_string(
        title, 
        validator=validate_file, 
        validation_msg="File does not exist or is not a regular file."
    )
    return os.path.expanduser(path)
