"""Custom Click group with default command fallback and typo correction.

If the user types a command name that doesn't exist, we suggest the
closest match using difflib (e.g., 'trancribe' → 'Did you mean transcribe?').
"""

import difflib

import click


class DefaultGroup(click.Group):
    """Invokes a default subcommand if the subcommand is missing.

    Also provides typo correction for mistyped command names.
    """

    def __init__(self, *args, **kwargs):
        self.default_command = kwargs.pop("default_command", None)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        if not args and self.default_command is not None:
            args.insert(0, self.default_command)
            return super().parse_args(ctx, args)

        # Words that should never fall through to the default command.
        # These are common user expectations that aren't files.
        _meta_words = {"help", "version"}

        # If we have arguments, check if the first one is a known command or an option.
        if (
            args
            and args[0] not in self.commands
            and not args[0].startswith("-")
            and args[0].lower() not in _meta_words
            and self.default_command is not None
        ):
            args.insert(0, self.default_command)

        # Redirect bare "help" to --help
        if args and args[0].lower() == "help" and args[0] not in self.commands:
            args[0] = "--help"

        return super().parse_args(ctx, args)

    def resolve_command(self, ctx, args):
        """Override to add typo correction when a command is not found."""
        cmd_name = args[0] if args else None

        # Let Click resolve normally first
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            pass

        # Command not found — try fuzzy matching
        if cmd_name and cmd_name not in self.commands:
            close = difflib.get_close_matches(
                cmd_name,
                list(self.commands.keys()),
                n=1,
                cutoff=0.5,
            )
            if close:
                hint = f"Error: No such command '{cmd_name}'.\n\nDid you mean: {close[0]}?"
                raise click.UsageError(hint)

        raise click.UsageError(f"Error: No such command '{cmd_name}'.")

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format commands into logical groups in the help text."""
        commands = []
        for cmd in self.list_commands(ctx):
            c = self.get_command(ctx, cmd)
            if c is None or c.hidden:
                continue
            commands.append((cmd, c))

        if not commands:
            return

        groups = {
            "Core Actions": ["repl", "transcribe", "listen"],
            "Analysis & Exploration": [
                "ask", "chat", "summarize", "vocab", "bookmark", "analyze", "inspect", "audio", "play", "speak"
            ],
            "Data & Management": ["history", "search", "export", "delete", "show", "clean"],
            "Configuration & Jobs": ["config", "jobs", "download", "system", "info", "doctor", "cleanup", "preset", "status"]
        }

        categorized = {k: [] for k in groups}
        uncategorized = []

        for name, cmd in commands:
            placed = False
            for group_name, cmds in groups.items():
                if name in cmds:
                    categorized[group_name].append((name, cmd))
                    placed = True
                    break
            if not placed:
                uncategorized.append((name, cmd))

        for group_name, cmds in categorized.items():
            if not cmds:
                continue
            # Use formatter.write_paragraph to add some space
            formatter.write(f"\n{group_name}:\n")
            with formatter.indentation():
                # We format it manually since formatter.write_dl doesn't support grouping natively
                # in older click versions cleanly without messing up alignments.
                rows = []
                for name, cmd in cmds:
                    help_text = cmd.get_short_help_str(limit=formatter.width) or ""
                    rows.append((name, help_text))
                formatter.write_dl(rows)

        if uncategorized:
            formatter.write("\nOther Commands:\n")
            with formatter.indentation():
                rows = []
                for name, cmd in uncategorized:
                    help_text = cmd.get_short_help_str(limit=formatter.width) or ""
                    rows.append((name, help_text))
                formatter.write_dl(rows)
