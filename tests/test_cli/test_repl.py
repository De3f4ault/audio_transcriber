"""Tests for REPL session state and dispatch."""

from __future__ import annotations

import click


class TestReplSession:
    """Test ReplSession state management."""

    def _make_session(self):
        from audiobench.cli.repl.session import ReplSession

        group = click.Group("test")
        return ReplSession(group)

    def test_initial_state(self):
        session = self._make_session()
        assert session.last_id is None
        assert session.focus is None
        assert session._command_count == 0

    def test_prompt_no_context(self):
        session = self._make_session()
        prompt = session.prompt
        assert "❯" in str(prompt)
        assert "#" not in str(prompt)

    def test_expand_vars_no_context(self):
        session = self._make_session()
        args = ["show", "$last"]
        result = session.expand_vars(args)
        assert result == ["show", "$last"]  # No expansion without context

    def test_expand_vars_with_context(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=42, label="Transcript #42")
        args = ["show", "$last"]
        result = session.expand_vars(args)
        assert result == ["show", "42"]

    def test_auto_inject_id_context_aware(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=10, label="Transcript #10")
        args = ["show"]
        result = session.auto_inject_id(args)
        assert result == ["show", "10"]

    def test_auto_inject_id_skips_explicit(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=10, label="Transcript #10")
        args = ["show", "99"]
        result = session.auto_inject_id(args)
        assert result == ["show", "99"]  # User-specified ID preserved

    def test_auto_inject_id_skips_non_aware(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=10, label="Transcript #10")
        args = ["history"]
        result = session.auto_inject_id(args)
        assert result == ["history"]  # history is not context-aware

    def test_clear_context(self):
        session = self._make_session()
        from audiobench.core.focused_entity import FocusedEntity

        session.focus = FocusedEntity(type="transcript", id=42, label="Transcript #42")
        session.focus = None
        assert session.last_id is None
        assert session.focus is None


class TestSlashCommands:
    """Test slash command routing."""

    def test_exit_returns_true(self):
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.slash_commands import handle_slash_command

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/exit", session) is True

    def test_quit_returns_true(self):
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.slash_commands import handle_slash_command

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/quit", session) is True

    def test_unknown_returns_false(self):
        from audiobench.cli.repl.session import ReplSession
        from audiobench.cli.repl.slash_commands import handle_slash_command

        group = click.Group("test")
        session = ReplSession(group)
        assert handle_slash_command("/nonexistent", session) is False


class TestAliases:
    """Test bare-word alias mapping."""

    def test_help_alias(self):
        from audiobench.cli.repl.slash_commands import ALIASES

        assert ALIASES["help"] == "/help"

    def test_exit_alias(self):
        from audiobench.cli.repl.slash_commands import ALIASES

        assert ALIASES["exit"] == "/exit"

    def test_quit_alias(self):
        from audiobench.cli.repl.slash_commands import ALIASES

        assert ALIASES["quit"] == "/exit"
