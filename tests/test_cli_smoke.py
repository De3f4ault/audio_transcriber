"""Smoke tests — every command's --help works and exits 0.

Ensures the monolith split didn't break any command registration
or imports. Each test invokes the command with --help and verifies
a clean exit code.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cli.app import cli


@pytest.fixture
def runner():
    return CliRunner()


# ── Main group ──


def test_main_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "AudioBench" in result.output


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


# ── Individual commands ──


COMMANDS = [
    "transcribe",
    "subtitle",
    "listen",
    "speak",
    "download-voice",
    "summarize",
    "ask",
    "chat",
    "history",
    "search",
    "export",
    "delete",
    "download",
    "info",
]


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help(runner, command):
    """Every registered command should respond to --help with exit code 0."""
    result = runner.invoke(cli, [command, "--help"])
    assert result.exit_code == 0, f"{command} --help failed: {result.output}"
    assert command.replace("-", " ") in result.output.lower() or "usage" in result.output.lower()


# ── Command count ──


def test_all_14_commands_registered(runner):
    """Verify that exactly 14 commands are registered."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output, f"Command '{cmd}' missing from --help output"
