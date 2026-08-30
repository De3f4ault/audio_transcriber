import json
from pathlib import Path
from unittest.mock import patch

from audiobench.cli.commands.memory_cmd import (
    SearchSessionState,
    _load_search_prefs,
    _parse_slash_command,
    _save_search_prefs,
)


def test_search_session_state_default_layout():
    state = SearchSessionState()
    assert state.layout == 'book'
    assert state.wrap_cap is None


def test_search_prefs_save_and_load(tmp_path: Path):
    prefs_file = tmp_path / 'search_prefs.json'
    with patch('audiobench.cli.commands.memory_cmd._prefs_path', return_value=prefs_file):
        state = SearchSessionState()
        state.layout = 'book'
        state.wrap_cap = 88
        state.preset = 'deep'
        state.diversity_weight = 0.8
        state.model = 'qwen2.5:32b'

        _save_search_prefs(state)
        assert prefs_file.exists()

        loaded = _load_search_prefs()
        assert loaded['layout'] == 'book'
        assert loaded['wrap_cap'] == 88
        assert loaded['preset'] == 'deep'
        assert loaded['diversity_weight'] == 0.8
        assert loaded['model'] == 'qwen2.5:32b'


def test_slash_command_persists_settings(tmp_path: Path):
    prefs_file = tmp_path / 'search_prefs.json'
    with patch('audiobench.cli.commands.memory_cmd._prefs_path', return_value=prefs_file):
        state = SearchSessionState()
        _parse_slash_command('/set layout book', state)
        assert state.layout == 'book'
        loaded = _load_search_prefs()
        assert loaded.get('layout') == 'book'

        _parse_slash_command('/set width 92', state)
        assert state.wrap_cap == 92
        loaded = _load_search_prefs()
        assert loaded.get('wrap_cap') == 92

        _parse_slash_command('/set fast', state)
        assert state.preset == 'fast'
        loaded = _load_search_prefs()
        assert loaded.get('preset') == 'fast'

        _parse_slash_command('/set autocomplete off', state)
        assert state.autocomplete is False
        loaded = _load_search_prefs()
        assert loaded.get('autocomplete') is False

        _parse_slash_command('/set autocomplete on', state)
        assert state.autocomplete is True
        loaded = _load_search_prefs()
        assert loaded.get('autocomplete') is True


def test_settings_card_commands():
    state = SearchSessionState()
    for cmd in ('/settings', '/config', '/show config', '/show settings'):
        out = _parse_slash_command(cmd, state)
        assert out is not None
        assert 'Search & Session Settings' in out
        assert 'Preset' in out
        assert 'Layout' in out

    # Bare /set should return usage help, not the full card
    set_out = _parse_slash_command('/set', state)
    assert set_out is not None
    assert 'Usage:' in set_out
    assert '/set <option> <value>' in set_out
    assert '/settings' in set_out
