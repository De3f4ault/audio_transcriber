"""System commands — info, download, doctor."""

from __future__ import annotations

import sys

import click

from cli.theme import (
    ACCENT,
    APP_NAME,
    APP_VERSION,
    BOLD,
    DIM,
    SUCCESS,
    console,
    error_panel,
    make_table,
)
from src.audiobench.config.settings import get_settings


# ── Info Command ────────────────────────────────────────────


@click.command()
def info() -> None:
    """Show system info and current settings."""
    settings = get_settings()

    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    device_str = settings.resolve_device()
    device_label = f"{device_str} ({'CUDA' if cuda_available else 'CPU only'})"

    table = make_table(
        f"{APP_NAME} v{APP_VERSION}",
        [
            ("Setting", {"style": BOLD}),
            ("Value", {}),
        ],
    )

    table.add_row("Model", settings.model_name)
    table.add_row("Device", device_label)
    table.add_row("Compute Type", settings.resolve_compute_type())
    table.add_row("CPU Threads", str(settings.resolve_cpu_threads()))
    table.add_row("Speed Preset", settings.speed_preset)
    table.add_row("Beam Size", str(settings.resolve_beam_size()))
    table.add_row("Batch Size", str(settings.resolve_batch_size()))
    table.add_row("Language", settings.language or "auto-detect")
    table.add_row("Output Format", settings.output_format)
    table.add_row("Word Timestamps", str(settings.word_timestamps))
    table.add_row("Diarization", str(settings.enable_diarization))
    table.add_row("Database", settings.database_url)
    table.add_row("Models Dir", str(settings.models_dir))
    table.add_row("HF Token", "✓ set" if settings.hf_token else "– not set")
    table.add_row("Log Level", settings.log_level)

    console.print(table)

    # Engines
    from src.audiobench.engines.factory import list_engines

    console.print(f"  [{DIM}]Engines: {', '.join(list_engines())}[/]")

    # Formats
    from src.audiobench.core.ffmpeg import AudioLoader

    formats = AudioLoader.get_supported_formats()
    console.print(f"  [{DIM}]Audio: {', '.join(sorted(formats['audio']))}[/]")
    console.print(f"  [{DIM}]Video: {', '.join(sorted(formats['video']))}[/]")

    # Presets
    console.print()
    pt = make_table(
        "Speed Presets",
        [
            ("Preset", {}),
            ("Beam", {"justify": "center", "width": 6}),
            ("Batch", {"justify": "center", "width": 6}),
            ("Description", {}),
        ],
    )
    pt.add_row("fast", "1", "4", "Maximum speed, good quality")
    pt.add_row("balanced", "3", "4", "Good balance (default)")
    pt.add_row("accurate", "5", "1", "Best quality, slower")
    console.print(pt)


# ── Download Command ────────────────────────────────────────


@click.command()
@click.argument(
    "model_name",
    type=click.Choice(
        [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v3",
            "large-v3-turbo",
        ]
    ),
)
def download(model_name: str) -> None:
    """Pre-download a Whisper model for offline use."""
    console.print(f"  [{ACCENT}]Downloading model:[/] {model_name} ...")

    try:
        from faster_whisper import WhisperModel

        WhisperModel(model_name, device="cpu", compute_type="int8")
        console.print(f"  [{SUCCESS}]✓[/] Model '{model_name}' downloaded and cached.")
    except Exception as e:
        console.print(error_panel(f"Download failed: {model_name}", str(e)))
        sys.exit(1)
