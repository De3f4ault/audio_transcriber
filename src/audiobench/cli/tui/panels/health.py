"""HealthPanel — disk usage with btop-style gradient fill bars."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.widgets import Static


# ── Gradient bar helpers ───────────────────────────────────────────────────────

_BAR_FILLED = "▮"
_BAR_EMPTY  = "░"
_BAR_WIDTH  = 14


def _bar(used: int, total: int) -> Text:
    """Return a gradient fill bar: ▮▮▮▮░░░░ with colour by fill %."""
    if total <= 0:
        pct = 0.0
    else:
        pct = min(used / total, 1.0)

    filled = int(pct * _BAR_WIDTH)
    empty  = _BAR_WIDTH - filled

    # Interpolate green→amber→red
    if pct < 0.6:
        color = "#00e676"
    elif pct < 0.85:
        color = "#ffb300"
    else:
        color = "#ef5350"

    t = Text(no_wrap=True)
    t.append(_BAR_FILLED * filled, style=color)
    t.append(_BAR_EMPTY  * empty,  style="#37474f")
    return t


def _size_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _fmt(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b //= 1024
    return f"{b:.1f} TB"


class HealthPanel(Static):
    """System health panel with gradient fill bars, refreshes every 15s."""

    _DB_WARN_MB   = 512    # warn bar colour threshold
    _DB_CRIT_MB   = 2048   # critical

    def on_mount(self) -> None:
        self.set_interval(15.0, self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.update(self._build())
        except Exception as exc:
            self.update(Text(f"Health unavailable: {exc}", style="#ef5350"))

    def _build(self) -> Text:
        from audiobench.core.settings import get_settings
        settings = get_settings()

        db_path      = Path(settings.database_url.replace("sqlite:///", ""))
        journal_path = settings.data_dir / "journal.db"
        models_dir   = settings.models_dir
        logs_dir     = settings.data_dir / "logs"

        db_bytes      = _size_bytes(db_path)
        journal_bytes = _size_bytes(journal_path)
        model_bytes   = _size_bytes(models_dir)
        log_bytes     = _size_bytes(logs_dir)

        # Use 2 GB as the "full" scale for bars (arbitrary upper bound)
        SCALE = 2 * 1024 * 1024 * 1024

        t = Text(no_wrap=True)

        def row(label: str, used: int, scale: int = SCALE) -> None:
            t.append(f"  {label:<18}", style="bold #b0bec5")
            t.append_text(_bar(used, scale))
            t.append(f"  {_fmt(used)}\n", style="#90a4ae")

        t.append("  Disk\n", style="bold #64b5f6")
        row("transcriptions.db", db_bytes)
        row("journal.db",        journal_bytes)
        row("model cache",       model_bytes)
        row("logs/",             log_bytes)

        # LanceDB vectors
        try:
            import lancedb
            ldb_path = settings.data_dir / "lancedb"
            if ldb_path.exists():
                ldb = lancedb.connect(str(ldb_path))
                total = sum(
                    ldb.open_table(t_name).count_rows()
                    for t_name in ldb.table_names()
                )
                t.append("\n  Vectors\n", style="bold #64b5f6")
                # Scale bar by 100k rows = "full"
                vec_bar = _bar(total, 100_000)
                t.append(f"  {'LanceDB rows':<18}", style="bold #b0bec5")
                t.append_text(vec_bar)
                t.append(f"  {total:,}\n", style="#90a4ae")
        except Exception:
            pass

        return t
