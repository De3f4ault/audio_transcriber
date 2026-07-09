from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, DataTable
from textual.binding import Binding

from audiobench.daemon.client import DaemonClient


class InferencesFeed(Static):
    """Observatory panel for viewing and actioning system inferences.

    Polling: refreshes every 10s via the Observatory app's tick.
    Key bindings:
        [c]     Confirm selected inference → calls DaemonClient.confirm_inference()
        [r]     Reject selected inference  → calls DaemonClient.reject_inference()
        [Enter] Expand full content in a notify popup

    The selected row's expression_id is tracked in _selected_id so action
    methods can call the daemon without re-querying the table widget.
    """

    BINDINGS = [
        Binding("c", "confirm_inference", "Confirm Inference", key_display="C"),
        Binding("r", "reject_inference", "Reject Inference", key_display="R"),
        Binding("enter", "expand_content", "View Details", key_display="Enter"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = DaemonClient()
        self._selected_id: int | None = None
        self._rows: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("System Inferences", id="inferences_header"),
            DataTable(id="inferences_table")
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("ID", "Type", "Content Preview")
        table.cursor_type = "row"
        self._refresh_feed()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the selected row's expression_id."""
        if event.row_key is not None:
            try:
                self._selected_id = int(str(event.row_key.value))
            except (ValueError, TypeError):
                self._selected_id = None

    def _refresh_feed(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        try:
            self._rows = self._client.get_inferences()
        except Exception:
            self._rows = []

        for inf in self._rows:
            preview = (inf.get("content") or "")[:80].replace("\n", " ")
            table.add_row(
                str(inf["id"]),
                inf.get("source_type", ""),
                preview,
                key=str(inf["id"]),
            )

        if self._rows and self._selected_id is None:
            self._selected_id = self._rows[0]["id"]

    def action_confirm_inference(self) -> None:
        if self._selected_id is None:
            self.notify("No inference selected", severity="warning")
            return
        try:
            self._client.confirm_inference(self._selected_id)
            self.notify(f"Confirmed inference #{self._selected_id}", timeout=2)
            self._refresh_feed()
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def action_reject_inference(self) -> None:
        if self._selected_id is None:
            self.notify("No inference selected", severity="warning")
            return
        try:
            self._client.reject_inference(self._selected_id)
            self.notify(f"Rejected inference #{self._selected_id}", timeout=2)
            self._refresh_feed()
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def action_expand_content(self) -> None:
        if self._selected_id is None:
            return
        for inf in self._rows:
            if inf["id"] == self._selected_id:
                self.notify(inf.get("content", ""), timeout=8)
                return
