from pathlib import Path


class DirectoryNavigator:
    def __init__(self, start_path="."):
        self.current_path = Path(start_path).resolve()
        self.selected = 0
        self.show_hidden = False
        self.selected_files = set()  # Set of Paths

    def toggle_hidden_visibility(self):
        """Toggle visibility of hidden files and directories"""
        self.show_hidden = not self.show_hidden
        self.selected = 0

    def list_items(self):
        """Return list of directories and files (directories first)"""
        try:
            all_items = []
            for item in Path(self.current_path).iterdir():
                if not self.show_hidden and item.name.startswith("."):
                    continue

                from audiobench.transcribe.audio_converter import ALL_SUPPORTED_FORMATS

                if item.is_file() and item.suffix.lstrip(".").lower() not in ALL_SUPPORTED_FORMATS:
                    continue

                all_items.append(item)

            # Sort: directories first, then files, both alphabetically
            sorted_items = sorted(all_items, key=lambda x: (not x.is_dir(), x.name.lower()))
            return [item.name for item in sorted_items]
        except PermissionError:
            return []

    def get_current_item_path(self):
        """Get the full path of the currently selected item"""
        items = self.list_items()
        if items and 0 <= self.selected < len(items):
            return self.current_path / items[self.selected]
        return None

    def enter(self):
        """Handle directory entry"""
        items = self.list_items()
        if not items:
            return

        selected_path = self.current_path / items[self.selected]

        if selected_path.is_dir():
            self.current_path = selected_path
            self.selected = 0

    def go_up(self):
        """Navigate to parent directory"""
        parent = self.current_path.parent
        if parent != self.current_path:
            self.current_path = parent
            self.selected = 0

    def select_next(self):
        """Move selection down"""
        items = self.list_items()
        if items:
            self.selected = (self.selected + 1) % len(items)

    def select_prev(self):
        """Move selection up"""
        items = self.list_items()
        if items:
            self.selected = (self.selected - 1) % len(items)

    def toggle_selection(self):
        """Toggle file selection state (spacebar)"""
        current = self.get_current_item_path()
        if current and current.is_file():
            if current in self.selected_files:
                self.selected_files.remove(current)
            else:
                self.selected_files.add(current)

    def get_item_count_info(self):
        try:
            all_items = list(Path(self.current_path).iterdir())
            total_items = len(all_items)
            visible_items = len(self.list_items())
            hidden_items = total_items - visible_items

            return {"total": total_items, "visible": visible_items, "hidden": hidden_items}
        except PermissionError:
            return {"total": 0, "visible": 0, "hidden": 0}
