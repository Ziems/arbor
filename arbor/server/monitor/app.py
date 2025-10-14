from typing import List

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Header, Static

from .mock_data import MockJob, generate_mock_jobs
from .widgets import JobTable


class ArborMonitorApp(App):
    """Main application for monitoring Arbor jobs"""

    CSS = """
    Screen {
        layout: vertical;
    }

    .header-info {
        height: 1;
        padding: 0 1;
        background: $primary;
        color: $text;
        text-align: center;
    }

    JobTable {
        height: 1fr;
        margin: 1;
        border: solid $primary;
    }

    .status-container {
        height: 3;
        background: $surface;
        border: solid $primary;
        margin: 0 1;
    }

    .status-line {
        height: 1;
        padding: 0 1;
        background: $surface;
    }

    .help-line {
        height: 1;
        padding: 0 1;
        background: $panel;
        text-align: center;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.jobs: List[MockJob] = generate_mock_jobs()
        self.selected_job_id = 1

    def compose(self) -> ComposeResult:
        # TOP: header
        yield Header()
        yield Static(f"Jobs ({len(self.jobs)} running)", classes="header-info")

        # MIDDLE: shows list of jobs w/ info
        job_table = JobTable()
        job_table.id = "job_table"
        yield job_table

        # BOTTOM: shows selected job summary info
        with Container(classes="status-container"):
            yield Static(
                f"Selected: {self._get_selected_job_name()}",
                classes="status-line",
                id="status_line",
            )
            yield Static("[↑↓] Navigate [q] Quit", classes="help-line")

    def on_mount(self):
        """Initialize the table with mock data"""
        table = self.query_one(JobTable)  # Can query by class instead of id
        table.populate_jobs(self.jobs)
        table.focus()  # Focus the table for keyboard navigation

    @on(DataTable.RowHighlighted)
    def row_highlighted(self, event: DataTable.RowHighlighted):
        """Handle row highlighting (cursor movement)"""
        if event.cursor_row is not None:
            # DataTable rows are 0-indexed, our job IDs are 1-indexed
            self.selected_job_id = event.cursor_row + 1
            self._update_status_line()

    def _update_status_line(self):
        """Update the selected job display"""
        status_line = self.query_one("#status_line", Static)
        selected_job = self._get_selected_job()
        if selected_job:
            status_text = f"Selected: {selected_job.name} (Job ID: {selected_job.job_id}) - {selected_job.status}"
        else:
            status_text = "Selected: None"
        status_line.update(status_text)

    def _get_selected_job_name(self) -> str:
        """Get the name of the currently selected job"""
        job = self._get_selected_job()
        return job.name if job else "None"

    def _get_selected_job(self) -> MockJob | None:
        """Get the currently selected job object"""
        if 1 <= self.selected_job_id <= len(self.jobs):
            return self.jobs[self.selected_job_id - 1]
        return None
