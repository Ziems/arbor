from typing import List

from rich.text import Text
from textual.widgets import DataTable

from .mock_data import MockJob


class JobTable(DataTable):
    """Table widget for displaying Arbor jobs"""

    def __init__(self):
        super().__init__(cursor_type="row", zebra_stripes=True)
        self.add_columns("ID", "Job ID", "Name", "Progress", "Runtime", "GPU Usage")

    def populate_jobs(self, jobs: List[MockJob]):
        """Fill table with job data"""
        # Clear existing rows
        if self.row_count > 0:
            for row_key in list(self.rows.keys()):
                self.remove_row(row_key)

        # Add rows
        rows_data = []
        for job in jobs:
            rows_data.append(
                [
                    str(job.id),
                    job.job_id,
                    self._truncate_name(job.name),
                    self._format_progress(job.progress_percent),
                    self._format_runtime(job.runtime_seconds),
                    self._format_gpu_usage(job.gpu_ids, job.gpu_utilization),
                ]
            )

        self.add_rows(rows_data)

    def _truncate_name(self, name: str, max_length: int = 15) -> str:
        """Truncate long job names"""
        if len(name) > max_length:
            return name[: max_length - 2] + ".."
        return name

    def _format_progress(self, percent: int) -> Text:
        """Create progress bar: ████░ 45%"""
        filled = "█" * (percent // 10)
        empty = "░" * (10 - (percent // 10))
        return Text(f"{filled}{empty} {percent}%")

    def _format_runtime(self, seconds: int) -> str:
        """Convert seconds to human readable: 2h 15m"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def _format_gpu_usage(self, gpu_ids: List[int], utilization: int) -> str:
        """Format: GPU 1,2: 85%"""
        gpu_str = ",".join(map(str, gpu_ids))
        return f"GPU {gpu_str}: {utilization}%"
