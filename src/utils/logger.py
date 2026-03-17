"""
TensorBoard + CSV logging. Stable metrics CSV schema for later plotting.
"""
from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


# CSV columns (stable schema for plot_results.py later)
CSV_COLUMNS = [
    "global_step",
    "episode",
    "train_return",
    "train_episode_length",
    "loss",
    "epsilon",
    "eval_mean_return",
    "eval_std_return",
]


class RunLogger:
    """Writes metrics to TensorBoard and a single metrics.csv per run."""

    def __init__(self, run_dir: Path, csv_flush_interval: int = 1, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_flush_interval = csv_flush_interval
        mode = "a" if resume else "w"
        self._csv_file = open(self.csv_path, mode, newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not resume:
            self._csv_writer.writeheader()
        self._csv_file.flush()
        self._row_count = 0

    def log_metrics(self, metrics: dict, global_step: int):
        """Write one row to CSV and scalar metrics to TensorBoard."""
        row = {k: v for k, v in metrics.items() if k in CSV_COLUMNS}
        if "global_step" not in row:
            row["global_step"] = global_step
        self._csv_writer.writerow(row)
        self._row_count += 1
        if self._row_count % self.csv_flush_interval == 0:
            self._csv_file.flush()
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, global_step)

    def close(self):
        self._csv_file.flush()
        self._csv_file.close()
        self.writer.close()
