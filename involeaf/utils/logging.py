"""Run logging. Every number the report cites comes from these JSON files."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def get_logger(name: str = "involeaf") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class RunLogger:
    """Accumulates one run's record and writes it atomically at the end.

    Results are never hand-copied into the report; ``scripts/make_tables.py`` reads
    these files instead.
    """

    def __init__(self, run_name: str, config: dict, device_info: dict,
                 results_dir: Path | None = None) -> None:
        self.run_name = run_name
        self.dir = Path(results_dir or RESULTS_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "run_name": run_name,
            "config": config,
            "device": device_info,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epochs": [],
        }
        self.log = get_logger()

    def log_epoch(self, epoch: int, **metrics: Any) -> None:
        row = {"epoch": epoch, **metrics}
        self.record["epochs"].append(row)
        pretty = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self.log.info(f"[{self.run_name}] epoch {epoch:>3}  {pretty}")

    def set(self, key: str, value: Any) -> None:
        self.record[key] = value

    def save(self) -> Path:
        self.record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        path = self.dir / f"{self.run_name}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.record, indent=2), encoding="utf-8")
        tmp.replace(path)
        self.log.info(f"[{self.run_name}] results written to {path}")
        return path
