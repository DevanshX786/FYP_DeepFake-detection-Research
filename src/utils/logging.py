import json
import logging
import os
import sys
from typing import Any, Dict


def get_logger(name: str = "deepfake_detector", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger instance with console and optional file handlers.

    Args:
        name: Name of the logger.
        log_file: Path to optional log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricTracker:
    """Track and average metrics across batches during training/validation."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0


def save_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Save dictionary of metrics to JSON file with proper indentation.

    Args:
        metrics: Dictionary of metric values.
        filepath: Destination file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, default=str)
