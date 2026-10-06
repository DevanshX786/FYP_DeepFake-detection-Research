from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import calculate_metrics, calculate_per_manipulation_metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    progress_bar: bool = False,
    mixed_precision: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Dict[str, Any]]:
    """Evaluate model on a validation/test dataloader.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss criterion.
        device: Device to run evaluation on.
        threshold: Decision threshold for classification.
        progress_bar: Whether to display a tqdm progress bar.
        mixed_precision: Whether to use FP16 automatic mixed precision.

    Returns:
        Tuple of (metrics_dict, y_true, y_probs, per_method_metrics).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []
    all_methods = []

    iterator = tqdm(dataloader, desc="Validating", leave=False) if progress_bar else dataloader

    for batch in iterator:
        rgb = batch["rgb"].to(device, non_blocking=True)
        freq = batch["freq"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        methods = batch.get("manipulation_method", ["unknown"] * len(labels))

        # Forward pass with AMP
        with torch.amp.autocast('cuda', enabled=mixed_precision and device.type == "cuda"):
            logits = model(rgb, freq).squeeze(-1)
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_methods.extend(methods if isinstance(methods, list) else list(methods))

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    metrics = calculate_metrics(y_true, y_probs, threshold=threshold)
    metrics["val_loss"] = avg_loss

    per_method_metrics = calculate_per_manipulation_metrics(y_true, y_probs, all_methods, threshold=threshold)

    return metrics, y_true, y_probs, per_method_metrics
