import csv
import os
import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.validate import validate_epoch
from src.utils.checkpointing import CheckpointManager
from src.utils.logging import get_logger, MetricTracker


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epochs: int = 30,
    device: Optional[torch.device] = None,
    mixed_precision: bool = True,
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 5,
    checkpoint_manager: Optional[CheckpointManager] = None,
    log_dir: str = "experiments/default",
    logger=None,
) -> Dict[str, Any]:
    """Execute complete model training and validation workflow.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training split.
        val_loader: DataLoader for validation split.
        criterion: Loss function module.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning rate scheduler.
        epochs: Maximum number of epochs.
        device: Torch device (CUDA/CPU).
        mixed_precision: Whether to use FP16 automatic mixed precision.
        gradient_clip_val: Maximum gradient norm for clipping.
        early_stopping_patience: Number of epochs without improvement before stopping.
        checkpoint_manager: Instance of CheckpointManager.
        log_dir: Experiment output directory.
        logger: Configured logger.

    Returns:
        Dictionary containing summary training results and best metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if logger is None:
        logger = get_logger("training", log_file=os.path.join(log_dir, "train.log"))

    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Starting training for {epochs} epochs on {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")

    model = model.to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision and device.type == "cuda")

    os.makedirs(log_dir, exist_ok=True)
    csv_log_path = os.path.join(log_dir, "training_log.csv")
    csv_fields = ["epoch", "train_loss", "val_loss", "val_acc", "val_bal_acc", "val_f1", "val_auc", "lr", "epoch_time_s"]

    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    patience_counter = 0
    best_val_auc = float("-inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_tracker = MetricTracker()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for batch in pbar:
            rgb = batch["rgb"].to(device, non_blocking=True)
            freq = batch["freq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=mixed_precision and device.type == "cuda"):
                logits = model(rgb, freq).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            if gradient_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

            scaler.step(optimizer)
            scaler.update()

            train_loss_tracker.update(loss.item(), n=len(labels))
            pbar.set_postfix({"loss": f"{train_loss_tracker.avg:.4f}"})

        # Run Validation
        val_metrics, _, _, _ = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            progress_bar=False,
            mixed_precision=mixed_precision,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["val_loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss_tracker.avg:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Bal Acc: {val_metrics.get('balanced_accuracy', 0.0):.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Write to CSV log
        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss_tracker.avg,
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["accuracy"],
                "val_bal_acc": val_metrics.get("balanced_accuracy", 0.0),
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["roc_auc"],
                "lr": current_lr,
                "epoch_time_s": epoch_time,
            })

        # Checkpoint Saving & Early Stopping
        is_best = False
        if checkpoint_manager is not None:
            is_best = checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
            )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {patience_counter} epochs).")
                break

    logger.info(f"Training completed. Best Val AUC: {best_val_auc:.4f}")
    return {"best_metrics": best_metrics, "best_val_auc": best_val_auc}
