import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.evaluation.metrics import calculate_metrics, calculate_per_manipulation_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from src.training.losses import get_loss_criterion
from src.utils.checkpointing import CheckpointManager
from src.utils.logging import get_logger, save_metrics_to_json


def evaluate_checkpoint(
    model: nn.Module,
    test_csv_path: str,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "experiments/evaluation",
    batch_size: int = 8,
    num_frames: int = 16,
    device: Optional[torch.device] = None,
    model_name: str = "CDTC-Net",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a trained model checkpoint on a test split CSV.

    Args:
        model: PyTorch model architecture.
        test_csv_path: Path to test split CSV.
        checkpoint_path: Path to model weights (.pt). If None, model is evaluated as is.
        output_dir: Directory where test metrics and plots will be saved.
        batch_size: Batch size for evaluation.
        num_frames: Number of sampled frames.
        device: Torch device.
        model_name: Name of model for plotting and logs.
        threshold: Decision threshold.

    Returns:
        Evaluation summary metrics dictionary.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger("evaluation", log_file=os.path.join(output_dir, "evaluate.log"))

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        chk_mgr = CheckpointManager(checkpoint_dir=os.path.dirname(checkpoint_path))
        chk_mgr.load(model=model, checkpoint_path=checkpoint_path, device=device)

    model = model.to(device)
    model.eval()

    df = pd.read_csv(test_csv_path)
    logger.info(f"Evaluating {model_name} on {len(df)} test samples from: {test_csv_path}")

    dataset = DeepfakeVideoDataset(
        dataframe=df,
        num_frames=num_frames,
        sampling_strategy="uniform",
        cache_dir="data/processed/face_crops",
    )
    test_loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, seed=42)

    criterion = get_loss_criterion("BCEWithLogitsLoss")

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_probs = []
    all_methods = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            freq = batch["freq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            methods = batch.get("manipulation_method", ["unknown"] * len(labels))

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
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
    metrics["test_loss"] = avg_loss
    per_method = calculate_per_manipulation_metrics(y_true, y_probs, all_methods, threshold=threshold)

    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall:    {metrics['recall']:.4f}")
    logger.info(f"Test F1-score:  {metrics['f1']:.4f}")
    logger.info(f"Test ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Generate plots
    plot_confusion_matrix(y_true, y_probs, save_path=os.path.join(output_dir, "confusion_matrix.png"), threshold=threshold, title=f"Confusion Matrix — {model_name}")
    plot_roc_curve(y_true, y_probs, save_path=os.path.join(output_dir, "roc_curve.png"), model_name=model_name)
    plot_precision_recall_curve(y_true, y_probs, save_path=os.path.join(output_dir, "precision_recall_curve.png"), model_name=model_name)

    # Save metrics JSON
    full_results = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "test_dataset": test_csv_path,
        "overall_metrics": metrics,
        "per_manipulation_method_metrics": per_method,
    }
    save_metrics_to_json(full_results, os.path.join(output_dir, "test_metrics.json"))
    logger.info(f"Evaluation complete. Artifacts saved to: {output_dir}")

    return full_results
