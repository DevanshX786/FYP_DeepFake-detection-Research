import os
from typing import Any, Dict, Optional
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.evaluate import evaluate_checkpoint
from src.utils.logging import get_logger, save_metrics_to_json


def evaluate_cross_dataset(
    model: nn.Module,
    celebdf_csv_path: str,
    checkpoint_path: str,
    output_dir: str = "experiments/cross_dataset_celebdf",
    batch_size: int = 8,
    num_frames: int = 16,
    device: Optional[torch.device] = None,
    model_name: str = "CDTC-Net",
) -> Dict[str, Any]:
    """Execute zero-shot Cross-Dataset generalization test on Celeb-DF v2.

    Guarantees:
    - Zero fine-tuning on Celeb-DF.
    - Evaluates out-of-domain generalization performance directly.

    Args:
        model: Model architecture.
        celebdf_csv_path: Path to Celeb-DF metadata/test CSV.
        checkpoint_path: Path to FF++ trained checkpoint (.pt).
        output_dir: Output directory for plots and JSON report.
        batch_size: Batch size.
        num_frames: Sampled frames per video (16).
        device: Torch device.
        model_name: Model identifier.

    Returns:
        Evaluation results dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = get_logger("cross_dataset", log_file=os.path.join(output_dir, "cross_dataset.log"))
    logger.info(f"=== Starting Cross-Dataset Generalization Test: FF++ -> Celeb-DF ({model_name}) ===")

    results = evaluate_checkpoint(
        model=model,
        test_csv_path=celebdf_csv_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        batch_size=batch_size,
        num_frames=num_frames,
        device=device,
        model_name=f"{model_name} (Zero-Shot Celeb-DF)",
    )

    logger.info(f"Cross-Dataset Celeb-DF ROC-AUC: {results['overall_metrics']['roc_auc']:.4f}")
    logger.info(f"Cross-Dataset Celeb-DF F1-Score: {results['overall_metrics']['f1']:.4f}")
    return results
