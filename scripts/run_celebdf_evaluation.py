import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.evaluation.metrics import calculate_metrics, calculate_per_manipulation_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from src.training.losses import get_loss_criterion
from src.utils.checkpointing import CheckpointManager
from src.utils.logging import get_logger, save_metrics_to_json
from scripts.run_single_experiment import build_model


EXPERIMENTS = [
    {"exp_id": "EXP-1", "name": "ResNet-50 + BiLSTM", "chk_path": "experiments/exp_1/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "EXP-2", "name": "ConvNeXt RGB Spatial-only", "chk_path": "experiments/exp_2/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "EXP-3", "name": "ConvNeXt RGB + Transformer", "chk_path": "experiments/exp_3/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "EXP-4", "name": "Frequency-only (2D-FFT)", "chk_path": "experiments/exp_4/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "EXP-5", "name": "Full Proposed CDTC-Net", "chk_path": "experiments/exp_5/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "ABL-1", "name": "Dual-Domain + Avg Pooling", "chk_path": "experiments/abl_1/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "ABL-2", "name": "Dual-Domain + Transformer (No Diff)", "chk_path": "experiments/abl_2/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "ABL-3", "name": "Signed-Only Diff Tokens", "chk_path": "experiments/abl_3/checkpoints/best_model.pt", "freq_method": "fft"},
    {"exp_id": "ABL-4", "name": "CDTC-Net with 2D-DCT", "chk_path": "experiments/abl_4/checkpoints/best_model.pt", "freq_method": "dct"},
]


def evaluate_single_celebdf(
    exp_info: dict,
    test_csv_path: str = "data/splits/celebdf_test.csv",
    batch_size: int = 8,
    num_frames: int = 16,
    device: torch.device = None,
    threshold: float = 0.5,
) -> dict:
    exp_id = exp_info["exp_id"]
    chk_path = exp_info["chk_path"]
    freq_method = exp_info["freq_method"]

    exp_dir_name = exp_id.lower().replace("-", "_")
    output_dir = f"experiments/{exp_dir_name}/cross_dataset/celebdf"
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(f"celebdf_{exp_id}", log_file=os.path.join(output_dir, "evaluate.log"))
    logger.info(f"==================================================================")
    logger.info(f"    ZERO-SHOT CELEB-DF EVALUATION: {exp_id} ({exp_info['name']})   ")
    logger.info(f"==================================================================")
    logger.info(f"Checkpoint: {chk_path}")

    if not os.path.exists(chk_path):
        raise FileNotFoundError(f"Checkpoint not found: {chk_path}")

    # Build model architecture
    model, model_freq = build_model(exp_id)
    chk_mgr = CheckpointManager(checkpoint_dir=os.path.dirname(chk_path))
    loaded_chk = chk_mgr.load(model=model, checkpoint_path=chk_path, device=device)
    model = model.to(device)
    model.eval()

    best_val_epoch = loaded_chk.get("epoch", "unknown")
    logger.info(f"Successfully loaded checkpoint from epoch {best_val_epoch}")

    # Dataset & DataLoader
    df = pd.read_csv(test_csv_path)
    dataset = DeepfakeVideoDataset(
        dataframe=df,
        num_frames=num_frames,
        sampling_strategy="uniform",
        freq_method=freq_method,
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
        for batch in tqdm(test_loader, desc=f"Evaluating {exp_id} on Celeb-DF v2"):
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

    logger.info(f"Accuracy:          {metrics['accuracy']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}")
    logger.info(f"Precision:         {metrics['precision']:.4f}")
    logger.info(f"Recall:            {metrics['recall']:.4f}")
    logger.info(f"F1-score:          {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:           {metrics['roc_auc']:.4f}")

    # Generate plots
    plot_confusion_matrix(y_true, y_probs, save_path=os.path.join(output_dir, "confusion_matrix.png"), threshold=threshold, title=f"Celeb-DF v2 Confusion Matrix — {exp_id}")
    plot_roc_curve(y_true, y_probs, save_path=os.path.join(output_dir, "roc_curve.png"), model_name=f"{exp_id} (Celeb-DF v2)")
    plot_precision_recall_curve(y_true, y_probs, save_path=os.path.join(output_dir, "precision_recall_curve.png"), model_name=f"{exp_id} (Celeb-DF v2)")

    full_results = {
        "exp_id": exp_id,
        "model_name": exp_info["name"],
        "checkpoint_path": chk_path,
        "checkpoint_epoch": best_val_epoch,
        "test_dataset": test_csv_path,
        "dataset_name": "Celeb-DF v2 (Zero-Shot)",
        "overall_metrics": metrics,
        "per_manipulation_method_metrics": per_method,
    }
    save_metrics_to_json(full_results, os.path.join(output_dir, "test_metrics.json"))

    # Also save to centralized cross_dataset directory
    central_dir = f"experiments/cross_dataset/celebdf/{exp_dir_name}"
    os.makedirs(central_dir, exist_ok=True)
    save_metrics_to_json(full_results, os.path.join(central_dir, "test_metrics.json"))

    return full_results


def run_all_celebdf_evaluations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running zero-shot Celeb-DF v2 evaluations on {device}...")

    summary_records = []
    central_summary_dir = "experiments/cross_dataset/celebdf"
    os.makedirs(central_summary_dir, exist_ok=True)

    for exp in EXPERIMENTS:
        print(f"\n>>> Running Celeb-DF evaluation for {exp['exp_id']} ({exp['name']})...")
        res = evaluate_single_celebdf(exp, device=device)
        ov = res["overall_metrics"]

        summary_records.append({
            "exp_id": exp["exp_id"],
            "model_name": exp["name"],
            "checkpoint_epoch": res.get("checkpoint_epoch", "unknown"),
            "accuracy": round(ov["accuracy"], 4),
            "balanced_accuracy": round(ov.get("balanced_accuracy", 0.0), 4),
            "precision": round(ov["precision"], 4),
            "recall": round(ov["recall"], 4),
            "f1": round(ov["f1"], 4),
            "roc_auc": round(ov["roc_auc"], 4),
            "tn": ov["true_negatives"],
            "fp": ov["false_positives"],
            "fn": ov["false_negatives"],
            "tp": ov["true_positives"],
            "test_loss": round(ov["test_loss"], 4),
        })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(central_summary_dir, "summary_table.csv"), index=False)
    
    with open(os.path.join(central_summary_dir, "cross_dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=4)

    print("\n==================================================================")
    print("      CELEB-DF V2 ZERO-SHOT CROSS-DATASET EVALUATION COMPLETE     ")
    print("==================================================================")
    print(summary_df.to_string(index=False))
    print("==================================================================")


if __name__ == "__main__":
    run_all_celebdf_evaluations()
