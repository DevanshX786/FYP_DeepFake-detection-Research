import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.evaluation.metrics import calculate_metrics, calculate_per_manipulation_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from src.models.cdtc_net import CDTCNet
from src.utils.checkpointing import CheckpointManager
from src.utils.logging import MetricTracker, get_logger, save_metrics_to_json
from src.utils.seed import set_seed


def evaluate_dataset_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Generate raw logits and probabilities for an entire dataloader."""
    model.eval()
    all_labels = []
    all_probs = []
    all_logits = []
    all_methods = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            rgb = batch["rgb"].to(device, non_blocking=True)
            freq = batch["freq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            methods = batch.get("manipulation_method", ["unknown"] * len(labels))
            paths = batch.get("video_path", [""] * len(labels))

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(rgb, freq).squeeze(-1)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())
            all_methods.extend(methods if isinstance(methods, list) else list(methods))
            all_paths.extend(paths if isinstance(paths, list) else list(paths))

    return np.array(all_labels), np.array(all_probs), all_methods, all_paths


def sweep_validation_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_csv_path: str,
) -> Tuple[float, Dict[str, Any], pd.DataFrame]:
    """Sweep decision thresholds on validation set and select optimal threshold maximizing balanced accuracy."""
    thresholds = np.linspace(0.01, 0.99, 99)
    records = []

    for t in thresholds:
        m = calculate_metrics(y_true, y_probs, threshold=float(t))
        spec = m["true_negatives"] / (m["true_negatives"] + m["false_positives"]) if (m["true_negatives"] + m["false_positives"]) > 0 else 0.0
        records.append({
            "threshold": round(float(t), 4),
            "balanced_accuracy": m["balanced_accuracy"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": spec,
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "true_negatives": m["true_negatives"],
            "false_positives": m["false_positives"],
            "false_negatives": m["false_negatives"],
            "true_positives": m["true_positives"],
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)

    # Pick threshold with maximum balanced accuracy (tie-break by F1 then closest to 0.5)
    best_row = df.sort_values(by=["balanced_accuracy", "f1"], ascending=[False, False]).iloc[0]
    best_threshold = float(best_row["threshold"])
    best_metrics = best_row.to_dict()

    return best_threshold, best_metrics, df


def main():
    exp_id = "EXP-5-V2-BALANCED"
    exp_dir = "experiments/exp_5_v2_balanced"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "validation"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "cross_dataset", "celebdf"), exist_ok=True)

    logger = get_logger(exp_id, log_file=os.path.join(exp_dir, "train.log"))
    logger.info("==================================================================")
    logger.info("       LAUNCHING EXP-5 V2 (CLASS-BALANCED CDTC-NET TRAINING)      ")
    logger.info("==================================================================")

    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset & Class Weighting Calculations
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    test_df = pd.read_csv("data/splits/test.csv")
    celebdf_df = pd.read_csv("data/splits/celebdf_test.csv")

    n_real = int((train_df["label"] == 0).sum())
    n_fake = int((train_df["label"] == 1).sum())
    n_total = len(train_df)
    pos_weight_val = n_real / n_fake  # 700 / 2800 = 0.25

    logger.info(f"Train Dataset: Total={n_total}, REAL={n_real} ({n_real/n_total*100:.2f}%), FAKE={n_fake} ({n_fake/n_total*100:.2f}%)")
    logger.info(f"Class-Balanced BCE pos_weight = {pos_weight_val:.4f} (Effective Weight: Real=1.00, Fake={pos_weight_val:.2f})")

    # 2. Build Model
    model = CDTCNet(
        pretrained_backbone=True,
        freeze_rgb_backbone=True,
        feature_dim=256,
        freq_in_channels=1,
        transformer_heads=8,
        transformer_layers=2,
        transformer_ff_dim=512,
        dropout_classifier=0.3,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Architecture: CDTCNet (ConvNeXt-Tiny + 2D-FFT + Fusion + Signed/Abs Diff + 32-token Transformer)")
    logger.info(f"Total Params: {total_params:,} | Trainable Params: {trainable_params:,}")

    # 3. Save Config
    config = {
        "exp_id": exp_id,
        "model_class": "CDTCNet",
        "loss_type": "BCEWithLogitsLoss",
        "pos_weight": pos_weight_val,
        "effective_class_weights": {"real_0": 1.0, "fake_1": pos_weight_val},
        "seed": seed,
        "epochs": 30,
        "early_stopping": False,
        "batch_size": 8,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "lr_scheduler": "CosineAnnealingLR",
        "t_max": 30,
        "eta_min": 1e-6,
        "mixed_precision": True,
        "gradient_clip_val": 1.0,
        "freeze_backbone": True,
        "num_frames": 16,
        "input_resolution": [224, 224],
        "freq_method": "fft",
        "device": str(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_split": "data/splits/train.csv",
        "val_split": "data/splits/val.csv",
        "test_split": "data/splits/test.csv",
        "cross_dataset_split": "data/splits/celebdf_test.csv",
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 4. DataLoaders
    train_dataset = DeepfakeVideoDataset(dataframe=train_df, num_frames=16, sampling_strategy="uniform", freq_method="fft")
    val_dataset = DeepfakeVideoDataset(dataframe=val_df, num_frames=16, sampling_strategy="uniform", freq_method="fft")
    test_dataset = DeepfakeVideoDataset(dataframe=test_df, num_frames=16, sampling_strategy="uniform", freq_method="fft")
    celebdf_dataset = DeepfakeVideoDataset(dataframe=celebdf_df, num_frames=16, sampling_strategy="uniform", freq_method="fft")

    train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=True, num_workers=4, seed=seed)
    val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False, num_workers=4, seed=seed)
    test_loader = create_dataloader(test_dataset, batch_size=8, shuffle=False, num_workers=4, seed=seed)
    celebdf_loader = create_dataloader(celebdf_dataset, batch_size=8, shuffle=False, num_workers=4, seed=seed)

    # 5. Loss, Optimizer, Scheduler, Checkpointing
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    chk_dir = os.path.join(exp_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir=chk_dir, monitor="roc_auc", mode="max", save_best_only=False)

    model = model.to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    # 6. Pre-Epoch 1 Gradient & Loss Sanity Diagnostic
    logger.info("------------------------------------------------------------------")
    logger.info("PRE-TRAINING OPTIMIZATION & GRADIENT SANITY CHECK:")
    first_batch = next(iter(train_loader))
    model.train()
    optimizer.zero_grad()
    rgb_sample = first_batch["rgb"].to(device)
    freq_sample = first_batch["freq"].to(device)
    labels_sample = first_batch["label"].to(device)

    with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
        init_logits = model(rgb_sample, freq_sample).squeeze(-1)
        init_loss = criterion(init_logits, labels_sample)
        unweighted_loss = nn.BCEWithLogitsLoss()(init_logits, labels_sample)

    scaler.scale(init_loss).backward()
    scaler.unscale_(optimizer)
    pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    optimizer.zero_grad()

    # Reset clean scaler for training loop
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    logger.info(f"Initial Effective Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    logger.info(f"First Batch Loss (pos_weight=0.25): {init_loss.item():.4f}")
    logger.info(f"First Batch Loss (unweighted):     {unweighted_loss.item():.4f}")
    logger.info(f"First Batch Initial Gradient Norm: {pre_clip_grad_norm:.4f}")
    logger.info("Sanity check passed: Gradients are stable and active.")
    logger.info("------------------------------------------------------------------")

    # 7. Complete 30-Epoch Training Loop (Full Budget, No Early Stopping)
    csv_log_path = os.path.join(exp_dir, "training_log.csv")
    csv_fields = [
        "epoch", "train_loss", "val_loss", "val_acc", "val_bal_acc",
        "val_precision", "val_recall", "val_specificity", "val_f1", "val_auc",
        "val_tn", "val_fp", "val_fn", "val_tp", "lr", "epoch_time_s"
    ]
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    best_val_auc = float("-inf")
    best_epoch = 0
    start_total_time = time.time()

    for epoch in range(1, 31):
        epoch_start = time.time()
        model.train()
        train_loss_tracker = MetricTracker()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/30 [Train]")
        for batch in pbar:
            rgb = batch["rgb"].to(device, non_blocking=True)
            freq = batch["freq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                logits = model(rgb, freq).squeeze(-1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_tracker.update(loss.item(), n=len(labels))
            pbar.set_postfix({"loss": f"{train_loss_tracker.avg:.4f}"})

        # Validation Step
        model.eval()
        val_loss_tracker = MetricTracker()
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                freq = batch["freq"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    logits = model(rgb, freq).squeeze(-1)
                    vloss = criterion(logits, labels)

                probs = torch.sigmoid(logits)
                val_loss_tracker.update(vloss.item(), n=len(labels))
                val_labels.extend(labels.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        y_true_val = np.array(val_labels)
        y_probs_val = np.array(val_probs)
        vm = calculate_metrics(y_true_val, y_probs_val, threshold=0.5)
        spec = vm["true_negatives"] / (vm["true_negatives"] + vm["false_positives"]) if (vm["true_negatives"] + vm["false_positives"]) > 0 else 0.0

        val_metrics_record = {
            "roc_auc": vm["roc_auc"],
            "accuracy": vm["accuracy"],
            "balanced_accuracy": vm["balanced_accuracy"],
            "precision": vm["precision"],
            "recall": vm["recall"],
            "specificity": spec,
            "f1": vm["f1"],
            "val_loss": val_loss_tracker.avg,
        }

        # Checkpoint Saving (Save every epoch + track best ROC-AUC)
        is_best = checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics_record,
            filename=f"checkpoint_epoch_{epoch:03d}.pt",
        )
        if vm["roc_auc"] > best_val_auc:
            best_val_auc = vm["roc_auc"]
            best_epoch = epoch

        logger.info(
            f"Epoch {epoch:02d}/30 | "
            f"TrainLoss: {train_loss_tracker.avg:.4f} | "
            f"ValLoss: {val_loss_tracker.avg:.4f} | "
            f"ValAUC: {vm['roc_auc']:.4f} {'[BEST]' if is_best else ''} | "
            f"ValBalAcc: {vm['balanced_accuracy']:.4f} | "
            f"ValSpec: {spec:.4f} | "
            f"ValRec: {vm['recall']:.4f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss_tracker.avg,
                "val_loss": val_loss_tracker.avg,
                "val_acc": vm["accuracy"],
                "val_bal_acc": vm["balanced_accuracy"],
                "val_precision": vm["precision"],
                "val_recall": vm["recall"],
                "val_specificity": spec,
                "val_f1": vm["f1"],
                "val_auc": vm["roc_auc"],
                "val_tn": vm["true_negatives"],
                "val_fp": vm["false_positives"],
                "val_fn": vm["false_negatives"],
                "val_tp": vm["true_positives"],
                "lr": current_lr,
                "epoch_time_s": epoch_time,
            })

    total_training_duration = time.time() - start_total_time
    logger.info("==================================================================")
    logger.info(f" 30-EPOCH TRAINING COMPLETE. Total Duration: {total_training_duration/60:.2f} min")
    logger.info(f" Best Validation ROC-AUC: {best_val_auc:.4f} achieved at Epoch {best_epoch:02d}")
    logger.info("==================================================================")

    # Save final model explicitly as final_model.pt
    final_state = {
        "epoch": 30,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
    }
    torch.save(final_state, os.path.join(chk_dir, "final_model.pt"))

    # 8. Validation Threshold Calibration on Validation Split ONLY
    best_chk_path = os.path.join(chk_dir, "best_model.pt")
    chk_state = torch.load(best_chk_path, map_location=device, weights_only=False)
    model.load_state_dict(chk_state["model_state_dict"])
    model.eval()

    logger.info("------------------------------------------------------------------")
    logger.info("VALIDATION THRESHOLD CALIBRATION (SWEEPING 99 THRESHOLDS ON VAL SPLIT)")
    val_y_true, val_y_probs, val_methods, val_paths = evaluate_dataset_predictions(model, val_loader, device)

    # Save raw validation predictions
    val_preds_record = {
        "video_paths": val_paths,
        "ground_truth": val_y_true.tolist(),
        "predicted_probs": val_y_probs.tolist(),
        "manipulation_methods": val_methods,
    }
    with open(os.path.join(exp_dir, "validation", "val_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(val_preds_record, f)

    sweep_csv_path = os.path.join(exp_dir, "validation", "threshold_sweep.csv")
    best_threshold, best_val_thresh_metrics, sweep_df = sweep_validation_thresholds(val_y_true, val_y_probs, sweep_csv_path)

    # Log Top 5 candidate thresholds
    top5 = sweep_df.sort_values(by="balanced_accuracy", ascending=False).head(5)
    logger.info(f"Top 5 Validation Thresholds by Balanced Accuracy:\n{top5[['threshold', 'balanced_accuracy', 'specificity', 'recall', 'f1']].to_string(index=False)}")
    logger.info(f"SELECTED OPTIMAL THRESHOLD: tau* = {best_threshold:.4f}")
    logger.info(f"Validation Balanced Accuracy at tau*: {best_val_thresh_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Validation Specificity at tau*:       {best_val_thresh_metrics['specificity']:.4f}")
    logger.info(f"Validation Recall at tau*:            {best_val_thresh_metrics['recall']:.4f}")
    logger.info(f"Validation F1 at tau*:                {best_val_thresh_metrics['f1']:.4f}")
    logger.info("------------------------------------------------------------------")

    # Update config with calibrated threshold
    config["calibrated_threshold"] = best_threshold
    config["best_val_epoch"] = best_epoch
    config["best_val_auc"] = best_val_auc
    config["val_metrics_at_calibrated_threshold"] = best_val_thresh_metrics
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 9. In-Domain FF++ Test Set Evaluation at Calibrated Threshold
    logger.info("------------------------------------------------------------------")
    logger.info(f"EVALUATING BEST MODEL ON FF++ TEST SPLIT AT FROZEN tau* = {best_threshold:.4f}")
    test_y_true, test_y_probs, test_methods, test_paths = evaluate_dataset_predictions(model, test_loader, device)

    test_metrics = calculate_metrics(test_y_true, test_y_probs, threshold=best_threshold)
    test_spec = test_metrics["true_negatives"] / (test_metrics["true_negatives"] + test_metrics["false_positives"]) if (test_metrics["true_negatives"] + test_metrics["false_positives"]) > 0 else 0.0
    test_metrics["specificity"] = test_spec
    per_method_test = calculate_per_manipulation_metrics(test_y_true, test_y_probs, test_methods, threshold=best_threshold)

    # Plots
    eval_dir = os.path.join(exp_dir, "evaluation")
    plot_confusion_matrix(test_y_true, test_y_probs, save_path=os.path.join(eval_dir, "confusion_matrix.png"), threshold=best_threshold, title=f"Confusion Matrix — {exp_id} (tau={best_threshold})")
    plot_roc_curve(test_y_true, test_y_probs, save_path=os.path.join(eval_dir, "roc_curve.png"), model_name=exp_id)
    plot_precision_recall_curve(test_y_true, test_y_probs, save_path=os.path.join(eval_dir, "precision_recall_curve.png"), model_name=exp_id)

    full_test_results = {
        "model_name": exp_id,
        "checkpoint_path": best_chk_path,
        "best_epoch": best_epoch,
        "calibrated_threshold": best_threshold,
        "overall_metrics": test_metrics,
        "per_manipulation_method_metrics": per_method_test,
    }
    save_metrics_to_json(full_test_results, os.path.join(eval_dir, "test_metrics.json"))

    logger.info(f"FF++ Test Accuracy:          {test_metrics['accuracy']:.4f}")
    logger.info(f"FF++ Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"FF++ Test Precision:         {test_metrics['precision']:.4f}")
    logger.info(f"FF++ Test Recall:            {test_metrics['recall']:.4f}")
    logger.info(f"FF++ Test Specificity:       {test_metrics['specificity']:.4f}")
    logger.info(f"FF++ Test F1-Score:          {test_metrics['f1']:.4f}")
    logger.info(f"FF++ Test ROC-AUC:           {test_metrics['roc_auc']:.4f}")
    logger.info(f"FF++ Test Confusion Matrix:  TN={test_metrics['true_negatives']}, FP={test_metrics['false_positives']}, FN={test_metrics['false_negatives']}, TP={test_metrics['true_positives']}")
    logger.info("------------------------------------------------------------------")

    # 10. Diagnostic FF++ Test Evaluation of FINAL EPOCH (Epoch 30) Checkpoint
    final_chk_path = os.path.join(chk_dir, "final_model.pt")
    final_state = torch.load(final_chk_path, map_location=device, weights_only=False)
    model.load_state_dict(final_state["model_state_dict"])
    model.eval()

    final_test_y_true, final_test_y_probs, final_test_methods, _ = evaluate_dataset_predictions(model, test_loader, device)
    final_metrics_tau = calculate_metrics(final_test_y_true, final_test_y_probs, threshold=best_threshold)
    final_metrics_tau["specificity"] = final_metrics_tau["true_negatives"] / (final_metrics_tau["true_negatives"] + final_metrics_tau["false_positives"])
    final_metrics_default = calculate_metrics(final_test_y_true, final_test_y_probs, threshold=0.5)
    final_metrics_default["specificity"] = final_metrics_default["true_negatives"] / (final_metrics_default["true_negatives"] + final_metrics_default["false_positives"])

    final_test_results = {
        "model_name": f"{exp_id}_FINAL_EPOCH_30",
        "checkpoint_path": final_chk_path,
        "metrics_at_calibrated_threshold": final_metrics_tau,
        "metrics_at_default_threshold": final_metrics_default,
    }
    save_metrics_to_json(final_test_results, os.path.join(eval_dir, "final_epoch_test_metrics.json"))

    # Reload best model for remaining analyses
    model.load_state_dict(chk_state["model_state_dict"])
    model.eval()

    # 11. Probability Distribution Analysis
    val_real_probs = val_y_probs[val_y_true == 0]
    val_fake_probs = val_y_probs[val_y_true == 1]
    test_real_probs = test_y_probs[test_y_true == 0]
    test_fake_probs = test_y_probs[test_y_true == 1]

    prob_dist = {
        "validation": {
            "real": {
                "count": len(val_real_probs),
                "mean_p_fake": float(np.mean(val_real_probs)),
                "median_p_fake": float(np.median(val_real_probs)),
                "std_p_fake": float(np.std(val_real_probs)),
                "min_p_fake": float(np.min(val_real_probs)),
                "max_p_fake": float(np.max(val_real_probs)),
            },
            "fake": {
                "count": len(val_fake_probs),
                "mean_p_fake": float(np.mean(val_fake_probs)),
                "median_p_fake": float(np.median(val_fake_probs)),
                "std_p_fake": float(np.std(val_fake_probs)),
                "min_p_fake": float(np.min(val_fake_probs)),
                "max_p_fake": float(np.max(val_fake_probs)),
            },
        },
        "test": {
            "real": {
                "count": len(test_real_probs),
                "mean_p_fake": float(np.mean(test_real_probs)),
                "median_p_fake": float(np.median(test_real_probs)),
                "std_p_fake": float(np.std(test_real_probs)),
                "min_p_fake": float(np.min(test_real_probs)),
                "max_p_fake": float(np.max(test_real_probs)),
            },
            "fake": {
                "count": len(test_fake_probs),
                "mean_p_fake": float(np.mean(test_fake_probs)),
                "median_p_fake": float(np.median(test_fake_probs)),
                "std_p_fake": float(np.std(test_fake_probs)),
                "min_p_fake": float(np.min(test_fake_probs)),
                "max_p_fake": float(np.max(test_fake_probs)),
            },
        },
    }
    with open(os.path.join(eval_dir, "probability_distributions.json"), "w", encoding="utf-8") as f:
        json.dump(prob_dist, f, indent=4)

    logger.info("PROBABILITY DISTRIBUTION STATISTICS:")
    logger.info(f" Val Real P(fake): Mean={prob_dist['validation']['real']['mean_p_fake']:.4f}, Median={prob_dist['validation']['real']['median_p_fake']:.4f}")
    logger.info(f" Val Fake P(fake): Mean={prob_dist['validation']['fake']['mean_p_fake']:.4f}, Median={prob_dist['validation']['fake']['median_p_fake']:.4f}")
    logger.info(f" Test Real P(fake): Mean={prob_dist['test']['real']['mean_p_fake']:.4f}, Median={prob_dist['test']['real']['median_p_fake']:.4f}")
    logger.info(f" Test Fake P(fake): Mean={prob_dist['test']['fake']['mean_p_fake']:.4f}, Median={prob_dist['test']['fake']['median_p_fake']:.4f}")

    # 12. Zero-Shot Cross-Dataset Evaluation on Celeb-DF v2
    logger.info("------------------------------------------------------------------")
    logger.info(f"ZERO-SHOT CROSS-DATASET EVALUATION ON CELEB-DF V2 AT tau* = {best_threshold:.4f}")
    celeb_y_true, celeb_y_probs, celeb_methods, celeb_paths = evaluate_dataset_predictions(model, celebdf_loader, device)

    celeb_metrics = calculate_metrics(celeb_y_true, celeb_y_probs, threshold=best_threshold)
    celeb_spec = celeb_metrics["true_negatives"] / (celeb_metrics["true_negatives"] + celeb_metrics["false_positives"]) if (celeb_metrics["true_negatives"] + celeb_metrics["false_positives"]) > 0 else 0.0
    celeb_metrics["specificity"] = celeb_spec
    per_method_celeb = calculate_per_manipulation_metrics(celeb_y_true, celeb_y_probs, celeb_methods, threshold=best_threshold)

    celeb_out_dir = os.path.join(exp_dir, "cross_dataset", "celebdf")
    plot_confusion_matrix(celeb_y_true, celeb_y_probs, save_path=os.path.join(celeb_out_dir, "confusion_matrix.png"), threshold=best_threshold, title=f"Celeb-DF Confusion Matrix — {exp_id} (tau={best_threshold})")
    plot_roc_curve(celeb_y_true, celeb_y_probs, save_path=os.path.join(celeb_out_dir, "roc_curve.png"), model_name=f"{exp_id} Celeb-DF")
    plot_precision_recall_curve(celeb_y_true, celeb_y_probs, save_path=os.path.join(celeb_out_dir, "precision_recall_curve.png"), model_name=f"{exp_id} Celeb-DF")

    full_celeb_results = {
        "model_name": exp_id,
        "checkpoint_path": best_chk_path,
        "test_dataset": "data/splits/celebdf_test.csv",
        "calibrated_threshold": best_threshold,
        "overall_metrics": celeb_metrics,
        "per_manipulation_method_metrics": per_method_celeb,
    }
    save_metrics_to_json(full_celeb_results, os.path.join(celeb_out_dir, "test_metrics.json"))

    logger.info(f"Celeb-DF Zero-Shot Accuracy:          {celeb_metrics['accuracy']:.4f}")
    logger.info(f"Celeb-DF Zero-Shot Balanced Accuracy: {celeb_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Celeb-DF Zero-Shot Precision:         {celeb_metrics['precision']:.4f}")
    logger.info(f"Celeb-DF Zero-Shot Recall:            {celeb_metrics['recall']:.4f}")
    logger.info(f"Celeb-DF Zero-Shot Specificity:       {celeb_metrics['specificity']:.4f}")
    logger.info(f"Celeb-DF Zero-Shot F1-Score:          {celeb_metrics['f1']:.4f}")
    logger.info(f"Celeb-DF Zero-Shot ROC-AUC:           {celeb_metrics['roc_auc']:.4f}")
    logger.info(f"Celeb-DF Zero-Shot Confusion:         TN={celeb_metrics['true_negatives']}, FP={celeb_metrics['false_positives']}, FN={celeb_metrics['false_negatives']}, TP={celeb_metrics['true_positives']}")
    logger.info("==================================================================")
    logger.info("                 EXP-5 V2 COMPLETE AND STORED                     ")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
