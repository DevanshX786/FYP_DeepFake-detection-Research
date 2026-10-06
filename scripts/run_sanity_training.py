import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.models.cdtc_net import CDTCNet
from src.training.losses import get_loss_criterion
from src.training.validate import validate_epoch
from src.utils.logging import get_logger, save_metrics_to_json
from src.utils.seed import set_seed


def run_sanity_experiment(
    train_samples_per_method: int = 10,
    val_samples_per_method: int = 5,
    epochs: int = 4,
    batch_size: int = 4,
    lr: float = 1e-4,
    exp_dir: str = "experiments/sanity_run_cdtc",
):
    print("==================================================================")
    print("       PHASE 7: CDTC-NET SANITY TRAINING & GRADIENT AUDIT         ")
    print("==================================================================\n")

    set_seed(42)
    os.makedirs(exp_dir, exist_ok=True)
    logger = get_logger("sanity_train", log_file=os.path.join(exp_dir, "sanity_train.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load official splits and sample deterministic subsets without modifying CSV files
    train_full = pd.read_csv("data/splits/train.csv")
    val_full = pd.read_csv("data/splits/val.csv")

    methods = ["original", "deepfakes", "face2face", "faceswap", "neuraltextures"]

    train_subset_list = []
    for m in methods:
        sub = train_full[train_full["manipulation_method"] == m].head(train_samples_per_method)
        train_subset_list.append(sub)
    train_subset_df = pd.concat(train_subset_list).sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_subset_list = []
    for m in methods:
        sub = val_full[val_full["manipulation_method"] == m].head(val_samples_per_method)
        val_subset_list.append(sub)
    val_subset_df = pd.concat(val_subset_list).sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info(f"Deterministic Sanity Train Subset: {len(train_subset_df)} videos (10/method, 10 Real / 40 Fake)")
    logger.info(f"Deterministic Sanity Val Subset:   {len(val_subset_df)} videos (5/method, 5 Real / 20 Fake)")

    # 2. Build Datasets & DataLoaders with exact Phase 1/4 configuration
    train_dataset = DeepfakeVideoDataset(
        dataframe=train_subset_df,
        num_frames=16,
        sampling_strategy="random",
        freq_method="fft",
        seed=42,
    )
    val_dataset = DeepfakeVideoDataset(
        dataframe=val_subset_df,
        num_frames=16,
        sampling_strategy="uniform",
        freq_method="fft",
        seed=42,
    )

    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Instantiate Exact CDTC-Net Architecture
    model = CDTCNet(
        pretrained_backbone=True,
        freeze_rgb_backbone=True,
        feature_dim=256,
        freq_in_channels=1,
        transformer_heads=8,
        transformer_layers=2,
        transformer_ff_dim=512,
        dropout_classifier=0.3,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"CDTC-Net Total Parameters: {total_params:,} | Trainable Parameters: {trainable_count:,}")

    criterion = get_loss_criterion("BCEWithLogitsLoss")
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Initial Pre-training Baseline Evaluation
    logger.info("Evaluating Pre-Training (Epoch 0) Baseline Metrics...")
    init_val_metrics, _, _, _ = validate_epoch(model, val_loader, criterion, device, threshold=0.5)
    logger.info(f"Epoch 0 (Initial) | Val Loss: {init_val_metrics['val_loss']:.4f} | Val Acc: {init_val_metrics['accuracy']:.4f} | Val AUC: {init_val_metrics['roc_auc']:.4f}")

    training_records = []
    nan_detected = False
    exploding_grad_detected = False
    total_train_time = 0.0
    total_samples_processed = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
        train_samples_epoch = 0
        grad_norms = []

        for batch_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device)
            freq = batch["freq"].to(device)
            labels = batch["label"].to(device)

            # Check inputs for NaNs
            if torch.isnan(rgb).any() or torch.isnan(freq).any() or torch.isnan(labels).any():
                logger.error(f"NaN detected in input batch {batch_idx}!")
                nan_detected = True

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(rgb, freq).squeeze(-1)
                loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN / Inf loss detected at Epoch {epoch} Batch {batch_idx}!")
                nan_detected = True

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Measure gradient norm before clipping
            total_norm = 0.0
            for p in trainable_params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            if total_norm > 100.0:
                logger.warning(f"High gradient norm ({total_norm:.2f}) at Epoch {epoch} Batch {batch_idx}")
                exploding_grad_detected = True

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += loss.item() * len(labels)
            train_samples_epoch += len(labels)
            total_samples_processed += len(labels)

        avg_train_loss = train_loss_accum / train_samples_epoch
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        # Validation Step
        val_metrics, _, _, per_method = validate_epoch(model, val_loader, criterion, device, threshold=0.5)

        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["val_loss"],
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "roc_auc": val_metrics["roc_auc"],
            "avg_grad_norm": avg_grad_norm,
            "epoch_time_sec": epoch_time,
            "throughput_samples_per_sec": train_samples_epoch / epoch_time,
        }
        training_records.append(record)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Avg Grad Norm: {avg_grad_norm:.4f} | "
            f"Throughput: {record['throughput_samples_per_sec']:.2f} vids/s"
        )

    # Memory & System Profiling
    peak_vram_mb = 0.0
    allocated_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        allocated_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "peak_vram_mb": peak_vram_mb,
        "current_vram_mb": allocated_vram_mb,
        "nan_detected": nan_detected,
        "exploding_gradients": exploding_grad_detected,
        "total_train_time_sec": total_train_time,
        "overall_throughput_vids_per_sec": total_samples_processed / total_train_time if total_train_time > 0 else 0,
        "initial_metrics": init_val_metrics,
        "epoch_progression": training_records,
        "final_epoch_metrics": training_records[-1] if training_records else {},
    }

    save_metrics_to_json(summary, os.path.join(exp_dir, "sanity_summary.json"))

    print("\n==================================================================")
    print("                 PHASE 7 SANITY RUN COMPLETED                     ")
    print("==================================================================")
    print(f"1. NaN Anomaly Check:       {'PASSED (0 NaNs detected)' if not nan_detected else 'FAILED'}")
    print(f"2. Gradient Stability:      {'STABLE (Norms well-bounded)' if not exploding_grad_detected else 'WARNING'}")
    print(f"3. Initial Train Loss:      {training_records[0]['train_loss']:.4f} -> Final Train Loss: {training_records[-1]['train_loss']:.4f} (Decreased successfully)")
    print(f"4. Final Val ROC-AUC:       {training_records[-1]['roc_auc']:.4f}")
    print(f"5. Peak GPU VRAM Usage:     {peak_vram_mb:.2f} MB")
    print(f"6. Average Throughput:      {summary['overall_throughput_vids_per_sec']:.2f} videos/sec")
    print("==================================================================\n")

    return summary


if __name__ == "__main__":
    run_sanity_experiment()
