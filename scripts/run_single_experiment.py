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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.models.cdtc_net import CDTCNet
from src.models.baselines.senior_resnet_bilstm import SeniorResNetBiLSTM
from src.models.baselines.rgb_only import RGBOnlyModel
from src.models.baselines.rgb_transformer import RGBTransformerModel
from src.models.baselines.frequency_only import FrequencyOnlyModel
from src.models.rgb_encoder import ConvNeXtRGBEncoder
from src.models.frequency_encoder import FrequencyEncoder
from src.models.fusion import CrossDomainFusion
from src.models.temporal_difference import TemporalDifferenceModule
from src.models.temporal_transformer import TemporalConsistencyTransformer
from src.training.losses import get_loss_criterion
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_checkpoint
from src.utils.checkpointing import CheckpointManager
from src.utils.logging import get_logger, save_metrics_to_json
from src.utils.seed import set_seed


class CDTCNetAblationA1(nn.Module):
    """Ablation A1: Dual Domain + Temporal Average Pooling (No Transformer, No Diff Tokens)."""

    def __init__(self, pretrained=True, freeze_backbone=True, feature_dim=256, dropout=0.3):
        super().__init__()
        self.rgb_encoder = ConvNeXtRGBEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone, output_dim=feature_dim)
        self.freq_encoder = FrequencyEncoder(in_channels=1, output_dim=feature_dim)
        self.fusion = CrossDomainFusion(rgb_dim=feature_dim, freq_dim=feature_dim, output_dim=feature_dim, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, rgb_frames, freq_maps):
        rgb_tokens = self.rgb_encoder(rgb_frames)
        freq_tokens = self.freq_encoder(freq_maps)
        frame_tokens = self.fusion(rgb_tokens, freq_tokens)
        video_repr = frame_tokens.mean(dim=1)
        return self.classifier(video_repr)


class CDTCNetAblationA2(nn.Module):
    """Ablation A2: Dual Domain + Transformer (No Difference Tokens)."""

    def __init__(self, pretrained=True, freeze_backbone=True, feature_dim=256, dropout=0.3):
        super().__init__()
        self.rgb_encoder = ConvNeXtRGBEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone, output_dim=feature_dim)
        self.freq_encoder = FrequencyEncoder(in_channels=1, output_dim=feature_dim)
        self.fusion = CrossDomainFusion(rgb_dim=feature_dim, freq_dim=feature_dim, output_dim=feature_dim, dropout=0.2)
        self.transformer = TemporalConsistencyTransformer(embed_dim=feature_dim, num_heads=8, num_layers=2, feedforward_dim=512, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, rgb_frames, freq_maps):
        rgb_tokens = self.rgb_encoder(rgb_frames)
        freq_tokens = self.freq_encoder(freq_maps)
        frame_tokens = self.fusion(rgb_tokens, freq_tokens)
        video_repr = self.transformer(frame_tokens, diff_tokens=None)
        return self.classifier(video_repr)


class CDTCNetAblationA3(nn.Module):
    """Ablation A3: Signed-Only Difference Tokens."""

    def __init__(self, pretrained=True, freeze_backbone=True, feature_dim=256, dropout=0.3):
        super().__init__()
        self.rgb_encoder = ConvNeXtRGBEncoder(pretrained=pretrained, freeze_backbone=freeze_backbone, output_dim=feature_dim)
        self.freq_encoder = FrequencyEncoder(in_channels=1, output_dim=feature_dim)
        self.fusion = CrossDomainFusion(rgb_dim=feature_dim, freq_dim=feature_dim, output_dim=feature_dim, dropout=0.2)
        self.diff_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.transformer = TemporalConsistencyTransformer(embed_dim=feature_dim, num_heads=8, num_layers=2, feedforward_dim=512, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, rgb_frames, freq_maps):
        rgb_tokens = self.rgb_encoder(rgb_frames)
        freq_tokens = self.freq_encoder(freq_maps)
        frame_tokens = self.fusion(rgb_tokens, freq_tokens)
        d_t = frame_tokens[:, 1:, :] - frame_tokens[:, :-1, :]
        diff_tokens = self.diff_proj(d_t)
        video_repr = self.transformer(frame_tokens, diff_tokens=diff_tokens)
        return self.classifier(video_repr)


def build_model(exp_id: str, freeze_backbone: bool = True):
    """Instantiate the model for the given experiment ID."""
    if exp_id == "EXP-1":
        return SeniorResNetBiLSTM(pretrained=True, freeze_backbone=freeze_backbone, hidden_dim=256, dropout=0.5), "fft"
    elif exp_id == "EXP-2":
        return RGBOnlyModel(pretrained=True, freeze_backbone=freeze_backbone, feature_dim=256, dropout=0.3), "fft"
    elif exp_id == "EXP-3":
        return RGBTransformerModel(pretrained=True, freeze_backbone=freeze_backbone, feature_dim=256, dropout=0.3), "fft"
    elif exp_id == "EXP-4":
        return FrequencyOnlyModel(in_channels=1, feature_dim=256, dropout=0.3), "fft"
    elif exp_id == "EXP-5":
        return CDTCNet(pretrained_backbone=True, freeze_rgb_backbone=freeze_backbone, feature_dim=256), "fft"
    elif exp_id == "ABL-1":
        return CDTCNetAblationA1(pretrained=True, freeze_backbone=freeze_backbone, feature_dim=256), "fft"
    elif exp_id == "ABL-2":
        return CDTCNetAblationA2(pretrained=True, freeze_backbone=freeze_backbone, feature_dim=256), "fft"
    elif exp_id == "ABL-3":
        return CDTCNetAblationA3(pretrained=True, freeze_backbone=freeze_backbone, feature_dim=256), "fft"
    elif exp_id == "ABL-4":
        return CDTCNet(pretrained_backbone=True, freeze_rgb_backbone=freeze_backbone, feature_dim=256), "dct"
    else:
        raise ValueError(f"Unknown experiment ID: {exp_id}")


def main():
    parser = argparse.ArgumentParser(description="Run a single deepfake detection experiment.")
    parser.add_argument("--exp_id", type=str, required=True, choices=["EXP-1", "EXP-2", "EXP-3", "EXP-4", "EXP-5", "ABL-1", "ABL-2", "ABL-3", "ABL-4"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--exp_dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.exp_dir is None:
        args.exp_dir = f"experiments/{args.exp_id.lower().replace('-', '_')}"
    os.makedirs(args.exp_dir, exist_ok=True)

    logger = get_logger(args.exp_id, log_file=os.path.join(args.exp_dir, "train.log"))
    logger.info(f"==================================================================")
    logger.info(f"             RUNNING EXPERIMENT: {args.exp_id}                   ")
    logger.info(f"==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Build Model
    model, freq_method = build_model(args.exp_id, freeze_backbone=args.freeze_backbone)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model.__class__.__name__} | Frequency: {freq_method}")
    logger.info(f"Total Params: {total_params:,} | Trainable Params: {trainable_params:,}")

    # 2. Save Config
    config = {
        "exp_id": args.exp_id,
        "model_class": model.__class__.__name__,
        "freq_method": freq_method,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": 0.01,
        "freeze_backbone": args.freeze_backbone,
        "num_frames": 16,
        "device": str(device),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_split": "data/splits/train.csv",
        "val_split": "data/splits/val.csv",
        "test_split": "data/splits/test.csv",
    }
    with open(os.path.join(args.exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 3. DataLoaders
    train_df = pd.read_csv("data/splits/train.csv")
    val_df = pd.read_csv("data/splits/val.csv")
    logger.info(f"Loaded train: {len(train_df)} samples | val: {len(val_df)} samples")

    train_dataset = DeepfakeVideoDataset(
        dataframe=train_df,
        num_frames=16,
        sampling_strategy="uniform",
        freq_method=freq_method,
    )
    val_dataset = DeepfakeVideoDataset(
        dataframe=val_df,
        num_frames=16,
        sampling_strategy="uniform",
        freq_method=freq_method,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # 4. Optimizer, Scheduler, Loss
    criterion = get_loss_criterion("BCEWithLogitsLoss")
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    chk_dir = os.path.join(args.exp_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir=chk_dir, monitor="roc_auc", mode="max", save_best_only=True)

    # 5. Train
    train_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        mixed_precision=True,
        gradient_clip_val=1.0,
        early_stopping_patience=5,
        checkpoint_manager=checkpoint_manager,
        log_dir=args.exp_dir,
        logger=logger,
    )

    best_chk_path = os.path.join(chk_dir, "best_model.pt")
    if not os.path.exists(best_chk_path):
        best_chk_path = os.path.join(chk_dir, "latest_model.pt")

    # 6. In-Domain FF++ Test Evaluation (Evaluated exactly once on test.csv)
    eval_dir = os.path.join(args.exp_dir, "evaluation")
    test_results = evaluate_checkpoint(
        model=model,
        test_csv_path="data/splits/test.csv",
        checkpoint_path=best_chk_path,
        output_dir=eval_dir,
        batch_size=args.batch_size,
        num_frames=16,
        device=device,
        model_name=args.exp_id,
        threshold=0.5,
    )

    logger.info(f"==================================================================")
    logger.info(f"         EXPERIMENT {args.exp_id} COMPLETED SUCCESSFULLY          ")
    logger.info(f" Test Accuracy:         {test_results['overall_metrics']['accuracy']:.4f}")
    logger.info(f" Test Balanced Accuracy:{test_results['overall_metrics'].get('balanced_accuracy', 0.0):.4f}")
    logger.info(f" Test Precision:        {test_results['overall_metrics']['precision']:.4f}")
    logger.info(f" Test Recall:           {test_results['overall_metrics']['recall']:.4f}")
    logger.info(f" Test F1-Score:         {test_results['overall_metrics']['f1']:.4f}")
    logger.info(f" Test ROC-AUC:          {test_results['overall_metrics']['roc_auc']:.4f}")
    logger.info(f"==================================================================")


if __name__ == "__main__":
    main()
