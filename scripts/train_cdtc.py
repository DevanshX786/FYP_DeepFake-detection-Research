import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import pandas as pd

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.data.splits import load_split_dataframe
from src.models.cdtc_net import CDTCNet
from src.training.losses import get_loss_criterion
from src.training.train import train_model
from src.utils.checkpointing import CheckpointManager
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train CDTC-Net (Cross-Domain Temporal Consistency Network).")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model YAML config.")
    parser.add_argument("--training_config", type=str, default="configs/training.yaml", help="Path to training YAML config.")
    parser.add_argument("--dataset_config", type=str, default="configs/dataset.yaml", help="Path to dataset YAML config.")
    parser.add_argument("--train_csv", type=str, default="data/splits/train.csv", help="Path to train split CSV.")
    parser.add_argument("--val_csv", type=str, default="data/splits/val.csv", help="Path to validation split CSV.")
    parser.add_argument("--exp_name", type=str, default=None, help="Custom experiment directory name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Unfreeze RGB backbone from start.")
    args = parser.parse_args()

    # Load configs
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.training_config, "r") as f:
        train_cfg = yaml.safe_load(f)
    with open(args.dataset_config, "r") as f:
        data_cfg = yaml.safe_load(f)

    # Seed
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Experiment dir
    exp_name = args.exp_name or train_cfg.get("experiment_name", "cdtc_ffpp_c23_default")
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = get_logger("train_cdtc", log_file=os.path.join(exp_dir, "train.log"))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initialized CDTC-Net Training Pipeline on: {device}")

    # Load Splits
    if not os.path.exists(args.train_csv) or not os.path.exists(args.val_csv):
        logger.warning("Split files not found. Creating synthetic demo splits for sanity check...")
        synthetic_df = pd.DataFrame([
            {"video_path": "synthetic_0.mp4", "label": 0, "source_video_id": "000", "manipulation_method": "original"},
            {"video_path": "synthetic_1.mp4", "label": 1, "source_video_id": "001", "manipulation_method": "deepfakes"},
        ])
        train_df, val_df = synthetic_df, synthetic_df
    else:
        train_df = load_split_dataframe(args.train_csv)
        val_df = load_split_dataframe(args.val_csv)

    # Build Datasets
    num_frames = data_cfg.get("frames_per_video", 16)
    freq_method = model_cfg.get("frequency_branch", {}).get("method", "fft")

    train_dataset = DeepfakeVideoDataset(
        dataframe=train_df,
        num_frames=num_frames,
        sampling_strategy="random",
        freq_method=freq_method,
        seed=seed,
    )
    val_dataset = DeepfakeVideoDataset(
        dataframe=val_df,
        num_frames=num_frames,
        sampling_strategy="uniform",
        freq_method=freq_method,
        seed=seed,
    )

    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize Model
    freeze_backbone = not args.unfreeze_backbone if args.unfreeze_backbone else model_cfg.get("rgb_branch", {}).get("freeze_backbone", True)
    model = CDTCNet(
        pretrained_backbone=model_cfg.get("rgb_branch", {}).get("pretrained", True),
        freeze_rgb_backbone=freeze_backbone,
        feature_dim=model_cfg.get("rgb_branch", {}).get("feature_dim", 256),
        freq_in_channels=model_cfg.get("frequency_branch", {}).get("in_channels", 1),
        transformer_heads=model_cfg.get("temporal_transformer", {}).get("num_heads", 8),
        transformer_layers=model_cfg.get("temporal_transformer", {}).get("num_layers", 2),
        transformer_ff_dim=model_cfg.get("temporal_transformer", {}).get("feedforward_dim", 512),
        dropout_classifier=model_cfg.get("classifier", {}).get("dropout", 0.3),
    )

    # Criterion & Optimizer
    criterion = get_loss_criterion(train_cfg.get("loss", {}).get("type", "BCEWithLogitsLoss"))
    lr = args.lr or train_cfg.get("optimizer", {}).get("lr", 0.0001)
    weight_decay = train_cfg.get("optimizer", {}).get("weight_decay", 0.01)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    epochs = args.epochs or train_cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    chk_mgr = CheckpointManager(
        checkpoint_dir=exp_dir,
        monitor=train_cfg.get("checkpointing", {}).get("monitor", "val_auc"),
        mode=train_cfg.get("checkpointing", {}).get("mode", "max"),
    )

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        mixed_precision=train_cfg.get("mixed_precision", True),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        early_stopping_patience=train_cfg.get("early_stopping", {}).get("patience", 5),
        checkpoint_manager=chk_mgr,
        log_dir=exp_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
