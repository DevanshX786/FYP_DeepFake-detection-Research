import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import torch
import pandas as pd

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.data.splits import load_split_dataframe
from src.models.baselines.senior_resnet_bilstm import SeniorResNetBiLSTM
from src.models.baselines.rgb_only import RGBOnlyModel
from src.models.baselines.rgb_transformer import RGBTransformerModel
from src.models.baselines.frequency_only import FrequencyOnlyModel
from src.training.losses import get_loss_criterion
from src.training.train import train_model
from src.utils.checkpointing import CheckpointManager
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train baseline models for Deepfake detection.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["senior_resnet_bilstm", "rgb_only", "rgb_transformer", "frequency_only"],
        help="Baseline model type.",
    )
    parser.add_argument("--training_config", type=str, default="configs/training.yaml", help="Path to training config.")
    parser.add_argument("--train_csv", type=str, default="data/splits/train.csv", help="Path to train CSV.")
    parser.add_argument("--val_csv", type=str, default="data/splits/val.csv", help="Path to val CSV.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    args = parser.parse_args()

    with open(args.training_config, "r") as f:
        train_cfg = yaml.safe_load(f)

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    exp_name = args.exp_name or f"baseline_{args.model_type}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = get_logger(f"train_{args.model_type}", log_file=os.path.join(exp_dir, "train.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Baseline [{args.model_type}] on {device}")

    # Load splits
    if not os.path.exists(args.train_csv) or not os.path.exists(args.val_csv):
        logger.warning("Splits not found. Creating synthetic demo splits...")
        dummy_df = pd.DataFrame([
            {"video_path": "syn_0.mp4", "label": 0, "source_video_id": "000", "manipulation_method": "original"},
            {"video_path": "syn_1.mp4", "label": 1, "source_video_id": "001", "manipulation_method": "deepfakes"},
        ])
        train_df, val_df = dummy_df, dummy_df
    else:
        train_df = load_split_dataframe(args.train_csv)
        val_df = load_split_dataframe(args.val_csv)

    train_dataset = DeepfakeVideoDataset(train_df, num_frames=16, sampling_strategy="random", seed=seed)
    val_dataset = DeepfakeVideoDataset(val_df, num_frames=16, sampling_strategy="uniform", seed=seed)

    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    if args.model_type == "senior_resnet_bilstm":
        model = SeniorResNetBiLSTM(pretrained=True, hidden_dim=256, dropout=0.5)
    elif args.model_type == "rgb_only":
        model = RGBOnlyModel(pretrained=True, freeze_backbone=True, feature_dim=256)
    elif args.model_type == "rgb_transformer":
        model = RGBTransformerModel(pretrained=True, freeze_backbone=True, feature_dim=256)
    elif args.model_type == "frequency_only":
        model = FrequencyOnlyModel(in_channels=1, feature_dim=256)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    criterion = get_loss_criterion("BCEWithLogitsLoss")
    lr = args.lr or train_cfg.get("optimizer", {}).get("lr", 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    epochs = args.epochs or train_cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    chk_mgr = CheckpointManager(checkpoint_dir=exp_dir)

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
        checkpoint_manager=chk_mgr,
        log_dir=exp_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
