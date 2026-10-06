import os
import sys
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import DeepfakeVideoDataset, create_dataloader
from src.models.cdtc_net import CDTCNet
from src.training.losses import get_loss_criterion


def test_real_pipeline():
    print("=== Phase 4: Data Pipeline Test on Actual Video Dataset ===")
    train_df = pd.read_csv("data/splits/train.csv")

    # Pick 1 Real and 1 Fake video from the real dataset
    real_sample = train_df[train_df["label"] == 0].iloc[0]
    fake_sample = train_df[train_df["label"] == 1].iloc[0]

    test_subset = pd.DataFrame([real_sample, fake_sample])
    print(f"Testing with 2 actual videos:")
    print(f"  Real video: {real_sample['video_path']} (Source: {real_sample['source_video_id']})")
    print(f"  Fake video: {fake_sample['video_path']} (Method: {fake_sample['manipulation_method']}, Source: {fake_sample['source_video_id']})")

    # Build dataset
    dataset = DeepfakeVideoDataset(
        dataframe=test_subset,
        num_frames=16,
        sampling_strategy="uniform",
        freq_method="fft",
    )
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # Fetch batch
    batch = next(iter(dataloader))
    rgb = batch["rgb"]
    freq = batch["freq"]
    labels = batch["label"]

    print(f"\nBatch extracted successfully:")
    print(f"  RGB tensor shape:   {rgb.shape} (Expected: [2, 16, 3, 224, 224])")
    print(f"  Freq tensor shape:  {freq.shape} (Expected: [2, 16, 1, 224, 224])")
    print(f"  Labels tensor:      {labels.tolist()} (Expected: [0.0, 1.0])")

    assert rgb.shape == (2, 16, 3, 224, 224), f"Unexpected RGB shape: {rgb.shape}"
    assert freq.shape == (2, 16, 1, 224, 224), f"Unexpected Freq shape: {freq.shape}"

    # Initialize CDTC-Net model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing CDTC-Net on device: {device}")
    model = CDTCNet(pretrained_backbone=False, freeze_rgb_backbone=False).to(device)

    rgb = rgb.to(device)
    freq = freq.to(device)
    labels = labels.to(device)

    # Forward pass
    logits = model(rgb, freq).squeeze(-1)
    probs = torch.sigmoid(logits)

    criterion = get_loss_criterion("BCEWithLogitsLoss")
    loss = criterion(logits, labels)

    print(f"  Forward pass logits: {logits.detach().cpu().numpy()}")
    print(f"  Predicted P(fake):   {probs.detach().cpu().numpy()}")
    print(f"  BCE Loss:            {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("  Backward pass & gradient calculation: SUCCESSFUL!")

    print("\n=======================================================")
    print(" PHASE 4 COMPLETE: REAL DATA PIPELINE FULLY FUNCTIONAL!")
    print("=======================================================")


if __name__ == "__main__":
    test_real_pipeline()
