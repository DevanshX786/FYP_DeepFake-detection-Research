import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import pandas as pd

from src.models.cdtc_net import CDTCNet
from src.models.baselines.senior_resnet_bilstm import SeniorResNetBiLSTM
from src.models.baselines.rgb_only import RGBOnlyModel
from src.models.baselines.rgb_transformer import RGBTransformerModel
from src.models.baselines.frequency_only import FrequencyOnlyModel
from src.data.preprocessing import compute_fft_magnitude, compute_dct_magnitude, preprocess_face_crops
from src.data.sampler import sample_frame_indices
from src.data.splits import create_identity_safe_splits
from src.evaluation.metrics import calculate_metrics, calculate_per_manipulation_metrics


def run_sanity_checks():
    print("=== Step 1: Testing Sampler ===")
    indices_uniform = sample_frame_indices(total_frames=100, num_frames=16, strategy="uniform")
    assert len(indices_uniform) == 16, f"Expected 16 frames, got {len(indices_uniform)}"
    print(f"Uniform sampled indices: {indices_uniform}")

    indices_random = sample_frame_indices(total_frames=100, num_frames=16, strategy="random", seed=42)
    assert len(indices_random) == 16
    print(f"Random sampled indices: {indices_random}")

    print("\n=== Step 2: Testing Frequency & RGB Preprocessing ===")
    dummy_face = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    fft_map = compute_fft_magnitude(dummy_face, shift_dc=True, log_scale=True)
    dct_map = compute_dct_magnitude(dummy_face, log_scale=True)
    assert fft_map.shape == (224, 224), f"Unexpected FFT shape: {fft_map.shape}"
    assert dct_map.shape == (224, 224), f"Unexpected DCT shape: {dct_map.shape}"
    print(f"FFT map range: [{fft_map.min():.2f}, {fft_map.max():.2f}], shape: {fft_map.shape}")
    print(f"DCT map range: [{dct_map.min():.2f}, {dct_map.max():.2f}], shape: {dct_map.shape}")

    face_crops = [dummy_face for _ in range(16)]
    rgb_t, freq_t = preprocess_face_crops(face_crops, freq_method="fft")
    assert rgb_t.shape == (16, 3, 224, 224), f"Unexpected RGB tensor shape: {rgb_t.shape}"
    assert freq_t.shape == (16, 1, 224, 224), f"Unexpected Freq tensor shape: {freq_t.shape}"
    print(f"Stacked RGB Tensor shape: {rgb_t.shape}, Freq Tensor shape: {freq_t.shape}")

    print("\n=== Step 3: Testing Identity-Safe Splits ===")
    mock_df = pd.DataFrame([
        {"video_path": f"v_{i}.mp4", "label": i % 2, "source_video_id": f"src_{i // 4}", "manipulation_method": "df"}
        for i in range(40)
    ])
    train_df, val_df, test_df = create_identity_safe_splits(mock_df, output_dir=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    train_ids = set(train_df["source_video_id"])
    val_ids = set(val_df["source_video_id"])
    test_ids = set(test_df["source_video_id"])
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected between val and test!"
    print(f"Splits verified leakage-free! Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n=== Step 4: Testing Model Architectures Forward Passes ===")
    # Batch size = 2, 16 frames per video
    batch_rgb = torch.randn(2, 16, 3, 224, 224)
    batch_freq = torch.randn(2, 16, 1, 224, 224)

    # 1. CDTC-Net
    print("Testing CDTC-Net forward pass...")
    cdtc = CDTCNet(pretrained_backbone=False, freeze_rgb_backbone=False)
    cdtc_out = cdtc(batch_rgb, batch_freq)
    assert cdtc_out.shape == (2, 1), f"Expected (2, 1), got {cdtc_out.shape}"
    loss = cdtc_out.sum()
    loss.backward()
    print(f"  CDTC-Net output: {cdtc_out.shape} -> Loss backward successful!")

    # 2. Senior ResNet50 + BiLSTM Baseline
    print("Testing Senior ResNet50 + BiLSTM baseline...")
    senior = SeniorResNetBiLSTM(pretrained=False)
    senior_out = senior(batch_rgb)
    assert senior_out.shape == (2, 1), f"Expected (2, 1), got {senior_out.shape}"
    print(f"  Senior baseline output: {senior_out.shape} -> Successful!")

    # 3. RGB Only Baseline
    print("Testing RGB-Only baseline...")
    rgb_only = RGBOnlyModel(pretrained=False, freeze_backbone=False)
    rgb_out = rgb_only(batch_rgb)
    assert rgb_out.shape == (2, 1), f"Expected (2, 1), got {rgb_out.shape}"
    print(f"  RGB-Only output: {rgb_out.shape} -> Successful!")

    # 4. RGB + Transformer Baseline (Ablation A0)
    print("Testing RGB + Transformer baseline...")
    rgb_trans = RGBTransformerModel(pretrained=False, freeze_backbone=False)
    trans_out = rgb_trans(batch_rgb)
    assert trans_out.shape == (2, 1), f"Expected (2, 1), got {trans_out.shape}"
    print(f"  RGB + Transformer output: {trans_out.shape} -> Successful!")

    # 5. Frequency Only Baseline
    print("Testing Frequency-Only baseline...")
    freq_only = FrequencyOnlyModel()
    freq_out = freq_only(freq_maps=batch_freq)
    assert freq_out.shape == (2, 1), f"Expected (2, 1), got {freq_out.shape}"
    print(f"  Frequency-Only output: {freq_out.shape} -> Successful!")

    print("\n=== Step 5: Testing Metrics Calculation ===")
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_probs = np.array([0.1, 0.2, 0.9, 0.85, 0.7, 0.3, 0.95, 0.15])
    methods = ["original", "original", "deepfakes", "face2face", "faceswap", "original", "neuraltextures", "original"]
    metrics = calculate_metrics(y_true, y_probs)
    per_manip = calculate_per_manipulation_metrics(y_true, y_probs, methods)
    assert metrics["accuracy"] == 1.0, f"Expected 1.0 acc, got {metrics['accuracy']}"
    assert metrics["roc_auc"] == 1.0, f"Expected 1.0 auc, got {metrics['roc_auc']}"
    print(f"Metrics: Acc = {metrics['accuracy']}, AUC = {metrics['roc_auc']}, F1 = {metrics['f1']}")
    print(f"Per manipulation categories tested: {list(per_manip.keys())}")

    print("\n==========================================")
    print(" ALL ARCHITECTURAL AND DATA PIPELINE SANITY CHECKS PASSED!")
    print("==========================================")


if __name__ == "__main__":
    run_sanity_checks()
