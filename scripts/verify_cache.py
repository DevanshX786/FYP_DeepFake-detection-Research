import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import DeepfakeVideoDataset
from src.data.sampler import extract_sampled_frames
from src.data.face_detection import FaceDetector
from src.data.preprocessing import preprocess_face_crops
from src.utils.logging import get_logger


def verify_cache_integrity(
    manifest_csv: str = "data/processed/cache_manifest.csv",
    cache_dir: str = "data/processed/face_crops",
    num_samples: int = 50,
    seed: int = 42,
) -> bool:
    """Verify cached face crop tensors against direct on-the-fly video extraction."""
    logger = get_logger("verify_cache", log_file="data/processed/verify_cache.log")
    logger.info("==================================================================")
    logger.info("          CACHE VERIFICATION & NUMERICAL INTEGRITY AUDIT          ")
    logger.info("==================================================================")

    if not os.path.exists(manifest_csv):
        # If manifest not yet fully written, read directly from metadata and check existing cached files
        metadata_csv = "data/metadata/ffpp_metadata.csv"
        df = pd.read_csv(metadata_csv)
    else:
        df = pd.read_csv(manifest_csv)

    # Filter to only cached files
    cached_df = df[df["video_path"].apply(lambda p: os.path.exists(os.path.join(cache_dir, os.path.splitext(p.split("faceforensics/")[1] if "faceforensics/" in p else os.path.basename(p))[0] + ".pt")))]

    total_cached = len(cached_df)
    logger.info(f"Total Cached Files Available: {total_cached}")

    if total_cached == 0:
        logger.error("No cached files found to verify!")
        return False

    sample_size = min(num_samples, total_cached)
    sampled = cached_df.sample(n=sample_size, random_state=seed)
    logger.info(f"Testing {sample_size} representative samples across manipulation methods and splits...")

    face_detector = FaceDetector(method="mtcnn", margin=0.15, target_size=(224, 224), confidence_threshold=0.5)

    cached_dataset = DeepfakeVideoDataset(dataframe=sampled, num_frames=16, freq_method="fft", cache_dir=cache_dir)
    dynamic_dataset = DeepfakeVideoDataset(dataframe=sampled, num_frames=16, freq_method="fft", cache_dir=None)

    shape_matches = 0
    numeric_matches = 0
    label_matches = 0
    meta_matches = 0

    max_rgb_diff = 0.0
    max_freq_diff = 0.0

    for idx in range(sample_size):
        cached_sample = cached_dataset[idx]
        dynamic_sample = dynamic_dataset[idx]

        # 1. Check shapes
        assert cached_sample["rgb"].shape == torch.Size([16, 3, 224, 224]), f"Wrong RGB shape: {cached_sample['rgb'].shape}"
        assert cached_sample["freq"].shape == torch.Size([16, 1, 224, 224]), f"Wrong Freq shape: {cached_sample['freq'].shape}"
        shape_matches += 1

        # 2. Check label & metadata
        if cached_sample["label"] == dynamic_sample["label"]:
            label_matches += 1
        if cached_sample["manipulation_method"] == dynamic_sample["manipulation_method"]:
            meta_matches += 1

        # 3. Check tensor numerical consistency
        rgb_diff = torch.max(torch.abs(cached_sample["rgb"] - dynamic_sample["rgb"])).item()
        freq_diff = torch.max(torch.abs(cached_sample["freq"] - dynamic_sample["freq"])).item()

        max_rgb_diff = max(max_rgb_diff, rgb_diff)
        max_freq_diff = max(max_freq_diff, freq_diff)

        if rgb_diff < 1e-4 and freq_diff < 1e-4:
            numeric_matches += 1

    logger.info(f"Shape Verification:    {shape_matches}/{sample_size} PASSED [16, 3, 224, 224] & [16, 1, 224, 224]")
    logger.info(f"Label & Meta Match:    {label_matches}/{sample_size} PASSED")
    logger.info(f"Numerical Consistency: {numeric_matches}/{sample_size} PASSED (Max RGB diff: {max_rgb_diff:.6f}, Max Freq diff: {max_freq_diff:.6f})")

    # Measure loading throughput on cached dataset
    import time
    from torch.utils.data import DataLoader
    loader = DataLoader(cached_dataset, batch_size=8, shuffle=False, num_workers=0)
    t0 = time.time()
    count = 0
    for batch in loader:
        count += len(batch["label"])
    cached_speed = count / (time.time() - t0)
    logger.info(f"Cached DataLoader Throughput: {cached_speed:.2f} videos/sec (~{cached_speed*16:.1f} frames/sec)")
    
    # Calculate estimated epoch time for 4250 videos (3500 train + 750 val)
    estimated_epoch_sec = 4250 / cached_speed
    logger.info(f"Estimated Epoch Time on Cached Data: {estimated_epoch_sec:.1f} seconds ({estimated_epoch_sec/60:.2f} minutes) vs ~180 minutes on raw video!")

    passed = (shape_matches == sample_size) and (label_matches == sample_size) and (numeric_matches == sample_size)
    if passed:
        logger.info("==================================================================")
        logger.info("         CACHE VERIFICATION AUDIT PASSED WITH ZERO ERRORS         ")
        logger.info("==================================================================")
    else:
        logger.error("Cache verification failed integrity checks!")

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify cached face tensors.")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    verify_cache_integrity(num_samples=args.samples)
