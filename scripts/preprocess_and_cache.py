import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.face_detection import FaceDetector
from src.data.preprocessing import preprocess_face_crops
from src.data.sampler import extract_sampled_frames
from src.utils.logging import get_logger


def get_cache_rel_path(video_path: str) -> str:
    """Generate a clean, deterministic relative path for caching."""
    norm_path = os.path.normpath(video_path).replace("\\", "/")
    # Remove leading data/raw/faceforensics/ if present
    if "data/raw/faceforensics/" in norm_path:
        rel = norm_path.split("data/raw/faceforensics/")[1]
    elif "faceforensics/" in norm_path:
        rel = norm_path.split("faceforensics/")[1]
    else:
        # Fallback to hashed name
        h = hashlib.md5(norm_path.encode("utf-8")).hexdigest()
        rel = f"hashed/{h}_{os.path.basename(norm_path)}"
    
    # Replace video extension with .pt
    base, _ = os.path.splitext(rel)
    return base + ".pt"


def preprocess_all_ffpp(
    metadata_csv: str = "data/metadata/ffpp_metadata.csv",
    output_base_dir: str = "data/processed/face_crops",
    manifest_csv: str = "data/processed/cache_manifest.csv",
    num_frames: int = 16,
    batch_size: int = 1,
):
    """Pre-extract and cache face crops for all videos in FF++ metadata."""
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)

    logger = get_logger("preprocess_cache", log_file="data/processed/preprocess.log")
    logger.info("==================================================================")
    logger.info("          ONE-TIME PREPROCESSING & FACE CROP CACHING              ")
    logger.info("==================================================================")

    df = pd.read_csv(metadata_csv)
    total_videos = len(df)
    logger.info(f"Loaded {total_videos} videos from metadata: {metadata_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Face Detector Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    face_detector = FaceDetector(method="mtcnn", margin=0.15, target_size=(224, 224), confidence_threshold=0.5)

    manifest_records = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    t_start = time.time()

    for idx, row in enumerate(df.itertuples()):
        video_path = row.video_path
        rel_cache_path = get_cache_rel_path(video_path)
        full_cache_path = os.path.join(output_base_dir, rel_cache_path)

        os.makedirs(os.path.dirname(full_cache_path), exist_ok=True)

        if os.path.exists(full_cache_path) and os.path.getsize(full_cache_path) > 1000:
            skipped_count += 1
            manifest_records.append({
                "video_path": video_path,
                "cache_path": full_cache_path,
                "label": row.label,
                "dataset": getattr(row, "dataset", "faceforensics"),
                "manipulation_method": getattr(row, "manipulation_method", "unknown"),
                "source_video_id": getattr(row, "source_video_id", "unknown"),
                "split": getattr(row, "split", "unknown"),
                "status": "cached",
            })
            continue

        try:
            # 1. Extract sampled frames
            raw_frames, sampled_indices = extract_sampled_frames(
                video_path=video_path,
                num_frames=num_frames,
                strategy="uniform",
                seed=42,
            )

            # 2. Face detection and cropping
            crops, crop_meta = face_detector.process_frame_sequence(raw_frames)

            # crops is List[np.ndarray] of shape (224, 224, 3) uint8
            # Stack into numpy array [16, 224, 224, 3] uint8
            crops_array = np.stack(crops, axis=0).astype(np.uint8)
            crops_tensor = torch.from_numpy(crops_array)  # [16, 224, 224, 3] uint8

            # Save dictionary containing face crops and detection metadata
            save_payload = {
                "face_crops": crops_tensor,  # uint8 tensor [16, 224, 224, 3]
                "sampled_indices": sampled_indices,
                "crop_meta": crop_meta,
                "video_path": video_path,
                "label": row.label,
                "source_video_id": getattr(row, "source_video_id", "unknown"),
                "manipulation_method": getattr(row, "manipulation_method", "unknown"),
                "split": getattr(row, "split", "unknown"),
            }

            torch.save(save_payload, full_cache_path)
            processed_count += 1

            manifest_records.append({
                "video_path": video_path,
                "cache_path": full_cache_path,
                "label": row.label,
                "dataset": getattr(row, "dataset", "faceforensics"),
                "manipulation_method": getattr(row, "manipulation_method", "unknown"),
                "source_video_id": getattr(row, "source_video_id", "unknown"),
                "split": getattr(row, "split", "unknown"),
                "status": "success",
            })

        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to process video {video_path}: {e}")
            manifest_records.append({
                "video_path": video_path,
                "cache_path": "",
                "label": row.label,
                "dataset": getattr(row, "dataset", "faceforensics"),
                "manipulation_method": getattr(row, "manipulation_method", "unknown"),
                "source_video_id": getattr(row, "source_video_id", "unknown"),
                "split": getattr(row, "split", "unknown"),
                "status": f"failed: {str(e)}",
            })

        if (idx + 1) % 100 == 0 or (idx + 1) == total_videos:
            elapsed = time.time() - t_start
            vids_per_sec = (processed_count + skipped_count) / elapsed if elapsed > 0 else 0
            eta_min = (total_videos - (idx + 1)) / (vids_per_sec * 60) if vids_per_sec > 0 else 0
            logger.info(
                f"Progress: {idx + 1}/{total_videos} videos ({(idx + 1)/total_videos*100:.1f}%) | "
                f"Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count} | "
                f"Rate: {vids_per_sec:.2f} vids/s | ETA: {eta_min:.1f} min"
            )

    # Save manifest
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(manifest_csv, index=False)
    logger.info(f"Manifest written to: {manifest_csv}")

    total_time = time.time() - t_start
    logger.info("==================================================================")
    logger.info("               PREPROCESSING & CACHING COMPLETED                  ")
    logger.info(f" Total Time:          {total_time:.2f} s ({total_time/60:.2f} min)")
    logger.info(f" Videos Processed:    {processed_count}")
    logger.info(f" Videos Skipped:      {skipped_count}")
    logger.info(f" Videos Failed:       {failed_count}")
    logger.info(f" Output Directory:    {output_base_dir}")
    logger.info("==================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and cache FF++ face crops.")
    parser.add_argument("--metadata_csv", type=str, default="data/metadata/ffpp_metadata.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/face_crops")
    parser.add_argument("--manifest_csv", type=str, default="data/processed/cache_manifest.csv")
    args = parser.parse_args()

    preprocess_all_ffpp(
        metadata_csv=args.metadata_csv,
        output_base_dir=args.output_dir,
        manifest_csv=args.manifest_csv,
    )
