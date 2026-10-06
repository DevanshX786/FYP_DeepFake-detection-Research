import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.face_detection import FaceDetector
from src.data.sampler import extract_sampled_frames
from src.utils.logging import get_logger


def prepare_celebdf_test_split(
    test_list_path: str = "data/raw/celebdf/List_of_testing_videos.txt",
    celebdf_root: str = "data/raw/celebdf",
    output_csv_path: str = "data/splits/celebdf_test.csv",
    cache_output_dir: str = "data/processed/face_crops/celebdf",
):
    logger = get_logger("celebdf_prep")
    logger.info("==================================================================")
    logger.info("       PREPARING CELEB-DF V2 ZERO-SHOT TEST SPLIT & CROPS         ")
    logger.info("==================================================================")

    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Celeb-DF test list not found: {test_list_path}")

    records = []
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                parts = line.split()
            celeb_flag, rel_path = parts[0], parts[1]
            # In official Celeb-DF List_of_testing_videos.txt:
            # 1 = Real video -> label 0
            # 0 = Synthesis video -> label 1
            if celeb_flag == "1":
                label = 0
                manipulation_method = "real"
            else:
                label = 1
                manipulation_method = "celeb_synthesis"

            video_path = os.path.join(celebdf_root, rel_path).replace("\\", "/")
            source_id = os.path.basename(rel_path).split("_")[0]

            records.append({
                "video_path": video_path,
                "label": label,
                "dataset": "celeb_df_v2",
                "manipulation_method": manipulation_method,
                "source_video_id": source_id,
                "rel_path": rel_path,
            })

    df = pd.DataFrame(records)
    logger.info(f"Total test videos parsed: {len(df)}")
    logger.info(f"Real videos (label 0): {(df['label'] == 0).sum()}")
    logger.info(f"Fake videos (label 1): {(df['label'] == 1).sum()}")

    # Verify all files exist
    missing = [p for p in df["video_path"] if not os.path.exists(p)]
    if missing:
        logger.error(f"Missing {len(missing)} videos on disk! First missing: {missing[0]}")
        raise FileNotFoundError(f"Missing videos: {missing[:5]}")
    else:
        logger.info("All 518 Celeb-DF v2 test videos verified present on disk.")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Celeb-DF v2 test split saved to: {output_csv_path}")

    # Pre-extract face crops for deterministic evaluation
    os.makedirs(cache_output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Extracting face crops with MTCNN on {device}...")
    detector = FaceDetector(method="mtcnn", margin=0.15, target_size=(224, 224), confidence_threshold=0.5)

    cached_count = 0
    skipped_count = 0
    t0 = time.time()

    for idx, row in enumerate(df.itertuples()):
        rel_base, _ = os.path.splitext(row.rel_path)
        cache_file = os.path.join(cache_output_dir, rel_base + ".pt")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
            skipped_count += 1
            continue

        raw_frames, sampled_indices = extract_sampled_frames(
            video_path=row.video_path,
            num_frames=16,
            strategy="uniform",
            seed=42,
        )

        crops, crop_meta = detector.process_frame_sequence(raw_frames)
        crops_array = np.stack(crops, axis=0).astype(np.uint8)
        crops_tensor = torch.from_numpy(crops_array)

        save_payload = {
            "face_crops": crops_tensor,
            "sampled_indices": sampled_indices,
            "crop_meta": crop_meta,
            "video_path": row.video_path,
            "label": row.label,
            "source_video_id": row.source_video_id,
            "manipulation_method": row.manipulation_method,
            "dataset": "celeb_df_v2",
        }
        torch.save(save_payload, cache_file)
        cached_count += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            logger.info(f"Processed {idx + 1}/{len(df)} videos (Cached: {cached_count}, Skipped: {skipped_count})")

    elapsed = time.time() - t0
    logger.info(f"Face crop extraction completed in {elapsed:.1f}s ({elapsed/60:.2f} min).")
    logger.info("==================================================================")


if __name__ == "__main__":
    prepare_celebdf_test_split()
