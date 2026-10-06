import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.face_detection import FaceDetector
from src.data.sampler import extract_sampled_frames


def main():
    parser = argparse.ArgumentParser(description="Extract and cache face crops offline for fast training.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Input metadata CSV.")
    parser.add_argument("--output_dir", type=str, default="data/processed/face_crops", help="Output directory.")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per video.")
    parser.add_argument("--face_detector", type=str, default="opencv_dnn", help="Detector method.")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata_csv)
    detector = FaceDetector(method=args.face_detector)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Extracting face crops for {len(df)} videos...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vpath = row["video_path"]
        if not os.path.exists(vpath):
            continue

        vid_name = os.path.splitext(os.path.basename(vpath))[0]
        vid_out_dir = os.path.join(args.output_dir, vid_name)
        os.makedirs(vid_out_dir, exist_ok=True)

        try:
            frames, _ = extract_sampled_frames(vpath, num_frames=args.num_frames, strategy="uniform")
            crops, _ = detector.process_frame_sequence(frames)

            for f_idx, crop in enumerate(crops):
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(vid_out_dir, f"frame_{f_idx:02d}.jpg")
                cv2.imwrite(out_path, crop_bgr)
        except Exception as e:
            print(f"Error processing video {vpath}: {e}")

    print("Offline preprocessing completed.")


if __name__ == "__main__":
    main()
