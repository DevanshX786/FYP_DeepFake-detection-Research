import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.sampler import extract_sampled_frames
from src.data.face_detection import FaceDetector
from src.data.preprocessing import preprocess_face_crops

def test_speed():
    df = pd.read_csv("data/splits/train.csv")
    fd = FaceDetector()
    sample_videos = df["video_path"].head(5).tolist()
    
    t0 = time.time()
    for i, path in enumerate(sample_videos):
        t_start = time.time()
        raw, idxs = extract_sampled_frames(path, num_frames=16, strategy="uniform")
        crops, meta = fd.process_frame_sequence(raw)
        rgb_t, freq_t = preprocess_face_crops(crops, freq_method="fft")
        print(f"Video {i+1}: {os.path.basename(path)} in {time.time()-t_start:.3f}s | rgb shape: {rgb_t.shape}, freq shape: {freq_t.shape}")
    total_t = time.time() - t0
    print(f"5 videos took {total_t:.2f}s (Average: {total_t/5:.3f}s/video)")

if __name__ == "__main__":
    test_speed()
