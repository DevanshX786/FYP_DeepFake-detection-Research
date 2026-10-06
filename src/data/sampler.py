import random
from typing import List, Optional, Tuple
import cv2
import numpy as np


def sample_frame_indices(
    total_frames: int,
    num_frames: int = 16,
    strategy: str = "uniform",
    seed: Optional[int] = None,
) -> List[int]:
    """Sample frame indices from a video sequence.

    Args:
        total_frames: Total number of frames available in video.
        num_frames: Number of frames to sample (default: 16).
        strategy: 'uniform' (deterministic evenly spaced), 'random' (random within uniform bins),
                  or 'deterministic' (identical to uniform).
        seed: Optional random seed for deterministic randomized sampling.

    Returns:
        List of 0-indexed frame indices of length `num_frames`.
    """
    if total_frames <= 0:
        return [0] * num_frames

    if total_frames < num_frames:
        # If video is shorter than requested frames, tile indices cyclically
        indices = [i % total_frames for i in range(num_frames)]
        return sorted(indices)

    if strategy in ("uniform", "deterministic"):
        # Evenly spaced sampling across entire video duration
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        return indices

    elif strategy == "random":
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        # Segment video into num_frames equal bins and choose one random frame per bin
        bin_edges = np.linspace(0, total_frames, num_frames + 1, dtype=int)
        indices = []
        for i in range(num_frames):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            if end <= start:
                chosen = min(start, total_frames - 1)
            else:
                chosen = rng.randint(start, end - 1)
            indices.append(int(chosen))
        return sorted(indices)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}. Choose 'uniform', 'random', or 'deterministic'.")


import os
import tempfile
import zipfile


def extract_sampled_frames(
    video_path: str,
    num_frames: int = 16,
    strategy: str = "uniform",
    seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """Read a video file and extract sampled frames as RGB numpy arrays.
    Supports both direct video file paths and embedded ZIP paths ('archive.zip::internal_path.mp4').

    Args:
        video_path: Path to video file or 'zip_path::internal_path.mp4'.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ('uniform', 'random', 'deterministic').
        seed: Optional seed.

    Returns:
        Tuple of (list of RGB numpy arrays [H, W, 3], list of frame indices sampled).
    """
    tmp_path = None
    actual_path = video_path

    if "::" in video_path:
        zip_file_path, internal_path = video_path.split("::", 1)
        if not os.path.exists(zip_file_path):
            raise IOError(f"Source ZIP not found: {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, "r") as zf:
            video_bytes = zf.read(internal_path)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        actual_path = tmp_path

    try:
        cap = cv2.VideoCapture(actual_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sample_frame_indices(total_frames, num_frames=num_frames, strategy=strategy, seed=seed)

        frames = []
        current_idx = 0
        target_set = set(indices)
        frame_dict = {}

        # Read frames sequentially for speed and video codec stability
        max_target = max(indices) if indices else 0
        while cap.isOpened() and current_idx <= max_target:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in target_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_dict[current_idx] = frame_rgb
            current_idx += 1

        cap.release()

        # Reconstruct in order of sampled indices
        fallback_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        for idx in indices:
            if idx in frame_dict:
                frames.append(frame_dict[idx])
            elif len(frames) > 0:
                frames.append(frames[-1].copy())
            else:
                frames.append(fallback_frame.copy())

        return frames, indices
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

