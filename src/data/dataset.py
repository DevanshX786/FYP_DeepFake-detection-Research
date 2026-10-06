import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.face_detection import FaceDetector
from src.data.preprocessing import preprocess_face_crops
from src.data.sampler import extract_sampled_frames, sample_frame_indices


class DeepfakeVideoDataset(Dataset):
    """PyTorch Dataset for multi-domain video-level Deepfake detection."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_frames: int = 16,
        sampling_strategy: str = "uniform",
        freq_method: str = "fft",
        cache_dir: Optional[str] = "data/processed/face_crops",
        face_detector: Optional[FaceDetector] = None,
        transform: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        """Initialize DeepfakeVideoDataset.

        Args:
            dataframe: DataFrame containing at least ['video_path', 'label'].
            num_frames: Number of sampled frames per video (default: 16).
            sampling_strategy: 'uniform', 'random', or 'deterministic'.
            freq_method: 'fft' or 'dct'.
            cache_dir: Directory containing pre-extracted face crops (.pt).
            face_detector: Instance of FaceDetector. If None, initialized with defaults.
            transform: Optional additional augmentations.
            seed: Optional seed for frame sampler.
        """
        self.df = dataframe.reset_index(drop=True)
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.freq_method = freq_method
        self.cache_dir = cache_dir
        self.face_detector = face_detector if face_detector is not None else FaceDetector()
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def _get_cache_path(self, video_path: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        norm_path = os.path.normpath(video_path).replace("\\", "/")
        if "data/raw/faceforensics/" in norm_path:
            rel = norm_path.split("data/raw/faceforensics/")[1]
        elif "faceforensics/" in norm_path:
            rel = norm_path.split("faceforensics/")[1]
        elif "data/raw/celebdf/" in norm_path:
            rel = "celebdf/" + norm_path.split("data/raw/celebdf/")[1]
        elif "celebdf/" in norm_path:
            rel = "celebdf/" + norm_path.split("celebdf/")[1]
        else:
            rel = os.path.basename(norm_path)
        base, _ = os.path.splitext(rel)
        cache_file = os.path.join(self.cache_dir, base + ".pt")
        return cache_file if os.path.exists(cache_file) else None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        video_path = str(row["video_path"])
        label = float(row["label"])

        cache_path = self._get_cache_path(video_path)
        if cache_path is not None:
            try:
                cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                crops_tensor = cached_data["face_crops"]  # [16, 224, 224, 3] uint8
                if isinstance(crops_tensor, torch.Tensor):
                    crops_list = [crops_tensor[i].numpy() for i in range(crops_tensor.shape[0])]
                else:
                    crops_list = [crops_tensor[i] for i in range(len(crops_tensor))]

                rgb_tensor, freq_tensor = preprocess_face_crops(crops_list, freq_method=self.freq_method)

                return {
                    "rgb": rgb_tensor,          # Shape: [N, 3, 224, 224]
                    "freq": freq_tensor,        # Shape: [N, 1, 224, 224]
                    "label": torch.tensor(label, dtype=torch.float32),  # Shape: scalar
                    "video_path": video_path,
                    "manipulation_method": row.get("manipulation_method", "unknown"),
                    "dataset": row.get("dataset", "unknown"),
                    "source_video_id": row.get("source_video_id", "unknown"),
                }
            except Exception:
                pass  # Fall back to dynamic extraction if cache read fails

        # Dynamic fallback
        if os.path.exists(video_path):
            raw_frames, sampled_indices = extract_sampled_frames(
                video_path=video_path,
                num_frames=self.num_frames,
                strategy=self.sampling_strategy,
                seed=self.seed,
            )
        else:
            raw_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        face_crops, crop_meta = self.face_detector.process_frame_sequence(raw_frames)
        rgb_tensor, freq_tensor = preprocess_face_crops(face_crops, freq_method=self.freq_method)

        return {
            "rgb": rgb_tensor,          # Shape: [N, 3, 224, 224]
            "freq": freq_tensor,        # Shape: [N, 1, 224, 224]
            "label": torch.tensor(label, dtype=torch.float32),  # Shape: scalar
            "video_path": video_path,
            "manipulation_method": row.get("manipulation_method", "unknown"),
            "dataset": row.get("dataset", "unknown"),
            "source_video_id": row.get("source_video_id", "unknown"),
        }


import random

def seed_worker(worker_id: int) -> None:
    """Worker initialization function to enforce deterministic seed per DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass



def create_dataloader(
    dataset: DeepfakeVideoDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = 42,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """Create an optimized DataLoader for DeepfakeVideoDataset.

    Args:
        dataset: DeepfakeVideoDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Subprocess workers for data loading (default: 2).
        pin_memory: If True, pin memory for faster GPU transfer.
        drop_last: Whether to drop last incomplete batch.
        seed: Random seed for deterministic worker generation.
        prefetch_factor: Number of batches loaded in advance by each worker (default: 2).

    Returns:
        Configured DataLoader instance.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "drop_last": drop_last,
    }
    if generator is not None:
        kwargs["generator"] = generator

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor if prefetch_factor is not None else 2
        kwargs["worker_init_fn"] = seed_worker

    return DataLoader(dataset, **kwargs)

