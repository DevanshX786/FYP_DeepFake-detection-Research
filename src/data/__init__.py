"""Data pipeline, sampling, face detection, frequency transforms, and datasets."""
from src.data.sampler import sample_frame_indices, extract_sampled_frames
from src.data.face_detection import FaceDetector
from src.data.preprocessing import compute_fft_magnitude, compute_dct_magnitude, preprocess_face_crops
from src.data.splits import create_identity_safe_splits, load_split_dataframe
from src.data.dataset import DeepfakeVideoDataset, create_dataloader

__all__ = [
    "sample_frame_indices",
    "extract_sampled_frames",
    "FaceDetector",
    "compute_fft_magnitude",
    "compute_dct_magnitude",
    "preprocess_face_crops",
    "create_identity_safe_splits",
    "load_split_dataframe",
    "DeepfakeVideoDataset",
    "create_dataloader",
]
