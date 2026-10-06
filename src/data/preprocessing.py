from typing import List, Tuple, Union
import cv2
import numpy as np
import scipy.fftpack
import torch
import torchvision.transforms as T

# Standard ImageNet normalization for RGB backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

rgb_transform = T.Compose([
    T.ToTensor(),  # [0, 255] -> [0.0, 1.0], [H, W, C] -> [C, H, W]
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def compute_fft_magnitude(
    image_rgb: np.ndarray,
    shift_dc: bool = True,
    log_scale: bool = True,
) -> np.ndarray:
    """Compute 2D Fast Fourier Transform magnitude spectrum from an image.

    Args:
        image_rgb: Input RGB image [H, W, 3] or grayscale [H, W] (uint8 or float).
        shift_dc: If True, shift DC zero-frequency component to center.
        log_scale: If True, apply log(1 + |FFT|) scaling.

    Returns:
        Normalized 2D frequency spectrum magnitude [H, W] in range [0, 1].
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.copy()

    # Float conversion
    gray_float = gray.astype(np.float32) / 255.0

    # 2D FFT
    fft2d = np.fft.fft2(gray_float)

    if shift_dc:
        fft2d = np.fft.fftshift(fft2d)

    magnitude = np.abs(fft2d)

    if log_scale:
        magnitude = np.log1p(magnitude)

    # Min-max normalization to [0, 1]
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    if mag_max - mag_min > 1e-8:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)

    return magnitude.astype(np.float32)


def compute_dct_magnitude(
    image_rgb: np.ndarray,
    log_scale: bool = True,
) -> np.ndarray:
    """Compute 2D Discrete Cosine Transform (Type-II) magnitude spectrum.

    Args:
        image_rgb: Input RGB image [H, W, 3] or grayscale [H, W].
        log_scale: If True, apply log(1 + |DCT|) scaling.

    Returns:
        Normalized 2D DCT representation [H, W] in range [0, 1].
    """
    if image_rgb.ndim == 3:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_rgb.copy()

    gray_float = gray.astype(np.float32) / 255.0

    # 2D DCT using scipy.fftpack
    dct2d = scipy.fftpack.dct(scipy.fftpack.dct(gray_float.T, norm='ortho').T, norm='ortho')
    magnitude = np.abs(dct2d)

    if log_scale:
        magnitude = np.log1p(magnitude)

    mag_min = magnitude.min()
    mag_max = magnitude.max()
    if mag_max - mag_min > 1e-8:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude = np.zeros_like(magnitude)

    return magnitude.astype(np.float32)


def preprocess_face_crops(
    face_crops: List[np.ndarray],
    freq_method: str = "fft",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform list of cropped RGB face arrays into batch tensors for RGB and Frequency branches.

    Args:
        face_crops: List of N face crops [224, 224, 3] (uint8).
        freq_method: 'fft' or 'dct'.

    Returns:
        Tuple of:
            - rgb_tensor: torch.Tensor of shape [N, 3, 224, 224]
            - freq_tensor: torch.Tensor of shape [N, 1, 224, 224]
    """
    rgb_tensors = []
    freq_tensors = []

    for crop in face_crops:
        # RGB Tensor
        rgb_t = rgb_transform(crop)
        rgb_tensors.append(rgb_t)

        # Frequency Tensor
        if freq_method == "dct":
            freq_map = compute_dct_magnitude(crop, log_scale=True)
        else:
            freq_map = compute_fft_magnitude(crop, shift_dc=True, log_scale=True)

        freq_t = torch.from_numpy(freq_map).unsqueeze(0)  # [1, H, W]
        freq_tensors.append(freq_t)

    # Stack along sequence dimension: [N, 3, 224, 224] and [N, 1, 224, 224]
    rgb_stacked = torch.stack(rgb_tensors, dim=0)
    freq_stacked = torch.stack(freq_tensors, dim=0)

    return rgb_stacked, freq_stacked
