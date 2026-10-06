import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import cv2
import numpy as np
import torch

# Filter cosmetic PyTorch nested tensor warning
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.face_detection import FaceDetector
from src.data.preprocessing import preprocess_face_crops
from src.data.sampler import extract_sampled_frames, sample_frame_indices
from src.models.cdtc_net import CDTCNet
from src.utils.checkpointing import CheckpointManager

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class CDTCNetInferenceEngine:
    """Isolated, standalone inference engine for frozen CDTC-Net model (EXP-5 V2 Balanced)."""

    def __init__(
        self,
        checkpoint_path: str = "experiments/exp_5_v2_balanced/checkpoints/best_model.pt",
        device: Optional[Union[str, torch.device]] = None,
        num_frames: int = 16,
        face_confidence_threshold: float = 0.5,
        face_margin: float = 0.15,
        decision_threshold: float = 0.65,
    ):
        """Initialize the inference pipeline with model weights and detector.

        Args:
            checkpoint_path: Path to the frozen EXP-5 V2 best_model.pt checkpoint.
            device: Torch device or device string ('cuda', 'cpu', None for auto).
            num_frames: Number of deterministic uniform frames to extract (default: 16).
            face_confidence_threshold: Minimum MTCNN confidence score.
            face_margin: Bounding-box margin expansion factor.
            decision_threshold: Frozen validation-derived decision threshold (default: 0.65).
        """
        self.num_frames = num_frames
        self.checkpoint_path = checkpoint_path
        self.decision_threshold = decision_threshold

        # Setup compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                print(f"[WARNING] CUDA requested ({device}) but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        # Load Model
        self.model = self._load_model(self.checkpoint_path)

        # Initialize Face Detector
        self.detector = FaceDetector(
            method="mtcnn",
            margin=face_margin,
            target_size=(224, 224),
            fallback_strategy="previous_box",
            confidence_threshold=face_confidence_threshold,
        )

    def _load_model(self, checkpoint_path: str) -> CDTCNet:
        """Instantiate CDTC-Net architecture and load frozen weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CDTC-Net checkpoint file not found at: {checkpoint_path}")

        try:
            # Instantiate model (pretrained_backbone=False since we load weights from checkpoint)
            model = CDTCNet(
                pretrained_backbone=False,
                freeze_rgb_backbone=False,
                feature_dim=256,
                freq_in_channels=1,
                transformer_heads=8,
                transformer_layers=2,
                transformer_ff_dim=512,
                dropout_classifier=0.3,
            )

            # Load checkpoint using existing CheckpointManager
            chk_mgr = CheckpointManager(checkpoint_dir=os.path.dirname(checkpoint_path))
            chk_mgr.load(model=model, checkpoint_path=checkpoint_path, device=self.device)

            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load CDTC-Net model from {checkpoint_path}: {str(e)}") from e

    def _validate_video_path(self, video_path: str) -> None:
        """Validate existence and supported format for target video."""
        if not video_path:
            raise ValueError("Provided video path is empty.")

        # Check for zip-embedded paths (e.g. 'archive.zip::path/to/video.mp4')
        if "::" in video_path:
            zip_file_path, _ = video_path.split("::", 1)
            if not os.path.exists(zip_file_path):
                raise FileNotFoundError(f"ZIP archive not found: {zip_file_path}")
            return

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        _, ext = os.path.splitext(video_path)
        if ext.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{ext}'. Supported extensions: {sorted(list(SUPPORTED_VIDEO_EXTENSIONS))}"
            )

    def _get_video_frame_count(self, video_path: str) -> int:
        """Obtain total video frame count."""
        if "::" in video_path:
            # For zip archive paths, extract_sampled_frames handles reading
            return -1

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return -1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    @torch.no_grad()
    def predict_video(
        self,
        video_path: str,
        include_frames_in_result: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run end-to-end inference on a single video file.

        Args:
            video_path: Absolute or relative path to target video file.
            include_frames_in_result: Whether to attach raw frames and face crops in output dictionary.
            progress_callback: Optional callable fn(stage_name: str, step: int, total_steps: int)

        Returns:
            Structured dictionary containing predictions, probabilities, metadata, and timing.
        """
        start_time = time.time()

        if progress_callback:
            progress_callback("Loading video", 1, 8)

        self._validate_video_path(video_path)
        total_frames = self._get_video_frame_count(video_path)

        # 1. Frame Sampling & Video Decoding
        if progress_callback:
            progress_callback("Sampling 16 frames", 2, 8)

        sampling_start = time.time()
        try:
            frames_rgb, sampled_indices = extract_sampled_frames(
                video_path=video_path,
                num_frames=self.num_frames,
                strategy="uniform",
            )
        except Exception as e:
            raise RuntimeError(f"Failed to read/sample frames from video '{video_path}': {str(e)}") from e

        if not frames_rgb or len(frames_rgb) == 0:
            raise RuntimeError(f"No frames could be extracted from video '{video_path}'.")
        sampling_time = time.time() - sampling_start

        # 2. Face Detection and Cropping
        if progress_callback:
            progress_callback("Detecting faces", 3, 8)

        detection_start = time.time()
        face_crops, face_metadata = self.detector.process_frame_sequence(frames_rgb)
        detection_time = time.time() - detection_start

        faces_detected = sum(1 for m in face_metadata if m.get("detected", False))
        fallback_crops = len(face_metadata) - faces_detected

        # 3. Dual-Domain Tensor Preprocessing (RGB + 2D-FFT)
        prep_start = time.time()
        rgb_tensor, freq_tensor = preprocess_face_crops(face_crops, freq_method="fft")
        prep_time = time.time() - prep_start

        # Add batch dimension [1, N, C, H, W]
        rgb_batch = rgb_tensor.unsqueeze(0).to(self.device, non_blocking=True)
        freq_batch = freq_tensor.unsqueeze(0).to(self.device, non_blocking=True)

        # 4. Neural Network Forward Pass with 8-stage progress tracking
        forward_start = time.time()
        use_cuda_autocast = self.device.type == "cuda"
        with torch.amp.autocast('cuda', enabled=use_cuda_autocast):
            if progress_callback:
                progress_callback("Extracting RGB features", 4, 8)
            rgb_tokens = self.model.rgb_encoder(rgb_batch)

            if progress_callback:
                progress_callback("Extracting frequency features", 5, 8)
            freq_tokens = self.model.freq_encoder(freq_batch)

            frame_tokens = self.model.fusion(rgb_tokens, freq_tokens)

            if progress_callback:
                progress_callback("Computing temporal differences", 6, 8)
            diff_tokens = self.model.temp_diff(frame_tokens)

            if progress_callback:
                progress_callback("Temporal Transformer inference", 7, 8)
            video_repr = self.model.transformer(frame_tokens, diff_tokens)

            if progress_callback:
                progress_callback("Generating prediction", 8, 8)
            logits = self.model.classifier(video_repr)

            prob = torch.sigmoid(logits).item()
            raw_logit = logits.item()

            # Compute genuine L2 transition magnitude across consecutive frame tokens
            # D_t = F_{t+1} - F_t for t = 0 ... 14
            d_t_vectors = frame_tokens[0, 1:, :] - frame_tokens[0, :-1, :]  # [15, 256]
            transition_mags = torch.norm(d_t_vectors, p=2, dim=-1).cpu().numpy().tolist()

        forward_time = time.time() - forward_start
        total_elapsed = time.time() - start_time

        # 5. Final Classification (Frozen 0.65 threshold)
        threshold = self.decision_threshold
        prediction = "FAKE" if prob >= threshold else "REAL"
        real_prob = 1.0 - prob

        result: Dict[str, Any] = {
            "video_path": video_path,
            "prediction": prediction,
            "fake_probability": float(prob),
            "real_probability": float(real_prob),
            "raw_logit": float(raw_logit),
            "decision_threshold": threshold,
            "total_video_frames": total_frames,
            "num_sampled_frames": len(sampled_indices),
            "sampled_frame_indices": sampled_indices,
            "faces_detected": faces_detected,
            "fallback_crops": fallback_crops,
            "face_metadata": face_metadata,
            "transition_magnitudes": [float(v) for v in transition_mags],
            "device": str(self.device),
            "inference_time_seconds": total_elapsed,
            "timing_breakdown": {
                "frame_sampling_s": sampling_time,
                "face_detection_s": detection_time,
                "preprocessing_s": prep_time,
                "forward_pass_s": forward_time,
                "total_s": total_elapsed,
            },
        }

        if include_frames_in_result:
            result["raw_frames"] = frames_rgb
            result["face_crops"] = face_crops
            # Extract frequency magnitude maps as uint8 for display [16, 224, 224]
            freq_np = freq_tensor.squeeze(1).cpu().numpy()  # [16, 224, 224]
            result["freq_maps"] = [(freq_np[i] * 255.0).astype(np.uint8) for i in range(len(freq_np))]

        return result


def run_inference(
    video_path: str,
    checkpoint_path: str = "experiments/exp_5_v2_balanced/checkpoints/best_model.pt",
    device: Optional[str] = None,
    decision_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Convenience functional interface for video inference."""
    engine = CDTCNetInferenceEngine(
        checkpoint_path=checkpoint_path,
        device=device,
        decision_threshold=decision_threshold,
    )
    return engine.predict_video(video_path)


def main():
    parser = argparse.ArgumentParser(description="Standalone CDTC-Net Deepfake Detection Inference.")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input video file.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default="experiments/exp_5_v2_balanced/checkpoints/best_model.pt",
        help="Path to CDTC-Net model checkpoint (.pt).",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.65,
        help="Decision threshold (default: 0.65).",
    )
    parser.add_argument("--device", "-d", type=str, default=None, help="Device ('cuda', 'cpu', or auto).")
    args = parser.parse_args()

    try:
        engine = CDTCNetInferenceEngine(
            checkpoint_path=args.checkpoint,
            device=args.device,
            decision_threshold=args.threshold,
        )
        result = engine.predict_video(args.video, include_frames_in_result=False)

        print("\nCDTC-Net Inference")
        print("------------------")
        print(f"Video: {result['video_path']}")
        print(f"Frames sampled: {result['num_sampled_frames']}")
        print(f"Faces detected: {result['faces_detected']}")
        print(f"Fallback crops: {result['fallback_crops']}")
        print(f"Raw logit: {result['raw_logit']:.4f}")
        print(f"Fake probability: {result['fake_probability']:.4f}")
        print(f"Real probability: {result['real_probability']:.4f}")
        print(f"Prediction: {result['prediction']}")
        print(f"Inference time: {result['inference_time_seconds']:.3f} s\n")

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
