import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class FaceDetector:
    """Dedicated face detection and cropping pipeline for video frames."""

    def __init__(
        self,
        method: str = "mtcnn",
        margin: float = 0.15,
        target_size: Tuple[int, int] = (224, 224),
        fallback_strategy: str = "center_crop",
        confidence_threshold: float = 0.5,
    ):
        """Initialize FaceDetector.

        Args:
            method: 'mtcnn', 'opencv', or 'center_crop'.
            margin: Relative bounding-box expansion factor (default: 0.15).
            target_size: Output (width, height) resolution (default: (224, 224)).
            fallback_strategy: 'center_crop' or 'previous_box'.
            confidence_threshold: Minimum detection confidence score.
        """
        self.method = method
        self.margin = margin
        self.target_size = target_size
        self.fallback_strategy = fallback_strategy
        self.confidence_threshold = confidence_threshold
        self.detector_backend = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize detection backend."""
        if self.method == "mtcnn":
            try:
                from facenet_pytorch import MTCNN
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.detector_backend = MTCNN(keep_all=True, device=device, post_process=False)
            except Exception:
                self.detector_backend = None

    def detect_primary_face(self, frame_rgb: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Detect the primary (largest) face bounding box in a single frame.

        Args:
            frame_rgb: Frame in RGB format [H, W, 3].

        Returns:
            Tuple of (bbox (x1, y1, x2, y2) or None, confidence score).
        """
        h, w = frame_rgb.shape[:2]

        if self.detector_backend is not None:
            try:
                boxes, probs = self.detector_backend.detect(frame_rgb)
                if boxes is not None and len(boxes) > 0:
                    best_idx = int(np.argmax(probs))
                    conf = float(probs[best_idx])
                    if conf >= self.confidence_threshold:
                        box = boxes[best_idx]
                        x1, y1, x2, y2 = [int(v) for v in box]
                        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2)), conf
            except Exception:
                pass

        return None, 0.0

    def crop_and_align_face(
        self,
        frame_rgb: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        last_known_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
        """Crop and square-align a face region using margin expansion and fallback logic.

        Args:
            frame_rgb: Frame in RGB format [H, W, 3].
            bbox: Detected bounding box (x1, y1, x2, y2) or None.
            last_known_bbox: Last successful bounding box across sequence for tracking.

        Returns:
            Tuple of (cropped & resized face RGB [target_h, target_w, 3], used_bbox, success_flag).
        """
        h, w = frame_rgb.shape[:2]
        success = True
        chosen_box = bbox

        if chosen_box is None:
            if self.fallback_strategy == "previous_box" and last_known_bbox is not None:
                chosen_box = last_known_bbox
                success = False
            else:
                # Center square crop fallback
                min_dim = min(h, w)
                cx, cy = w // 2, h // 2
                chosen_box = (cx - min_dim // 2, cy - min_dim // 2, cx + min_dim // 2, cy + min_dim // 2)
                success = False

        x1, y1, x2, y2 = chosen_box
        bw = x2 - x1
        bh = y2 - y1

        # Expand margin
        margin_x = int(bw * self.margin)
        margin_y = int(bh * self.margin)

        nx1 = max(0, x1 - margin_x)
        ny1 = max(0, y1 - margin_y)
        nx2 = min(w, x2 + margin_x)
        ny2 = min(h, y2 + margin_y)

        # Make crop square centered on face
        cw = nx2 - nx1
        ch = ny2 - ny1
        max_side = max(cw, ch)
        center_x = (nx1 + nx2) // 2
        center_y = (ny1 + ny2) // 2

        sq_x1 = max(0, center_x - max_side // 2)
        sq_y1 = max(0, center_y - max_side // 2)
        sq_x2 = min(w, sq_x1 + max_side)
        sq_y2 = min(h, sq_y1 + max_side)

        crop = frame_rgb[sq_y1:sq_y2, sq_x1:sq_x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            crop = frame_rgb

        resized_crop = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        return resized_crop, (sq_x1, sq_y1, sq_x2, sq_y2), success

    def process_frame_sequence(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Process a sequence of frames, detecting and cropping faces with temporal fallback.

        Args:
            frames: List of RGB video frames.

        Returns:
            Tuple of (list of cropped face RGB arrays [224, 224, 3], list of metadata dicts).
        """
        crops = []
        metadata = []
        last_box = None

        batch_boxes = None
        batch_probs = None
        if self.detector_backend is not None and len(frames) > 0:
            try:
                batch_boxes, batch_probs = self.detector_backend.detect(frames)
            except Exception:
                batch_boxes = None
                batch_probs = None

        for idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            bbox = None
            conf = 0.0

            if batch_boxes is not None and idx < len(batch_boxes):
                boxes = batch_boxes[idx]
                probs = batch_probs[idx]
                if boxes is not None and len(boxes) > 0:
                    best_idx = int(np.argmax(probs))
                    conf = float(probs[best_idx])
                    if conf >= self.confidence_threshold:
                        box = boxes[best_idx]
                        x1, y1, x2, y2 = [int(v) for v in box]
                        bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            else:
                bbox, conf = self.detect_primary_face(frame)

            crop, used_box, success = self.crop_and_align_face(frame, bbox, last_known_bbox=last_box)
            if success:
                last_box = used_box

            crops.append(crop)
            metadata.append({
                "frame_idx": idx,
                "detected": success,
                "confidence": conf,
                "bbox": used_box,
            })

        return crops, metadata
