import base64
import io
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from demo.inference import CDTCNetInferenceEngine

FROZEN_CHECKPOINT = "experiments/exp_5_v2_balanced/checkpoints/best_model.pt"
FROZEN_THRESHOLD = 0.65
STATIC_DIR = os.path.join(PROJECT_ROOT, "demo", "static")
SHOWCASE_DIR = os.path.join(PROJECT_ROOT, "demo", "demo_samples_research_showcase")

# Initialize FastAPI app
app = FastAPI(
    title="CDTC-Net Web Prototype",
    description="Cross-Domain Temporal Consistency Network for Deepfake Video Detection",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Lazy-loaded inference engine
_engine: Optional[CDTCNetInferenceEngine] = None


def get_engine() -> CDTCNetInferenceEngine:
    global _engine
    if _engine is None:
        print("[CDTC-Net Server] Initializing model inference engine...")
        _engine = CDTCNetInferenceEngine(
            checkpoint_path=FROZEN_CHECKPOINT,
            decision_threshold=FROZEN_THRESHOLD,
        )
        print("[CDTC-Net Server] Inference engine loaded successfully.")
    return _engine


def encode_image_base64(img_rgb: np.ndarray) -> str:
    """Encode RGB or Grayscale numpy array to base64 JPEG string."""
    if len(img_rgb.shape) == 3:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_rgb
    success, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


class SampleAnalyzeRequest(BaseModel):
    filename: str


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>CDTC-Net Web Prototype: index.html not found</h1>"


@app.get("/api/samples")
async def get_showcase_samples():
    """Return available research showcase samples and their canonical metadata."""
    meta_path = os.path.join(SHOWCASE_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        return JSONResponse({"samples": []})
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return JSONResponse({"samples": meta})


@app.get("/api/sample-video/{filename}")
async def get_sample_video(filename: str):
    """Stream a sample video file for browser preview."""
    # Prevent path traversal
    safe_name = os.path.basename(filename)
    file_path = os.path.join(SHOWCASE_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(file_path, media_type="video/mp4")


@app.post("/api/analyze-sample")
async def analyze_sample(req: SampleAnalyzeRequest):
    """Run inference on one of the prepared research showcase samples."""
    safe_name = os.path.basename(req.filename)
    file_path = os.path.join(SHOWCASE_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Sample '{safe_name}' not found")
    
    engine = get_engine()
    try:
        raw_res = engine.predict_video(file_path, include_frames_in_result=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Process frames and frequency maps to base64
    crops_b64 = [encode_image_base64(c) for c in raw_res.get("face_crops", [])]
    freqs_b64 = [encode_image_base64(f) for f in raw_res.get("freq_maps", [])]

    # Calculate exact dynamic margins
    fake_prob = raw_res["fake_probability"]
    thresh = raw_res["decision_threshold"]
    margin_pp = (fake_prob - thresh) * 100.0

    return {
        "success": True,
        "filename": safe_name,
        "video_url": f"/api/sample-video/{safe_name}",
        "prediction": raw_res["prediction"],
        "fake_probability": fake_prob,
        "real_probability": raw_res["real_probability"],
        "raw_logit": raw_res["raw_logit"],
        "decision_threshold": thresh,
        "margin_pp": margin_pp,
        "is_near_threshold": abs(fake_prob - thresh) <= 0.02,
        "num_sampled_frames": raw_res["num_sampled_frames"],
        "faces_detected": raw_res["faces_detected"],
        "fallback_crops": raw_res["fallback_crops"],
        "inference_time_seconds": raw_res["inference_time_seconds"],
        "transition_magnitudes": raw_res.get("transition_magnitudes", []),
        "face_crops_b64": crops_b64,
        "freq_maps_b64": freqs_b64,
    }


@app.post("/api/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)):
    """Accept an uploaded video file and execute end-to-end inference."""
    suffix = os.path.splitext(file.filename)[1].lower()
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if suffix not in valid_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{suffix}'. Supported: {sorted(list(valid_exts))}")
    
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tfile.write(contents)
        tfile.flush()
        tfile.close()

        engine = get_engine()
        raw_res = engine.predict_video(tfile.name, include_frames_in_result=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tfile.name):
            try:
                os.remove(tfile.name)
            except Exception:
                pass

    crops_b64 = [encode_image_base64(c) for c in raw_res.get("face_crops", [])]
    freqs_b64 = [encode_image_base64(f) for f in raw_res.get("freq_maps", [])]

    fake_prob = raw_res["fake_probability"]
    thresh = raw_res["decision_threshold"]
    margin_pp = (fake_prob - thresh) * 100.0

    return {
        "success": True,
        "filename": file.filename,
        "prediction": raw_res["prediction"],
        "fake_probability": fake_prob,
        "real_probability": raw_res["real_probability"],
        "raw_logit": raw_res["raw_logit"],
        "decision_threshold": thresh,
        "margin_pp": margin_pp,
        "is_near_threshold": abs(fake_prob - thresh) <= 0.02,
        "num_sampled_frames": raw_res["num_sampled_frames"],
        "faces_detected": raw_res["faces_detected"],
        "fallback_crops": raw_res["fallback_crops"],
        "inference_time_seconds": raw_res["inference_time_seconds"],
        "transition_magnitudes": raw_res.get("transition_magnitudes", []),
        "face_crops_b64": crops_b64,
        "freq_maps_b64": freqs_b64,
    }


def run_server(port: int = 8501, host: str = "0.0.0.0"):
    import uvicorn
    print(f"Starting CDTC-Net academic research server on http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(port=port)
