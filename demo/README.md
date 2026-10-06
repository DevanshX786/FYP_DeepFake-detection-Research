# CDTC-Net Demonstration Application

Interactive demonstration interface for the **Cross-Domain Temporal Consistency Network (CDTC-Net)** for Deepfake Video Detection.

This application provides a demonstration of the finalized **EXP-5 V2 (Class-Balanced)** research model, analyzing spatial RGB facial appearance, 2D Fast Fourier Transform (2D-FFT) spectral characteristics, and explicit inter-frame temporal transitions across video sequences.

---

## 1. System Requirements & Dependencies

- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python Version:** Python 3.10, 3.11, or 3.12
- **Hardware Requirements:**
  - **GPU (Recommended):** NVIDIA GPU with CUDA support (VRAM $\ge 4\text{ GB}$). Inference time: $\sim 0.6\text{--}1.8\text{ s}$ per video.
  - **CPU (Fallback):** Multi-core x86_64 CPU. Inference time: $\sim 4.0\text{--}8.0\text{ s}$ per video.
- **Key Python Packages:**
  - `torch`, `torchvision`, `torchaudio`
  - `fastapi`, `uvicorn`
  - `facenet-pytorch`
  - `opencv-python`
  - `scipy`, `numpy`, `pandas`

---

## 2. Quickstart & Launch Command

From the root directory of the repository, execute:

```bash
python demo/server.py 8501
```

The application will launch and be accessible in your web browser at:
`http://localhost:8501`

---

## 3. Supported File Formats & Input Specifications

- **Supported Video Formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- **Video Duration:** Any length (deterministic uniform sampling extracts exactly 16 frames spanning the sequence)
- **Resolution:** Any input resolution (automatically detected, cropped, and resized to $224 \times 224$ face crops)
- **Demo Samples:** Curated panel research showcase clips (including true negatives, true positives, a real false positive failure case, and a boundary case) are provided in `demo/demo_samples_research_showcase/` for instant loading in the UI.

---

## 4. Architecture & Checkpoint Traceability

- **Model Class:** `CDTCNet` ([`src/models/cdtc_net.py`](../src/models/cdtc_net.py))
- **Model Architecture:**
  - **Spatial Stream:** EfficientNet-B0 RGB encoder ($256\text{-D}$)
  - **Spectral Stream:** Custom 4-layer CNN on 2D-FFT DC-shifted log-magnitude spectra ($256\text{-D}$)
  - **Fusion:** Cross-Domain Linear Projection ($512 \to 256\text{-D}$)
  - **Temporal Difference Module:** Signed ($D_t$) + Absolute ($|D_t|$) transitions ($512 \to 256\text{-D}$)
  - **Temporal Consistency Transformer:** 2 layers, 8 heads, $d_{\text{model}}=256$, $\text{FFN}=512$ ($31\text{ tokens}$)
  - **Classifier:** Multi-Layer Perceptron ($256 \to 128 \to 1$)
- **Model Checkpoint Path:** [`experiments/exp_5_v2_balanced/checkpoints/best_model.pt`](../experiments/exp_5_v2_balanced/checkpoints/best_model.pt)
- **Validation ROC-AUC at Checkpoint:** `0.7663` (Epoch 14)
- **Decision Threshold:** $\tau^* = 0.6500$ (Frozen validation-calibrated threshold)

---

## 5. Offline & Security Verification

- **100% Offline Operation:** No internet connectivity, cloud API calls, or remote resource downloads occur during runtime.
- **Read-Only Model Weights:** Checkpoints and experiment logs are strictly opened in read-only mode (`torch.load`).
- **Temporary File Lifecycle:** Uploaded video bytes are written to an isolated temporary file and automatically deleted after inference completes.

---

## 6. Known Behavioral Characteristics & Scientific Context

- **Class-Balanced Optimization (V2):** Trained with $\text{pos\_weight} = 0.25$ ($\omega_{\text{real}}=1.0, \omega_{\text{fake}}=0.25$) to mitigate the positive-class bias of V1. Real video specificity was increased from $34.67\%$ to $64.00\%$ on FaceForensics++ c23 and from $12.36\%$ to $75.28\%$ on Celeb-DF v2 zero-shot.
- **Calibrated Decision Boundary ($\tau^* = 0.65$):** Selected strictly using validation set balanced accuracy. Real videos consistently score lower $P(\text{fake})$ ($0.10\text{--}0.35$), providing clean separation from manipulated videos.
- **Academic Integrity:** The demo displays genuine intermediate tensors (face crops, 2D FFT spectra, $L_2$ transition dynamics) and does not synthesize artificial heatmap explanations or override predictions.
