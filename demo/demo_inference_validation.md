# CDTC-Net V2 Demo Inference Equivalence Validation Report

**Date:** 2026-10-06  
**Model Checkpoint:** `experiments/exp_5_v2_balanced/checkpoints/best_model.pt` (Epoch 14, Val AUC 0.7663)  
**Decision Threshold:** $\tau^* = 0.6500$ (Frozen validation-derived)  
**Target Agreement:** 100% categorical agreement and $< 10^{-4}$ numerical logit difference.  

## Validation Protocol
1. **Path A (Research Evaluation Pipeline):** `DeepfakeVideoDataset` + `CDTCNet` forward pass directly reproducing the finalized EXP-5 V2 evaluation protocol.
2. **Path B (Demo Inference Pipeline):** `demo/inference.py` (`CDTCNetInferenceEngine.predict_video`) powering the interactive Streamlit demonstration.
3. **Test Videos:** 10 deterministic test split videos across 5 manipulation categories (Original Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures).

## Quantitative Results

| Vid | Category | Video Path | GT Label | Path A Logit | Path B Logit | Logit Diff | Path A P(Fake) | Path B P(Fake) | Prob Diff | Pred A | Pred B | Equivalence |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `original` | `data/raw/faceforensics/original/006.mp4` | REAL | -1.9268 | -1.9268 | 0.00e+00 | 0.1271 | 0.1271 | 0.00e+00 | **REAL** | **REAL** | `PASS` |
| 2 | `original` | `data/raw/faceforensics/original/025.mp4` | REAL | 1.8652 | 1.8652 | 0.00e+00 | 0.8657 | 0.8657 | 0.00e+00 | **FAKE** | **FAKE** | `PASS` |
| 3 | `deepfakes` | `data/raw/faceforensics/Deepfakes/006_002.mp4` | FAKE | -1.4570 | -1.4570 | 0.00e+00 | 0.1890 | 0.1890 | 0.00e+00 | **REAL** | **REAL** | `PASS` |
| 4 | `deepfakes` | `data/raw/faceforensics/Deepfakes/025_067.mp4` | FAKE | 1.9873 | 1.9873 | 0.00e+00 | 0.8794 | 0.8794 | 0.00e+00 | **FAKE** | **FAKE** | `PASS` |
| 5 | `face2face` | `data/raw/faceforensics/Face2Face/006_002.mp4` | FAKE | 0.1812 | 0.1812 | 0.00e+00 | 0.5449 | 0.5449 | 0.00e+00 | **REAL** | **REAL** | `PASS` |
| 6 | `face2face` | `data/raw/faceforensics/Face2Face/025_067.mp4` | FAKE | 2.5137 | 2.5137 | 0.00e+00 | 0.9253 | 0.9253 | 0.00e+00 | **FAKE** | **FAKE** | `PASS` |
| 7 | `faceswap` | `data/raw/faceforensics/FaceSwap/006_002.mp4` | FAKE | -1.9893 | -1.9893 | 0.00e+00 | 0.1204 | 0.1204 | 0.00e+00 | **REAL** | **REAL** | `PASS` |
| 8 | `faceswap` | `data/raw/faceforensics/FaceSwap/025_067.mp4` | FAKE | 2.5234 | 2.5234 | 0.00e+00 | 0.9258 | 0.9258 | 0.00e+00 | **FAKE** | **FAKE** | `PASS` |
| 9 | `neuraltextures` | `data/raw/faceforensics/NeuralTextures/006_002.mp4` | FAKE | 0.5391 | 0.5391 | 0.00e+00 | 0.6318 | 0.6318 | 0.00e+00 | **REAL** | **REAL** | `PASS` |
| 10 | `neuraltextures` | `data/raw/faceforensics/NeuralTextures/025_067.mp4` | FAKE | 2.1367 | 2.1367 | 0.00e+00 | 0.8945 | 0.8945 | 0.00e+00 | **FAKE** | **FAKE** | `PASS` |

## Equivalence Summary
- **Total Test Videos:** 10
- **Categorical Prediction Agreement:** 10/10 (100.0%)
- **Max Absolute Logit Difference:** 0.00e+00
- **Max Absolute Probability Difference:** 0.00e+00
- **Overall Validation Status:** `PASSED - EXACT NUMERICAL EQUIVALENCE`

## Conclusion
The standalone demo inference pipeline (`demo/inference.py`) produces exact numerical equivalence to the research evaluation pipeline for CDTC-Net V2. No discrepancies exist.