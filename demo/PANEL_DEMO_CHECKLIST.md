# Final Year Project Panel Presentation Demo Checklist

> **Project Title:** CDTC-Net: Cross-Domain Temporal Consistency Network for Deepfake Video Detection  
> **Model Checkpoint:** Frozen `EXP-5 V2 (Class-Balanced)` (`experiments/exp_5_v2_balanced/checkpoints/best_model.pt`)  
> **Decision Threshold:** Frozen at $\tau^* = 0.6500$ (Validation-Derived)  

---

## 1. Quick Launch & Pre-Flight Check

1. **Open Terminal** in the project root (`d:\Project\DeepFake Detection_FYP`).
2. **Launch Application:**
   ```bash
   streamlit run demo/app.py
   ```
3. **Verify Browser Access:** Open `http://localhost:8501`.
4. **Ensure Device Configuration:** Verify `EXP-5 V2 (Class-Balanced)` and `Threshold: 0.65` are displayed in the sidebar.

---

## 2. Research Showcase Demonstration Samples (`demo/demo_samples_research_showcase/`)

| Index | Clip Name | Video Source | Ground Truth | Expected Prediction | Model Probability | Showcase Role & Presentation Talking Points |
| :---: | :---| :---| :---: | :---: | :---: | :---|
| **1** | `01_REAL_CORRECT_032.mp4` | FF++ Original `032.mp4` | **REAL** | **`REAL`** | $P_{\text{real}} = 95.2\%$ ($P_{\text{fake}} = 4.8\%$) | **True Negative (Capability):** Observed output shows high authentic probability ($P_{\text{real}} = 95.2\%$). The low prediction is consistent with authentic facial dynamics. |
| **2** | `02_REAL_CORRECT_057.mp4` | FF++ Original `057.mp4` | **REAL** | **`REAL`** | $P_{\text{real}} = 95.2\%$ ($P_{\text{fake}} = 4.8\%$) | **True Negative (Capability):** Authentic sequence with head motion; model maintains low fake probability ($4.8\%$) across 16 sampled frames. |
| **3** | `03_REAL_FALSE_POSITIVE_459.mp4` | FF++ Original `459.mp4` | **REAL** | **`FAKE`** | $P_{\text{real}} = 6.9\%$ ($P_{\text{fake}} = 93.1\%$) | **Failure Case — False Positive (Limitation):** Honest presentation of model limitation. Hypothesized contributing factors include complex studio shadows and compression gradients, though exact feature attribution is not experimentally isolated here. |
| **4** | `04_FAKE_CORRECT_DEEPFAKES_030_193.mp4` | FF++ Deepfakes `030_193.mp4` | **FAKE** | **`FAKE`** | $P_{\text{fake}} = 93.3\%$ ($P_{\text{real}} = 6.7\%$) | **True Positive (Capability):** Autoencoder face synthesis; observed output is consistent with manipulation detection across the video sequence. |
| **5** | `05_FAKE_CORRECT_FACETOFACE_025_067.mp4` | FF++ Face2Face `025_067.mp4` | **FAKE** | **`FAKE`** | $P_{\text{fake}} = 92.5\%$ ($P_{\text{real}} = 7.5\%$) | **True Positive (Capability):** Re-enactment manipulation; observed prediction reflects elevated fake probability. |
| **6** | `06_FAKE_CORRECT_NEURALTEXTURES_030_193.mp4` | FF++ NeuralTextures `030_193.mp4` | **FAKE** | **`FAKE`** | $P_{\text{fake}} = 93.3\%$ ($P_{\text{real}} = 6.7\%$) | **True Positive (Capability):** GAN rendering manipulation; observed prediction is consistent with synthetic facial texture detection. |
| **7** | `07_BORDERLINE_REAL_027.mp4` | FF++ Original `027.mp4` | **REAL** | **`REAL`** | $P_{\text{fake}} = 64.9\%$ ($P_{\text{real}} = 35.1\%$) | **Near-threshold / borderline prediction:** Authentic video scoring within $0.0006$ of the frozen $\tau = 0.65$ threshold, illustrating decision boundary sensitivity. |

---

## 3. Presentation Walkthrough Script

### A. Main Screen Walkthrough (1–2 minutes)
1. **Selection / Upload:** Select a prepared demo sample from the dropdown or upload a custom video file.
2. **Execution Stages:** Click **Analyze Video** and guide the panel through the 8 progress stages:
   - *Loading video $\to$ Sampling 16 frames $\to$ Detecting faces $\to$ RGB feature extraction $\to$ Frequency feature extraction $\to$ Temporal difference computation $\to$ Temporal Transformer inference $\to$ Prediction generation*.
3. **Verdict & Probabilities:**
   - Highlight the prominent **Prediction Banner** (`REAL` / `FAKE`).
   - Point out the distinction between model probabilities ($P(\text{real})$ vs $P(\text{fake})$) and the frozen decision boundary ($\tau^* = 0.65$).
   - Show the compact **Processing Summary** (16 frames, 16 faces detected, 0 fallbacks, $<1.5\text{ s}$ latency).
4. **Failure Case Presentation (`03_REAL_FALSE_POSITIVE_459.mp4`):**
   - Direct attention to the explicit callout: *“Failure Case — False Positive: Ground truth is REAL, but model predicts FAKE ($P=93.1\%$).”*
   - Emphasize that live deepfake detection systems must acknowledge false-positive failure modes (54 false positives out of 150 real videos in the full test evaluation), and discuss potential contributing factors such as lighting gradients and codec compression.

### B. Research Deep-Dive & Visualizations (2–3 minutes)
1. **Expand "🔬 Frame Analysis & Internal Representations":**
   - **Sampled Face Crops:** Show the 16 MTCNN-detected face crops ($224 \times 224$) with consistent spatial bounding boxes.
   - **2D-FFT Magnitude Representations:** Explain how 2D Fourier transforms compute log-power spectra to display frequency energy distribution.
   - **Temporal Feature Transitions:** Show the line chart of $\|F_{t+1} - F_t\|_2$, illustrating how explicit signed and absolute differences ($[D_t, |D_t|]$) quantify inter-frame feature dynamics.
2. **Expand "ℹ️ How CDTC-Net Works":**
   - Walk through the dual-stream EfficientNet-B0 + Frequency CNN encoders, multi-domain fusion, 31-token sequence ($16\text{ frames} + 15\text{ differences}$), and 2-layer temporal Transformer.
3. **Expand "📑 Experimental Protocol & Research Context":**
   - Reference the canonical benchmark evaluation on FaceForensics++ (c23, 750 test videos: $\text{TN}=96, \text{FP}=54, \text{FN}=214, \text{TP}=386$; Accuracy: $64.27\%$, Specificity: $64.00\%$, ROC-AUC: $70.12\%$) and zero-shot cross-dataset evaluation on Celeb-DF v2.

---

## 4. Addressing Panel Questions

| Potential Panel Question | Recommended Technical Response |
| :--- | :--- |
| *“Why was the decision threshold set to 0.65 instead of 0.50?”* | *“In our V2 class-balanced training, the loss function was weighted with $\text{pos\_weight} = 0.25$ to combat severe false positives. A threshold sweep on the validation set determined that $\tau^* = 0.65$ maximizes balanced accuracy to $71.83\%$, aligning with the natural score separation between real and fake distributions.”* |
| *“How did class-balanced training improve the model over V1?”* | *“V1 had a severe false-positive bias, with only $34.67\%$ specificity on FF++ and $12.36\%$ on Celeb-DF. V2 training increased real-video specificity to $64.00\%$ on FF++ and $75.28\%$ on Celeb-DF, substantially reducing false alarms on authentic content.”* |
| *“Are the visual frequency maps and transition plots generated post-hoc?”* | *“No, all visualizations are genuine intermediate tensors extracted directly from the inference pipeline forward pass: the 16 face crops from MTCNN, the 16 2D-FFT magnitude arrays from the spectral preprocessor, and the 15 transition $L_2$ norms from the temporal difference module.”* |
| *“Why did the model misclassify video 459 (the false positive)?”* | *“In our test evaluation, 54 of 150 authentic videos scored above the 0.65 threshold. Hypothesized contributing factors include complex shadow patterns, lighting gradients across the face, and codec compression artifacts that produce spectral distributions resembling manipulated faces. However, establishing exact component-level causality would require dedicated attribution analysis.”* |
