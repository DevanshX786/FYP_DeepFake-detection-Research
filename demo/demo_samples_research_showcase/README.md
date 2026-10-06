# CDTC-Net Research Showcase & Evaluation Demonstration Set

This directory (`demo/demo_samples_research_showcase/`) contains representative video samples selected from the complete held-out **FaceForensics++ c23 test benchmark (750 videos)** using the frozen **EXP-5 V2 (Class-Balanced)** model and frozen decision threshold ($\tau^* = 0.6500$).

---

## 1. Scientific Purpose & Selection Principle

The purpose of this showcase set is to provide a scientifically grounded live demonstration of CDTC-Net that illustrates **both model capabilities and known empirical limitations**:
- Demonstrates observed model behavior on authentic videos and major manipulation categories (*Deepfakes, Face2Face, NeuralTextures*).
- Explicitly features an **actual false-positive failure case** (`03_REAL_FALSE_POSITIVE_459.mp4`) from the held-out test split, illustrating how subtle studio compression and lighting shadows can correlate with elevated fake probabilities.
- Explicitly features a **near-threshold / borderline prediction** (`07_BORDERLINE_REAL_027.mp4`) that scores directly adjacent ($\Delta = -0.0006$) to the $\tau = 0.65$ decision boundary.

> **Important Research Integrity Notice:**  
> This demonstration set is **not** a metric for model performance. The canonical performance of CDTC-Net is established exclusively by the complete held-out evaluation ($N=750$ videos: $\text{TN}=96, \text{FP}=54, \text{FN}=214, \text{TP}=386$; Accuracy: $64.27\%$, Specificity: $64.00\%$, ROC-AUC: $70.12\%$). No models or thresholds were tuned on these demonstration samples.

---

## 2. Distinction of Results, Observed Behavior, and Hypotheses

In evaluating and presenting these showcase clips:
1. **Measured Results:** Exact raw logits, sigmoid probabilities $P(\text{fake})$, face detection counts, and inference latency measured directly from the frozen pipeline.
2. **Observed Model Behavior:** The model outputs low $P(\text{fake})$ for clean authentic sequences, high $P(\text{fake})$ for manipulated clips and certain shadowed authentic clips, and intermediate scores near the decision boundary.
3. **Hypotheses (Non-Causal):** Explanations regarding whether specific spatial blending, compression gradients, or high-frequency spectral cues drove an individual prediction are hypotheses. This demonstration does not establish which specific internal feature representation caused any individual prediction, as formal per-sample attribution analysis was not conducted.

---

## 3. Showcase Sample Inventory

| # | Filename | Original Source Video | Manipulation Type | Ground Truth | Frozen Prediction | Model Probability | Showcase Evaluation Role |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---|
| **1** | `01_REAL_CORRECT_032.mp4` | `data/raw/faceforensics/original/032.mp4` | Original Real | **REAL** | **`REAL`** | $P_{\text{fake}} = 4.8\%$ ($P_{\text{real}} = 95.2\%$) | **True Negative (Capability):** Observed output shows high authentic probability and low feature divergence. |
| **2** | `02_REAL_CORRECT_057.mp4` | `data/raw/faceforensics/original/057.mp4` | Original Real | **REAL** | **`REAL`** | $P_{\text{fake}} = 4.8\%$ ($P_{\text{real}} = 95.2\%$) | **True Negative (Capability):** Authentic sequence with facial movement; model maintains low fake probability. |
| **3** | `03_REAL_FALSE_POSITIVE_459.mp4` | `data/raw/faceforensics/original/459.mp4` | Original Real | **REAL** | **`FAKE`** | $P_{\text{fake}} = 93.1\%$ ($P_{\text{real}} = 6.9\%$) | **Failure Case — False Positive (Limitation):** Genuine authentic video misclassified as FAKE. Hypothesized factors include compression and shadow gradients. |
| **4** | `04_FAKE_CORRECT_DEEPFAKES_030_193.mp4` | `data/raw/faceforensics/Deepfakes/030_193.mp4` | Deepfakes | **FAKE** | **`FAKE`** | $P_{\text{fake}} = 93.3\%$ ($P_{\text{real}} = 6.7\%$) | **True Positive (Capability):** Deepfakes autoencoder manipulation; observed output is consistent with manipulation detection. |
| **5** | `05_FAKE_CORRECT_FACETOFACE_025_067.mp4` | `data/raw/faceforensics/Face2Face/025_067.mp4` | Face2Face | **FAKE** | **`FAKE`** | $P_{\text{fake}} = 92.5\%$ ($P_{\text{real}} = 7.5\%$) | **True Positive (Capability):** Face2Face expression re-enactment; observed output reflects elevated fake probability. |
| **6** | `06_FAKE_CORRECT_NEURALTEXTURES_030_193.mp4` | `data/raw/faceforensics/NeuralTextures/030_193.mp4` | NeuralTextures | **FAKE** | **`FAKE`** | $P_{\text{fake}} = 93.3\%$ ($P_{\text{real}} = 6.7\%$) | **True Positive (Capability):** NeuralTextures GAN rendering synthesis; observed output is consistent with manipulation detection. |
| **7** | `07_BORDERLINE_REAL_027.mp4` | `data/raw/faceforensics/original/027.mp4` | Original Real | **REAL** | **`REAL`** | $P_{\text{fake}} = 64.9\%$ ($P_{\text{real}} = 35.1\%$) | **Near-threshold / borderline prediction:** Authentic video scoring within $0.0006$ of the frozen $\tau = 0.65$ threshold. |

---

## 4. Machine-Readable Metadata

Full metadata including measured raw logits, execution timestamps, and frame detection statistics is saved in [`metadata.json`](./metadata.json).
